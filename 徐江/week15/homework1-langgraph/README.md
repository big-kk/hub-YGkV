# 作业1 · 情感陪伴助手（LangGraph 版）

用 **LangGraph** 实现“情感识别 → 分档回复”两段式流程，配 FastAPI 后端 + Streamlit 前端。

## 一、作业要求与本项目的映射

| 作业要求 | 实现 |
|---|---|
| 进程1：接受请求进行情感识别 | LangGraph 节点 `sentiment_analysis`（`backend/graph.py`） |
| 进程2：正向鼓励 / 负面排解 | LangGraph 节点 `generate_reply`，按 `sentiment` 分三条策略（正面鼓励、负面排解、中性温和回应） |
| 后端进程：接受前端输入，串联进程1和进程2 | FastAPI（`backend/main.py`）+ `run_pipeline()` 调用编译后的图 |
| 前端进程：用户输入 | Streamlit（`frontend/app.py`） |

数据流：

```
Streamlit 前端 ──POST /chat──> FastAPI 后端
                                   │
                                   ▼
                        LangGraph 图（StateGraph）
                        START → sentiment_analysis（进程1）
                              → generate_reply（进程2）
                              → END
                                   │
                                   ▼
                     {sentiment, confidence, reason, reply, trace}
```

## 二、目录结构

```
homework1-langgraph/
├── backend/
│   ├── __init__.py
│   ├── config.py        # 配置：mock/real 模式、DeepSeek、端口
│   ├── llm.py           # 情感识别 + 回复生成（mock 规则 / DeepSeek 双模式）
│   ├── graph.py         # LangGraph：EmotionState + 两个节点 + 编译图
│   └── main.py          # FastAPI：POST /chat、GET /mode、GET /health
├── frontend/
│   └── app.py           # Streamlit 前端
├── Dockerfile           # Docker 镜像构建文件
├── docker-start.sh      # 容器入口：同时启动前后端
├── docker-compose.yml   # docker compose 编排
├── .dockerignore        # 排除 .venv / .env 等不入镜像的文件
├── requirements.txt
├── .env.example
└── README.md
```

## 三、快速开始（Mock 模式，无需 API Key）

```bash
cd homework1-langgraph
python -m venv .venv
.venv/Scripts/activate          # Windows；macOS/Linux 用 source .venv/bin/activate
pip install -r requirements.txt

# 终端1：启动后端（默认 8000 端口）
uvicorn backend.main:app --port 8000

# 终端2：启动前端（默认 8501 端口）
streamlit run frontend/app.py
```

浏览器打开 http://localhost:8501 ，输入文字后点击“发送”，可以看到：
- 进程1 的识别结果（正面/负面/中性 + 置信度 + 理由）
- 进程2 的陪伴回复
- 完整流程 trace（展开“查看完整流程 trace”）

## 四、切换到 DeepSeek 真实模型

1. 复制 `.env.example` 为 `.env`
2. 到 [DeepSeek 开放平台](https://platform.deepseek.com) 申请 API Key
3. 修改 `.env`：

```env
EMOTION_LLM_MODE=real
DEEPSEEK_API_KEY=sk-你的key
DEEPSEEK_MODEL=deepseek-chat
```

4. 重启后端即可。mock/real 只影响 `backend/llm.py` 内部的模型调用，图结构、节点、接口完全不变。前端右上角会显示当前模式。

## 五、接口说明

`POST /chat`

请求：
```json
{"text": "最近压力好大，天天失眠"}
```

响应：
```json
{
  "sentiment": "negative",
  "sentiment_cn": "负面",
  "confidence": 0.95,
  "reason": "检测到消极词：压力、失眠",
  "reply": "抱抱你。「最近压力好大，天天失眠」——心里一定挺难受的吧？……",
  "fallback": false,
  "trace": ["[后端] 接收前端输入（12 字）", "[进程1] 情感识别完成 → 负面（置信度 0.95）", "……"],
  "mode": "mock"
}
```

## 六、兜底策略（全场景）

| 场景 | 行为 |
|---|---|
| 空输入 / 超长输入（>500字） | 后端返回 400，前端提示 |
| 模型调用异常（网络、超时、无 key） | 归入中性 + 返回固定安抚话术，`fallback=true` |
| 情感无法判断 | 归入中性，走“温和回应”策略 |
| 回复生成异常 | 返回固定安抚话术，`fallback=true` |

## 七、Docker 打包与运行

项目根目录已提供 `Dockerfile`、`docker-start.sh`、`.dockerignore`、`docker-compose.yml`。镜像采用**单容器设计**：容器内同时运行 FastAPI 后端（8000）和 Streamlit 前端（8501），由入口脚本统一管理。

### 1. 构建镜像

```bash
cd homework1-langgraph
docker build -t homework1-langgraph .
# 国内网络下载依赖慢时，可指定 pip 镜像源：
# docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple -t homework1-langgraph .
```

### 2. 运行容器

```bash
docker run -d -p 8000:8000 -p 8501:8501 --name homework1-langgraph homework1-langgraph
```

浏览器访问 http://localhost:8501 （8000 为后端 API 端口）。

### 3. 使用 docker compose（推荐）

```bash
docker compose up --build -d   # 构建并后台启动
docker compose logs -f         # 查看日志
docker compose down            # 停止并删除容器
```

### 4. 切换 DeepSeek 真实模型

```bash
docker run -d -p 8000:8000 -p 8501:8501 \
  -e EMOTION_LLM_MODE=real \
  -e DEEPSEEK_API_KEY=sk-你的key \
  homework1-langgraph
```

### 镜像说明

| 项 | 说明 |
|---|---|
| 基础镜像 | `python:3.12-slim` |
| 暴露端口 | 8000（后端 API）、8501（前端页面） |
| 进程管理 | `docker-start.sh` 同时拉起 uvicorn 与 streamlit，任一进程退出容器即停止（配合 `restart: unless-stopped` 自动重启） |
| 健康检查 | 每 30s 请求 `/health`，不健康可被编排系统感知 |
| 不入镜像 | `.venv`、`.env`、`__pycache__`（见 `.dockerignore`）；镜像内依赖由 `requirements.txt` 全新安装，key 通过 `-e` 或 compose 的 `environment` 注入，不写进镜像 |
