# 作业1 · 情感陪伴助手（OpenAI Agents 版）

用 **OpenAI Agents SDK** 实现同样的“情感识别 → 分档回复”流程，配 FastAPI 后端 + Streamlit 前端。与 `homework1-langgraph` 逻辑一致，编排方式不同。

## 一、作业要求与本项目的映射

| 作业要求 | 实现 |
|---|---|
| 进程1：接受请求进行情感识别 | `SentimentAnalyst` Agent，`output_type=SentimentResult` 结构化输出 |
| 进程2：正向鼓励 / 负面排解 | `ResponseAgent` Agent，按进程1 结果动态注入「鼓励/排解/温和回应」策略指令 |
| 后端进程：接受前端输入，串联进程1和进程2 | FastAPI（`backend/main.py`）+ `orchestrator.py` 依次 `Runner.run()` 两个 Agent |
| 前端进程：用户输入 | Streamlit（`frontend/app.py`） |

数据流：

```
Streamlit 前端 ──POST /chat──> FastAPI 后端
                                   │
                                   ▼
                    Runner.run(SentimentAnalyst, 用户输入)      # 进程1
                        → SentimentResult(sentiment, confidence, reason)
                                   │
                                   ▼
                    Runner.run(ResponseAgent, 用户输入)          # 进程2
                        （instructions 按识别结果注入对应策略）
                        → 陪伴回复
```

## 二、目录结构

```
homework1-openai-agents/
├── backend/
│   ├── __init__.py
│   ├── config.py          # 配置：mock/real 模式、DeepSeek、端口
│   ├── schemas.py         # SentimentResult（进程1 的结构化输出模型）
│   ├── mock_model.py      # MockModel：实现 SDK Model 接口，mock 模式零 API 依赖
│   ├── agents_def.py      # 两个 Agent 的定义 + 模型选择（MockModel / DeepSeek）
│   ├── orchestrator.py    # 后端编排：串联两个 Agent + trace + 兜底
│   └── main.py            # FastAPI：POST /chat、GET /mode、GET /health
├── frontend/
│   └── app.py             # Streamlit 前端
├── Dockerfile             # Docker 镜像构建文件
├── docker-start.sh        # 容器入口：同时启动前后端
├── docker-compose.yml     # docker compose 编排
├── .dockerignore          # 排除 .venv / .env 等不入镜像的文件
├── requirements.txt
├── .env.example
└── README.md
```

## 三、快速开始（Mock 模式，无需 API Key）

```bash
cd homework1-openai-agents
python -m venv .venv
.venv/Scripts/activate          # Windows；macOS/Linux 用 source .venv/bin/activate
pip install -r requirements.txt

# 终端1：启动后端（默认 8001 端口，避免与 LangGraph 版的 8000 冲突）
uvicorn backend.main:app --port 8001

# 终端2：启动前端（默认 8502 端口）
streamlit run frontend/app.py
```

浏览器打开 http://localhost:8502 。

> Mock 模式下两个 Agent 依然走完整的 Agents SDK 链路（Agent 定义 → Runner 运行 → 结构化输出校验 → final_output），只是底层模型换成了本地实现的 `MockModel`（`backend/mock_model.py`），因此不调用任何 API 也能演示 SDK 的全部机制。

## 四、切换到 DeepSeek 真实模型

1. 复制 `.env.example` 为 `.env`
2. 到 [DeepSeek 开放平台](https://platform.deepseek.com) 申请 API Key
3. 修改 `.env`：

```env
EMOTION_LLM_MODE=real
DEEPSEEK_API_KEY=sk-你的key
DEEPSEEK_MODEL=deepseek-chat
```

4. 重启后端。`backend/agents_def.py` 中的 `_get_model()` 会自动改用 `OpenAIChatCompletionsModel`（chat completions 接口，DeepSeek 兼容），Agent 定义与编排逻辑不变。

## 五、接口说明

`POST /chat`，请求/响应格式与 LangGraph 版完全一致：

```json
// 请求
{"text": "今天考试成绩出来了，全部通过，好开心！"}

// 响应
{
  "sentiment": "positive",
  "sentiment_cn": "正面",
  "confidence": 0.79,
  "reason": "检测到积极词：开心",
  "reply": "哇，恭喜你！……",
  "fallback": false,
  "trace": ["[后端] 接收前端输入（19 字）", "[后端] 调度进程1：SentimentAnalyst（情感识别 Agent）", "[进程1] 识别结果 → 正面（置信度 0.79）", "[后端] 调度进程2：ResponseAgent（「鼓励」策略）", "[进程2] 「鼓励」回复生成完毕", "[后端] 结果返回前端"],
  "mode": "mock"
}
```

## 六、与 LangGraph 版的差异

| 维度 | LangGraph 版 | OpenAI Agents 版 |
|---|---|---|
| 进程抽象 | 图节点（函数） | Agent（instructions + model + output_type） |
| 串联方式 | `StateGraph` 边 + 共享 State | `Runner.run()` 依次调用，结果在代码中传递 |
| 结构化输出 | 节点内自行解析 JSON | SDK 的 `output_type` 自动校验解析 |
| 中间状态 | State 字典贯穿全图 | 每次 `Runner.run` 独立，trace 由后端记录 |
| 扩展方式 | 加节点/条件边 | 加 Agent/handoff/tool |

## 七、兜底策略（全场景）

| 场景 | 行为 |
|---|---|
| 空输入 / 超长输入（>500字） | 后端返回 400，前端提示 |
| Agent 调用异常（网络、超时、无 key） | 归入中性 + 返回固定安抚话术，`fallback=true` |
| 情感无法判断 | 归入中性，走“温和回应”策略 |
| 回复生成异常 | 返回固定安抚话术，`fallback=true` |

## 八、Docker 打包与运行

项目根目录已提供 `Dockerfile`、`docker-start.sh`、`.dockerignore`、`docker-compose.yml`。镜像采用**单容器设计**：容器内同时运行 FastAPI 后端（8001）和 Streamlit 前端（8502），由入口脚本统一管理。

### 1. 构建镜像

```bash
cd homework1-openai-agents
docker build -t homework1-openai-agents .
# 国内网络下载依赖慢时，可指定 pip 镜像源：
# docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple -t homework1-openai-agents .
```

### 2. 运行容器

```bash
docker run -d -p 8001:8001 -p 8502:8502 --name homework1-openai-agents homework1-openai-agents
```

浏览器访问 http://localhost:8502 （8001 为后端 API 端口）。

### 3. 使用 docker compose（推荐）

```bash
docker compose up --build -d   # 构建并后台启动
docker compose logs -f         # 查看日志
docker compose down            # 停止并删除容器
```

### 4. 切换 DeepSeek 真实模型

```bash
docker run -d -p 8001:8001 -p 8502:8502 \
  -e EMOTION_LLM_MODE=real \
  -e DEEPSEEK_API_KEY=sk-你的key \
  homework1-openai-agents
```

### 镜像说明

| 项 | 说明 |
|---|---|
| 基础镜像 | `python:3.12-slim` |
| 暴露端口 | 8001（后端 API）、8502（前端页面） |
| 进程管理 | `docker-start.sh` 同时拉起 uvicorn 与 streamlit，任一进程退出容器即停止（配合 `restart: unless-stopped` 自动重启） |
| 健康检查 | 每 30s 请求 `/health`，不健康可被编排系统感知 |
| 不入镜像 | `.venv`、`.env`、`__pycache__`（见 `.dockerignore`）；镜像内依赖由 `requirements.txt` 全新安装，key 通过 `-e` 或 compose 的 `environment` 注入，不写进镜像 |
