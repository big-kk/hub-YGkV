# 作业1 · 情感陪伴助手（LangGraph 版）Docker 镜像
# 单容器内同时运行：FastAPI 后端（8000）+ Streamlit 前端（8501）
FROM python:3.12-slim

# 可选：构建时指定国内 pip 镜像源加速依赖安装，例如
#   docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple -t homework1-langgraph .
ARG PIP_INDEX_URL=

WORKDIR /app

# 先复制依赖清单并安装（充分利用 Docker 层缓存：改业务代码不必重装依赖）
COPY requirements.txt ./
RUN pip install --no-cache-dir ${PIP_INDEX_URL:+--index-url "$PIP_INDEX_URL"} -r requirements.txt

# 复制应用代码（.venv / .env / __pycache__ 等已被 .dockerignore 排除，不会进镜像）
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY docker-start.sh ./
RUN chmod +x docker-start.sh

# 环境变量：默认 mock 模式开箱即用；容器内前后端通过 127.0.0.1 互访
ENV EMOTION_LLM_MODE=mock \
    BACKEND_URL=http://127.0.0.1:8000 \
    BACKEND_HOST=0.0.0.0 \
    BACKEND_PORT=8000 \
    FRONTEND_PORT=8501

EXPOSE 8000 8501

# 健康检查：探活 FastAPI 后端
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('BACKEND_PORT','8000')+'/health',timeout=3)" || exit 1

CMD ["./docker-start.sh"]
