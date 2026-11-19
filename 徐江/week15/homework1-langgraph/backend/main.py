"""FastAPI 后端：接收前端请求，串联 LangGraph 情感图的两个进程。"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import config
from .graph import run_pipeline

app = FastAPI(title="情感陪伴助手（LangGraph 版）", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    text: str = Field(..., description="用户输入的文本")


class ChatResponse(BaseModel):
    sentiment: str
    sentiment_cn: str
    confidence: float
    reason: str
    reply: str
    fallback: bool
    trace: list[str]
    mode: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/mode")
def mode():
    return config.config_summary()


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    text = (req.text or "").strip()

    # ---- 全场景兜底 1：输入校验 ----
    if not text:
        raise HTTPException(status_code=400, detail="输入不能为空，请先说点什么吧～")
    if len(text) > config.MAX_INPUT_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"输入过长（超过 {config.MAX_INPUT_LEN} 字），请精简后再发送",
        )

    # ---- 串联进程1（情感识别）与进程2（分档回复） ----
    return run_pipeline(text)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.BACKEND_HOST, port=config.BACKEND_PORT)
