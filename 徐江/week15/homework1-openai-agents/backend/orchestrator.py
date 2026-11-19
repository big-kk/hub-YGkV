"""后端编排：接收前端输入，串联两个 Agent（进程1 情感识别 → 进程2 分档回复）。

mock 模式下两个 Agent 由 MockModel 驱动；real 模式下换成 DeepSeek，编排逻辑不变。
"""

from agents import Runner

from . import config
from .agents_def import build_reply_agent, build_sentiment_agent
from .mock_model import FALLBACK_REPLY

SENTIMENT_CN = {"positive": "正面", "negative": "负面", "neutral": "中性"}
BRANCH_CN = {"positive": "鼓励", "negative": "排解", "neutral": "温和回应"}


async def run_pipeline(text: str) -> dict:
    trace = [f"[后端] 接收前端输入（{len(text)} 字）"]

    # ---------- 进程1：情感识别 Agent ----------
    trace.append("[后端] 调度进程1：SentimentAnalyst（情感识别 Agent）")
    try:
        r1 = await Runner.run(build_sentiment_agent(), text)
        result = r1.final_output
        sentiment = result.sentiment if result.sentiment in SENTIMENT_CN else "neutral"
        confidence = min(max(float(result.confidence), 0.0), 1.0)
        reason = str(result.reason)[:50]
        trace.append(
            f"[进程1] 识别结果 → {SENTIMENT_CN[sentiment]}（置信度 {confidence:.2f}）"
        )
    except Exception as e:
        # 兜底：进程1 异常 → 中性 + 固定话术
        trace.append(f"[进程1] Agent 调用异常（{e}）→ 归入中性并触发兜底")
        trace.append("[后端] 结果返回前端")
        return {
            "sentiment": "neutral",
            "sentiment_cn": "中性",
            "confidence": 0.0,
            "reason": f"识别失败：{e}",
            "reply": FALLBACK_REPLY,
            "fallback": True,
            "trace": trace,
            "mode": config.LLM_MODE,
        }

    # ---------- 进程2：回复 Agent ----------
    trace.append(f"[后端] 调度进程2：ResponseAgent（「{BRANCH_CN[sentiment]}」策略）")
    try:
        r2 = await Runner.run(build_reply_agent(sentiment), text)
        reply = str(r2.final_output).strip()
        trace.append(f"[进程2] 「{BRANCH_CN[sentiment]}」回复生成完毕")
        fallback = False
    except Exception as e:
        # 兜底：进程2 异常 → 固定话术
        trace.append(f"[进程2] Agent 调用异常（{e}）→ 返回兜底话术")
        reply = FALLBACK_REPLY
        fallback = True

    trace.append("[后端] 结果返回前端")
    return {
        "sentiment": sentiment,
        "sentiment_cn": SENTIMENT_CN[sentiment],
        "confidence": round(confidence, 2),
        "reason": reason,
        "reply": reply,
        "fallback": fallback,
        "trace": trace,
        "mode": config.LLM_MODE,
    }
