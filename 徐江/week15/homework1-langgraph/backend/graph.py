"""LangGraph 情感处理图。

两个节点对应作业中的两个「进程」：
- sentiment_analysis：进程1，接受请求进行情感识别
- generate_reply    ：进程2，根据识别结果分档回复（正面鼓励 / 负面排解 / 中性温和回应）

数据流：START → sentiment_analysis → generate_reply → END
"""

from typing import Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from . import config, llm

SENTIMENT_CN = {"positive": "正面", "negative": "负面", "neutral": "中性"}
BRANCH_CN = {"positive": "鼓励", "negative": "排解", "neutral": "温和回应"}


class EmotionState(TypedDict, total=False):
    input_text: str          # 用户输入
    sentiment: str           # positive / negative / neutral
    confidence: float        # 置信度 0~1
    reason: str              # 识别理由
    reply: str               # 最终回复
    fallback: bool           # 是否触发兜底
    trace: list[str]         # 流程记录（用于前端展示串联过程）


# ---------- 进程1：情感识别 ----------
def sentiment_analysis_node(state: EmotionState) -> EmotionState:
    text = state["input_text"]
    trace = list(state.get("trace", []))
    try:
        result = llm.analyze_sentiment(text)
        sentiment = result["sentiment"]
        trace.append(
            f"[进程1] 情感识别完成 → {SENTIMENT_CN[sentiment]}"
            f"（置信度 {result['confidence']:.2f}）"
        )
        return {
            "sentiment": sentiment,
            "confidence": result["confidence"],
            "reason": result["reason"],
            "trace": trace,
        }
    except llm.LLMError as e:
        # 模型异常：归入中性并标记兜底，保证链路不断
        trace.append(f"[进程1] 模型调用异常（{e}）→ 归入中性并触发兜底")
        return {
            "sentiment": "neutral",
            "confidence": 0.0,
            "reason": f"识别失败：{e}",
            "fallback": True,
            "trace": trace,
        }


# ---------- 进程2：分档回复 ----------
def generate_reply_node(state: EmotionState) -> EmotionState:
    sentiment = state.get("sentiment", "neutral")
    trace = list(state.get("trace", []))
    trace.append(f"[进程2] 进入回复生成，命中「{BRANCH_CN[sentiment]}」策略")

    if state.get("fallback"):
        reply = llm.FALLBACK_REPLY
        trace.append("[进程2] 兜底模式 → 返回固定安抚话术")
    else:
        try:
            reply = llm.generate_reply(state["input_text"], sentiment)
            trace.append(f"[进程2] 「{BRANCH_CN[sentiment]}」回复生成完毕")
        except llm.LLMError as e:
            reply = llm.FALLBACK_REPLY
            trace.append(f"[进程2] 回复模型异常（{e}）→ 返回兜底话术")
            return {"reply": reply, "fallback": True, "trace": trace}

    return {"reply": reply, "trace": trace}


def build_graph():
    graph = StateGraph(EmotionState)
    graph.add_node("sentiment_analysis", sentiment_analysis_node)  # 进程1
    graph.add_node("generate_reply", generate_reply_node)          # 进程2
    graph.add_edge(START, "sentiment_analysis")
    graph.add_edge("sentiment_analysis", "generate_reply")
    graph.add_edge("generate_reply", END)
    return graph.compile()


emotion_graph = build_graph()


def run_pipeline(text: str) -> dict:
    """后端串联入口：接收前端输入，跑完整图，返回结果。"""
    trace = [f"[后端] 接收前端输入（{len(text)} 字）"]
    result = emotion_graph.invoke(
        {"input_text": text, "sentiment": "", "confidence": 0.0,
         "reason": "", "reply": "", "fallback": False, "trace": trace}
    )
    result["trace"].append("[后端] 结果返回前端")
    return {
        "sentiment": result["sentiment"],
        "sentiment_cn": SENTIMENT_CN[result["sentiment"]],
        "confidence": result["confidence"],
        "reason": result["reason"],
        "reply": result["reply"],
        "fallback": result["fallback"],
        "trace": result["trace"],
        "mode": config.LLM_MODE,
    }
