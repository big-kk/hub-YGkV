"""两个 Agent 的定义与模型选择。

进程1：SentimentAnalyst（情感识别 Agent），output_type=SentimentResult 结构化输出
进程2：ResponseAgent（回复 Agent），根据识别结果动态注入「鼓励 / 排解 / 温和回应」策略指令
"""

from agents import Agent, OpenAIChatCompletionsModel

from . import config
from .mock_model import MockModel
from .schemas import SentimentResult

SENTIMENT_INSTRUCTIONS = """你是一个情感分析引擎，擅长识别用户文本中的情感倾向。

工作要求：
1. 只输出 JSON，不要输出任何其他内容（不要 Markdown 代码块）。
2. JSON 格式：{"sentiment": "positive|negative|neutral", "confidence": 0到1之间的小数, "reason": "20字以内的中文理由"}
3. sentiment 取值：positive（正面）/ negative（负面）/ neutral（中性或情绪不明显）
4. 无法判断时归入 neutral。"""

RESPONSE_INSTRUCTIONS_TEMPLATE = """你是一位温暖、真诚的情感陪伴助手，像一位善于倾听的朋友。

当前用户的情感识别结果为「{sentiment_cn}」，你必须执行「{strategy}」策略：
{strategy_text}

请直接输出给用户的回复内容，不要输出任何解释。"""

STRATEGY_TEXT = {
    "positive": "为用户感到高兴，一起庆祝，语气热情真诚，100 字以内。",
    "negative": "先共情接纳用户的感受，再温和安抚，可给 1-2 个简单易行的小建议，不说教、不套话，120 字以内。",
    "neutral": "温和自然地回应，表达关心并邀请用户继续分享，80 字以内。",
}

STRATEGY_CN = {"positive": "鼓励", "negative": "排解", "neutral": "温和回应"}


def _get_model():
    """mock 模式用 MockModel；real 模式用 DeepSeek（chat completions 兼容接口）。"""
    if config.LLM_MODE == "real":
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
            timeout=config.REQUEST_TIMEOUT,
        )
        return OpenAIChatCompletionsModel(model=config.DEEPSEEK_MODEL, openai_client=client)
    return MockModel()


def build_sentiment_agent() -> Agent:
    """进程1：情感识别 Agent"""
    return Agent(
        name="SentimentAnalyst",
        instructions=SENTIMENT_INSTRUCTIONS,
        model=_get_model(),
        output_type=SentimentResult,
    )


def build_reply_agent(sentiment: str) -> Agent:
    """进程2：回复 Agent（根据进程1 的识别结果注入对应策略）"""
    return Agent(
        name="ResponseAgent",
        instructions=RESPONSE_INSTRUCTIONS_TEMPLATE.format(
            sentiment_cn={"positive": "正面", "negative": "负面", "neutral": "中性"}[sentiment],
            strategy=STRATEGY_CN[sentiment],
            strategy_text=STRATEGY_TEXT[sentiment],
        ),
        model=_get_model(),
    )
