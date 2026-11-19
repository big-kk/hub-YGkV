"""LLM 客户端：统一封装「情感识别」和「生成回复」两种能力。

- mock 模式：本地关键词规则 + 模板回复，零依赖跑通全流程
- real 模式：调用 DeepSeek（OpenAI 兼容接口）
"""

import hashlib
import json

from . import config


class LLMError(Exception):
    """模型调用异常（网络错误、超时、返回无法解析等）"""


# ==================== mock：关键词情感规则 ====================

POSITIVE_WORDS = {
    "开心": 2, "高兴": 2, "快乐": 2, "幸福": 2, "满意": 1, "喜欢": 1,
    "感谢": 1, "温暖": 1, "顺利": 1, "不错": 1, "加油": 1, "期待": 1,
    "放松": 1, "舒服": 1, "美好": 2, "成功": 2, "晋升": 2, "录取": 2,
    "上岸": 2, "过啦": 2, "棒": 2, "惊喜": 2, "甜": 1, "笑": 1,
    "阳光": 1, "治愈": 1, "被爱": 2, "拥抱": 1, "奖": 1, "满分": 2,
}
NEGATIVE_WORDS = {
    "难过": 2, "伤心": 2, "崩溃": 3, "焦虑": 2, "抑郁": 3, "累": 2,
    "疲惫": 2, "烦": 2, "讨厌": 2, "痛苦": 3, "孤独": 2, "想哭": 2,
    "哭": 2, "失败": 2, "挂科": 2, "失业": 3, "分手": 2, "吵架": 2,
    "压力": 2, "emo": 2, "低落": 2, "委屈": 2, "生气": 2, "愤怒": 3,
    "害怕": 2, "恐惧": 2, "迷茫": 1, "无助": 2, "失眠": 2, "确诊": 2,
    "不顺": 2, "搞砸": 2, "后悔": 2, "想放弃": 3, "堵": 1, "糟": 2,
}

POSITIVE_REPLIES = [
    "看到你状态这么好，真替你开心！{topic}这份积极的能量很珍贵，记得也把这份光带给身边的人。继续加油，我一直在～",
    "哇，恭喜你！{topic}这种好事值得好好庆祝一下～保持这份热情和节奏，你比自己想象中更有力量。",
    "读着你的文字都能感受到那份开心，太棒了！{topic}愿你之后的日子也一直这样顺顺利利、闪闪发光。",
]
NEGATIVE_REPLIES = [
    "抱抱你。{topic}心里一定挺难受的吧？难过的时候先别硬撑，允许自己休息一下。要不要跟我说说发生了什么？我会一直在这里听。",
    "听起来你真的不容易。{topic}别急着责怪自己，每个人都会有难熬的时刻。先深呼吸几次，喝口温水，我们慢慢来，我陪你一起面对。",
    "你的感受是真实的，也值得被认真对待。{topic}如果现在很累，就先放下一切好好睡一觉，天大的事睡醒再说。我陪着你。",
]
NEUTRAL_REPLIES = [
    "嗯，我在认真听。{topic}如果心里还有别的想法，无论是开心还是难过，都可以慢慢讲给我听。",
    "收到～{topic}今天过得还顺利吗？不管是想分享还是想吐槽，我都在。",
    "好的，我记下了。{topic}随时想聊点什么都可以，我在这儿陪着你。",
]
FALLBACK_REPLY = (
    "抱歉，我现在有点不在状态，没能好好回应你。"
    "你可以先喝口水、休息一下，稍后再和我聊聊，我一直都在。"
)


def _pick_variant(variants: list[str], text: str) -> str:
    idx = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % len(variants)
    return variants[idx]


def _mock_topic(text: str) -> str:
    """从输入中截取一小段作为回复中的引用，增加亲切感"""
    t = text.strip().replace("\n", " ")
    return f"「{t[:18]}」——" if t else ""


def mock_sentiment(text: str) -> dict:
    pos_hits = [w for w in POSITIVE_WORDS if w in text]
    neg_hits = [w for w in NEGATIVE_WORDS if w in text]
    pos_score = sum(POSITIVE_WORDS[w] for w in pos_hits)
    neg_score = sum(NEGATIVE_WORDS[w] for w in neg_hits)
    score = pos_score - neg_score

    if score > 0:
        sentiment = "positive"
        confidence = min(0.95, 0.55 + 0.12 * score)
        reason = f"检测到积极词：{'、'.join(pos_hits)}"
    elif score < 0:
        sentiment = "negative"
        confidence = min(0.95, 0.55 + 0.12 * abs(score))
        reason = f"检测到消极词：{'、'.join(neg_hits)}"
    else:
        sentiment = "neutral"
        confidence = 0.5
        reason = "未检测到明显情绪词，情绪平稳"

    return {"sentiment": sentiment, "confidence": round(confidence, 2), "reason": reason}


def mock_reply(text: str, sentiment: str) -> str:
    topic = _mock_topic(text)
    table = {
        "positive": POSITIVE_REPLIES,
        "negative": NEGATIVE_REPLIES,
        "neutral": NEUTRAL_REPLIES,
    }
    return _pick_variant(table[sentiment], text).format(topic=topic)


# ==================== real：DeepSeek（OpenAI 兼容接口） ====================

def _get_client():
    if not config.DEEPSEEK_API_KEY:
        raise LLMError("real 模式未配置 DEEPSEEK_API_KEY")
    from openai import OpenAI

    return OpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
        timeout=config.REQUEST_TIMEOUT,
    )


def real_sentiment(text: str) -> dict:
    client = _get_client()
    prompt = (
        "你是情感分析引擎。分析用户文本的情感倾向，只输出 JSON，不要输出其他内容。\n"
        '输出格式：{"sentiment": "positive|negative|neutral", '
        '"confidence": 0到1之间的小数, "reason": "20字以内的中文理由"}\n'
        f"待分析文本：{text}"
    )
    try:
        resp = client.chat.completions.create(
            model=config.DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        data = json.loads(resp.choices[0].message.content)
        sentiment = str(data.get("sentiment", "")).lower()
        if sentiment not in ("positive", "negative", "neutral"):
            sentiment = "neutral"  # 情感无法判断 → 归中性
        confidence = float(data.get("confidence", 0.5))
        confidence = min(max(confidence, 0.0), 1.0)
        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 2),
            "reason": str(data.get("reason", "模型未给出理由"))[:50],
        }
    except LLMError:
        raise
    except Exception as e:
        raise LLMError(f"情感识别模型调用失败: {e}") from e


def real_reply(text: str, sentiment: str) -> str:
    client = _get_client()
    strategy = {
        "positive": "用户表达了积极情绪。请用热情、真诚的语言为其高兴和鼓励，可适当一起庆祝，100字以内。",
        "negative": "用户表达了消极情绪。请先共情接纳他的感受，再温和安抚，可给1-2个简单易行的小建议，不要说教，不套话，120字以内。",
        "neutral": "用户情绪平和。请温和自然地回应，表达关心并邀请他继续分享，80字以内。",
    }[sentiment]
    try:
        resp = client.chat.completions.create(
            model=config.DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": "你是一位温暖、真诚的情感陪伴助手，像一位善于倾听的朋友。"},
                {"role": "user", "content": f"{strategy}\n用户原文：{text}"},
            ],
            temperature=0.8,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        raise LLMError(f"回复生成模型调用失败: {e}") from e


# ==================== 统一入口（按模式路由） ====================

def analyze_sentiment(text: str) -> dict:
    if config.LLM_MODE == "real":
        return real_sentiment(text)
    return mock_sentiment(text)


def generate_reply(text: str, sentiment: str) -> str:
    if config.LLM_MODE == "real":
        return real_reply(text, sentiment)
    return mock_reply(text, sentiment)
