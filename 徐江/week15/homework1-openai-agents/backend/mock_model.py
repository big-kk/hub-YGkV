"""MockModel：实现 Agents SDK 的 Model 接口，在 mock 模式下本地模拟大模型。

作用：不调用真实 API，也能完整跑通「Agent 定义 → Runner 运行 → 结构化输出」的 SDK 全流程。
- 若 Agent 声明了 output_type（进程1 情感识别 Agent）→ 返回 JSON 文本消息，
  SDK 会自动按 output_type 校验并解析为 SentimentResult 实例；
- 若 Agent 无 output_type（进程2 回复 Agent）→ 根据 system_instructions 中的策略
  （鼓励/排解/温和回应）返回对应的模板回复。

切换到 real 模式后，两个 Agent 改用的就是 DeepSeek 真实模型，其余代码不变。
"""

import hashlib
import json
import re

from openai.types.responses import (
    ResponseOutputMessage,
    ResponseOutputText,
)

from agents.items import ModelResponse
from agents.models.interface import Model, ModelTracing
from agents.usage import Usage

# ==================== mock：关键词情感规则（与 LangGraph 版相同的规则） ====================

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

# 全场景兜底：模型调用异常时返回的固定安抚话术
FALLBACK_REPLY = (
    "抱歉，我现在有点不在状态，没能好好回应你。"
    "你可以先喝口水、休息一下，稍后再和我聊聊，我一直都在。"
)


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


def _mock_reply(text: str, instructions: str) -> str:
    # 从指令中解析当前执行的策略（指令形如：你必须执行「排解」策略）
    m = re.search(r"执行「([^」]+)」策略", instructions)
    strategy = m.group(1) if m else ""
    if strategy == "排解":
        table = NEGATIVE_REPLIES
    elif strategy == "鼓励":
        table = POSITIVE_REPLIES
    else:
        table = NEUTRAL_REPLIES
    idx = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % len(table)
    topic = f"「{text.strip()[:18]}」——" if text.strip() else ""
    return table[idx].format(topic=topic)


def _extract_user_text(input_items) -> str:
    """从 Responses 格式的输入中提取用户文本"""
    if isinstance(input_items, str):
        return input_items
    texts = []
    for item in input_items or []:
        content = item.get("content") if isinstance(item, dict) else None
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.extend(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") in ("input_text", "output_text", "text")
            )
    return "\n".join(t for t in texts if t).strip()


def _build_message(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        type="message",
        id=f"msg_mock_{abs(hash(text)) % 10**8}",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(text=text, type="output_text", annotations=[], logprobs=[])],
    )


class MockModel(Model):
    """本地模拟模型：按 Agent 角色返回确定性结果，零 API 依赖。"""

    async def get_response(
        self,
        system_instructions: str | None,
        input,
        model_settings,
        tools,
        output_schema,
        handoffs,
        tracing: ModelTracing,
        *,
        previous_response_id=None,
        conversation_id=None,
        prompt=None,
    ) -> ModelResponse:
        text = _extract_user_text(input)

        if output_schema is not None and not output_schema.is_plain_text():
            # 进程1 情感识别 Agent：返回 JSON，SDK 自动解析为 SentimentResult
            payload = json.dumps(mock_sentiment(text), ensure_ascii=False)
        else:
            # 进程2 回复 Agent：按 instructions 中的策略返回模板回复
            payload = _mock_reply(text, system_instructions or "")

        return ModelResponse(
            output=[_build_message(payload)],
            usage=Usage(),
            response_id=f"resp_mock_{abs(hash(payload)) % 10**8}",
        )

    async def stream_response(self, *args, **kwargs):
        raise NotImplementedError("MockModel 不支持流式输出")
