"""数据模型：进程1（情感识别 Agent）的结构化输出。"""

from pydantic import BaseModel, Field


class SentimentResult(BaseModel):
    """情感识别 Agent 的结构化输出（output_type）"""

    sentiment: str = Field(description="情感倾向：positive / negative / neutral")
    confidence: float = Field(ge=0.0, le=1.0, description="置信度，0 到 1 之间")
    reason: str = Field(description="20 字以内的中文识别理由")
