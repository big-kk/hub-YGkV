import os

from dotenv import load_dotenv

load_dotenv()

# ============ 运行模式 ============
# mock : MockModel 本地模拟两个 Agent，无需 API Key，可完整跑通 Agents SDK 全流程
# real : DeepSeek（OpenAI 兼容接口，chat completions 模式），需配置 DEEPSEEK_API_KEY
LLM_MODE = os.getenv("EMOTION_LLM_MODE", "mock").strip().lower()
if LLM_MODE not in ("mock", "real"):
    LLM_MODE = "mock"

# ============ DeepSeek 配置（real 模式生效） ============
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "30"))

# ============ 服务配置 ============
BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8001"))
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "8502"))

MAX_INPUT_LEN = 500


def config_summary() -> dict:
    return {
        "mode": LLM_MODE,
        "model": DEEPSEEK_MODEL if LLM_MODE == "real" else "MockModel（本地模拟）",
        "api_key_configured": bool(DEEPSEEK_API_KEY) if LLM_MODE == "real" else None,
        "base_url": DEEPSEEK_BASE_URL if LLM_MODE == "real" else None,
    }
