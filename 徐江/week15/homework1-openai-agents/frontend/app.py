"""Streamlit 前端：用户输入 → 调用 FastAPI 后端 → 展示两个 Agent 的识别结果与陪伴回复。

运行：streamlit run frontend/app.py
"""

import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8001")

SENTIMENT_STYLE = {
    "positive": ("#e6f4ea", "#1e7e34", "😊 正面"),
    "negative": ("#fdecea", "#c62828", "😔 负面"),
    "neutral": ("#fff8e1", "#9a6b00", "😐 中性"),
}


def fetch_mode(backend_url: str):
    try:
        return requests.get(f"{backend_url}/mode", timeout=5).json()
    except Exception:
        return None


def main():
    st.set_page_config(page_title="情感陪伴助手 · OpenAI Agents 版", page_icon="💬", layout="centered")
    st.title("💬 情感陪伴助手")
    st.caption("OpenAI Agents 版 ｜ 前端输入 → 后端串联「Agent1 情感识别 → Agent2 分档回复」")

    # ---------- 侧边栏 ----------
    with st.sidebar:
        st.header("⚙️ 设置")
        backend_url = st.text_input("后端地址", BACKEND_URL).rstrip("/")

        st.markdown("**运行模式**")
        mode_info = fetch_mode(backend_url)
        if mode_info is None:
            st.error("连不上后端，请先启动 FastAPI")
        elif mode_info.get("mode") == "real":
            st.success(f"🤖 真实模式：{mode_info.get('model')}")
        else:
            st.info("🧪 Mock 模式：MockModel 本地模拟，无需 API Key")

        st.markdown("---")
        st.markdown("**流程架构**")
        st.markdown(
            "- 前端（Streamlit）：用户输入\n"
            "- 后端（FastAPI）：接收并调度 Agent\n"
            "- 进程1 Agent：SentimentAnalyst，结构化输出情感结果\n"
            "- 进程2 Agent：ResponseAgent，正面鼓励 / 负面排解 / 中性温和回应"
        )

    # ---------- 输入区 ----------
    text = st.text_area(
        "说说你现在的心情，或最近遇到的事：",
        height=120,
        placeholder="例如：今天和女朋友吵架了，心情很差",
    )

    if st.button("发送", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("请输入一些内容再发送～")
            return
        with st.spinner("正在倾听你的话..."):
            try:
                resp = requests.post(f"{backend_url}/chat", json={"text": text}, timeout=60)
            except requests.exceptions.ConnectionError:
                st.error("❌ 连接后端失败：请先运行 `uvicorn backend.main:app --port 8001`")
                return
            except requests.exceptions.Timeout:
                st.error("❌ 后端响应超时，请稍后重试")
                return

        if resp.status_code != 200:
            detail = resp.json().get("detail", resp.text)
            st.error(f"⚠️ {detail}")
            return

        data = resp.json()

        # ---------- 识别结果 ----------
        bg, fg, label = SENTIMENT_STYLE.get(data["sentiment"], SENTIMENT_STYLE["neutral"])
        st.markdown("### 🔍 进程1 Agent · 情感识别结果")
        st.markdown(
            f'<div style="background:{bg};color:{fg};padding:12px 16px;border-radius:10px;'
            f'font-size:18px;font-weight:600">{label}</div>',
            unsafe_allow_html=True,
        )
        st.progress(data["confidence"], text=f"置信度：{data['confidence']:.2f}")
        st.caption(f"识别理由：{data['reason']}")

        # ---------- 最终回复 ----------
        st.markdown("### 💗 进程2 Agent · 陪伴回复")
        if data.get("fallback"):
            st.warning("⚠️ 模型服务异常，本次为兜底安抚话术")
        st.info(data["reply"])

        # ---------- 流程 trace ----------
        with st.expander("🔎 查看完整流程 trace"):
            for i, step in enumerate(data["trace"], 1):
                st.markdown(f"{i}. `{step}`")


if __name__ == "__main__":
    main()
