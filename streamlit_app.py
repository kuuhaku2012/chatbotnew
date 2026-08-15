"""Public Streamlit demo entrypoint.

The UI calls the chatbot pipeline in-process so the deployment needs only one
Streamlit service (no separate localhost FastAPI server).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "app"))

# The public deployment uses a platform-independent NumPy index. Chroma's
# persisted HNSW files were built on Windows and may fail when queried on the
# Linux workers used by Streamlit Community Cloud.
os.environ.setdefault("CHATBOT_PORTABLE_RETRIEVAL", "1")

# Community Cloud exposes values entered in App settings > Secrets through
# st.secrets. Keep local .env support in app.main for development.
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except FileNotFoundError:
    pass

from main import process_chat  # noqa: E402


st.set_page_config(
    page_title="Trợ lý Công an xã An Viễn",
    page_icon="👮",
    layout="centered",
)

st.title("👮 Trợ lý Công an xã An Viễn")
st.caption(
    "Hỗ trợ tra cứu thủ tục hành chính, an toàn PCCC và thông tin pháp luật. "
    "Bản demo không thay thế hướng dẫn của cơ quan có thẩm quyền."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Câu hỏi gợi ý")
    samples = [
        "Đăng ký tạm trú cần giấy tờ gì?",
        "Khi có nhiều khói trong nhà thì thoát ra bằng cách nào?",
        "Muốn hỏi cán bộ phụ trách PCCC tại An Viễn thì gọi ai?",
        "Bình chữa cháy nặng hơn 18 kg được đặt cao tối đa bao nhiêu?",
    ]
    selected_sample = None
    for index, sample in enumerate(samples):
        if st.button(sample, key=f"sample_{index}", use_container_width=True):
            selected_sample = sample
    if st.button("Xóa cuộc trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

typed_prompt = st.chat_input("Nhập câu hỏi của anh/chị...")
prompt = typed_prompt or selected_sample

if prompt:
    prompt = prompt.strip()[:2000]
    history = st.session_state.messages[-12:]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu..."):
            try:
                result = process_chat(prompt, history=history)
                answer = result.get("response") or "Không nhận được phản hồi."
                st.markdown(answer)
            except Exception:
                # Keep technical details out of the public UI, but preserve the
                # traceback in Streamlit's protected app logs for diagnosis.
                import logging

                logging.exception("Chat request failed")
                answer = (
                    "Hệ thống đang tạm thời chưa xử lý được yêu cầu. Anh/chị vui lòng "
                    "thử lại sau hoặc liên hệ cán bộ phụ trách."
                )
                st.error(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

st.divider()
st.caption("Trường hợp khẩn cấp về cháy, nổ hoặc cứu nạn, gọi 114 ngay.")
