"""
main.py

FastAPI backend nối toàn bộ pipeline:
1. Guardrail check
2. TieredRetriever
3. LLM Generation (Groq Llama-3.3-70b-versatile, đọc từ api_key request hoặc GROQ_API_KEY server)
4. Audit Logging (logs/chat_log.jsonl)
"""

from __future__ import annotations
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, Callable

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq

from dotenv import load_dotenv

# Tải biến môi trường từ file .env (tìm ở thư mục hiện tại và các thư mục cha)
load_dotenv()

# Import modules từ retriever.py và guardrail.py
sys.path.insert(0, os.path.dirname(__file__))
from guardrail import guardrail_check
from retriever import TieredRetriever, GUARDRAIL_TIER2

# Khởi tạo FastAPI App
app = FastAPI(title="Chatbot Công An Phường API")

# Xac dinh thu muc goc du an (app/ nam ngay duoi goc, chi can len 1 cap)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(BASE_DIR, "chroma_db")
KB_CHUNKS_PATH = os.path.join(DATA_DIR, "kb_chunks.jsonl")

# Khởi tạo TieredRetriever với ngưỡng đã hiệu chỉnh
retriever = TieredRetriever(db_path=DB_PATH, kb_chunks_path=KB_CHUNKS_PATH, kb_max_distance=0.13, legal_max_distance=0.17)

# Đảm bảo thư mục logs tồn tại
LOGS_DIR = os.path.join(BASE_DIR, "logs")
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, "chat_log.jsonl")


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []
    api_key: Optional[str] = None


def format_history(history: list[dict], max_turns: int = 6) -> str:
    if not history:
        return "Chưa có lịch sử hội thoại."
    recent = history[-max_turns:]
    formatted_lines = []
    for msg in recent:
        role = "Người dân" if msg.get("role") == "user" else "Trợ lý ảo"
        content = msg.get("content", "").strip()
        formatted_lines.append(f"- {role}: {content}")
    return "\n".join(formatted_lines)


def log_turn(
    message: str,
    guardrail_res: dict,
    tier: Any,
    system_prompt: Optional[str],
    response: str,
    latency_ms: int,
    matched_chunk_id: Optional[str] = None,
    matched_dieu: Optional[str] = None,
    retrieval_distance: Optional[float] = None
):
    # Đảm bảo KHÔNG ghi api_key vào log dưới bất kỳ hình thức nào
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "input": message,
        "guardrail": guardrail_res,
        "tier": tier,
        "matched_chunk_id": matched_chunk_id,
        "matched_dieu": matched_dieu,
        "retrieval_distance": retrieval_distance,
        "system_prompt": system_prompt,
        "response": response,
        "latency_ms": latency_ms
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def call_llm(system_prompt: str, user_message: str, api_key: Optional[str] = None) -> str:
    # Ưu tiên dùng API Key từ người dùng gửi lên, nếu không có thì fallback về biến môi trường GROQ_API_KEY
    effective_key = (api_key.strip() if api_key and api_key.strip() else None) or os.getenv("GROQ_API_KEY")
    if not effective_key:
        raise ValueError("Vui lòng nhập Groq API Key ở sidebar hoặc cấu hình GROQ_API_KEY trên server")
    
    client = Groq(api_key=effective_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    return response.choices[0].message.content.strip()


def process_chat(
    message: str,
    history: list[dict] = [],
    api_key: Optional[str] = None,
    llm_fn: Optional[Callable[[str, str], str]] = None
) -> Dict[str, Any]:
    start_time = time.time()
    
    # Thiết lập hàm gọi LLM (tự động truyền api_key nếu dùng hàm mặc định)
    if llm_fn is None:
        llm_fn = lambda sp, usr_msg: call_llm(sp, usr_msg, api_key=api_key)
    
    # 1. Chạy Guardrail Check
    gr_res = guardrail_check(message)
    if gr_res["triggered"]:
        cat = gr_res.get("category", "emergency")
        tier_str = f"guardrail_{cat}"
        response_text = gr_res["message"]
        latency_ms = int((time.time() - start_time) * 1000)
        log_turn(
            message=message,
            guardrail_res=gr_res,
            tier=tier_str,
            system_prompt=None,
            response=response_text,
            latency_ms=latency_ms,
            matched_chunk_id=None,
            matched_dieu=None,
            retrieval_distance=None
        )
        return {
            "response": response_text,
            "tier": tier_str
        }

    # 2. Gọi TieredRetriever (CHỈ DÙNG DUY NHẤT TIN NHẮN MỚI NHẤT, KHÔNG NỐI/GHÉP HISTORY)
    ret_res = retriever.retrieve(message)
    tier = ret_res.get("tier")

    hist_str = format_history(history)

    # 3. Điều hướng theo TIER
    if tier == 1:
        canonical_answer = ret_res.get("canonical_answer") or ""
        chunk_guardrail = ret_res.get("guardrail") or ""
        req_entities = ret_res.get("required_entities") or []
        clarifying_q = ret_res.get("clarifying_question_if_missing") or ""
        matched_chunk_id = ret_res.get("chunk_id")
        distance = ret_res.get("distance")

        system_prompt = (
            f"Bạn là trợ lý ảo hỗ trợ công an phường.\n\n"
            f"LỊCH SỬ HỘI THOẠI GẦN ĐÂY:\n{hist_str}\n\n"
            f"THÔNG TIN CẦN CÓ TỪ NGUỜI DÙNG: {req_entities}\n\n"
            f"BƯỚC 1: Đọc lịch sử hội thoại và tin nhắn mới nhất của người dùng, xác định xem tổng hợp cả quá trình trao đổi người dùng đã cung cấp đủ các thông tin trên chưa.\n\n"
            f"BƯỚC 2A - NẾU THIẾU: chỉ trả lời bằng ĐÚNG MỘT câu hỏi ngắn gọn, lịch sự, xưng 'Anh/chị', hỏi TRỰC TIẾP người dùng (dựa trên nội dung: {clarifying_q}). DỪNG Ở ĐÂY, không nói gì thêm.\n\n"
            f"BƯỚC 2B - NẾU ĐÃ ĐỦ: dùng NGUYÊN NỘI DUNG căn cứ sau, diễn đạt lại tự nhiên (không copy y nguyên từng chữ), TRẢ LỜI THẲNG vào điều người dùng hỏi, KHÔNG hỏi lại, KHÔNG hỏi xác nhận lại thông tin họ vừa cho:\n"
            f"\"{canonical_answer}\"\n"
            f"Giữ đúng quy tắc: \"{chunk_guardrail}\"."
        )

        response_text = llm_fn(system_prompt, message)
        latency_ms = int((time.time() - start_time) * 1000)
        log_turn(
            message=message,
            guardrail_res=gr_res,
            tier=1,
            system_prompt=system_prompt,
            response=response_text,
            latency_ms=latency_ms,
            matched_chunk_id=matched_chunk_id,
            matched_dieu=None,
            retrieval_distance=distance
        )
        return {
            "response": response_text,
            "tier": 1,
            "matched_chunk_id": matched_chunk_id,
            "retrieval_distance": distance
        }

    elif tier == 2:
        dieu_num = ret_res.get("dieu_number")
        src_title = ret_res.get("source_title")
        excerpt = ret_res.get("excerpt") or ""
        matched_dieu_str = f"Điều {dieu_num} {src_title}" if dieu_num else None
        distance = ret_res.get("distance")

        system_prompt = (
            f"Bạn là trợ lý ảo hỗ trợ thông tin pháp luật công an phường.\n\n"
            f"LỊCH SỬ HỘI THOẠI GẦN ĐÂY:\n{hist_str}\n\n"
            f"Quy tắc tuyệt đối: {GUARDRAIL_TIER2}\n\n"
            f"Nội dung điều luật trích dẫn (Điều {dieu_num} {src_title}):\n\"{excerpt}\"\n\n"
            f"BẮT BUỘC câu trả lời phải có dạng:\n"
            f"\"Theo Điều {dieu_num} {src_title}: [tóm tắt ngắn]. Đây là thông tin luật gốc, anh/chị nên liên hệ cán bộ phụ trách để được hướng dẫn chính xác theo trường hợp cụ thể.\""
        )
        response_text = llm_fn(system_prompt, message)
        latency_ms = int((time.time() - start_time) * 1000)
        log_turn(
            message=message,
            guardrail_res=gr_res,
            tier=2,
            system_prompt=system_prompt,
            response=response_text,
            latency_ms=latency_ms,
            matched_chunk_id=None,
            matched_dieu=matched_dieu_str,
            retrieval_distance=distance
        )
        return {
            "response": response_text,
            "tier": 2,
            "matched_dieu": matched_dieu_str,
            "retrieval_distance": distance
        }

    else:
        # ESCALATE
        response_text = "Không tìm thấy thông tin phù hợp ở cả hai tầng dữ liệu. Anh/chị vui lòng liên hệ trực tiếp cán bộ trực ban Công an phường để được hỗ trợ cụ thể đối với yêu cầu này."
        latency_ms = int((time.time() - start_time) * 1000)
        kb_dist = ret_res.get("kb_distance")
        log_turn(
            message=message,
            guardrail_res=gr_res,
            tier="escalate",
            system_prompt=None,
            response=response_text,
            latency_ms=latency_ms,
            matched_chunk_id=None,
            matched_dieu=None,
            retrieval_distance=kb_dist
        )
        return {
            "response": response_text,
            "tier": "escalate",
            "retrieval_distance": kb_dist
        }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    try:
        return process_chat(req.message, req.history, api_key=req.api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))