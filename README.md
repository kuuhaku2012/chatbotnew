# Chatbot Công an phường (bản demo)

Chatbot trả lời thủ tục hành chính và thông tin pháp luật cơ bản cho công an
phường, dùng kiến trúc RAG 2 tầng: tầng 1 trả lời từ 101 câu hỏi/thủ tục đã
được duyệt sẵn, tầng 2 trích dẫn trực tiếp văn bản luật gốc khi câu hỏi nằm
ngoài 101 intent đó.

## Kiến trúc tổng quan

```
Người dùng
   │
   ▼
Chat UI (Streamlit, app/ui.py)
   │  HTTP POST /chat
   ▼
Backend (FastAPI, app/main.py)
   │
   ├─ 1. guardrail.py — chặn khẩn cấp (113/114/115) và thông tin nhạy cảm
   │     (OTP/mật khẩu/STK) TRƯỚC KHI vào retrieval, không tốn LLM call
   │
   ├─ 2. retriever.py (TieredRetriever) — 2 collection Chroma:
   │     • kb_variants (2100 câu hỏi mẫu của 101 intent đã duyệt)
   │     • legal_articles (3448 chunk luật/PCCC)
   │     Query kb_variants trước; nếu không đủ gần mới rơi xuống
   │     legal_articles; nếu cả 2 đều xa thì escalate.
   │
   └─ 3. LLM (Groq, Llama-3.3-70b-versatile) — diễn đạt câu trả lời:
         • Tầng 1: dùng canonical_answer đã duyệt, tự tin trả lời
         • Tầng 2: CHỈ trích dẫn + khuyến nghị gặp cán bộ, không tự
           kết luận (vì đây là luật gốc chưa qua diễn giải chuyên môn)
         • Escalate: câu trả lời cố định, không gọi LLM
```

## Cấu trúc thư mục

```
.
├── app/                  Backend + giao diện
│   ├── main.py             FastAPI: nối guardrail → retriever → LLM → log
│   ├── retriever.py         TieredRetriever, ngưỡng đã hiệu chỉnh
│   ├── guardrail.py          Chặn khẩn cấp + thông tin nhạy cảm
│   └── ui.py                  Streamlit chat UI (đa lượt, ô nhập API key)
│
├── ingest/                Pipeline dựng dữ liệu (chạy 1 lần / khi cập nhật)
│   ├── parse_kb_chunks.py         docx → data/kb_chunks.jsonl
│   ├── parse_source_registry.py    docx → data/source_registry.json
│   ├── crawl_legal_docs.py          source_registry.json → PDF + text thô
│   ├── ocr_fallback.py               OCR lại PDF scan (pypdf đọc không ra)
│   ├── chunk_legal_by_dieu.py        text → chunk theo từng Điều luật
│   └── build_vectorstore.py           embed + nạp vào ./chroma_db
│
├── data/
│   ├── kb_source.docx               Tài liệu KB_CHUNK gốc ban đầu (97 intent)
│   ├── kb_chunks.jsonl                101 intent (2100 câu hỏi mẫu)
│   ├── source_registry.json            24 nguồn luật chính thống
│   └── legal_docs/
│       ├── legal_documents_ocr.jsonl     24 văn bản (4 text layer + 14 OCR)
│       ├── legal_chunks_ocr.jsonl         558 Điều đã tách — dùng để embed
│       └── raw_pdf/                        18 PDF gốc, giữ để đối chiếu/audit
│
├── chroma_db/            Vector store (5548 embeddings, build sẵn)
├── logs/chat_log.jsonl     Log mọi lượt hỏi-đáp (audit)
├── tessdata/                Gói ngôn ngữ Tesseract (chỉ cần khi re-run OCR)
├── tests/                     Script test độc lập từng phần
├── .env.example                 Mẫu biến môi trường — copy thành .env
└── requirements.txt
```

## Cài đặt

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Chỉ cần nếu re-run OCR (không cần cho chạy chatbot bình thường):
# sudo apt-get install tesseract-ocr tesseract-ocr-vie poppler-utils
```

Sao chép `.env.example` thành `.env`, điền `GROQ_API_KEY` thật (lấy free tại
console.groq.com). **Không commit `.env` lên git, không dán key vào bất kỳ
đâu công khai.**

## Chạy chatbot (dữ liệu đã build sẵn trong `chroma_db/`)

Mở 2 terminal:

```bash
# Terminal 1 — backend
cd app
uvicorn main:app --reload --port 8001

# Terminal 2 — giao diện
cd app
streamlit run ui.py
```

Nếu cổng 8001 bị chiếm, đổi `--port` và set biến môi trường
`CHATBOT_BACKEND_URL` khớp theo trước khi chạy `ui.py`.

Người dùng cũng có thể tự nhập Groq API Key riêng ngay trên giao diện
(sidebar) thay vì dùng `GROQ_API_KEY` cấu hình sẵn trên server.

## Dựng lại toàn bộ dữ liệu từ đầu (chỉ cần khi tài liệu gốc thay đổi)

Chạy lần lượt từ thư mục gốc dự án:

```bash
python ingest/parse_kb_chunks.py data/kb_source.docx data/kb_chunks.jsonl
python ingest/parse_source_registry.py data/kb_source.docx data/source_registry.json

# Cần máy có mạng thật ra ngoài (không chạy được trong môi trường bị chặn mạng)
python ingest/crawl_legal_docs.py data/source_registry.json data/legal_docs/

python ingest/ocr_fallback.py data/legal_docs/legal_documents.jsonl \
    data/legal_docs/raw_pdf/ data/legal_docs/legal_documents_ocr.jsonl

python ingest/chunk_legal_by_dieu.py data/legal_docs/legal_documents_ocr.jsonl \
    data/legal_docs/legal_chunks_ocr.jsonl

python ingest/build_vectorstore.py --kb data/kb_chunks.jsonl \
    --legal data/legal_docs/legal_chunks_ocr.jsonl --db_path ./chroma_db
```

Sau khi build lại vector store, **nên chạy lại bước hiệu chỉnh ngưỡng**
(`kb_max_distance`/`legal_max_distance` trong `retriever.py`) — phân bố
khoảng cách sẽ đổi khi 2 collection thay đổi kích thước.

## Test

```bash
cd tests
python test_guardrail.py        # test guardrail độc lập, không cần API key
python test_main_pipeline.py     # test pipeline đầy đủ, cần GROQ_API_KEY
python test_8_turns.py             # test hội thoại nhiều lượt
```

## Giới hạn đã biết (đọc trước khi dùng cho việc thật)

- **OCR có sai số**: 14/24 văn bản luật phải OCR lại (bản gốc là scan, không
  có text layer). Đã kiểm tra tay một phần Điều 1 của Luật Cư trú, phát
  hiện vài lỗi chính tả từng chữ (bình thường với OCR) nhưng nội dung vẫn
  đọc hiểu được. Nên đối chiếu tay các văn bản dùng nhiều nhất trước khi
  triển khai thật.
- **Ngưỡng retrieval mới hiệu chỉnh trên 35 câu hỏi mẫu** — đủ cho demo,
  chưa đủ lớn để tin tuyệt đối cho production. Nên mở rộng bộ test khi có
  dữ liệu sử dụng thực tế.
- **`SRC-I-01`** (thông tin liên hệ địa phương: số trực ban, kênh Zalo...)
  hiện là **dữ liệu demo kỹ thuật**, chưa được Công an phường thực tế xác
  nhận. Bắt buộc phải thay bằng thông tin đã được phường phê duyệt trước
  khi dùng cho sản phẩm thật.
- **`GROQ_API_KEY` cũ đã từng bị lộ nhiều lần trong quá trình phát triển**
  (dán vào code/chat). Nếu chưa revoke key `gsk_Ku6KXFJl...` ở
  console.groq.com, **cần làm ngay** trước khi dùng dự án này cho bất kỳ
  việc gì khác ngoài test cá nhân.
- Guardrail dựa trên từ khóa/regex, đã qua nhiều vòng test đối kháng nhưng
  không thể phủ hết mọi cách diễn đạt tiếng Việt — nên tiếp tục bổ sung từ
  khóa khi phát hiện trường hợp bị bỏ sót trong quá trình vận hành thực tế.
