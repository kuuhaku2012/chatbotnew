"""
retriever.py

Truy hoi 2 tang:
  Tang 1 (kb_variants)   : neu co match du gan -> tra CANONICAL_ANSWER
                            da duyet, tu tin tra loi.
  Tang 2 (legal_articles): chi dung khi tang 1 KHONG co match du gan ->
                            tra trich dan Dieu luat, KHONG duoc tu ket
                            luan thu tuc cu the (xem GUARDRAIL_TIER2 ben
                            duoi, dua vao prompt cho LLM o buoc sau).
  Khong tang nao match   : escalate cho can bo truc ban.

Nguong (threshold) la COSINE DISTANCE (0 = giong het, cang lon cang
khac nhau) vi ChromaDB o day dung "hnsw:space": "cosine" trong
build_vectorstore.py. Can hieu chinh 2 nguong nay bang du lieu that
cua ban (chay thu vai chuc cau hoi, xem distance thuc te) truoc khi
dua vao production -- gia tri mac dinh o day chi la diem khoi dau.

Cach dung:
    from retriever import TieredRetriever
    r = TieredRetriever(db_path="./chroma_db")
    result = r.retrieve("thủ tục đăng ký tạm trú cần giấy tờ gì")
"""

import json
import re
import chromadb
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "intfloat/multilingual-e5-base"

GUARDRAIL_TIER2 = (
    "Chỉ trích dẫn và tóm tắt ngắn gọn nội dung điều luật được cung cấp. "
    "Không tự suy luận thủ tục cụ thể, không xác nhận hồ sơ, không kết luận "
    "cho trường hợp cá nhân. Luôn khuyến nghị liên hệ cán bộ phụ trách để "
    "được hướng dẫn chính xác theo trường hợp cụ thể."
)


ABBREVIATION_PATTERNS = (
    (r"\bs[đd]t\b", "số điện thoại"),
    (r"\bca\s*xã\b", "công an xã"),
    (r"\bca\s*phường\b", "công an phường"),
)


def normalize_query(text: str) -> str:
    """Expand common citizen abbreviations without changing query intent."""
    normalized = " ".join((text or "").strip().split())
    for pattern, replacement in ABBREVIATION_PATTERNS:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    return normalized


def decide_tier(kb_best_distance, legal_best_distance, kb_max_distance,
                legal_max_distance, kb_preference_margin=0.009):
    """Ham logic thuan (khong phu thuoc embedding/DB) -- de test rieng."""
    kb_is_good = kb_best_distance is not None and kb_best_distance <= kb_max_distance
    legal_is_good = (
        legal_best_distance is not None and legal_best_distance <= legal_max_distance
    )

    # A generic Tier-1 intent can be barely closer than a precise legal excerpt.
    # Prefer the curated answer only when it has a meaningful similarity lead;
    # otherwise retain the safer, source-bound Tier-2 response.
    if kb_is_good and (
        not legal_is_good
        or kb_best_distance + kb_preference_margin < legal_best_distance
    ):
        return "tier1"
    if legal_is_good:
        return "tier2"
    if kb_is_good:
        return "tier1"
    return "escalate"


class TieredRetriever:
    def __init__(self, db_path="./chroma_db", kb_chunks_path="data/kb_chunks.jsonl",
                 kb_max_distance=0.13, legal_max_distance=0.17, top_k=3):
        self.model = SentenceTransformer(EMBED_MODEL_NAME)
        client = chromadb.PersistentClient(path=db_path)
        self.kb_coll = client.get_collection("kb_variants")
        self.legal_coll = client.get_collection("legal_articles")
        self.kb_max_distance = kb_max_distance
        self.legal_max_distance = legal_max_distance
        self.top_k = top_k

        # lookup chunk_id -> KB_CHUNK day du (de lay CANONICAL_ANSWER, GUARDRAIL...)
        self.kb_lookup = {}
        with open(kb_chunks_path, encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                self.kb_lookup[c["chunk_id"]] = c

    def _embed_query(self, text):
        normalized = normalize_query(text)
        return self.model.encode(
            [f"query: {normalized}"], normalize_embeddings=True
        ).tolist()

    def retrieve_raw(self, query):
        q_emb = self._embed_query(query)

        kb_res = self.kb_coll.query(query_embeddings=q_emb, n_results=self.top_k)
        legal_res = self.legal_coll.query(query_embeddings=q_emb, n_results=self.top_k)

        kb_best_dist = kb_res["distances"][0][0] if kb_res["distances"][0] else None
        legal_best_dist = legal_res["distances"][0][0] if legal_res["distances"][0] else None

        kb_chunk_id = kb_res["metadatas"][0][0]["chunk_id"] if (kb_res["metadatas"] and kb_res["metadatas"][0]) else None

        legal_meta = legal_res["metadatas"][0][0] if (legal_res["metadatas"] and legal_res["metadatas"][0]) else None
        legal_src_id = legal_meta["src_id"] if legal_meta else None
        legal_dieu_number = legal_meta["dieu_number"] if legal_meta else None
        legal_dieu_title = legal_meta["dieu_title"] if legal_meta else None

        return {
            "kb_best_distance": kb_best_dist,
            "kb_chunk_id": kb_chunk_id,
            "legal_best_distance": legal_best_dist,
            "legal_src_id": legal_src_id,
            "legal_dieu_number": legal_dieu_number,
            "legal_dieu_title": legal_dieu_title,
        }

    def retrieve(self, query):
        q_emb = self._embed_query(query)

        kb_res = self.kb_coll.query(query_embeddings=q_emb, n_results=self.top_k)
        legal_res = self.legal_coll.query(query_embeddings=q_emb, n_results=self.top_k)

        kb_best_dist = kb_res["distances"][0][0] if kb_res["distances"][0] else None
        legal_best_dist = legal_res["distances"][0][0] if legal_res["distances"][0] else None

        tier = decide_tier(kb_best_dist, legal_best_dist,
                            self.kb_max_distance, self.legal_max_distance)

        normalized_query = normalize_query(query).lower()
        asks_for_contact = any(
            marker in normalized_query
            for marker in (
                "số điện thoại",
                "số trực ban",
                "liên hệ trực ban",
                "gọi trực ban",
            )
        )
        if tier == "tier2" and asks_for_contact:
            tier = "escalate"

        if tier == "tier1":
            chunk_id = kb_res["metadatas"][0][0]["chunk_id"]
            chunk = self.kb_lookup[chunk_id]
            return {
                "tier": 1,
                "distance": kb_best_dist,
                "chunk_id": chunk_id,
                "canonical_answer": chunk.get("canonical_answer"),
                "guardrail": chunk.get("guardrail"),
                "required_entities": chunk.get("required_entities"),
                "clarifying_question_if_missing": chunk.get("clarifying_question_if_missing"),
                "handoff_or_emergency_rule": chunk.get("handoff_or_emergency_rule"),
            }

        if tier == "tier2":
            meta = legal_res["metadatas"][0][0]
            return {
                "tier": 2,
                "distance": legal_best_dist,
                "src_id": meta["src_id"],
                "source_title": meta["source_title"],
                "dieu_number": meta["dieu_number"],
                "dieu_title": meta["dieu_title"],
                "excerpt": legal_res["documents"][0][0],
                "guardrail": GUARDRAIL_TIER2,
            }

        return {
            "tier": "escalate",
            "kb_distance": kb_best_dist,
            "legal_distance": legal_best_dist,
            "message": "Không tìm thấy nội dung đủ liên quan ở cả hai tầng — cần chuyển cán bộ trực ban.",
        }
