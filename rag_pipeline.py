from __future__ import annotations

import re
import math
import logging

import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from config import (
    EMBEDDING_MODEL,
    CROSS_ENCODER_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    RRF_K,
    CANDIDATE_MULTIPLIER,
    MIN_CANDIDATES,
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------
# Tokenizer for BM25 (better than naive .split())
# ----------------------------------------------------------
_WORD_RE = re.compile(r"\b\w+\b")


def tokenize_for_bm25(text: str) -> list[str]:
    """Lowercase + extract word tokens, stripping punctuation."""
    return _WORD_RE.findall(text.lower())


# ----------------------------------------------------------
# Confidence helpers
# ----------------------------------------------------------
def sigmoid(x: float) -> float:
    """Map unbounded cross-encoder logit → 0-1 probability."""
    return 1.0 / (1.0 + math.exp(-x))


def get_confidence_label(chunks: list[dict]) -> str:
    """Return High / Medium / Low based on sigmoid-normalized scores."""
    if not chunks:
        return "Low"

    avg_score = sigmoid(sum(c["score"] for c in chunks) / len(chunks))

    if avg_score >= CONFIDENCE_HIGH:
        return "High"
    elif avg_score >= CONFIDENCE_MEDIUM:
        return "Medium"
    else:
        return "Low"


# ----------------------------------------------------------
# RAG Pipeline
# ----------------------------------------------------------
class RAGPipeline:
    def __init__(
        self,
        embedder: SentenceTransformer | None = None,
        cross_encoder: CrossEncoder | None = None,
    ):
        # Accept pre-loaded (cached) models or create new ones
        self.embedder = embedder or SentenceTransformer(EMBEDDING_MODEL)
        self.cross_encoder = cross_encoder or CrossEncoder(CROSS_ENCODER_MODEL)

        self.stores: dict[str, dict] = {
            "resume": {"chunks": [], "index": None, "bm25": None},
            "job_description": {"chunks": [], "index": None, "bm25": None},
        }

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=CHUNK_SEPARATORS,
        )

    # ----- PDF parsing -----
    def extract_text_from_pdf(self, uploaded_file) -> str:
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        return text.strip()

    # ----- Chunking -----
    def split_text_into_chunks(self, text: str) -> list[str]:
        return self.text_splitter.split_text(text)

    # ----- Index building -----
    def _build_store_for_source(self, source: str, text: str) -> None:
        chunks = self.split_text_into_chunks(text)
        chunk_objects = [
            {"chunk_id": f"{source}-{i}", "source": source, "text": chunk}
            for i, chunk in enumerate(chunks, start=1)
        ]

        if not chunk_objects:
            self.stores[source] = {"chunks": [], "index": None, "bm25": None}
            return

        chunk_texts = [c["text"] for c in chunk_objects]

        # 1. Dense index (FAISS)
        embeddings = self.embedder.encode(
            chunk_texts, convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        # 2. Sparse index (BM25 with proper tokenization)
        tokenized_corpus = [tokenize_for_bm25(doc) for doc in chunk_texts]
        bm25 = BM25Okapi(tokenized_corpus)

        self.stores[source] = {
            "chunks": chunk_objects,
            "index": index,
            "bm25": bm25,
        }

    def build_vector_store(self, documents: list[dict]) -> None:
        # Reset stores
        self.stores = {
            "resume": {"chunks": [], "index": None, "bm25": None},
            "job_description": {"chunks": [], "index": None, "bm25": None},
        }

        for doc in documents:
            source = doc.get("source")
            text = doc.get("text", "").strip()
            if source in self.stores and text:
                self._build_store_for_source(source, text)

    # ----- Hybrid retrieval -----
    def retrieve_from_source(
        self, query: str, source: str, top_k: int = 4
    ) -> list[dict]:
        if source not in self.stores:
            return []

        store = self.stores[source]
        if store["index"] is None or not store["chunks"] or store["bm25"] is None:
            return []

        candidate_k = min(
            max(MIN_CANDIDATES, top_k * CANDIDATE_MULTIPLIER),
            len(store["chunks"]),
        )
        if candidate_k == 0:
            return []

        # 1. Dense retrieval (FAISS)
        query_embedding = self.embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

        faiss_scores, faiss_indices = store["index"].search(
            query_embedding, candidate_k
        )

        # 2. Sparse retrieval (BM25)
        tokenized_query = tokenize_for_bm25(query)
        bm25_scores = store["bm25"].get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[::-1][:candidate_k]

        # 3. Reciprocal Rank Fusion
        all_candidate_indices = set(faiss_indices[0]).union(set(bm25_indices))

        faiss_rank = {idx: rank for rank, idx in enumerate(faiss_indices[0])}
        bm25_rank = {idx: rank for rank, idx in enumerate(bm25_indices)}

        fused_candidates = []
        for idx in all_candidate_indices:
            f_rank = faiss_rank.get(idx, 1000)
            b_rank = bm25_rank.get(idx, 1000)
            rrf_score = 1.0 / (RRF_K + f_rank) + 1.0 / (RRF_K + b_rank)
            fused_candidates.append({"chunk_idx": idx, "rrf_score": rrf_score})

        fused_candidates.sort(key=lambda x: x["rrf_score"], reverse=True)
        fused_candidates = fused_candidates[:candidate_k]

        if not fused_candidates:
            return []

        # 4. Cross-encoder re-ranking
        chunk_texts = [
            store["chunks"][c["chunk_idx"]]["text"] for c in fused_candidates
        ]
        cross_inp = [[query, text] for text in chunk_texts]

        try:
            cross_scores = self.cross_encoder.predict(cross_inp)
        except Exception as e:
            logger.warning("Cross-encoder re-ranking failed: %s — using RRF order", e)
            cross_scores = [c["rrf_score"] for c in fused_candidates]

        for i, c in enumerate(fused_candidates):
            c["cross_score"] = float(cross_scores[i])

        fused_candidates.sort(key=lambda x: x["cross_score"], reverse=True)

        # Build final results
        results = []
        seen_ids: set[str] = set()
        for c in fused_candidates[:top_k]:
            idx = c["chunk_idx"]
            chunk = store["chunks"][idx]
            if chunk["chunk_id"] in seen_ids:
                continue
            seen_ids.add(chunk["chunk_id"])
            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "source": chunk["source"],
                    "text": chunk["text"],
                    "score": c["cross_score"],
                }
            )

        return results

    def retrieve(
        self, query: str, top_k: int = 4, sources: list[str] | None = None
    ) -> list[dict]:
        if sources is None:
            sources = ["resume", "job_description"]

        all_results: list[dict] = []
        for source in sources:
            all_results.extend(
                self.retrieve_from_source(query, source, top_k=top_k)
            )

        # Deduplicate across sources
        seen: set[str] = set()
        unique = []
        for r in all_results:
            if r["chunk_id"] not in seen:
                seen.add(r["chunk_id"])
                unique.append(r)

        unique.sort(key=lambda x: x["score"], reverse=True)
        return unique[:top_k]

    def total_chunks(self) -> int:
        return sum(len(self.stores[s]["chunks"]) for s in self.stores)