# import fitz  # PyMuPDF
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer


# class RAGPipeline:
#     def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
#         self.embedder = SentenceTransformer(embedding_model_name)
#         self.index = None
#         self.chunks = []

#     def extract_text_from_pdf(self, uploaded_file) -> str:
#         text = ""
#         pdf_bytes = uploaded_file.read()
#         doc = fitz.open(stream=pdf_bytes, filetype="pdf")

#         for page in doc:
#             text += page.get_text() + "\n"

#         return text.strip()

#     def split_text_into_chunks(
#         self,
#         text: str,
#         chunk_size: int = 400,
#         overlap: int = 80
#     ) -> list:
#         words = text.split()
#         chunks = []

#         if not words:
#             return chunks

#         start = 0
#         while start < len(words):
#             end = start + chunk_size
#             chunk = " ".join(words[start:end]).strip()

#             if chunk:
#                 chunks.append(chunk)

#             if end >= len(words):
#                 break

#             start += chunk_size - overlap

#         return chunks

#     def build_vector_store(self, documents: list) -> None:
#         """
#         documents format:
#         [
#             {"source": "resume", "text": "..."},
#             {"source": "job_description", "text": "..."}
#         ]
#         """
#         self.chunks = []

#         for doc in documents:
#             text = doc.get("text", "").strip()
#             source = doc.get("source", "unknown")

#             if not text:
#                 continue

#             doc_chunks = self.split_text_into_chunks(text)

#             for chunk in doc_chunks:
#                 self.chunks.append({
#                     "source": source,
#                     "text": chunk
#                 })

#         if not self.chunks:
#             raise ValueError("No valid text chunks were created from the uploaded documents.")

#         chunk_texts = [chunk["text"] for chunk in self.chunks]

#         embeddings = self.embedder.encode(
#             chunk_texts,
#             convert_to_numpy=True,
#             normalize_embeddings=True
#         )

#         embeddings = embeddings.astype("float32")
#         dimension = embeddings.shape[1]

#         # cosine similarity style search using normalized vectors
#         self.index = faiss.IndexFlatIP(dimension)
#         self.index.add(embeddings)

#     def retrieve(self, query: str, top_k: int = 4, source_filter: str = None) -> list:
#         if self.index is None or not self.chunks:
#             return []

#         query_embedding = self.embedder.encode(
#             [query],
#             convert_to_numpy=True,
#             normalize_embeddings=True
#         ).astype("float32")

#         scores, indices = self.index.search(query_embedding, top_k * 3)

#         results = []
#         for idx, score in zip(indices[0], scores[0]):
#             if 0 <= idx < len(self.chunks):
#                 chunk = self.chunks[idx]

#                 if source_filter is not None and chunk["source"] != source_filter:
#                     continue

#                 results.append({
#                     "source": chunk["source"],
#                     "text": chunk["text"],
#                     "score": float(score)
#                 })

#                 if len(results) >= top_k:
#                     break

#         return results

import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi


class RAGPipeline:
    def __init__(
        self, 
        embedding_model_name: str = "all-MiniLM-L6-v2",
        cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.embedder = SentenceTransformer(embedding_model_name)
        self.cross_encoder = CrossEncoder(cross_encoder_name)

        self.stores = {
            "resume": {
                "chunks": [],
                "index": None,
                "bm25": None
            },
            "job_description": {
                "chunks": [],
                "index": None,
                "bm25": None
            }
        }
        
        # Initialize the advanced chunker
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=80,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def extract_text_from_pdf(self, uploaded_file) -> str:
        text = ""
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        for page in doc:
            text += page.get_text() + "\n"

        return text.strip()

    def split_text_into_chunks(self, text: str) -> list:
        # Use LangChain's chunker instead of the naive word splitter
        return self.text_splitter.split_text(text)

    def _build_store_for_source(self, source: str, text: str):
        chunks = self.split_text_into_chunks(text)
        chunk_objects = []

        for i, chunk in enumerate(chunks, start=1):
            chunk_objects.append({
                "chunk_id": f"chunk-{i}",
                "source": source,
                "text": chunk
            })

        if not chunk_objects:
            self.stores[source]["chunks"] = []
            self.stores[source]["index"] = None
            self.stores[source]["bm25"] = None
            return

        chunk_texts = [c["text"] for c in chunk_objects]
        
        # 1. Build Dense Index (FAISS)
        embeddings = self.embedder.encode(
            chunk_texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        
        # 2. Build Sparse Index (BM25)
        tokenized_corpus = [doc.lower().split() for doc in chunk_texts]
        bm25 = BM25Okapi(tokenized_corpus)

        self.stores[source]["chunks"] = chunk_objects
        self.stores[source]["index"] = index
        self.stores[source]["bm25"] = bm25

    def build_vector_store(self, documents: list) -> None:
        self.stores = {
            "resume": {"chunks": [], "index": None, "bm25": None},
            "job_description": {"chunks": [], "index": None, "bm25": None}
        }

        for doc in documents:
            source = doc.get("source")
            text = doc.get("text", "").strip()

            if source in self.stores and text:
                self._build_store_for_source(source, text)

    def retrieve_from_source(self, query: str, source: str, top_k: int = 4) -> list:
        if source not in self.stores:
            return []

        store = self.stores[source]
        if store["index"] is None or not store["chunks"] or store["bm25"] is None:
            return []

        # We will retrieve more candidates to re-rank
        candidate_k = max(10, top_k * 2)
        candidate_k = min(candidate_k, len(store["chunks"])) # don't exceed corpus size
        
        if candidate_k == 0:
            return []

        # 1. Dense Retrieval (FAISS)
        query_embedding = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        faiss_scores, faiss_indices = store["index"].search(query_embedding, candidate_k)
        
        # 2. Sparse Retrieval (BM25)
        tokenized_query = query.lower().split()
        bm25_scores = store["bm25"].get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[::-1][:candidate_k]
        
        # Combine candidate sets (Union of indices)
        all_candidate_indices = set(faiss_indices[0]).union(set(bm25_indices))
        
        # We need to map index -> rank to compute RRF (Reciprocal Rank Fusion)
        faiss_rank = {idx: rank for rank, idx in enumerate(faiss_indices[0])}
        bm25_rank = {idx: rank for rank, idx in enumerate(bm25_indices)}
        
        k_rrf = 60 # standard RRF constant
        fused_candidates = []
        
        for idx in all_candidate_indices:
            # lower rank is better
            f_rank = faiss_rank.get(idx, 1000) 
            b_rank = bm25_rank.get(idx, 1000)
            
            # Simple RRF formula
            rrf_score = 1.0 / (k_rrf + f_rank) + 1.0 / (k_rrf + b_rank)
            fused_candidates.append({
                "chunk_idx": idx,
                "rrf_score": rrf_score
            })
            
        # Sort by RRF score
        fused_candidates.sort(key=lambda x: x["rrf_score"], reverse=True)
        # Take top candidate_k for reranking
        fused_candidates = fused_candidates[:candidate_k]

        if not fused_candidates:
            return []
            
        # 3. Cross-Encoder Re-ranking
        chunk_texts_to_rerank = [store["chunks"][c["chunk_idx"]]["text"] for c in fused_candidates]
        cross_inp = [[query, text] for text in chunk_texts_to_rerank]
        cross_scores = self.cross_encoder.predict(cross_inp)
        
        # Attach cross-encoder scores
        for i, c in enumerate(fused_candidates):
            c["cross_score"] = float(cross_scores[i])
            
        # Sort by cross-encoder score
        fused_candidates.sort(key=lambda x: x["cross_score"], reverse=True)

        # Build final returning list
        results = []
        for c in fused_candidates[:top_k]:
            idx = c["chunk_idx"]
            chunk = store["chunks"][idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "text": chunk["text"],
                "score": c["cross_score"]  # Note: Cross encoder scores can be positive/negative logits
            })

        return results

    def retrieve(self, query: str, top_k: int = 4, sources: list = None) -> list:
        if sources is None:
            sources = ["resume", "job_description"]

        all_results = []
        per_source_k = max(1, top_k)

        for source in sources:
            all_results.extend(self.retrieve_from_source(query, source, top_k=per_source_k))

        # Re-sort everything using their cross-encoder scores again 
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]

    def total_chunks(self) -> int:
        return sum(len(self.stores[source]["chunks"]) for source in self.stores)