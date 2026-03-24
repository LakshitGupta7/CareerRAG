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
from sentence_transformers import SentenceTransformer


class RAGPipeline:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embedding_model_name)

        self.stores = {
            "resume": {
                "chunks": [],
                "index": None
            },
            "job_description": {
                "chunks": [],
                "index": None
            }
        }

    def extract_text_from_pdf(self, uploaded_file) -> str:
        text = ""
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        for page in doc:
            text += page.get_text() + "\n"

        return text.strip()

    def split_text_into_chunks(self, text: str, chunk_size: int = 400, overlap: int = 80) -> list:
        words = text.split()
        chunks = []

        if not words:
            return chunks

        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end]).strip()

            if chunk:
                chunks.append(chunk)

            if end >= len(words):
                break

            start += chunk_size - overlap

        return chunks

    def _build_store_for_source(self, source: str, text: str):
        chunks = self.split_text_into_chunks(text)
        chunk_objects = []

        for i, chunk in enumerate(chunks, start=1):
            chunk_objects.append({
                "chunk_id": f"{source}-{i}",
                "source": source,
                "text": chunk
            })

        if not chunk_objects:
            self.stores[source]["chunks"] = []
            self.stores[source]["index"] = None
            return

        chunk_texts = [c["text"] for c in chunk_objects]
        embeddings = self.embedder.encode(
            chunk_texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        self.stores[source]["chunks"] = chunk_objects
        self.stores[source]["index"] = index

    def build_vector_store(self, documents: list) -> None:
        self.stores = {
            "resume": {"chunks": [], "index": None},
            "job_description": {"chunks": [], "index": None}
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
        if store["index"] is None or not store["chunks"]:
            return []

        query_embedding = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = store["index"].search(query_embedding, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if 0 <= idx < len(store["chunks"]):
                chunk = store["chunks"][idx]
                results.append({
                    "chunk_id": chunk["chunk_id"],
                    "source": chunk["source"],
                    "text": chunk["text"],
                    "score": float(score)
                })

        return results

    def retrieve(self, query: str, top_k: int = 4, sources: list = None) -> list:
        if sources is None:
            sources = ["resume", "job_description"]

        all_results = []
        per_source_k = max(1, top_k)

        for source in sources:
            all_results.extend(self.retrieve_from_source(query, source, top_k=per_source_k))

        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]

    def total_chunks(self) -> int:
        return sum(len(self.stores[source]["chunks"]) for source in self.stores)