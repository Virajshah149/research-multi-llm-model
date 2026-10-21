"""
Vector Indexing and Top-K Passage Retrieval Engine.
Builds dedicated FAISS Inner Product (Cosine Similarity) indexes
for each embedding model (MiniLM, BGE, Gemini Embeddings).
"""

import sys
import time
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from config import EMBEDDING_CONFIGS, GEMINI_API_KEY, TOP_K


class EmbeddingEngine:
    """Manages loading and vector inference across different embedding backends."""

    def __init__(self):
        self._st_models = {}

    def _get_sentence_transformer(self, model_id: str):
        if model_id not in self._st_models:
            from sentence_transformers import SentenceTransformer
            print(f"[Embedder] Loading SentenceTransformer: {model_id}...")
            self._st_models[model_id] = SentenceTransformer(model_id)
        return self._st_models[model_id]

    def embed_texts(self, texts: List[str], config_name: str, is_query: bool = False) -> np.ndarray:
        """Computes L2-normalized embedding vectors for a list of texts."""
        cfg = EMBEDDING_CONFIGS[config_name]
        backend = cfg["type"]

        if backend == "sentence_transformers":
            model = self._get_sentence_transformer(cfg["model_id"])
            if is_query and cfg.get("query_prefix"):
                texts = [cfg["query_prefix"] + t for t in texts]
            embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        elif backend == "gemini":
            embeddings = self._embed_gemini(texts, cfg["model_id"], is_query=is_query)
        else:
            raise ValueError(f"Unknown embedding backend: {backend}")

        # L2-normalize vectors so IndexFlatIP computes exact Cosine Similarity
        faiss.normalize_L2(embeddings)
        return embeddings.astype(np.float32)

    def _embed_gemini(self, texts: List[str], model_id: str, is_query: bool = False) -> np.ndarray:
        """Embeds text in batches using the Google Gemini API."""
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in environment or .env file.")

        task_type = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)

        batch_size = 16
        vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for attempt in range(4):
                try:
                    res = client.models.embed_content(
                        model=model_id,
                        contents=batch,
                        config={"task_type": task_type}
                    )
                    for emb in res.embeddings:
                        vectors.append(emb.values)
                    break
                except Exception as e:
                    if attempt == 3:
                        raise e
                    print(f"   [Gemini Embed Backoff] {e}. Retrying in 2s...")
                    time.sleep(2.0)
            time.sleep(0.2)  # Avoid rate spikes

        return np.array(vectors, dtype=np.float32)


class RetrievalPipeline:
    """Creates, caches, and queries FAISS indexes for each embedding configuration."""

    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        self.texts = [c["text"] for c in chunks]
        self.engine = EmbeddingEngine()
        self.indexes: Dict[str, faiss.IndexFlatIP] = {}

    def build_all_indexes(self):
        """Constructs separate FAISS FlatIP indices for all configured embedding models."""
        for name in EMBEDDING_CONFIGS.keys():
            self.build_index_for(name)

    def build_index_for(self, config_name: str):
        """Builds a FAISS Index for a specific embedding configuration."""
        print(f"[Retrieval] Generating embeddings & building FAISS index for: {config_name}...")
        vectors = self.engine.embed_texts(self.texts, config_name, is_query=False)
        dim = vectors.shape[1]

        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        self.indexes[config_name] = index
        print(f"[Retrieval] Index ready for {config_name} ({index.ntotal} chunks indexed).")

    def retrieve(self, config_name: str, query: str, k: int = TOP_K) -> List[Dict[str, Any]]:
        """
        Retrieves top-k chunks for the given query using the specified embedding model.
        Returns chunk metadata with similarity scores.
        """
        if config_name not in self.indexes:
            self.build_index_for(config_name)

        q_vector = self.engine.embed_texts([query], config_name, is_query=True)
        scores, indices = self.indexes[config_name].search(q_vector, k)

        retrieved_passages = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            chunk["rank"] = rank + 1
            retrieved_passages.append(chunk)

        return retrieved_passages


def diagnose_retrieval(pipeline: RetrievalPipeline, query: str, k: int = 3):
    """Diagnostic tool to inspect retrieved passages side-by-side across all models."""
    print(f"\n{'='*70}\nDIAGNOSTIC RETRIEVAL FOR QUERY:\n'{query}'\n{'='*70}")
    for emb_name in EMBEDDING_CONFIGS.keys():
        print(f"\n--- Model: {emb_name} ---")
        try:
            results = pipeline.retrieve(emb_name, query, k=k)
            for r in results:
                print(f"Rank {r['rank']} [Page {r['page']}] (Cosine Score: {r['score']:.4f}):")
                print(f"   {r['text'][:180].replace(chr(10), ' ')}...\n")
        except Exception as e:
            print(f"Error retrieving for {emb_name}: {e}")
