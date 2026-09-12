"""
Retrieval index builder.
Builds BM25 and dense embedding indices over historical resolved cases.
"""

import json
import joblib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

try:
    from fastembed import TextEmbedding
    HAS_FASTEMBED = True
except ImportError:
    HAS_FASTEMBED = False

INDEX_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "models"
KB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "retrieval_knowledge_base.jsonl"

def tokenize(text: str) -> List[str]:
    return [w.lower() for w in text.split() if len(w) > 2]

class KnowledgeBaseIndex:
    """Manages indexing and serialization of historical support cases."""
    def __init__(self):
        self.cases: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.embeddings: Optional[np.ndarray] = None
        self._embed_model = None

    def load_cases(self, kb_path: Path = KB_PATH, limit: int = 5000):
        print(f"[Index] Loading historical cases from {kb_path} (limit={limit})...")
        self.cases = []
        with open(kb_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.cases.append(json.loads(line))
                    if len(self.cases) >= limit:
                        break
        print(f"[Index] Loaded {len(self.cases):,} historical cases.")

    def build_bm25_index(self):
        print("[Index] Building BM25 index over customer queries and resolutions...")
        corpus = []
        for c in self.cases:
            combined = f"{c['customer_message']} {c.get('reference_resolution', '')}"
            corpus.append(tokenize(combined))
        self.bm25 = BM25Okapi(corpus)
        print("[Index] BM25 index built successfully.")

    def build_dense_index(self):
        if not HAS_FASTEMBED:
            print("[Index] FastEmbed not installed; skipping dense indexing.")
            return
            
        print("[Index] Generating dense embeddings using BGE model...")
        self._embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        texts = [f"{c['customer_message']} -> {c.get('reference_resolution', '')[:120]}" for c in self.cases]
        embeddings_gen = self._embed_model.embed(texts)
        self.embeddings = np.array(list(embeddings_gen), dtype=np.float32)
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / np.maximum(norms, 1e-9)
        print(f"[Index] Dense index built with shape: {self.embeddings.shape}")

    def save(self, index_dir: Path = INDEX_DIR):
        index_dir.mkdir(parents=True, exist_ok=True)
        meta_file = index_dir / "kb_cases.json"
        bm25_file = index_dir / "bm25.pkl"
        embed_file = index_dir / "dense_embeddings.npy"

        print(f"[Index] Saving indexed cases to {meta_file}...")
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(self.cases, f, ensure_ascii=False)

        if self.bm25:
            print(f"[Index] Saving BM25 index to {bm25_file}...")
            joblib.dump(self.bm25, bm25_file)

        if self.embeddings is not None:
            print(f"[Index] Saving dense embeddings to {embed_file}...")
            np.save(embed_file, self.embeddings)

        print("[Index] Index serialization complete.")

    def load(self, index_dir: Path = INDEX_DIR):
        meta_file = index_dir / "kb_cases.json"
        bm25_file = index_dir / "bm25.pkl"
        embed_file = index_dir / "dense_embeddings.npy"

        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                self.cases = json.load(f)
        if bm25_file.exists():
            self.bm25 = joblib.load(bm25_file)
        if embed_file.exists():
            self.embeddings = np.load(embed_file)
            if HAS_FASTEMBED:
                self._embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

def build_and_save_index(limit: int = 5000):
    idx = KnowledgeBaseIndex()
    idx.load_cases(limit=limit)
    idx.build_bm25_index()
    idx.build_dense_index()
    idx.save()
    return idx

if __name__ == "__main__":
    build_and_save_index()
