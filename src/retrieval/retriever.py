"""
Retriever implementations:
1. TrivialRetriever (Baseline 1)
2. SimpleBM25Retriever (Baseline 2)
3. HybridRAGRetriever (Production RAG: BM25 + Dense Semantic Fusion)
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from .index import KnowledgeBaseIndex, tokenize, HAS_FASTEMBED

class TrivialRetriever:
    """Trivial Baseline: returns the first or most common canned case."""
    def __init__(self, fixed_case: Optional[Dict[str, Any]] = None):
        self.fixed_case = fixed_case or {
            "case_id": "case_trivial",
            "customer_message": "Where is my order?",
            "reference_resolution": "I'm sorry for the delay! We'd like to take a further look into this with you! Please reach us here: https://amazon.com/help",
            "similarity_score": 1.0
        }

    def retrieve(self, query: str, context: Optional[List[str]] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        return [self.fixed_case] * min(top_k, 1)

class SimpleBM25Retriever:
    """Simple Baseline: BM25 lexical matching."""
    def __init__(self, kb_index: KnowledgeBaseIndex):
        self.index = kb_index

    def retrieve(self, query: str, context: Optional[List[str]] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.index.bm25 or not self.index.cases:
            return []
        tokens = tokenize(query)
        if context:
            tokens.extend(tokenize(" ".join(context)))
        scores = self.index.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        max_score = max(scores[top_indices[0]], 1e-6)
        for idx in top_indices:
            case = dict(self.index.cases[idx])
            case["similarity_score"] = float(round(scores[idx] / max_score, 4))
            results.append(case)
        return results

class HybridRAGRetriever:
    """
    Production RAG Retriever:
    Fuses BM25 sparse keyword matching and Dense BGE semantic embeddings
    using Reciprocal Rank Fusion / normalized weighted score.
    """
    def __init__(self, kb_index: KnowledgeBaseIndex, alpha: float = 0.55):
        self.index = kb_index
        self.alpha = alpha  # weight for dense similarity
        self.bm25_retriever = SimpleBM25Retriever(kb_index)

    def retrieve(self, query: str, context: Optional[List[str]] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.index.cases:
            return []
            
        full_query = (" ".join(context) + " " + query) if context else query
        
        # 1. BM25 scoring
        bm25_results = self.bm25_retriever.retrieve(query, context, top_k=top_k * 4)
        bm25_dict = {c["case_id"]: c["similarity_score"] for c in bm25_results}

        # 2. Dense semantic scoring (if embeddings are loaded)
        dense_dict = {}
        if self.index.embeddings is not None and self.index._embed_model is not None:
            q_emb = list(self.index._embed_model.embed([full_query]))[0]
            q_emb = q_emb / max(np.linalg.norm(q_emb), 1e-9)
            sims = np.dot(self.index.embeddings, q_emb)
            dense_top_indices = np.argsort(sims)[::-1][:top_k * 4]
            for idx in dense_top_indices:
                cid = self.index.cases[idx]["case_id"]
                dense_dict[cid] = float(sims[idx])

        # 3. Fuse scores
        all_candidate_ids = set(bm25_dict.keys()) | set(dense_dict.keys())
        case_lookup = {c["case_id"]: c for c in self.index.cases}
        
        scored_candidates = []
        for cid in all_candidate_ids:
            b_score = bm25_dict.get(cid, 0.0)
            d_score = dense_dict.get(cid, 0.0)
            combined = (self.alpha * d_score) + ((1 - self.alpha) * b_score)
            
            c_data = dict(case_lookup[cid])
            c_data["similarity_score"] = float(round(combined, 4))
            c_data["dense_score"] = float(round(d_score, 4))
            c_data["bm25_score"] = float(round(b_score, 4))
            scored_candidates.append(c_data)

        scored_candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
        return scored_candidates[:top_k]
