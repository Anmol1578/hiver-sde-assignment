"""
Data preparation pipeline entry point.
Downloads/extracts raw tweets, reconstructs threads, and prepares train/retrieval/eval splits.
"""

import os
import json
import random
from pathlib import Path
from .download import extract_brand_dataset
from .threads import reconstruct_conversation_threads

PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

def run_data_preparation(brand: str = "AmazonHelp", max_rows: int = 1_000_000, seed: int = 42):
    """
    Complete data pipeline:
    1. Extracts brand tweets from archive.zip
    2. Reconstructs conversation threads
    3. Splits into Retrieval Knowledge Base (80%) and Evaluation Pool (20%)
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw_csv = extract_brand_dataset(brand=brand, max_rows=max_rows)
    
    threads = reconstruct_conversation_threads(raw_csv_path=raw_csv, brand_name=brand)
    
    # Save all processed conversations
    all_conv_file = PROCESSED_DIR / "amazon_conversations.jsonl"
    print(f"[Prepare] Saving all {len(threads):,} conversations to {all_conv_file}...")
    with open(all_conv_file, "w", encoding="utf-8") as f:
        for t in threads:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
            
    # Deterministic train/retrieval vs eval split (preventing data leakage)
    random.seed(seed)
    shuffled = list(threads)
    random.shuffle(shuffled)
    
    # Reserve 2,000 cases for evaluation candidate pool, remaining for historical retrieval KB
    eval_pool = shuffled[:2000]
    retrieval_kb = shuffled[2000:12000] # 10,000 cases for fast, high-coverage retrieval KB
    
    eval_file = PROCESSED_DIR / "eval_pool.jsonl"
    kb_file = PROCESSED_DIR / "retrieval_knowledge_base.jsonl"
    
    print(f"[Prepare] Writing {len(eval_pool):,} cases to evaluation pool: {eval_file}")
    with open(eval_file, "w", encoding="utf-8") as f:
        for item in eval_pool:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"[Prepare] Writing {len(retrieval_kb):,} cases to retrieval knowledge base: {kb_file}")
    with open(kb_file, "w", encoding="utf-8") as f:
        for item in retrieval_kb:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print("[Prepare] Data preparation completed successfully!")
    return {
        "total_conversations": len(threads),
        "eval_pool_size": len(eval_pool),
        "retrieval_kb_size": len(retrieval_kb)
    }

if __name__ == "__main__":
    run_data_preparation()
