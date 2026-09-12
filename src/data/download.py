"""
Dataset acquisition and brand subsample extraction module.
Extracts tweets for the chosen brand (AmazonHelp) from archive.zip / twcs.csv.
"""

import os
import io
import csv
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"

def find_archive_path() -> Path | None:
    """Finds archive.zip or twcs.zip across common repository locations."""
    candidates = [
        RAW_DIR / "archive.zip",
        DATA_DIR / "archive.zip",
        Path.cwd() / "archive.zip",
        Path(__file__).resolve().parent.parent.parent / "archive.zip",
        Path(__file__).resolve().parent.parent.parent.parent / "archive.zip",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def extract_brand_dataset(brand: str = "AmazonHelp", max_rows: int = 1_000_000, output_filename: str = "amazon_tweets.csv") -> Path:
    """
    Extracts tweets related to the specified brand from archive.zip.
    Includes both inbound messages to the brand and outbound replies from the brand.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / output_filename
    
    if out_path.exists() and out_path.stat().st_size > 100_000:
        print(f"[Data] Found existing extracted brand data at: {out_path} ({out_path.stat().st_size:,} bytes)")
        return out_path
    
    archive_path = find_archive_path()
    if not archive_path:
        raise FileNotFoundError(
            "Raw Kaggle archive.zip not found.\n"
            "Preprocessed artifacts are already committed in data/processed/ for fast reproduction.\n"
            "To re-run the full raw-data preparation pipeline from scratch, download 'twcs.csv' / 'archive.zip' "
            "from Kaggle (https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) "
            "and place 'archive.zip' in the repository root or under data/raw/."
        )
    
    print(f"[Data] Extracting conversations for '{brand}' from {archive_path.name} (scanning up to {max_rows:,} rows)...")
    
    # First pass: collect brand reply IDs and parent IDs
    brand_rows = []
    parent_ids_needed = set()
    brand_reply_ids = set()
    
    with zipfile.ZipFile(archive_path) as z:
        with z.open("twcs/twcs.csv") as f:
            wrapper = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
            reader = csv.DictReader(wrapper)
            
            for i, row in enumerate(reader):
                if i >= max_rows:
                    break
                if row["author_id"] == brand:
                    brand_rows.append(row)
                    brand_reply_ids.add(row["tweet_id"])
                    if row["in_response_to_tweet_id"]:
                        parent_ids_needed.add(row["in_response_to_tweet_id"])
    
    print(f"[Data] Collected {len(brand_rows):,} outbound replies from {brand}. Fetching {len(parent_ids_needed):,} customer parents...")
    
    # Second pass: collect customer tweets that triggered these brand replies
    customer_rows = []
    with zipfile.ZipFile(ARCHIVE_PATH) as z:
        with z.open("twcs/twcs.csv") as f:
            wrapper = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
            reader = csv.DictReader(wrapper)
            
            for i, row in enumerate(reader):
                if i >= max_rows:
                    break
                tid = row["tweet_id"]
                if tid in parent_ids_needed:
                    customer_rows.append(row)
                    # Also look for grandparent if multi-turn
                    if row["in_response_to_tweet_id"]:
                        pass
    
    all_rows = brand_rows + customer_rows
    # Sort by tweet_id
    all_rows.sort(key=lambda r: int(r["tweet_id"]) if r["tweet_id"].isdigit() else 0)
    
    print(f"[Data] Writing {len(all_rows):,} total tweets (customer + brand) to {out_path}...")
    fieldnames = ["tweet_id", "author_id", "inbound", "created_at", "text", "response_tweet_id", "in_response_to_tweet_id"]
    with open(out_path, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
        
    print(f"[Data] Successfully saved {out_path} ({out_path.stat().st_size:,} bytes).")
    return out_path

if __name__ == "__main__":
    extract_brand_dataset()
