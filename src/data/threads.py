"""
Thread reconstruction module.
Reconstructs multi-turn customer-brand conversation threads from raw tweets.
"""

import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from .clean import clean_tweet_text, clean_reference_reply, is_predominantly_english

def reconstruct_conversation_threads(
    raw_csv_path: Path,
    brand_name: str = "AmazonHelp",
    min_text_len: int = 15,
    english_only: bool = True
) -> List[Dict[str, Any]]:
    """
    Parses raw tweets and links customer queries to brand resolutions.
    Extracts customer message, previous conversation context, and historical brand resolution.
    """
    tweets_by_id = {}
    brand_replies = []
    
    print(f"[Threads] Reading tweets from {raw_csv_path}...")
    with open(raw_csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row["tweet_id"]
            tweets_by_id[tid] = row
            if row["author_id"] == brand_name and row["inbound"] == "False" and row["in_response_to_tweet_id"]:
                brand_replies.append(row)
                
    print(f"[Threads] Loaded {len(tweets_by_id):,} total tweets; found {len(brand_replies):,} brand replies.")
    
    threads = []
    seen_customer_messages = set()
    
    for reply in brand_replies:
        parent_id = reply["in_response_to_tweet_id"]
        if parent_id not in tweets_by_id:
            continue
        
        cust_tweet = tweets_by_id[parent_id]
        if cust_tweet["inbound"] != "True":
            continue
            
        cust_raw_text = cust_tweet["text"]
        reply_raw_text = reply["text"]
        
        if english_only:
            if not is_predominantly_english(cust_raw_text) or not is_predominantly_english(reply_raw_text):
                continue
                
        cleaned_cust_text = clean_tweet_text(cust_raw_text, remove_mentions=True, remove_urls=False)
        cleaned_reply = clean_reference_reply(reply_raw_text)
        
        if len(cleaned_cust_text) < min_text_len or len(cleaned_reply) < min_text_len:
            continue
            
        # Deduplicate identical customer tweets
        norm_key = cleaned_cust_text.lower().strip()
        if norm_key in seen_customer_messages:
            continue
        seen_customer_messages.add(norm_key)
        
        # Build prior context if the customer tweet was itself a reply
        context_chain = []
        curr = cust_tweet
        visited = {curr["tweet_id"]}
        
        # Trace ancestor turns (up to 4 turns back)
        while curr.get("in_response_to_tweet_id"):
            ancestor_id = curr["in_response_to_tweet_id"]
            if ancestor_id not in tweets_by_id or ancestor_id in visited:
                break
            visited.add(ancestor_id)
            ancestor = tweets_by_id[ancestor_id]
            role = "Brand" if ancestor["author_id"] == brand_name else "Customer"
            clean_anc = clean_tweet_text(ancestor["text"])
            if clean_anc:
                context_chain.append(f"{role}: {clean_anc}")
            curr = ancestor
            if len(context_chain) >= 4:
                break
                
        context_chain.reverse() # chronological order
        
        threads.append({
            "case_id": f"case_{reply['tweet_id']}",
            "customer_tweet_id": cust_tweet["tweet_id"],
            "reply_tweet_id": reply["tweet_id"],
            "customer_message": cleaned_cust_text,
            "raw_customer_message": cust_raw_text,
            "context": context_chain,
            "reference_resolution": cleaned_reply,
            "created_at": cust_tweet["created_at"],
            "has_context": len(context_chain) > 0
        })
        
    print(f"[Threads] Successfully reconstructed {len(threads):,} clean, validated conversation cases.")
    return threads
