"""
Unit and Integration Tests for Customer Support AI Pipeline.
Tests data cleaning, intent classification, escalation policy, RAG retrieval, and output schema.
"""

import json
import pytest
from pathlib import Path

from src.data.clean import clean_tweet_text, clean_reference_reply, is_predominantly_english
from src.intent.classifier import TrivialMajorityClassifier, ProductionIntentClassifier
from src.escalation.policy import ProductionEscalationEngine
from src.generation.reply import GroundedReplyGenerator
from src.pipeline import CustomerSupportAgent

BASE_DIR = Path(__file__).resolve().parent.parent

def test_data_cleaning():
    raw_tweet = "@AmazonHelp @115820 Where is my package??? It's 3 days late &amp; ruined! https://t.co/xyz123 ^TN"
    cleaned = clean_tweet_text(raw_tweet, remove_mentions=True, remove_urls=False)
    
    assert "@AmazonHelp" not in cleaned
    assert "@115820" not in cleaned
    assert "^TN" not in cleaned
    assert "&amp;" not in cleaned
    assert "&" in cleaned
    assert "[LINK]" in cleaned
    
    # Test reference reply cleaning
    raw_reply = "@115820 We're happy to help! Please check https://amazon.com/help ^AG"
    clean_rep = clean_reference_reply(raw_reply)
    assert clean_rep.startswith("We're happy to help!")
    assert not clean_rep.endswith("^AG")

def test_english_filter():
    assert is_predominantly_english("Where is my order? It was scheduled for today.") is True
    assert is_predominantly_english("こんにちは、アマゾン公式です。Fire TV Stickが見れない") is False

def test_trivial_baseline_classifier():
    clf = TrivialMajorityClassifier(majority_class="delivery_status_tracking")
    intent, conf = clf.predict("I need to return this broken toy")
    assert intent == "delivery_status_tracking"
    assert conf == 1.0

def test_production_intent_classifier():
    clf = ProductionIntentClassifier()
    
    # Account security
    intent, conf = clf.predict("Someone hacked into my Amazon account and changed the password!")
    assert intent == "account_and_payment_security"
    assert conf >= 0.90
    
    # Damaged item
    intent, conf = clf.predict("My parcel arrived broken and the perfume bottle was smashed.")
    assert intent == "damaged_or_missing_item"
    assert conf >= 0.90
    
    # Cancellation
    intent, conf = clf.predict("Please cancel my order before it ships!")
    assert intent == "order_change_or_cancel"
    assert conf >= 0.90

def test_escalation_engine():
    engine = ProductionEscalationEngine()
    
    # Critical security inquiry must escalate to human
    decision, reason = engine.evaluate(
        customer_message="My credit card was charged $400 without my authorization!",
        intent="account_and_payment_security",
        intent_confidence=0.98,
        retrieved_cases=[]
    )
    assert decision == "human"
    assert "security" in reason.lower() or "unauthorized" in reason.lower()
    
    # Explicit human request must escalate to human
    decision, reason = engine.evaluate(
        customer_message="I want to speak to a real human supervisor immediately!",
        intent="general_feedback_and_complaints",
        intent_confidence=0.95,
        retrieved_cases=[]
    )
    assert decision == "human"
    assert "human" in reason.lower()
    
    # Standard delivery inquiry with good evidence should auto-handle
    mock_case = {
        "case_id": "case_101",
        "customer_message": "Where is my package?",
        "reference_resolution": "Track your package in Your Orders.",
        "similarity_score": 0.88
    }
    decision, reason = engine.evaluate(
        customer_message="When is my parcel estimated to arrive?",
        intent="delivery_status_tracking",
        intent_confidence=0.92,
        retrieved_cases=[mock_case]
    )
    assert decision == "auto_handle"
    assert len(reason) > 10

def test_agent_output_schema():
    agent = CustomerSupportAgent()
    res = agent.process_message("How do I return this shirt that does not fit?")
    
    # Required keys according to assignment spec
    assert "intent" in res
    assert "reply" in res
    assert "decision" in res
    assert "reason" in res
    assert "evidence" in res
    
    assert res["decision"] in ["auto_handle", "human", "escalate_to_human"]
    assert isinstance(res["evidence"], list)
    assert len(res["reply"]) > 10
    assert len(res["reason"]) > 5

def test_golden_set_integrity():
    golden_file = BASE_DIR / "evaluation" / "golden.jsonl"
    assert golden_file.exists(), "golden.jsonl must exist"
    
    with open(golden_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
        
    assert len(records) == 200, f"Expected 200 golden records, found {len(records)}"
    
    intents = {r["intent"] for r in records}
    assert len(intents) == 8, f"Expected 8 distinct intents in golden set, found {len(intents)}"
    
    actions = {r["expected_action"] for r in records}
    assert "auto_handle" in actions
    assert "human" in actions
    
    # Validate required fields
    for r in records:
        assert "id" in r and r["id"].startswith("golden_")
        assert "customer_message" in r and len(r["customer_message"]) > 0
        assert "context" in r and isinstance(r["context"], list)
        assert "intent" in r
        assert "expected_action" in r
        assert "reason" in r and len(r["reason"]) > 0
        assert "reference_resolution" in r
