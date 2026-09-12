"""
Baselines for Customer Support Agent:
1. TrivialBaseline: Majority intent + most common historical canned reply + always auto-handle.
2. SimpleBaseline: TF-IDF + Logistic Regression for intent + BM25 retrieval for reply + keyword escalation.
"""

from typing import Dict, Any, List, Optional
from src.intent.classifier import TrivialMajorityClassifier, SimpleLogisticClassifier
from src.retrieval.retriever import TrivialRetriever, SimpleBM25Retriever
from src.escalation.policy import TrivialEscalationPolicy, SimpleKeywordEscalationPolicy

class TrivialBaselineAgent:
    """Trivial Baseline Agent: Majority class and static fallback."""
    def __init__(self):
        self.classifier = TrivialMajorityClassifier()
        self.retriever = TrivialRetriever()
        self.escalation = TrivialEscalationPolicy()

    def process_message(self, customer_message: str, context: Optional[List[str]] = None) -> Dict[str, Any]:
        intent, _ = self.classifier.predict(customer_message, context)
        retrieved = self.retriever.retrieve(customer_message, context, top_k=1)
        decision, reason = self.escalation.decide()
        
        reply = (
            "I'm sorry for the delay! We'd like to take a further look into this with you! "
            "Please reach us here: https://amazon.com/help"
        )
        evidence = [retrieved[0]["case_id"]] if retrieved else []

        return {
            "intent": intent,
            "reply": reply,
            "decision": decision,
            "reason": reason,
            "evidence": evidence
        }

class SimpleBaselineAgent:
    """Simple Baseline Agent: TF-IDF Logistic Regression + BM25 retrieval + Keyword escalation."""
    def __init__(self, bm25_retriever: Optional[SimpleBM25Retriever] = None):
        self.classifier = SimpleLogisticClassifier()
        self.retriever = bm25_retriever
        self.escalation = SimpleKeywordEscalationPolicy()

    def fit_classifier(self, texts: List[str], labels: List[str]):
        self.classifier.fit(texts, labels)

    def process_message(self, customer_message: str, context: Optional[List[str]] = None) -> Dict[str, Any]:
        intent, _ = self.classifier.predict(customer_message, context)
        
        retrieved = []
        if self.retriever:
            retrieved = self.retriever.retrieve(customer_message, context, top_k=2)
            
        decision, reason = self.escalation.decide(customer_message=customer_message)
        
        if retrieved:
            reply = retrieved[0].get("reference_resolution", "Please contact customer support at amazon.com/help.")
            evidence = [r["case_id"] for r in retrieved]
        else:
            reply = "Please contact our customer support team for assistance at https://amazon.com/help."
            evidence = []

        return {
            "intent": intent,
            "reply": reply,
            "decision": decision,
            "reason": reason,
            "evidence": evidence
        }
