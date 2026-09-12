"""
End-to-end Customer Support AI Pipeline.
Connects:
Customer Message -> Context -> Intent Classifier -> Historical Retrieval -> Escalation Policy -> Reply Generator -> Output
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from src.intent.classifier import ProductionIntentClassifier
from src.retrieval.index import KnowledgeBaseIndex
from src.retrieval.retriever import HybridRAGRetriever
from src.escalation.policy import ProductionEscalationEngine
from src.generation.reply import GroundedReplyGenerator

class CustomerSupportAgent:
    """Production AI Customer Support Agent for AmazonHelp."""
    def __init__(
        self,
        intent_classifier: Optional[ProductionIntentClassifier] = None,
        retriever: Optional[HybridRAGRetriever] = None,
        escalation_engine: Optional[ProductionEscalationEngine] = None,
        reply_generator: Optional[GroundedReplyGenerator] = None
    ):
        self.classifier = intent_classifier or ProductionIntentClassifier()
        self.retriever = retriever
        self.escalation = escalation_engine or ProductionEscalationEngine()
        self.generator = reply_generator or GroundedReplyGenerator()

    def process_message(
        self,
        customer_message: str,
        context: Optional[List[str]] = None,
        top_k: int = 2
    ) -> Dict[str, Any]:
        """
        Processes an incoming customer message and produces structured output.
        """
        # 1. Intent Classification
        intent, confidence = self.classifier.predict(customer_message, context)

        # 2. Historical Case Retrieval
        retrieved_cases = []
        if self.retriever:
            retrieved_cases = self.retriever.retrieve(customer_message, context, top_k=top_k)

        # 3. Escalation Decision
        decision, reason = self.escalation.evaluate(
            customer_message=customer_message,
            intent=intent,
            intent_confidence=confidence,
            retrieved_cases=retrieved_cases,
            context=context
        )

        # 4. Reply Generation (Conditioned on Intent, Evidence, and Escalation State)
        is_escalated = (decision in ["human", "escalate_to_human"])
        gen_result = self.generator.generate(
            customer_message=customer_message,
            intent=intent,
            evidence_cases=retrieved_cases,
            context=context,
            is_escalated=is_escalated,
            escalation_reason=reason
        )

        # 5. Formatted JSON Deliverable
        return {
            "intent": intent,
            "reply": gen_result["reply"],
            "decision": decision,
            "reason": reason,
            "evidence": gen_result.get("evidence", [])
        }

def create_default_agent() -> CustomerSupportAgent:
    """Factory function to build and load a fully initialized support agent."""
    idx = KnowledgeBaseIndex()
    idx.load()
    retriever = HybridRAGRetriever(idx)
    classifier = ProductionIntentClassifier()
    return CustomerSupportAgent(intent_classifier=classifier, retriever=retriever)
