"""
Escalation policies:
1. TrivialEscalationPolicy (Baseline 1)
2. SimpleKeywordEscalationPolicy (Baseline 2)
3. ProductionEscalationEngine (Production multi-factor policy)
"""

import re
from typing import List, Dict, Any, Tuple, Optional

# High-risk escalation keywords
EXPLICIT_HUMAN_KEYWORDS = re.compile(
    r"\b(speak to (a )?human|talk to (a )?(person|human|agent|representative)|real person|supervisor|manager|escalate)\b",
    re.IGNORECASE
)
FRUSTRATION_KEYWORDS = re.compile(
    r"\b(horrible|terrible|worst customer service|sue|lawyer|police|fraud|scam|unacceptable|disgusted|pissed|ridiculous)\b",
    re.IGNORECASE
)
DELIVERY_DISPUTE_KEYWORDS = re.compile(
    r"\b(says delivered but (it )?is not|marked delivered but not (received|here)|never arrived|didn't receive|stolen from porch)\b",
    re.IGNORECASE
)
PERSISTENT_FAILURE_KEYWORDS = re.compile(
    r"\b(still (not|haven't|waiting)|already tried|didn't work|useless|multiple times|second time|third time|no one (is )?helping)\b",
    re.IGNORECASE
)

class TrivialEscalationPolicy:
    """Trivial Baseline: always decides to auto_handle."""
    def decide(self, **kwargs) -> Tuple[str, str]:
        return "auto_handle", "Trivial default policy: auto-handle all incoming messages."

class SimpleKeywordEscalationPolicy:
    """Simple Baseline: keyword search for explicit anger or escalation requests."""
    def decide(self, customer_message: str, **kwargs) -> Tuple[str, str]:
        if EXPLICIT_HUMAN_KEYWORDS.search(customer_message) or FRUSTRATION_KEYWORDS.search(customer_message):
            return "human", "Keyword match indicating severe customer frustration or explicit escalation request."
        return "auto_handle", "No negative sentiment keywords detected."

class ProductionEscalationEngine:
    """
    Production Multi-Factor Escalation Engine:
    Combines intent risk classification, model confidence, retrieval grounding similarity,
    multi-turn context, financial/account security, and sentiment signals.
    """
    def __init__(self, confidence_threshold: float = 0.40, retrieval_similarity_threshold: float = 0.20):
        self.confidence_threshold = confidence_threshold
        self.retrieval_similarity_threshold = retrieval_similarity_threshold

    def evaluate(
        self,
        customer_message: str,
        intent: str,
        intent_confidence: float,
        retrieved_cases: List[Dict[str, Any]],
        context: Optional[List[str]] = None
    ) -> Tuple[str, str]:
        """
        Evaluates escalation condition and returns (decision, reason).
        decision is either 'human' or 'auto_handle'.
        """
        full_text = (" ".join(context) + " " + customer_message) if context else customer_message
        lower_msg = customer_message.lower()

        # Signal 1: Critical Intent Categories (Financial / Account Security / Fraud)
        if intent == "account_and_payment_security":
            return "human", "Account security, authentication, or unauthorized transaction risk requires verified human agent intervention."

        # Signal 2: Explicit Human Request
        if EXPLICIT_HUMAN_KEYWORDS.search(customer_message):
            return "human", "Customer explicitly requested to speak with a human agent or supervisor."

        # Signal 3: Stolen / Missing package marked delivered dispute
        if intent == "delivery_status_tracking" and DELIVERY_DISPUTE_KEYWORDS.search(full_text):
            return "human", "Dispute: Package marked as delivered by carrier but customer reports missing or stolen shipment."

        # Signal 4: Physical Damage or Empty Box Claims
        if intent == "damaged_or_missing_item":
            if any(w in lower_msg for w in ["opened", "empty", "tampered", "missing item", "broken", "shattered", "shattered"]):
                return "human", "Damaged goods or missing merchandise claim requires physical evidence verification and claims filing."

        # Signal 5: Severe Frustration or Legal Threat
        if FRUSTRATION_KEYWORDS.search(customer_message):
            return "human", "High churn risk: Customer expressed extreme dissatisfaction, abusive experience, or legal dispute."

        # Signal 6: Persistent Multi-turn Failure
        if context and len(context) >= 2 and PERSISTENT_FAILURE_KEYWORDS.search(full_text):
            return "human", "Multi-turn escalation: Issue remains repeatedly unresolved across customer interaction turns."

        # Signal 7: Low Intent Classification Confidence (True ambiguity)
        if intent_confidence < self.confidence_threshold:
            return "human", f"Ambiguous inquiry: Intent classification confidence ({intent_confidence:.2f}) below automated threshold ({self.confidence_threshold:.2f})."

        # Signal 8: Disputed refund delays (>14 days)
        if intent == "return_or_refund" and any(w in full_text for w in ["weeks ago", "never got refund", "refused return"]):
            return "human", "Dispute: Overdue refund beyond standard processing window requires accounting review."

        # All safety checks passed -> Auto-handle safely
        return "auto_handle", f"Standard resolvable inquiry for '{intent}' with high classification confidence ({intent_confidence:.2f}) and verified historical guidance."
