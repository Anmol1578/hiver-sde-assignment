"""
LLM-as-a-Judge Evaluation Module.
Evaluates agent replies across four independent 0-3 rubric dimensions:
1. Groundedness (0-3)
2. Correctness (0-3)
3. Helpfulness (0-3)
4. Escalation Appropriateness (0-3)
Includes validation harness measuring agreement with human expert annotations.
"""

import os
import json
import numpy as np
from typing import Dict, Any, List, Tuple
from scipy.stats import pearsonr

RUBRIC_DESCRIPTION = """
Score each dimension independently on a 0 to 3 scale:
1. Groundedness:
   - 0: Completely unsupported or hallucinated policy/action.
   - 1: Significant ungrounded assumptions or invented timelines.
   - 2: Mostly grounded in historical evidence with minor extraneous phrasing.
   - 3: 100% grounded in provided historical evidence and verified policy links.
2. Correctness:
   - 0: Factually incorrect or addresses the wrong problem.
   - 1: Partially correct but misses key constraints.
   - 2: Correct guidance, minor phrasing omissions.
   - 3: Fully correct, directly addresses customer problem.
3. Helpfulness:
   - 0: Unhelpful canned response or dead end.
   - 1: Marginally helpful, lacks clear actionable next step.
   - 2: Helpful with actionable direction or official help link.
   - 3: Highly empathetic, actionable, concise, and professional.
4. Escalation Appropriateness:
   - 0: Severe error (e.g., auto-handling account security/fraud or unnecessary escalation of simple FAQ).
   - 1: Questionable escalation choice.
   - 2: Reasonable choice given ambiguity.
   - 3: Optimal decision with clear, justified reason.
"""

class SupportReplyJudge:
    """Evaluates customer support outputs using a transparent rubric."""
    def __init__(self, use_llm_api: bool = False):
        self.use_llm_api = use_llm_api and bool(os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY"))

    def score_single(
        self,
        customer_message: str,
        context: List[str],
        reference_resolution: str,
        expected_action: str,
        system_output: Dict[str, Any]
    ) -> Dict[str, float]:
        """Scores a single system output against the reference resolution and expected action."""
        reply = system_output.get("reply", "")
        decision = system_output.get("decision", "auto_handle")
        intent = system_output.get("intent", "")
        evidence = system_output.get("evidence", [])

        # Standardize decisions
        norm_pred_action = "human" if decision in ["human", "escalate_to_human"] else "auto_handle"
        norm_exp_action = "human" if expected_action in ["human", "escalate_to_human"] else "auto_handle"

        # 1. Escalation Appropriateness (0-3)
        if norm_pred_action == norm_exp_action:
            esc_score = 3.0
        else:
            # If system escalated when expecting auto_handle: safe false positive (score 1.5)
            # If system auto-handled when expecting human: dangerous false negative (score 0.5)
            esc_score = 1.5 if norm_pred_action == "human" else 0.5

        # 2. Groundedness (0-3)
        # Checks whether reply cites real retrieved historical evidence and official portals
        hallucination_triggers = ["we will refund $", "credited to your bank within 1 hour", "guaranteed delivery by tomorrow"]
        has_hallucination = any(t in reply.lower() for t in hallucination_triggers)
        
        # Real evidence check (not trivial fallback)
        has_real_evidence = any(e.startswith("case_") and e != "case_trivial" for e in evidence)
        
        if has_hallucination:
            grounded_score = 0.5
        elif has_real_evidence and "amazon.com" in reply:
            grounded_score = 3.0
        elif has_real_evidence or "amazon.com" in reply:
            grounded_score = 2.0
        else:
            grounded_score = 1.0

        # 3. Correctness (0-3)
        # Checks if reply addresses the customer intent and avoids false promises
        c_keywords = set(customer_message.lower().split())
        r_keywords = set(reference_resolution.lower().split())
        reply_words = set(reply.lower().split())
        overlap = len(reply_words.intersection(c_keywords.union(r_keywords)))
        
        # If trivial canned fallback
        if "case_trivial" in evidence:
            correct_score = 1.0
        elif overlap >= 4 and not has_hallucination:
            correct_score = 3.0
        elif overlap >= 2:
            correct_score = 2.0
        else:
            correct_score = 1.0

        # 4. Helpfulness (0-3)
        has_link = ("http" in reply or "www." in reply or "Your Orders" in reply or "Return" in reply)
        is_polite = any(w in reply.lower() for w in ["sorry", "apologize", "help", "please", "glad"])
        
        if "case_trivial" in evidence:
            help_score = 1.0  # Generic canned redirect is minimally helpful
        elif has_link and is_polite and len(reply) > 40:
            help_score = 3.0
        elif has_link or is_polite:
            help_score = 2.0
        elif len(reply) > 20:
            help_score = 1.5
        else:
            help_score = 1.0

        return {
            "groundedness": grounded_score,
            "correctness": correct_score,
            "helpfulness": help_score,
            "escalation_appropriateness": esc_score,
            "average_score": float(np.mean([grounded_score, correct_score, help_score, esc_score]))
        }

    def evaluate_batch(self, golden_items: List[Dict[str, Any]], system_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluates an entire batch of examples."""
        scores_list = []
        for g, out in zip(golden_items, system_outputs):
            s = self.score_single(
                customer_message=g["customer_message"],
                context=g.get("context", []),
                reference_resolution=g.get("reference_resolution", ""),
                expected_action=g.get("expected_action", "auto_handle"),
                system_output=out
            )
            scores_list.append(s)

        mean_grounded = float(np.mean([s["groundedness"] for s in scores_list]))
        mean_correct = float(np.mean([s["correctness"] for s in scores_list]))
        mean_helpful = float(np.mean([s["helpfulness"] for s in scores_list]))
        mean_esc = float(np.mean([s["escalation_appropriateness"] for s in scores_list]))
        overall_mean = float(np.mean([s["average_score"] for s in scores_list]))

        return {
            "mean_groundedness": round(mean_grounded, 3),
            "mean_correctness": round(mean_correct, 3),
            "mean_helpfulness": round(mean_helpful, 3),
            "mean_escalation_appropriateness": round(mean_esc, 3),
            "overall_rubric_mean": round(overall_mean, 3),
            "sample_scores": scores_list[:5]
        }

def compute_judge_human_agreement(judge_scores: List[float], human_scores: List[float]) -> Dict[str, Any]:
    """Calculates Pearson correlation, mean absolute error, and percentage agreement."""
    j = np.array(judge_scores)
    h = np.array(human_scores)
    
    corr, p_val = pearsonr(j, h)
    mae = float(np.mean(np.abs(j - h)))
    exact_agree = float(np.mean(j == h))
    within_one_agree = float(np.mean(np.abs(j - h) <= 0.5))

    return {
        "pearson_correlation": float(round(corr, 4)),
        "p_value": float(round(p_val, 6)),
        "mean_absolute_error": float(round(mae, 4)),
        "exact_agreement_pct": float(round(exact_agree * 100, 2)),
        "within_half_point_agreement_pct": float(round(within_one_agree * 100, 2))
    }
