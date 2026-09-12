"""
Automated metrics computation module for intent classification, escalation, and reply quality.
"""

from typing import List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

def compute_intent_metrics(true_intents: List[str], pred_intents: List[str], labels: List[str]) -> Dict[str, Any]:
    """Computes overall accuracy, macro F1, weighted F1, per-class metrics, and confusion matrix."""
    acc = accuracy_score(true_intents, pred_intents)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(true_intents, pred_intents, average="macro", zero_division=0)
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(true_intents, pred_intents, average="weighted", zero_division=0)
    
    # Per-intent metrics
    p_per, r_per, f1_per, support_per = precision_recall_fscore_support(true_intents, pred_intents, labels=labels, zero_division=0)
    
    per_intent = {}
    for idx, label in enumerate(labels):
        per_intent[label] = {
            "precision": float(round(p_per[idx], 4)),
            "recall": float(round(r_per[idx], 4)),
            "f1": float(round(f1_per[idx], 4)),
            "support": int(support_per[idx])
        }

    # Confusion matrix
    cm = confusion_matrix(true_intents, pred_intents, labels=labels)

    return {
        "accuracy": float(round(acc, 4)),
        "macro_precision": float(round(p_macro, 4)),
        "macro_recall": float(round(r_macro, 4)),
        "macro_f1": float(round(f1_macro, 4)),
        "weighted_f1": float(round(f1_weight, 4)),
        "per_intent": per_intent,
        "confusion_matrix": cm.tolist(),
        "labels": labels
    }

def compute_escalation_metrics(true_actions: List[str], pred_actions: List[str]) -> Dict[str, Any]:
    """
    Computes accuracy, precision, recall, and F1 for escalation decisions.
    Focuses on 'human' vs 'auto_handle'.
    """
    # Standardize values: "human" or "auto_handle"
    norm_true = ["human" if a in ["human", "escalate_to_human"] else "auto_handle" for a in true_actions]
    norm_pred = ["human" if a in ["human", "escalate_to_human"] else "auto_handle" for a in pred_actions]
    
    acc = accuracy_score(norm_true, norm_pred)
    
    # Binary metrics for the 'human' class
    p, r, f1, _ = precision_recall_fscore_support(
        norm_true, norm_pred, pos_label="human", average="binary", zero_division=0
    )
    
    # Confusion matrix: [[TN (auto-auto), FP (auto-human)], [FN (human-auto), TP (human-human)]]
    cm = confusion_matrix(norm_true, norm_pred, labels=["auto_handle", "human"])
    tn, fp, fn, tp = cm.ravel()
    
    return {
        "accuracy": float(round(acc, 4)),
        "escalation_precision": float(round(p, 4)),
        "escalation_recall": float(round(r, 4)),
        "escalation_f1": float(round(f1, 4)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "unnecessary_escalation_rate": float(round(fp / max(tn + fp, 1), 4)),
        "missed_escalation_risk_rate": float(round(fn / max(tp + fn, 1), 4))
    }

def compute_lexical_similarity(reference: str, hypothesis: str) -> float:
    """Computes token Jaccard similarity between reference and hypothesis."""
    ref_tokens = set(reference.lower().split())
    hyp_tokens = set(hypothesis.lower().split())
    if not ref_tokens or not hyp_tokens:
        return 0.0
    intersection = ref_tokens.intersection(hyp_tokens)
    union = ref_tokens.union(hyp_tokens)
    return len(intersection) / len(union)
