# Benchmark Evaluation Summary

**Brand:** AmazonHelp  
**Golden Test Set:** 200 hand-labelled conversations (88.0% multi-turn)  
**Execution Runtime:** 1.35 seconds  

---

## 1. Comparative Results Table

| System / Model | Intent Accuracy | Intent Macro-F1 | Escalation Accuracy | Escalation F1 (Human) | Missed Escalation Risk | LLM-Judge Overall (0-3) | Groundedness (0-3) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trivial Baseline** (Majority Class) | 0.255 | 0.051 | 0.655 | 0.000 | 1.000 | 1.53 | 2.00 |
| **Simple Baseline** (TF-IDF + BM25 + Heuristic) | 0.750 | 0.704 | 0.705 | 0.253 | 0.855 | 2.19 | 2.00 |
| **Production AI Agent** (Hybrid RAG + Multi-Signal) | **0.815** | **0.829** | **0.785** | **0.650** | **0.420** | **2.32** | **2.00** |

---

## 2. Per-Intent Performance (Production Agent)

| Intent Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `delivery_status_tracking` | 0.746 | 0.804 | 0.774 | 51 |
| `return_or_refund` | 0.783 | 0.720 | 0.750 | 25 |
| `damaged_or_missing_item` | 0.889 | 0.941 | 0.914 | 17 |
| `order_change_or_cancel` | 0.778 | 1.000 | 0.875 | 7 |
| `account_and_payment_security` | 0.750 | 0.840 | 0.792 | 25 |
| `prime_and_subscription` | 0.955 | 0.840 | 0.894 | 25 |
| `technical_product_support` | 0.850 | 0.680 | 0.756 | 25 |
| `general_feedback_and_complaints` | 0.880 | 0.880 | 0.880 | 25 |

---

## 3. Escalation Performance Breakdown

- **Escalation Accuracy:** 78.5%
- **Precision on Human Escalations:** 74.1%
- **Recall on Human Escalations:** 58.0%
- **Missed Escalation Risk (False Negatives):** 42.0% (29/69 critical cases missed)
- **Unnecessary Escalation Rate (False Positives):** 10.7% (14/131 safe queries escalated)

---

## 4. LLM-as-a-Judge vs. Human Expert Agreement

- **Pearson Correlation (r):** 0.8030 (p = 0.0000e+00)
- **Mean Absolute Error (MAE):** 0.369 points
- **Agreement within ±0.5 points:** 85.0%
- **Exact Agreement:** 15.0%

