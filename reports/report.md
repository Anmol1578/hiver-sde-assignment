# Engineering Report: Grounded AI Customer Support Agent for AmazonHelp

**Candidate:** Anmol  
**Role:** SDE Intern — Take-Home Assignment  
**Company:** Hiver  
**Target Brand:** AmazonHelp (`Customer Support on Twitter` Dataset)  
**Deliverables Repository:** [hiver-sde-assignment](https://github.com/Anmol1578/hiver-sde-assignment)  
**Submission Notion Form:** [Hiver SDE Intern Submission Form](https://intelligent-bar-256.notion.site/39492cbf0da2800682cfc78a600a745f)  

---

## 1. Problem Framing & Scope

### 1.1 Selected Brand Rationale: Why AmazonHelp?
To build a credible AI support agent, the underlying dataset must contain diverse, actionable customer issues with realistic boundaries between self-service resolution and human escalation. We evaluated four candidate brands across the 2.81-million-tweet Kaggle dataset:
1. **AppleSupport** (106,860 tweets): While voluminous, over 70% of inbound conversations in late 2017 were dominated by a single iOS 11.1 autocorrect bug ("I" converting to "A [?]"), and brand replies were heavily canned ("Please send us a DM").
2. **SpotifyCares** (43,265 tweets) & **Uber_Support** (56,270 tweets): Overwhelmingly redirected users to private DMs without providing publicly verifiable troubleshooting steps or policy resolutions.
3. **AmazonHelp** (169,840 outbound tweets, 143,721 paired interactions): Represents the premier candidate. Customer inquiries span shipment tracking, return policies, damaged goods, payment/security failures, device technical support (Echo, Kindle, Fire TV), and digital subscriptions (Prime Video). Crucially, Amazon's responses contain distinct, verifiable operational resolutions and links (e.g., `amazon.com/returns`, `amazon.com/your-orders`, `amazon.com/help`), providing rich ground truth for retrieval-augmented generation.

### 1.2 Definition of "Good" for Amazon Customer Support
A trustworthy AI support agent must satisfy four strict operational invariants:
1. **Zero Policy Hallucination:** Never promise financial refunds, delivery timeframes, or compensation not grounded in historical evidence or official Amazon policy.
2. **Deterministic Risk Escalation:** Immediate, fail-safe handoff to human agents for account security, suspected fraud, unauthorized charges, or lost shipments marked delivered by the courier.
3. **Conversational Grounding:** Retain multi-turn conversational context so follow-up customer messages (e.g., "It still hasn't arrived" or "I already tried that") are correctly contextualized rather than answered as disconnected queries.
4. **Actionable & Empathetic Replies:** Provide direct, official self-service URLs and step-by-step resolution paths while maintaining customer trust.

### 1.3 Non-Goals (What We Intentionally Chose NOT to Build)
- **Direct Database Mutation / Transactional API Execution:** We do not execute real account closures, credit card refunds, or shipment re-routing directly from Twitter. Public social channels are insecure; the agent acts as an empathetic, grounded triage and guidance interface.
- **Unbounded Open-Domain Chat:** The agent does not engage in general banter or brand chit-chat. Non-support inquiries are politely directed to official Amazon channels.
- **Black-Box End-to-End LLM Prompting Without Retrieval:** We rejected zero-shot LLM generation without retrieval evidence because LLMs hallucinate return windows and refund policies when ungrounded.

---

## 2. System Architecture & Methodology

```
                   [ Incoming Customer Message ]
                                 │
                                 ▼
             [ Context Reconstructor (Multi-Turn Buffer) ]
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       [ Intent Classifier ]           [ Hybrid RAG Retriever ]
     (ML + Priority Patterns)        (Dense BGE + BM25 Okapi)
                 │                               │
                 │   Top-K Historical Evidence   │
                 └───────────────┬───────────────┘
                                 ▼
                  [ Multi-Factor Escalation Engine ]
             (Confidence + Intent Risk + Churn Signals)
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
            [ Decision: AUTO ]       [ Decision: HUMAN ]
                    │                         │
                    └────────────┬────────────┘
                                 ▼
                     [ Grounded Reply Generator ]
                    (Strict Evidence-Conditioned)
                                 │
                                 ▼
                     [ Structured Output JSON ]
```

### 2.1 Conversation Thread Reconstruction
Tweets in `twcs.csv` are flat records linked via `in_response_to_tweet_id` and `response_tweet_id`. We reconstructed conversational threads backwards from verified AmazonHelp outbound resolutions to root customer inquiries. This yielded **61,809 clean, multi-turn conversation cases**. Customer handles (`@115820`) and internal agent signatures (`^TN`, `^AG`) were sanitized while preserving official Amazon URLs and domain pathways.

### 2.2 Compact 8-Class Intent Taxonomy
Using TF-IDF cluster analysis and domain inspection of 50,000 AmazonHelp customer issues, we defined an 8-class taxonomy covering the support problem space:
1. `delivery_status_tracking`: Shipment tracking, courier delays, expected arrival dates.
2. `return_or_refund`: Return initiation, return window policies, drop-off locations, refund status.
3. `damaged_or_missing_item`: Broken goods, tampered boxes, missing items from parcel.
4. `order_change_or_cancel`: Cancelling items before dispatch, updating delivery address.
5. `account_and_payment_security`: 2FA/OTP failures, unauthorized charges, hacked accounts, account closures.
6. `prime_and_subscription`: Prime annual charges, student discounts, cancellation, streaming access.
7. `technical_product_support`: Hardware and software troubleshooting for Echo/Alexa, Kindle, Fire TV Stick.
8. `general_feedback_and_complaints`: Delivery driver misconduct, property damage, customer service complaints.

### 2.3 Historical Retrieval (Hybrid RAG)
Historical support cases serve as factual evidence. We built a dual-index architecture over 10,000 resolved historical cases:
- **Lexical BM25 (Okapi):** Captures exact product names, error codes, and order tracking terms.
- **Dense Vector Search (`BAAI/bge-small-en-v1.5`):** Captures semantic equivalence across paraphrased customer complaints.
- **Rank Fusion:** Fuses sparse and dense scores ($\alpha = 0.55$) to retrieve the top-2 most relevant historical cases, passed directly into the generation prompt as verifiable evidence (`evidence: ["case_10231", "case_8452"]`).

### 2.4 Multi-Factor Escalation Policy
Rather than relying on a single heuristic, the escalation engine evaluates six independent signals:
1. **Sensitive Intent Gate:** Inquiries classified as `account_and_payment_security` or courier property misconduct automatically route to human agents.
2. **Missing Delivery Dispute:** If carrier tracking claims "delivered" but customer reports non-receipt, escalate immediately.
3. **Intent Confidence Floor:** If classifier confidence drops below $0.65$, route to human triage.
4. **Retrieval Grounding Floor:** If the top retrieved historical case similarity falls below $0.40$, escalate due to lack of historical precedent.
5. **Customer Sentiment & Churn Signals:** Explicit requests for humans ("speak to an agent", "supervisor") or legal/abusive keywords trigger escalation.
6. **Multi-Turn Persisting Issue:** If conversation context exceeds 3 unresolved turns, hand off to human support.

---

## 3. Evaluation Strategy & The Golden Evaluation Set

### 3.1 Strict Train / Knowledge Base / Evaluation Separation
To guarantee zero data leakage:
- 10,000 historical cases were isolated into the **Retrieval Knowledge Base** (used strictly as retrieval evidence).
- 2,000 cases were reserved into an **Evaluation Pool** completely disjoint from the knowledge base.
- Exactly **200 examples** were hand-inspected and labelled to form the **Golden Evaluation Set** (`evaluation/golden.jsonl`).

### 3.2 Golden Evaluation Set Characteristics
- **Size:** 200 hand-labelled examples.
- **Multi-Turn Context:** 176 out of 200 examples (88.0%) include prior conversational context.
- **Action Balance:** 131 Auto-Handle cases, 69 Human Escalation cases.
- **Stratified Intent Distribution:** All 8 intents represented (including minority classes: 25 account security, 25 Prime subscriptions, 17 damaged items, 7 order cancellations).
- **Schema:**
  ```json
  {
    "id": "golden_001",
    "customer_message": "Can you still trace? It was supposed to be a gift for someone today.",
    "context": ["Customer: Order id# 4__credit_card__ was supposed to be delivered today. No sign of it yet! Delivery agent not taking call."],
    "intent": "delivery_status_tracking",
    "expected_action": "human",
    "reason": "Missing package dispute where courier agent is unresponsive on guaranteed delivery day.",
    "reference_resolution": "I'd like to have a closer look. Please reach out via our verified form..."
  }
  ```

### 3.3 LLM-as-a-Judge Rubric & Human Agreement Validation
To avoid vague "is it good?" scoring, we established four independent 0–3 point dimensions:
1. **Groundedness (0-3):** 3 = 100% supported by retrieved cases / official Amazon portals; 0 = hallucinated refund amount or false delivery guarantee.
2. **Correctness (0-3):** 3 = Addresses exact root problem; 0 = factually wrong or wrong intent.
3. **Helpfulness (0-3):** 3 = Empathetic, concise, provides actionable link/next step; 0 = unhelpful canned dead end.
4. **Escalation Appropriateness (0-3):** 3 = Optimal routing with clear reason; 0 = auto-handling security fraud or escalating trivial FAQ.

We validated the automated judge against a double-annotated human expert subset of 40 examples, calculating Pearson correlation ($r$), Mean Absolute Error (MAE), and agreement percentages.

---

## 4. Empirical Results & Baseline Comparison

### 4.1 Comparative Results Table

| System / Model | Intent Accuracy | Intent Macro-F1 | Escalation Accuracy | Escalation F1 (Human) | Missed Escalation Risk | LLM-Judge Overall (0-3) | Groundedness Score (0-3) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trivial Baseline** (Majority Intent + Canned Reply + Auto-Handle) | 0.255 | 0.051 | 0.655 | 0.000 | 1.000 (100% Missed) | 1.53 | 2.00 |
| **Simple Baseline** (TF-IDF + BM25 + Keyword Escalation) | 0.750 | 0.704 | 0.705 | 0.253 | 0.855 (85.5% Missed) | 2.19 | 2.00 |
| **Production AI Agent** (Hybrid RAG + Multi-Factor Escalation) | **0.815** | **0.829** | **0.785** | **0.650** | **0.420** (**42.0% Missed**) | **2.32** | **2.00** |

*(Note: Benchmark table reflects the deterministic offline reproducible evaluation mode running locally in ~2 seconds with zero API keys. When live LLM generation is optionally enabled via `USE_LIVE_LLM_API=true` per `.env.example`, generation scores reach 2.50 Overall / 2.52 Groundedness).*

### 4.2 Per-Intent Performance (Production Agent)

| Intent Category | Precision | Recall | F1-Score | Golden Support |
| :--- | :---: | :---: | :---: | :---: |
| `delivery_status_tracking` | 0.746 | 0.804 | 0.774 | 51 |
| `return_or_refund` | 0.783 | 0.720 | 0.750 | 25 |
| `damaged_or_missing_item` | 0.889 | 0.941 | 0.914 | 17 |
| `order_change_or_cancel` | 0.778 | 1.000 | 0.875 | 7 |
| `account_and_payment_security` | 0.750 | 0.840 | 0.792 | 25 |
| `prime_and_subscription` | 0.955 | 0.840 | 0.894 | 25 |
| `technical_product_support` | 0.850 | 0.680 | 0.756 | 25 |
| `general_feedback_and_complaints` | 0.880 | 0.880 | 0.880 | 25 |
| **Macro Average** | **0.829** | **0.838** | **0.829** | **200** |

### 4.3 Escalation Quality & Risk Analysis
- **Escalation Accuracy:** 78.5% (157/200 correct routing decisions).
- **Human Escalation Recall:** 58.0% (40/69 high-risk customer issues successfully caught).
- **Missed Escalation Risk (False Negatives):** Reduced from 100% in Trivial and 85.5% in Simple baseline down to **42.0%** in the Production AI system.
- **Unnecessary Escalation Rate (False Positives):** Only 10.7% (only 14 safe inquiries sent to human queues, preserving human agent capacity).

### 4.4 LLM Judge Reliability & Human Agreement
- **Pearson Correlation ($r$):** `0.8030` ($p < 10^{-10}$), demonstrating robust statistical alignment with human expert ratings.
- **Mean Absolute Error (MAE):** `0.369` on a 3.0 scale.
- **Agreement within $\pm 0.5$ points:** `85.0%`.
- **Exact Agreement:** `15.0%`.  
*(Note: Reflects deterministic offline mode; live LLM mode reaches $r = 0.8371$ / MAE = 0.376).*

---

## 5. Failure Analysis: Top 5 Real Failure Modes

Through granular inspection of the evaluation errors on the Golden Set, we identified the five most important failure modes:

### Failure Mode 1: Multi-Turn Context Ellipsis & Implicit Follow-ups
- **Real Example (`golden_042`):**
  - Context: `[Customer: "My package was supposed to arrive yesterday.", Brand: "Please check your tracking link..."]`
  - Latest Customer Message: *"It still says out for delivery from 9 AM yesterday. What should I do now?"*
- **Observed Failure:** Classifier predicts `delivery_status_tracking` and initially considers auto-handling with the same tracking link.
- **Root Cause:** The customer's latest utterance lacks the explicit words "missing" or "dispute", but within the multi-turn context, the delivery window has expired, converting an informational inquiry into an escalation.
- **Improvement Hypothesis:** Incorporate temporal difference analysis (e.g., comparing current time against estimated delivery timestamp in context) to trigger escalation when delay exceeds carrier cutoff.

### Failure Mode 2: Cross-Border & Regional Storefront Misdirection
- **Real Example (`golden_118`):**
  - Customer Message: *"Why can't I stream this movie on Prime Video in Ireland?"*
  - Retrieved Case: US Amazon Prime Help article (`amazon.com/help`).
- **Observed Failure:** The agent provided a generic `.com` help link rather than noting Amazon UK / Amazon Ireland regional digital licensing boundaries.
- **Root Cause:** Retrieval knowledge base lacks explicit storefront metadata (e.g., `.co.uk`, `.de`, `.com`) to filter region-specific resolutions.
- **Improvement Hypothesis:** Add metadata extraction for country/currency codes in the customer message and filter retrieval candidates by regional storefront.

### Failure Mode 3: Mixed-Intent Inquiries (Damaged Item + Refund Request)
- **Real Example (`golden_089`):**
  - Customer Message: *"My glass coffee pot arrived shattered in the box. I want my money back right now, I'm not waiting for a replacement!"*
- **Observed Failure:** Intent classified as `return_or_refund` rather than `damaged_or_missing_item`.
- **Root Cause:** High lexical overlap with refund keywords masked the root physical damage claim, leading to a standard return instructions reply rather than broken-glass safety handling.
- **Improvement Hypothesis:** Transition to hierarchical or multi-label intent classification where physical condition takes precedence over settlement preference.

### Failure Mode 4: Subtle Overdue Refund Thresholds
- **Real Example (`golden_134`):**
  - Customer Message: *"I dropped off my return package at UPS 4 days ago and still no refund."*
- **Observed Failure:** System escalated to a human agent citing "refund delay".
- **Root Cause:** Standard Amazon return policy allows 14 business days from courier scan before a refund is considered overdue. The 4-day inquiry is a standard self-service status check, resulting in an unnecessary human escalation (false positive).
- **Improvement Hypothesis:** Embed numerical policy regex rules: if return drop-off duration $< 7$ days, auto-handle with standard return processing timeframe; if $> 14$ days, escalate.

### Failure Mode 5: Hardware Defect vs. Routine Setup Confusion
- **Real Example (`golden_172`):**
  - Customer Message: *"My Echo Dot light ring is solid red and the mic button doesn't respond even after unplugging."*
- **Observed Failure:** System provided basic restart troubleshooting instructions instead of recognizing a frozen hardware failure.
- **Root Cause:** The phrase "unplugging" indicates the customer already executed the standard restart procedure, making a repeated restart instruction frustrating.
- **Improvement Hypothesis:** Add "prior action extraction" to detect whether standard self-service troubleshooting steps were already attempted before recommending them.

---

## 6. "What is Misleading About My Headline Number?" (Mandatory Section)

Our production system achieved a headline **81.5% Intent Accuracy** and **78.5% Escalation Accuracy**. While these numbers demonstrate marked improvement over the baselines (25.5% and 65.5% on Trivial; 75.0% and 70.5% on Simple), **relying solely on these headline figures is dangerous in production customer support**:

1. **Accuracy Hides Class Imbalance & Skews:**
   In real Twitter support data, `delivery_status_tracking` constitutes over 65% of raw inbound messages. A naive model that always predicts delivery tracking could achieve ~65% overall accuracy while failing completely on critical minority classes like `account_and_payment_security` or `damaged_or_missing_item`. Macro-F1 (**0.829**) and per-intent breakdown are far more informative than overall accuracy.
2. **Escalation Accuracy Treats All Errors Equally (The Asymmetry of Cost):**
   A false positive (escalating a safe tracking question to a human) costs ~$3–$5 in agent labor. A false negative (auto-handling an unauthorized credit card charge or account takeover with a generic canned link) can result in severe financial loss, fraud liability, regulatory penalties, and customer churn. A headline escalation accuracy of 78.5% sounds respectable, but the **42.0% missed escalation rate (29/69 critical cases missed)** represents the true operational risk that must be monitored and mitigated.
3. **Synthetic / Curated Test Set Representation:**
   Our 200-example Golden Set was carefully curated and balanced. Real-world social feeds contain sarcasm, memes, ASCII art, multi-brand mentions, and unintelligible rants. In open production, out-of-distribution performance will degrade without continuous active-learning triage.

---

## 7. What We Would Do With One More Week

If given one additional week of engineering time, we would prioritize three high-leverage initiatives:

1. **Temporal & Entity-Aware Policy Reasoner:**
   Integrate lightweight NER and date-parser extractors to automatically extract order dates, courier tracking numbers, and delivery promises from conversational context. This would directly resolve Failure Modes 1 and 4 by deterministically verifying whether an inquiry is within or outside standard policy windows.
2. **Multi-Label & Hierarchical Intent Classifier:**
   Refactor single-intent classification into a two-level hierarchical architecture:
   - Level 1: Risk & Problem Domain (Safety/Fraud vs Physical Goods vs Digital/Account).
   - Level 2: Specific Operational Intent.
   This prevents mixed-intent collisions (e.g. shattered package + refund request).
3. **Production Active Learning & Shadow Deployment:**
   Deploy the evaluation harness as a continuous shadow pipeline on live streaming tweets. Route low-confidence ($<0.70$) predictions to a human-in-the-loop review queue to continuously expand the Golden Evaluation Set with newly emerging edge cases (e.g. holiday shipping backlogs, new device releases).

---

## 8. Decision Log (15 Non-Obvious Decisions)

1. **Brand Selection: AmazonHelp over AppleSupport:** Chose AmazonHelp due to broader operational variety (shipping, returns, hardware, payments) compared to AppleSupport's extreme skew toward canned DM links and the November 2017 iOS 11 autocorrect bug.
2. **Compact 8-Class Intent Taxonomy:** Consolidated dozens of niche sub-intents into 8 actionable classes aligned directly with operational fulfillment routing.
3. **Exclusion of Transactional Execution:** Deliberately chose not to implement real database write actions (e.g. issuing refunds) over public Twitter for basic cybersecurity reasons.
4. **Golden Set Size (200 Examples):** Selected 200 hand-labelled examples as the optimal threshold providing statistical significance (confidence intervals $\pm 3.5\%$) while enabling 100% manual review.
5. **High Multi-Turn Ratio (88.0%) in Golden Set:** Specifically sampled threads with prior dialogue history rather than isolated single-turn tweets to rigorously test context preservation.
6. **Strict 80/20 Train-KB vs. Evaluation Split:** Partitioned the 12,000 processed cases into a 10,000-case knowledge base and a 2,000-case eval pool with zero overlap, preventing retrieval data leakage.
7. **Trivial Baseline Definition:** Designed the trivial baseline to predict the majority intent (`delivery_status_tracking`), the single most frequent canned response, and constant auto-handling to expose the true difficulty of the task.
8. **Simple Baseline Selection:** Used unigram+bigram TF-IDF with balanced Logistic Regression and BM25 retrieval to establish a strong, non-trivial traditional NLP benchmark.
9. **Hybrid RAG over Pure Generation:** Mandated historical case retrieval to enforce grounding and prevent LLM hallucination of refund amounts or delivery dates.
10. **Exposing Historical Resolution in Evidence:** Fed both the historical customer problem and the historical brand reply to the generation prompt, providing an operational template for the agent's tone and URL selection.
11. **Asymmetric Escalation Penalties:** Tuned the escalation engine to prioritize recall over precision on high-risk intents (`account_and_payment_security`), treating false negatives as 10x more costly than false positives.
12. **Independent 4-Scale LLM Judge Rubric:** Rejected binary "good/bad" scoring in favor of four distinct 0–3 dimensions (Groundedness, Correctness, Helpfulness, Escalation Appropriateness) to pinpoint exact failure modes.
13. **Local ONNX Dense Embeddings (`bge-small-en-v1.5`):** Selected FastEmbed ONNX runtime to enable dense semantic embeddings with zero GPU requirement, ensuring anyone can reproduce the pipeline on standard hardware in minutes.
14. **Deterministic Grounded Fallback Mode:** Built a high-fidelity local grounded generation mode so the complete test and evaluation pipeline runs offline without requiring external API keys.
15. **Public Hiver Notion Submission Integration:** Explicitly verified the official submission link (`https://intelligent-bar-256.notion.site/39492cbf0da2800682cfc78a600a745f`) and formatted the project structure for immediate reviewer evaluation.
