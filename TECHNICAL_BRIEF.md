# SupportMind: Technical Architecture Brief
**Advanced Confidence-Gated Ticket Orchestration**

---

## 1. Executive Summary
SupportMind is a production-grade ticket routing engine that moves beyond simple classification. It uses an **Ensemble Model** combined with **Uncertainty Quantification** to automate 80%+ of ticket triage while maintaining enterprise-level safety gates for ambiguous or complex customer issues.

---

## 2. Machine Learning Core
### Ensemble Architecture
We utilize a weighted soft-voting ensemble:
1.  **Semantic Layer**: DistilBERT (Fine-tuned on support-specific datasets) to capture context and intent.
2.  **Keyword Layer**: TF-IDF + Logistic Regression to handle explicit technical n-grams and error codes.

### Calibration & Precision Engineering
To ensure high-precision routing, we implemented:
*   **Temperature Scaling ($T=0.7$)**: Probabilities are sharpened to reduce background noise in unrelated categories.
*   **MC Dropout (Monte Carlo)**: During inference, we run multiple passes with active dropout to measure **Predictive Variance**. This allows us to detect when the model is "guessing."
*   **Shannon Entropy ($H$)**: We measure the information chaos of the output distribution. If $H > 1.2$, the system triggers a **Clarification Gate** instead of routing blindly.

---

## 3. Operational Orchestration
### Multi-Intent Segmentation
The engine split incoming text into semantic segments (using conjunction-aware regex). 
*   **Example**: "I can't log in AND I need an invoice."
*   **Action**: `MULTI-ROUTE` (Primary: Tech, Secondary: Billing).

### Impact-Driven SLA Risk
Unlike basic models that use confidence for priority, SupportMind uses a separate **Impact Engine**:
*   **Operational Danger**: Weighted flags for "crash", "outage", and "blocked".
*   **Complexity**: Analyzes technical depth (e.g., "API integration" vs. "password reset").
*   **Sentiment Penalty**: Negative sentiment directly escalates SLA risk to prevent churn.

---

## 4. Explainable AI (XAI)
To provide transparency to human agents, we integrated **SHAP (SHapley Additive exPlanations)**. This identifies exactly which words (e.g., "export", "failing") contributed to the model's decision, building trust in automated systems.

---

## 5. System Health & Monitoring
The system includes a live **Telemetry API** (`/metrics`) that tracks:
*   **Routing Distribution**: Auto-route vs. Clarify vs. Escalate.
*   **Latency Metrics**: 95th percentile response times (< 40ms on CPU).
*   **Model Lineage**: Tracking active model versions and fallback states.

---

**Developed by: Asmitha — SupportMind v2.0**
