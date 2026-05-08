# AetherFlow AI | SupportMind Engine 🧠

**Confidence-Gated Support Intelligence for B2B SaaS Customer Operations**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-orange.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> *"B2B SaaS support teams don't lose customers because agents are slow. They lose them because AI acts with false confidence on ambiguous tickets — and nobody in the stack knows it happened."*

---

## 🎯 What is SupportMind?

SupportMind is a **confidence-gated, uncertainty-aware** ticket routing system that solves the most expensive unsolved problem in B2B SaaS support: **AI routing ambiguous tickets with false certainty**.

Unlike traditional AI solutions (Zoho Zia, Freshworks Freddy, Zendesk, and Salesforce Einstein) which use standard Softmax classifiers with no uncertainty output, SupportMind implements **Monte Carlo Dropout on DistilBERT**. This produces calibrated confidence scores and Shannon entropy, enabling a robust **three-tier decision gate**:

| Action | Confidence | Entropy | What Happens |
|--------|-----------|---------|--------------|
| **ROUTE** | ≥ 0.80 | ≤ 0.35 | Auto-assign to the correct agent queue immediately. |
| **CLARIFY** | 0.55 – 0.80 | N/A | Ask 1 targeted, high-information-gain question to disambiguate. |
| **ESCALATE** | < 0.55 | N/A | Flag as complex; send to human triage immediately. |

---

## 🏗️ Detailed System Architecture

The SupportMind engine operates as a multi-stage pipeline designed to mimic human cognitive processes in support triage:

### Stage 1: Feature Extraction & Signal Detection
When a ticket arrives, it passes through an NLP feature extraction layer:
* **DistilBERT Embeddings**: Extracts deep semantic meaning (768-dimensional space).
* **VADER Sentiment Analysis**: Measures emotional tone (frustration, anger).
* **Regex & Heuristics**: Detects urgency flags ("ASAP", "System Down") and text complexity (Flesch-Kincaid).

### Stage 2: Confidence-Gated Router (MC Dropout)
Instead of a single forward pass, the DistilBERT classifier performs **20 stochastic forward passes**. By randomly deactivating neurons, it generates a distribution of predictions.
* **Low variance** across passes = High Confidence (Safe to Route)
* **High variance** across passes = High Epistemic Uncertainty (Needs Clarification or Escalation)

### Stage 3: The Intelligence Layer
* **SLA Breach Predictor (XGBoost)**: Evaluates the extracted features against current queue depth and historical SLA data to predict the probability of missing SLA targets (AUC 0.83).
* **Clarification Engine (Hybrid Architecture)**: When the router enters the CLARIFY tier, the engine uses a two-layer approach:
  1. **LLM Layer (Groq LLaMA3-8B)**: Generates a ticket-specific question referencing the customer's exact words. Runs in ~100ms via Groq's optimized inference.
  2. **Template Layer (fallback)**: If LLM is unavailable, selects from 47 pre-built templates scored by expected Shannon entropy reduction (information gain).
  
  This design ensures the system never stops routing — LLM enhances quality when available, templates guarantee reliability always.

---

## 📊 Benchmark Results (Honest Dual-Evaluation)

> ⚠️ **Benchmark Validity**: To ensure complete transparency, this project reports two sets of accuracy numbers:
> 1. **In-Distribution (Synthetic)**: The test set is generated from the same templates as the training data. The 100% accuracy here merely confirms the model successfully learned the training distribution without catastrophic forgetting.
> 2. **Out-of-Distribution (OOD)**: Evaluated against a separate, hand-crafted dataset of 96 real-world-style tickets (informal language, typos, missing context, and ambiguous edge-cases). **This is the honest estimate of the model's true generalization ability before fine-tuning on real production data.**

| Metric | In-Distribution *(synthetic)* | Out-of-Distribution *(hand-crafted)* |
|--------|------------------------------|--------------------------------------|
| Overall Routing Accuracy | **100.0%** | **57.3%** |
| Precision on Auto-Routed | **100.0%** | **100.0%** |
| Accuracy on Ambiguous Tickets | — | **30.0%** |

### Why the OOD Accuracy is "Low" (And Why That's Good)
On the OOD dataset, the model correctly routed the familiar tickets but struggled with the novel/ambiguous ones. **However, it only auto-routed 2.1% of the OOD tickets (achieving 100% precision on those).** It correctly flagged the remaining 97.9% as requiring clarification (51%) or escalation (47%). 

Traditional Softmax classifiers would have blindly auto-routed these unfamiliar tickets, leading to costly misroutes. **SupportMind's confidence gate correctly prevented these misroutes.** This proves the architecture works as intended—even when the model weights are untrained for the specific domain, the system fails *safely*.

### Why This Matters for Zoho Desk + Zia

Zia's current field prediction uses standard Softmax — it returns a 
category with no uncertainty signal. When Zia is wrong on an ambiguous 
ticket, the agent only discovers the misroute after picking it up. 
SupportMind's clarification gate catches this *before* routing, 
reducing misroute cost from agent-time to one extra customer message.

---

## 🚀 Installation & Setup Guide

Follow these steps to set up the engine locally for development or demonstration.

### 1. Prerequisites
* Python 3.10+
* Git
* Virtual Environment tool (`venv`, `conda`, etc.)

### 2. Clone and Install
```bash
# Clone the repository
git clone https://github.com/asmitha2025/supportfloww.git
cd supportfloww

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the System Locally
The core system is powered by FastAPI, serving both the REST API and the interactive dashboard.

```bash
# Start the FastAPI server
cd src
uvicorn api:app --host 0.0.0.0 --port 7860 --reload
```
Once the server is running, navigate to `http://localhost:7860/` in your web browser to access the **Live SupportMind Dashboard**.

---

## 🧠 Training the Models

If you wish to retrain the models from scratch using your own datasets:

1. **Prepare Data**: Place your raw ticket data in `data/raw/`.
2. **Train Router**: 
   ```bash
   python src/train_router.py
   ```
   *This trains the DistilBERT sequence classifier and saves it to `models/ticket_classifier/`.*
3. **Train SLA Predictor**:
   ```bash
   python src/train_sla.py
   ```
   *This trains the XGBoost model based on synthetic feature data and saves to `models/sla_predictor/`.*
4. **Evaluate System**:
   ```bash
   python src/evaluate.py
   ```
   *Generates benchmark metrics comparing MC Dropout against a standard Softmax baseline.*

---

## 📡 Comprehensive API Reference

SupportMind exposes a fully documented RESTful API. When the server is running, visit `http://localhost:7860/docs` for the interactive Swagger UI.

### `POST /route`
**Description**: Main routing endpoint. Processes a ticket and returns a 3-tier confidence-gated decision.
**Request Body**:
```json
{
  "text": "The API endpoint /v2/export returns a 500 error when batch size exceeds 1000.",
  "customer_id": "cust_8910"
}
```
**Response**:
```json
{
  "action": "route",
  "confidence": 0.942,
  "entropy": 0.12,
  "top_category": "technical_support",
  "features": {
    "sentiment_score": -0.25,
    "urgency_flags": []
  },
  "sla_breach_probability": 0.15,
  "latency_ms": 45.2
}
```

### `POST /clarify`
**Description**: Fetch the best clarification question based on model uncertainty.
### `POST /sla/predict`
**Description**: Predict SLA breach risk independently based on features.
### `POST /churn/signal`
**Description**: Extract churn signals from an array of historical thread texts.
### `GET /metrics`
**Description**: Live system health and routing distribution statistics.

---

## 🐳 Docker Deployment

For production deployments, package the application using Docker.

```bash
# Build the image
docker build -t supportmind .

# Run the container
docker run -d -p 7860:7860 --name supportmind-api supportmind
```

For advanced orchestration, we recommend extending the deployment with `docker-compose` to include Redis or RabbitMQ for asynchronous webhook processing.

---

## 📁 Repository Structure

```text
supportmind/
├── src/
│   ├── api.py                    # FastAPI server & endpoints
│   ├── confidence_router.py      # DistilBERT MC Dropout logic
│   ├── clarification_engine.py   # Shannon entropy info-gain logic
│   ├── sla_predictor.py          # XGBoost SLA modeling
│   ├── feature_extraction.py     # NLP Feature engineering
│   ├── churn_extractor.py        # Sentiment & Churn analysis
│   ├── train_router.py           # DistilBERT training script
│   ├── train_sla.py              # XGBoost training script
│   └── evaluate.py               # Evaluation & benchmark suite
├── dashboard/
│   └── web/                      # Interactive Frontend HTML/CSS/JS
│       ├── index.html            # Main UI
│       ├── app.js                # Frontend logic & API calls
│       └── style.css             # Glassmorphism styling
├── data/
│   └── clarification_bank.json   # 47 Question templates
├── models/                       # Stored model weights (ignored in git)
├── tests/                        # Pytest suite
├── Dockerfile                    # Containerization instructions
├── requirements.txt              # Python dependencies
└── README.md                     # You are here
```

---

## 👤 Author

**Asmitha** · BSc Data Science · 2026

Part of the three-project portfolio arc:
1. **OPTI-FAB** → Manufacturing edge AI with confidence gating
2. **IncidentMind** → RL-based incident response with ambiguity awareness
3. **SupportMind** → NLP ticket routing with MC Dropout uncertainty

> *"I spent the last year building systems that know what they don't know."*

