---
title: SupportMind
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# SupportMind

**Confidence-gated AI ticket routing for B2B SaaS support teams**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-orange.svg)](https://huggingface.co/)
[![CI Status](https://github.com/asmitha2025/supportfloww/actions/workflows/ci.yml/badge.svg)](https://github.com/asmitha2025/supportfloww/actions)

SupportMind is an uncertainty-aware support-routing engine built for the problem that most ticket classifiers ignore: **AI can be confidently wrong on ambiguous customer tickets**.

Instead of always forcing a category, SupportMind uses a three-tier decision gate:

| Decision | Signal | Action |
| --- | --- | --- |
| Route | High confidence, low ambiguity | Auto-assign to the best support queue. |
| Clarify | Medium confidence or competing intents | Ask one targeted question before routing. |
| Escalate | Low confidence or high-risk uncertainty | Send to human triage. |

This makes the system useful for real support operations where a wrong route can cost agent time, SLA performance, and customer trust.

## Why It Matters

Traditional softmax classifiers return a category even when the input is unclear. In customer support, that means tickets like "export is broken and my invoice is wrong" can be routed to only one team, even though two teams need to act.

SupportMind is designed to fail safely:

- Detects uncertainty with Monte Carlo Dropout and entropy.
- Handles multi-intent tickets such as billing plus technical support.
- Generates clarification questions when the model should not guess.
- Predicts SLA risk using operational features.
- Provides a lightweight fallback router so clean clones, demos, and CI still work without private model files.

## Demo Flow

Run the API and dashboard:

```bash
pip install -r requirements.txt
uvicorn src.api:app --host 0.0.0.0 --port 7860 --reload
```

Open:

```text
http://localhost:7860/
```

Try these tickets:

```text
The API endpoint /v2/export returns a 500 error when batch size exceeds 1000 records.
```

```text
The invoice is wrong, and also SSO login is broken for our managers.
```

```text
Could you please help resolve this? This is becoming difficult for our onboarding team and we are disappointed with repeated delays.
```

## Architecture

```text
Incoming ticket
  -> validation and cleaning
  -> feature extraction
  -> confidence-gated router
  -> multi-intent detection
  -> SLA risk scoring
  -> route, clarify, or escalate
```

Core modules:

- `src/api.py` - FastAPI app, dashboard serving, orchestration, guardrails.
- `src/ensemble_router.py` - Transformer plus sklearn ensemble with fallback mode.
- `src/confidence_router.py` - Monte Carlo Dropout transformer router.
- `src/clarification_engine.py` - LLM/template clarification question generation.
- `src/feature_extraction.py` - NLP, sentiment, urgency, and complexity signals.
- `src/sla_predictor.py` - SLA breach risk prediction.
- `tests/` - API, router, clarification, SLA, validator, and feature tests.

## Runtime Modes

SupportMind supports two runtime paths:

- **Full model mode:** uses transformer model files when available.
- **Fallback mode:** uses an embedded sklearn router so tests, demos, and CI can run without ignored local model artifacts.

Model artifacts are intentionally ignored from git:

```text
models/
data/raw/
data/processed/
```

This keeps the repository lightweight and makes the public project easier to clone.

## Benchmark Framing

SupportMind reports two types of evaluation:

| Metric | In-distribution synthetic | Out-of-distribution hand-crafted |
| --- | ---: | ---: |
| Routing accuracy | 100.0% | 57.3% |
| Auto-route precision | 100.0% | 100.0% |
| Ambiguous-ticket accuracy | N/A | 30.0% |

The OOD number is intentionally honest. The important result is not that the model guesses every unclear ticket correctly. The important result is that it avoids unsafe auto-routing when it is uncertain.

## Safety Approach

No AI router can guarantee that every unseen customer ticket will be correct. SupportMind handles that risk by making uncertainty part of the workflow:

- Low-confidence tickets are escalated instead of forced into a queue.
- Medium-confidence tickets trigger clarification instead of silent guessing.
- Multi-intent tickets expose primary and secondary queues.
- Non-neutral sentiment must include visible evidence in the response.
- Demo consistency tests verify that displayed queues match the probability chart.

For production use, this should still be paired with real support data, calibration checks, human review thresholds, and agent feedback loops. The demo is honest about that boundary.

## API

Interactive docs are available at:

```text
http://localhost:7860/docs
```

Main endpoint:

```http
POST /route
```

Example request:

```json
{
  "text": "The API endpoint /v2/export returns a 500 error when batch size exceeds 1000 records.",
  "customer_id": "cust_8910"
}
```

Example response:

```json
{
  "action": "route",
  "confidence": 0.94,
  "entropy": 0.12,
  "top_category": "technical_support",
  "sla_breach_probability": 0.15,
  "latency_ms": 45.2
}
```

Other endpoints:

- `POST /clarify` - generate a clarification question.
- `POST /sla/predict` - score SLA breach risk.
- `POST /churn/signal` - extract churn-risk signals from conversation history.
- `POST /explain` - return token-level explanation data.
- `GET /metrics` - live routing distribution and service stats.
- `GET /model/status` - active model/fallback status.

## Tests

Run:

```bash
python -m pytest -q
```

Current local result:

```text
55 passed
```

The test suite also passes in a clean-checkout simulation without local model files, matching the GitHub Actions environment.

For demo QA, add new edge-case tickets to `DEMO_TICKETS` in `tests/test_demo_consistency.py`. That test checks that multi-route labels match the probability chart and non-neutral sentiment has visible evidence.

## Docker

```bash
docker build -t supportmind .
docker run -p 7860:7860 supportmind
```

## Portfolio Positioning

SupportMind is built as a hiring-facing AI engineering project for support platforms such as Zoho Desk, Freshworks, Zendesk, Intercom, Salesforce Service Cloud, and similar customer operations products.

The project demonstrates:

- ML system design beyond a basic classifier.
- Practical uncertainty handling.
- FastAPI service design.
- Frontend demo experience.
- CI-tested behavior.
- Honest benchmark communication.

## Author

**Asmitha**
BSc Data Science, 2026

Portfolio theme: building AI systems that know when they should not guess.
