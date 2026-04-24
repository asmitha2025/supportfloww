# src/api.py
# FastAPI Server — SupportMind API
# SupportMind v1.0 — Asmitha

import os
import sys
import time
import logging
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App Setup ──────────────────────────────────────────────
app = FastAPI(
    title='SupportMind API',
    description='Confidence-Gated Support Intelligence for B2B SaaS Customer Operations',
    version='1.0.0',
    docs_url='/docs',
    redoc_url='/redoc',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy-load ML models ───────────────────────────────────
_router = None
_clarify = None
_sla_pred = None
_churn_ex = None
_feature_ext = None
_stats = {
    'total_routed': 0, 'total_clarified': 0, 'total_escalated': 0,
    'total_requests': 0, 'start_time': datetime.now().isoformat(),
}

def get_router():
    global _router
    if _router is None:
        from confidence_router import ConfidenceGatedRouter
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base, 'models', 'ticket_classifier')
        _router = ConfidenceGatedRouter(model_path)
    return _router

def get_clarify():
    global _clarify
    if _clarify is None:
        from clarification_engine import ClarificationEngine
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        bank_path = os.path.join(base, 'data', 'clarification_bank.json')
        _clarify = ClarificationEngine(bank_path)
    return _clarify

def get_sla():
    global _sla_pred
    if _sla_pred is None:
        from sla_predictor import SLABreachPredictor
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base, 'models', 'sla_predictor', 'sla_xgb.json')
        _sla_pred = SLABreachPredictor(model_path)
    return _sla_pred

def get_churn():
    global _churn_ex
    if _churn_ex is None:
        from churn_extractor import ChurnSignalExtractor
        _churn_ex = ChurnSignalExtractor()
    return _churn_ex

def get_features():
    global _feature_ext
    if _feature_ext is None:
        from feature_extraction import FeatureExtractor
        _feature_ext = FeatureExtractor()
    return _feature_ext


# ── Request/Response Models ───────────────────────────────
class TicketRequest(BaseModel):
    text: str
    customer_id: Optional[str] = None

class SLARequest(BaseModel):
    text_complexity_score: float = 8.0
    agent_queue_depth: int = 10
    customer_tier: int = 3
    hour_of_day: int = 14
    day_of_week: int = 2
    similar_ticket_avg_hrs: float = 4.5
    sentiment_score: float = 0.0
    repeat_issue: int = 0
    escalated_before: int = 0

class ThreadRequest(BaseModel):
    thread_texts: List[str]

class ClarifyRequest(BaseModel):
    text: str
    current_probs: Optional[List[float]] = None
    top_two_classes: Optional[List[str]] = None


# ── Endpoints ─────────────────────────────────────────────
@app.post('/route')
def route_ticket(req: TicketRequest):
    """Main routing endpoint — returns 3-tier confidence-gated decision."""
    start = time.time()
    _stats['total_requests'] += 1

    router = get_router()
    result = router.route(req.text)

    # Get features
    feat_ext = get_features()
    features = feat_ext.extract(req.text)

    # SLA prediction
    sla = get_sla()
    sla_features = {
        'text_complexity_score': features['text_complexity_score'],
        'agent_queue_depth': 15,
        'customer_tier': 3,
        'hour_of_day': datetime.now().hour,
        'day_of_week': datetime.now().weekday(),
        'similar_ticket_avg_hrs': features.get('similar_ticket_avg_hrs', 4.5),
        'sentiment_score': features['sentiment_score'],
        'repeat_issue': 0,
        'escalated_before': 0,
    }
    sla_risk = sla.predict(sla_features)

    # Update stats
    action = result['action']
    if action == 'route': _stats['total_routed'] += 1
    elif action == 'clarify': _stats['total_clarified'] += 1
    else: _stats['total_escalated'] += 1

    # If clarify, get the question
    clarification = None
    if action == 'clarify':
        import numpy as np
        clar = get_clarify()
        probs = np.array(list(result['all_probs'].values()))
        clarification = clar.select_question(probs, result['top_two_classes'])

    elapsed = round((time.time() - start) * 1000, 1)

    return {
        **result,
        'features': features,
        'sla_breach_probability': sla_risk,
        'clarification': clarification,
        'latency_ms': elapsed,
        'customer_id': req.customer_id,
    }


@app.post('/clarify')
def get_clarification(req: ClarifyRequest):
    """Get best clarification question for uncertain ticket."""
    import numpy as np
    clar = get_clarify()

    if req.current_probs:
        probs = np.array(req.current_probs)
    else:
        router = get_router()
        result = router.route(req.text)
        probs = np.array(list(result['all_probs'].values()))
        req.top_two_classes = result['top_two_classes']

    top_two = req.top_two_classes or ['billing', 'technical_support']
    return clar.select_question(probs, top_two)


@app.post('/sla/predict')
def predict_sla(req: SLARequest):
    """Predict SLA breach risk at ticket creation."""
    sla = get_sla()
    features = req.model_dump()
    result = sla.explain(features)
    return result


@app.post('/churn/signal')
def churn_signal(req: ThreadRequest):
    """Extract churn signal from thread history."""
    churn = get_churn()
    return churn.extract(req.thread_texts)


@app.get('/metrics')
def get_metrics():
    """Live system health and routing statistics."""
    total = _stats['total_requests'] or 1
    return {
        'total_requests': _stats['total_requests'],
        'routing_stats': {
            'routed': _stats['total_routed'],
            'clarified': _stats['total_clarified'],
            'escalated': _stats['total_escalated'],
        },
        'routing_distribution': {
            'route_pct': round(_stats['total_routed'] / total * 100, 1),
            'clarify_pct': round(_stats['total_clarified'] / total * 100, 1),
            'escalate_pct': round(_stats['total_escalated'] / total * 100, 1),
        },
        'start_time': _stats['start_time'],
        'model': 'distilbert-base-uncased (MC Dropout)',
    }


@app.get('/health')
def health():
    """Health check for deployment pipelines."""
    return {
        'status': 'ok',
        'model': 'distilbert-base-uncased-finetuned',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat(),
    }


# ── Serve web dashboard ──────────────────────────────────
dashboard_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dashboard', 'web')
if os.path.exists(dashboard_dir):
    app.mount("/dashboard", StaticFiles(directory=dashboard_dir, html=True), name="dashboard")

    @app.get('/')
    def serve_dashboard():
        return FileResponse(os.path.join(dashboard_dir, 'index.html'))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('api:app', host='0.0.0.0', port=7860, reload=True)

