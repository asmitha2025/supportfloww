# src/api.py
# FastAPI Server — SupportMind API
# SupportMind v1.0 — Asmitha

import os
import sys
import re
import time
import logging
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# Aggressive memory and backend management for Windows stability
os.environ['USE_TF'] = '0'
os.environ['USE_JAX'] = '0'
os.environ['USE_TORCH'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict

# Add project paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ticket_validator import TicketValidator
try:
    from interpretability import SupportMindExplainer
except OSError as e:
    print(f"Failed to load interpretability (PyTorch WinError 1455): {e}")
    SupportMindExplainer = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App Setup ──────────────────────────────────────────────
app = FastAPI(
    title='SupportMind API',
    description='Confidence-Gated Support Intelligence for B2B SaaS Customer Operations',
    version='1.0.0',
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
_validator = None
_explainer = None
_stats = {
    'total_routed': 0, 'total_clarified': 0, 'total_escalated': 0,
    'total_requests': 0, 'start_time': datetime.now().isoformat(),
}

@app.on_event("startup")
def startup_event():
    """Pre-load models on startup to prevent thread-lock issues."""
    logger.info("Initializing ML models on main thread...")
    get_router()
    get_clarify()
    get_sla()
    get_churn()
    get_features()
    get_validator()
    get_explainer()
    logger.info("All ML models loaded successfully.")

def get_router():
    global _router
    if _router is None:
        from ensemble_router import EnsembleRouter
        _router = EnsembleRouter(device='cpu')
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

def get_validator():
    global _validator
    if _validator is None:
        _validator = TicketValidator()
    return _validator

def get_explainer():
    global _explainer
    if _explainer is None:
        router = get_router()
        if router.model is not None:
            _explainer = SupportMindExplainer(router.model, router.tokenizer, device='cpu')
    return _explainer

# ── Request Models ─────────────────────────────────────────
class TicketRequest(BaseModel):
    text: str
    customer_id: Optional[str] = "CUST-DEMO"

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

class ExplainRequest(BaseModel):
    text: str
    target_class: Optional[str] = None

# ── Endpoints ─────────────────────────────────────────────
@app.post('/route')
def route_ticket(req: TicketRequest):
    """Main routing endpoint — returns 3-tier confidence-gated decision."""
    start = time.time()
    _stats['total_requests'] += 1
    
    # 1. Validation
    validator = get_validator()
    validation = validator.validate(req.text)

    if not validation['valid']:
        return {
            'action': 'invalid_input',
            'error_type': validation['error_type'],
            'response': validation['response'],
            'confidence': 0.0,
            'entropy': 0.0,
            'sla_risk': 0.0,
            'latency_ms': round((time.time() - start) * 1000, 1),
            'customer_id': req.customer_id,
        }

    clean_text = validation['cleaned_text']
    
    # 2. ML Routing & Features
    router = get_router()
    result = router.route(clean_text)
    
    feat_ext = get_features()
    features = feat_ext.extract(clean_text)
    
    # 3. Multi-Intent Detection (Segmentation)
    segments = [s.strip() for s in re.split(r'\.|\band\b|\balso\b', clean_text, flags=re.I) if len(s.strip().split()) > 3]
    segment_intents = []
    if len(segments) > 1:
        for seg in segments:
            seg_res = router.route(seg)
            if seg_res['confidence'] > 0.65:
                segment_intents.append(seg_res['top_category'])
    
    unique_intents = list(dict.fromkeys(segment_intents))
    is_multi_intent = len(unique_intents) >= 2

    # 4. Operational SLA Risk Engine
    urg_val = features.get('urgency_score', 0.0)
    comp_val = features.get('complexity_score', 0.0)
    sent_val = features.get('sentiment_score', 0.0)
    
    # Base risk: Urgency (50%) + Complexity (30%) + Sentiment Penalty (20%)
    raw_risk = (urg_val * 0.5) + (comp_val * 0.3)
    if sent_val < -0.4: raw_risk += 0.2
    sla_risk = min(max(raw_risk, 0.01), 1.0)

    # 5. Non-Support / Junk Detection
    is_junk = False
    if result['entropy'] > 1.6 and result['confidence'] < 0.4 and urg_val < 0.1 and not features.get('product_entities'):
        is_junk = True
    if features.get('token_count', 0) < 10 and urg_val < 0.1 and not features.get('has_question') and result['confidence'] < 0.6:
        is_junk = True

    # 6. Final Decision Orchestration
    final_decision = {
        'ticket_id': f"SM-{int(time.time()) % 100000:05d}",
        'action': 'route',
        'top_category': result['top_category'],
        'confidence': result['confidence'],
        'entropy': result['entropy'],
        'margin': result['margin'],
        'all_probs': result['all_probs'],
        'sla_risk': round(sla_risk, 4),
        'urgency_score': round(urg_val, 4),
        'complexity_score': round(comp_val, 4),
        'is_multi_intent': is_multi_intent,
        'features': {**features, 'latency_ms': round((time.time() - start) * 1000, 1)},
        'customer_id': req.customer_id,
        'latency_ms': round((time.time() - start) * 1000, 1),
    }

    if is_junk:
        final_decision.update({
            'action': 'invalid_input',
            'error_type': 'non_support',
            'response': "This doesn't appear to be a support request. Please provide more specific details about your issue.",
            'sla_risk': 0.01
        })
    elif is_multi_intent:
        final_decision.update({
            'action': 'multi_route',
            'primary_queue': unique_intents[0],
            'secondary_queue': unique_intents[1],
            'reason': f"Multiple intents detected: {', '.join(unique_intents)}",
        })
    elif result['entropy'] > 1.2 or result['margin'] < 0.22:
        final_decision['action'] = 'clarify'
    elif result['confidence'] < 0.62:
        final_decision['action'] = 'escalate'

    # Stats Tracking
    action = final_decision['action']
    if action == 'route': _stats['total_routed'] += 1
    elif action == 'clarify': _stats['total_clarified'] += 1
    elif action == 'multi_route': _stats['total_routed'] += 2
    else: _stats['total_escalated'] += 1

    # Clarification Generation
    if action == 'clarify':
        engine = get_clarify()
        from ensemble_router import CATEGORY_MAP
        probs = np.array([result['all_probs'].get(c, 0) for c in CATEGORY_MAP.values()])
        final_decision['clarification'] = engine.generate_question(clean_text, probs)

    return final_decision

@app.post('/clarify')
def get_clarification(req: ClarifyRequest):
    clar = get_clarify()
    if req.current_probs:
        probs = np.array(req.current_probs)
    else:
        router = get_router()
        res = router.route(req.text)
        probs = np.array(list(res['all_probs'].values()))
    return clar.generate_question(req.text, probs)

@app.get('/metrics')
def get_metrics():
    total = _stats['total_requests'] or 1
    router = get_router()
    bert_on = getattr(router, '_bert_available', False)
    return {
        'total_requests': _stats['total_requests'],
        'routing_stats': _stats,
        'routing_distribution': {
            'route_pct':   round(_stats['total_routed']    / total * 100, 1),
            'clarify_pct': round(_stats['total_clarified'] / total * 100, 1),
            'escalate_pct':round(_stats['total_escalated'] / total * 100, 1),
        },
        'model': 'Ensemble (BERT+LR)' if bert_on else 'Fallback (LR Only)',
        'bert_online': bert_on,
    }

@app.get('/health')
def health():
    return {'status': 'ok', 'version': '1.0.0', 'timestamp': datetime.now().isoformat()}

# ── Serve Dashboard ───────────────────────────────────────
dashboard_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dashboard', 'web')
if os.path.exists(dashboard_dir):
    app.mount("/dashboard", StaticFiles(directory=dashboard_dir, html=True), name="dashboard")
    @app.get('/')
    def serve_dashboard():
        return FileResponse(os.path.join(dashboard_dir, 'index.html'))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('api.app', host='0.0.0.0', port=7861, reload=False)
