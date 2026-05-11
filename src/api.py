# src/api.py
# FastAPI Server — SupportMind API
# SupportMind v1.0 — Asmitha

import os
import sys
from dotenv import load_dotenv
load_dotenv()

# Aggressive memory and backend management
os.environ['USE_TF'] = '0'
os.environ['USE_JAX'] = '0'
os.environ['USE_TORCH'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Prevent cuDNN WinError 1455 paging file crash

import time
import logging
import gc
from datetime import datetime


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
from interpretability import SupportMindExplainer

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

@app.on_event("startup")
def startup_event():
    """Pre-load all ML models into memory on the main thread.
    This prevents PyTorch segmentation faults and thread-lock issues
    that happen when lazy-loading large models inside FastAPI worker threads.
    """
    logger.info("Initializing ML models on main thread to prevent segfaults...")
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

_validator = None
_explainer = None

def get_validator():
    global _validator
    if _validator is None:
        _validator = TicketValidator()
    return _validator

def get_explainer():
    global _explainer
    if _explainer is None:
        router = get_router()
        # EnsembleRouter exposes .model and .tokenizer (None if BERT not loaded)
        if router.model is not None:
            _explainer = SupportMindExplainer(router.model, router.tokenizer, device='cpu')
        else:
            _explainer = None  # BERT not available; /explain will return 503
    return _explainer




# ── Request/Response Models ───────────────────────────────
class TicketRequest(BaseModel):
    text: str
    customer_id: Optional[str] = None

class SLARequest(BaseModel):
    """
    SLA breach prediction feature vector.

    **Production requirement**: `similar_ticket_avg_hrs` must be populated
    from a live historical data feed (e.g., a data warehouse query for the
    mean resolution time of similar resolved tickets in the past 30 days).
    The default value (4.5 hrs) is a static fallback for demonstration only
    and will produce under-calibrated predictions in real deployments.
    """
    text_complexity_score: float = 8.0
    agent_queue_depth: int = 10
    customer_tier: int = 3
    hour_of_day: int = 14
    day_of_week: int = 2
    similar_ticket_avg_hrs: float = 4.5  # ⚠️ Default fallback — must come from real historical feed in production
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

    # ── Validate input first ──────────────────────────
    validator = get_validator()
    validation = validator.validate(req.text)

    if not validation['valid']:
        return {
            'action': 'invalid_input',
            'error_type': validation['error_type'],
            'response': validation['response'],
            'confidence': 0.0,
            'entropy': 0.0,
            'top_category': None,
            'all_probs': {},
            'sla_breach_probability': 0.0,
            'clarification': None,
            'latency_ms': round((time.time() - start) * 1000, 1),
            'customer_id': req.customer_id,
        }

    # Use cleaned text for ML pipeline
    clean_text = validation['cleaned_text']

    router = get_router()
    result = router.route(clean_text)

    # Get features FIRST so we can use them for non-support gating
    feat_ext = get_features()
    features = feat_ext.extract(clean_text)

    # ── Non-support input detection ───────────────────
    # Reject things like "welcome to my channel", "subscribe and like", random text
    # that don't look like support tickets.
    # Classification uncertainty ≠ business risk. We reject these
    # instead of blindly escalating them to human agents.
    confidence = result.get('confidence', 0)
    entropy = result.get('entropy', 0)
    
    has_urgency = len(features.get('urgency_flags', [])) > 0
    has_product = len(features.get('product_entities', [])) > 0
    is_short = features.get('token_count', 0) < 10
    not_a_question = not features.get('has_question', False)

    is_junk = False
    # Condition 1: High uncertainty + no urgency (like random text)
    if entropy > 1.4 and confidence < 0.45 and not has_urgency:
        is_junk = True
        
    # Condition 2: Short, no urgency, no product, not a question, low confidence
    if is_short and not has_urgency and not has_product and not_a_question and confidence < 0.65:
        is_junk = True

    if is_junk:
        return {
            'action': 'invalid_input',
            'error_type': 'non_support',
            'response': "This doesn't appear to be a support request. "
                        "Could you describe a specific issue you're "
                        "experiencing with our product or service?",
            'confidence': round(confidence, 4),
            'entropy': round(entropy, 4),
            'top_category': result.get('top_category'),
            'all_probs': result.get('all_probs', {}),
            'sla_breach_probability': 0.0,
            'clarification': None,
            'latency_ms': round((time.time() - start) * 1000, 1),
            'customer_id': req.customer_id,
        }

    # ── SLA prediction (business-signal-driven formula) ──
    # SLA breach risk must reflect OPERATIONAL risk, not
    # classification uncertainty. We compute it from:
    #   - urgency flags (ASAP, blocking, production down)  → 40% weight
    #   - negative sentiment (frustrated customers)        → 25% weight
    #   - text complexity (complex issues take longer)     → 20% weight
    #   - churn risk probability                           → 15% weight
    # NOT from entropy or low confidence.
    urgency_score = features.get('urgency_score', 0.0)
    has_urgency = len(features.get('urgency_flags', [])) > 0
    sentiment = features.get('sentiment_score', 0.0)
    complexity = features.get('text_complexity_score', 0.0)
    margin = result.get('margin', 0.0)

    # Normalized components (each 0.0 → 1.0)
    urgency_component = min(urgency_score, 1.0)                          # already 0–1
    sentiment_component = max(0.0, -sentiment)                           # negative → high risk
    complexity_component = min(complexity / 15.0, 1.0)                   # normalize 0–15 scale
    churn_component = result.get('all_probs', {}).get('churn_risk', 0.0) # model's churn prob

    # Weighted combination
    raw_sla = (
        urgency_component  * 0.40 +
        sentiment_component * 0.25 +
        complexity_component * 0.20 +
        churn_component     * 0.15
    )

    # ── Gate: non-support / junk text should have near-zero SLA ──
    # If confidence is low AND sentiment is neutral/positive AND no
    # urgency flags, the text is likely not a real support issue.
    # Also check for very low margin (near-uniform = random text).
    if confidence < 0.50 and sentiment >= -0.1 and not has_urgency and margin < 0.10:
        sla_risk = round(max(0.01, raw_sla * 0.05), 4)  # suppress to ~0–2%
    else:
        sla_risk = round(min(max(raw_sla, 0.0), 1.0), 4)

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
        clarification = clar.select_question(
            probs,
            result['top_two_classes'],
            ticket_text=clean_text
        )


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
    return clar.select_question(
        probs,
        top_two,
        ticket_text=req.text
    )


@app.post('/sla/predict')
def predict_sla(req: SLARequest):
    """
    Predict SLA breach risk at ticket creation.

    **Production note**: The `similar_ticket_avg_hrs` field defaults to 4.5 hrs
    when omitted. In production, this value **must** be sourced from a real
    historical data feed (e.g., average resolution time for similar resolved
    tickets). Without it, breach probability estimates are not reliable.
    """
    sla = get_sla()
    features = req.model_dump()
    result = sla.explain(features)
    return result


@app.post('/explain')
def explain_prediction(req: ExplainRequest):
    """Generate SHAP word-level importance for a ticket."""
    from ensemble_router import CATEGORY_REVERSE

    explainer = get_explainer()
    if explainer is None:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail='SHAP explainer unavailable until DistilBERT training completes.'
        )

    target_idx = None
    if req.target_class and req.target_class in CATEGORY_REVERSE:
        target_idx = CATEGORY_REVERSE[req.target_class]

    return explainer.explain(req.text, target_class_idx=target_idx)



@app.post('/churn/signal')
def churn_signal(req: ThreadRequest):
    """Extract churn signal from thread history."""
    churn = get_churn()
    return churn.extract(req.thread_texts)


@app.get('/metrics')
def get_metrics():
    """Live system health and routing statistics."""
    total = _stats['total_requests'] or 1
    router = get_router()
    bert_on = getattr(router, '_bert_available', False)
    return {
        'total_requests': _stats['total_requests'],
        'routing_stats': {
            'routed':    _stats['total_routed'],
            'clarified': _stats['total_clarified'],
            'escalated': _stats['total_escalated'],
        },
        'routing_distribution': {
            'route_pct':   round(_stats['total_routed']    / total * 100, 1),
            'clarify_pct': round(_stats['total_clarified'] / total * 100, 1),
            'escalate_pct':round(_stats['total_escalated'] / total * 100, 1),
        },
        'start_time': _stats['start_time'],
        'model': (
            f"ensemble: {router._bert_router.model.config.model_type}-finetuned + tfidf-lr (MC Dropout)"
            if bert_on else
            'ensemble: tfidf-lr baseline (GPU training in progress)'
        ),
        'bert_online': bert_on,
    }


@app.get('/health')
def health():
    """Health check for deployment pipelines."""
    router = get_router()
    bert_on = getattr(router, '_bert_available', False)
    return {
        'status': 'ok',
        'model': f"ensemble ({router._bert_router.model.config.model_type} + tfidf-lr)" if bert_on else 'ensemble (tfidf-lr only)',
        'bert_online': bert_on,
        'version': '2.0.0',
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
    uvicorn.run('api:app', host='0.0.0.0', port=7861, reload=False)

