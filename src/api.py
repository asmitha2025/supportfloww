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
except Exception as e:
    print(f"Failed to load optional interpretability module: {e}")
    SupportMindExplainer = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App Setup ──────────────────────────────────────────────
app = FastAPI(
    title='SupportMind API',
    description='Confidence-Gated Support Intelligence for B2B SaaS Customer Operations',
    version='1.0.0',
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv('CORS_ALLOW_ORIGINS', '*').split(',')
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials='*' not in allowed_origins,
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
    'total_multi_route': 0,
    'total_requests': 0, 'start_time': datetime.now().isoformat(),
}

CATEGORY_NAMES = [
    'billing',
    'technical_support',
    'account_management',
    'feature_request',
    'compliance_legal',
    'onboarding',
    'general_inquiry',
    'churn_risk',
]
CATEGORY_INDEX = {category: idx for idx, category in enumerate(CATEGORY_NAMES)}

CATEGORY_SIGNAL_PATTERNS = {
    'billing': [
        r'\b(?:invoice|billing|bill|refund|charge|payment|paid|duplicate payment|credit)\b',
    ],
    'technical_support': [
        r'\b(?:error|bug|crash|broken|failing|not working|api|http\s*\d{3}|500|timeout|integration|export)\b',
    ],
    'account_management': [
        r'\b(?:password|login|log in|locked out|reset|permission|access|account|sso|user role|admin)\b',
    ],
    'feature_request': [
        r'\b(?:feature request|new feature|new capability|enhancement|could you add|can you add|please add|dark mode|support for)\b',
    ],
    'compliance_legal': [
        r'\b(?:gdpr|compliance|legal|audit|privacy|dpa|data processing|regulatory)\b',
    ],
    'onboarding': [
        r'\b(?:setup|set up|configure|getting started|onboard|new user|import data|walkthrough|training)\b',
    ],
    'general_inquiry': [
        r'\b(?:how do i|how can i|question|where can i|what is|information about)\b',
    ],
    'churn_risk': [
        r'\b(?:cancel|cancelling|canceling|switching|competitor|leaving|terminate|churn)\b',
    ],
}

EXPLANATION_KEYWORDS = {
    'billing': ['invoice', 'billing', 'bill', 'refund', 'charge', 'payment', 'paid', 'credit', 'subscription', 'plan'],
    'technical_support': ['error', 'bug', 'crash', 'broken', 'failing', 'working', 'api', 'http', '500', 'timeout', 'integration', 'export'],
    'account_management': ['password', 'login', 'locked', 'reset', 'permission', 'access', 'account', 'sso', 'user', 'admin'],
    'feature_request': ['feature', 'request', 'enhancement', 'add', 'support', 'capability', 'roadmap'],
    'compliance_legal': ['gdpr', 'compliance', 'legal', 'audit', 'privacy', 'dpa', 'regulatory', 'security'],
    'onboarding': ['setup', 'configure', 'started', 'onboard', 'new', 'import', 'walkthrough', 'training'],
    'general_inquiry': ['how', 'question', 'where', 'what', 'information', 'demo', 'trial', 'pricing'],
    'churn_risk': ['cancel', 'switching', 'competitor', 'leaving', 'terminate', 'frustrated', 'renewal'],
}

SUPPORT_INTENT_PATTERNS = [
    r'\b(?:please|help|fix|resolve|issue|problem|ticket|support|need help|can you|could you)\b',
    r"\b(?:forgot|reset|unable|cannot|can't|wrong|incorrect|failed|failing|broken)\b",
]

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
    # Explainability can be expensive with transformer models, so keep it lazy.
    # The /explain endpoint initializes it only when an explanation is requested.
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
        if SupportMindExplainer is not None and router.model is not None:
            _explainer = SupportMindExplainer(router.model, router.tokenizer, device='cpu')
    return _explainer

# ── Request Models ─────────────────────────────────────────
class TicketRequest(BaseModel):
    text: str
    customer_id: Optional[str] = "CUST-DEMO"
    clarification_choice: Optional[str] = None
    clarification_target: Optional[str] = None
    clarification_question_id: Optional[str] = None

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


def _extract_clarification_signal(req: TicketRequest) -> Dict[str, Optional[str]]:
    target = req.clarification_target
    choice = req.clarification_choice

    if not target:
        marker = re.search(
            r'\[Clarification:\s*(?P<target>[a-z_]+)\s*-\s*(?P<choice>[^\]]+)\]',
            req.text,
            flags=re.I,
        )
        if marker:
            target = marker.group('target').lower()
            choice = choice or marker.group('choice').strip()

    if target:
        target = target.strip().lower()
    if target not in CATEGORY_NAMES:
        return {'target': None, 'choice': choice}

    return {'target': target, 'choice': choice}


def _resolved_clarification_result(target: str,
                                   choice: Optional[str],
                                   question_id: Optional[str]) -> Dict:
    all_probs = {
        category: round(0.10 / (len(CATEGORY_NAMES) - 1), 4)
        for category in CATEGORY_NAMES
    }
    all_probs[target] = 0.90
    ranking = sorted(all_probs.items(), key=lambda item: item[1], reverse=True)
    return {
        'action': 'route',
        'queue': target,
        'top_category': target,
        'confidence': 0.90,
        'entropy': 0.35,
        'margin': 0.75,
        'all_probs': all_probs,
        'std_probs': {category: 0.0 for category in CATEGORY_NAMES},
        'category_ranking': ranking,
        'top_two_classes': [ranking[0][0], ranking[1][0]],
        'mc_passes': 0,
        'reason': (
            f"Clarification answer resolved the ambiguity toward {target}."
        ),
        'clarification_applied': True,
        'clarification_choice': choice,
        'clarification_question_id': question_id,
    }


def _has_direct_category_signal(text: str, category: str) -> bool:
    return _category_signal_strength(text, category) > 0


def _category_signal_strength(text: str, category: str) -> int:
    patterns = CATEGORY_SIGNAL_PATTERNS.get(category, [])
    return sum(
        len(re.findall(pattern, text, flags=re.I))
        for pattern in patterns
    )


def _first_signal_position(text: str, category: str) -> int:
    positions = []
    for pattern in CATEGORY_SIGNAL_PATTERNS.get(category, []):
        match = re.search(pattern, text, flags=re.I)
        if match:
            positions.append(match.start())
    return min(positions) if positions else 10**9


def _direct_signal_intents(text: str) -> List[str]:
    strengths = {
        category: _category_signal_strength(text, category)
        for category in CATEGORY_NAMES
    }
    intents = []

    account_access = re.search(
        r'\b(?:forgot|reset|password|locked out|login|log in|access|sso|admin)\b',
        text,
        flags=re.I,
    )

    for category, strength in strengths.items():
        if strength <= 0:
            continue
        if category == 'technical_support':
            # "SSO login is broken" is an access-management signal, and
            # "billing error" is a billing signal. Treat them as technical
            # only when a concrete product/API failure marker is present.
            if (account_access or strengths.get('billing', 0) > 0) and not re.search(
                r'\b(?:api|http\s*\d{3}|500|timeout|integration|export|crash)\b',
                text,
                flags=re.I,
            ):
                continue
        if category == 'account_management' and not account_access:
            continue
        intents.append(category)

    return sorted(
        intents,
        key=lambda category: (_first_signal_position(text, category), CATEGORY_NAMES.index(category)),
    )


def _result_forced_to_category(result: Dict, category: str, confidence: float, reason: str) -> Dict:
    adjusted = dict(result)
    probs = dict(result.get('all_probs') or {})
    other_total = sum(v for key, v in probs.items() if key != category)
    remaining = max(0.0, 1.0 - confidence)

    for key in CATEGORY_NAMES:
        if key == category:
            probs[key] = confidence
        else:
            original = float(probs.get(key, 0.0))
            probs[key] = (original / other_total * remaining) if other_total else remaining / (len(CATEGORY_NAMES) - 1)

    ranking = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    top_two = [ranking[0][0], ranking[1][0]]
    entropy = float(-sum(p * np.log(p + 1e-9) for p in probs.values()))
    margin = float(ranking[0][1] - ranking[1][1])

    adjusted.update({
        'top_category': category,
        'confidence': round(confidence, 4),
        'entropy': round(entropy, 4),
        'margin': round(margin, 4),
        'all_probs': {key: round(float(value), 4) for key, value in probs.items()},
        'category_ranking': [(key, round(float(value), 4)) for key, value in ranking],
        'top_two_classes': top_two,
        'reason': reason,
        'direct_signal_override': True,
    })
    return adjusted


def _update_result_probabilities(result: Dict, probs: Dict[str, float]) -> Dict:
    adjusted = dict(result)
    total = sum(max(float(value), 0.0) for value in probs.values())
    if total <= 0:
        return adjusted

    normalized = {
        category: max(float(probs.get(category, 0.0)), 0.0) / total
        for category in CATEGORY_NAMES
    }
    ranking = sorted(normalized.items(), key=lambda item: item[1], reverse=True)
    entropy = float(-sum(p * np.log(p + 1e-9) for p in normalized.values()))
    margin = float(ranking[0][1] - ranking[1][1])

    adjusted.update({
        'top_category': ranking[0][0],
        'confidence': round(float(ranking[0][1]), 4),
        'entropy': round(entropy, 4),
        'margin': round(margin, 4),
        'all_probs': {key: round(float(value), 4) for key, value in normalized.items()},
        'category_ranking': [(key, round(float(value), 4)) for key, value in ranking],
        'top_two_classes': [ranking[0][0], ranking[1][0]],
    })
    return adjusted


def _has_explicit_churn_signal(text: str) -> bool:
    return bool(re.search(
        r'\b(?:cancel|cancelling|canceling|switching|switch to|competitor|'
        r'leaving|terminate|churn|not renew|non-renew|renewal risk)\b',
        text,
        flags=re.I,
    ))


def _apply_probability_guardrails(result: Dict, text: str) -> Dict:
    probs = dict(result.get('all_probs') or {})
    churn_prob = float(probs.get('churn_risk', 0.0))

    if churn_prob > 0.05 and not _has_explicit_churn_signal(text):
        probs['churn_risk'] = 0.04
        adjusted = _update_result_probabilities(result, probs)
        adjusted['probability_guardrail'] = 'churn_dampened_without_explicit_churn_signal'
        return adjusted

    return result


def _apply_direct_signal_overrides(result: Dict, text: str, direct_intents: List[str]) -> Dict:
    if len(direct_intents) >= 2:
        return result

    account_strength = _category_signal_strength(text, 'account_management')
    account_access = re.search(
        r'\b(?:forgot|reset|password|locked out|login|log in|access|sso|admin)\b',
        text,
        flags=re.I,
    )
    if account_strength >= 2 and account_access and result.get('top_category') != 'account_management':
        return _result_forced_to_category(
            result,
            'account_management',
            confidence=max(0.78, float(result.get('all_probs', {}).get('account_management', 0.0))),
            reason='Direct account-access signal detected: password/login/admin access.',
        )

    billing_strength = _category_signal_strength(text, 'billing')
    onboarding_strength = _category_signal_strength(text, 'onboarding')
    if (
        billing_strength > 0
        and onboarding_strength == 0
        and result.get('top_category') == 'onboarding'
    ):
        return _result_forced_to_category(
            result,
            'billing',
            confidence=max(0.74, float(result.get('all_probs', {}).get('billing', 0.0))),
            reason='Direct billing signal detected without onboarding evidence.',
        )

    return result


def _order_intents_by_probability(intents: List[str], result: Dict) -> List[str]:
    probs = result.get('all_probs') or {}
    original_rank = {intent: idx for idx, intent in enumerate(intents)}
    return sorted(
        intents,
        key=lambda intent: (-float(probs.get(intent, 0.0)), original_rank[intent]),
    )


def _has_support_intent(text: str, features: Dict, result: Dict) -> bool:
    if any(re.search(pattern, text, flags=re.I) for pattern in SUPPORT_INTENT_PATTERNS):
        return True
    if features.get('product_entities') or features.get('has_question'):
        return True
    return any(_has_direct_category_signal(text, category) for category in CATEGORY_NAMES)


def _can_route_by_direct_signal(result: Dict, text: str) -> bool:
    if result.get('top_category') == 'compliance_legal':
        return False

    category = result.get('top_category', '')
    confidence = result.get('confidence', 0.0)
    margin = result.get('margin', 0.0)
    signal_strength = _category_signal_strength(text, category)

    if category == 'feature_request' and signal_strength >= 2 and confidence >= 0.55 and margin >= 0.30:
        return True

    if (
        category == 'account_management'
        and signal_strength >= 3
        and re.search(r'\b(?:forgot|reset|password|locked out|login|access)\b', text, flags=re.I)
    ):
        return True

    if signal_strength >= 3 and confidence >= 0.58 and margin >= 0.20:
        return True

    return signal_strength > 0 and confidence >= 0.62 and margin >= 0.35


def _needs_clarification(result: Dict, text: str) -> bool:
    confidence = result.get('confidence', 0.0)
    entropy = result.get('entropy', 0.0)
    margin = result.get('margin', 0.0)

    # The sklearn fallback keeps more probability mass in non-winning classes,
    # so entropy alone can be high even when the top class is clearly ahead.
    if (confidence >= 0.62 and margin >= 0.35) or _can_route_by_direct_signal(result, text):
        return False

    return margin < 0.22 or (entropy > 1.2 and margin < 0.35)


def _heuristic_explanation(text: str, target_class: Optional[str] = None) -> Dict:
    """Lightweight explainability fallback when transformer SHAP is unavailable."""
    target = (target_class or '').strip().lower()
    if target not in CATEGORY_NAMES:
        try:
            target = get_router().route(text).get('top_category', 'general_inquiry')
        except Exception:
            target = 'general_inquiry'

    keywords = EXPLANATION_KEYWORDS.get(target, [])
    tokens = re.findall(r"[A-Za-z0-9_@./:-]+|[^\s]", text or '')
    values = []

    for token in tokens:
        normalized = token.lower().strip(".,!?;:'\"()[]{}")
        if not normalized:
            values.append(0.0)
            continue

        value = 0.0
        if normalized in keywords:
            value += 0.28
        elif any(normalized in keyword or keyword in normalized for keyword in keywords if len(keyword) > 3):
            value += 0.16

        for category, other_keywords in EXPLANATION_KEYWORDS.items():
            if category == target:
                continue
            if normalized in other_keywords:
                value -= 0.08
                break

        values.append(round(value, 4))

    return {
        'tokens': tokens,
        'values': values,
        'base_value': 0.0,
        'target_class': CATEGORY_INDEX.get(target, CATEGORY_INDEX['general_inquiry']),
        'target_category': target,
        'prediction_value': round(sum(values), 4),
        'source': 'heuristic_keywords',
        'note': 'Transformer SHAP is unavailable in the current runtime, so keyword evidence is shown instead.',
    }

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
    clarification_signal = _extract_clarification_signal(req)
    
    # 2. ML Routing & Features
    feat_ext = get_features()
    features = feat_ext.extract(clean_text)

    if clarification_signal['target']:
        result = _resolved_clarification_result(
            clarification_signal['target'],
            clarification_signal['choice'],
            req.clarification_question_id,
        )
        is_multi_intent = False
        unique_intents = []
    else:
        router = get_router()
        result = router.route(clean_text)
    
        # 3. Multi-Intent Detection (Segmentation)
        direct_intents = _direct_signal_intents(clean_text)
        segments = [s.strip() for s in re.split(r'\.|,|\band\b|\balso\b', clean_text, flags=re.I) if len(s.strip().split()) > 3]
        segment_intents = []
        if len(segments) > 1:
            for seg in segments:
                for direct_intent in _direct_signal_intents(seg):
                    if direct_intent not in segment_intents:
                        segment_intents.append(direct_intent)
                seg_res = router.route(seg)
                top_category = seg_res['top_category']
                if (
                    seg_res['confidence'] > 0.65
                    and _category_signal_strength(seg, top_category) > 0
                    and top_category not in segment_intents
                ):
                    segment_intents.append(seg_res['top_category'])

        unique_intents = list(dict.fromkeys(segment_intents or direct_intents))
        is_multi_intent = len(unique_intents) >= 2
        result = _apply_direct_signal_overrides(result, clean_text, unique_intents)
        result = _apply_probability_guardrails(result, clean_text)
        if is_multi_intent:
            unique_intents = _order_intents_by_probability(unique_intents, result)

    # 4. Operational SLA Risk Engine
    urg_val = features.get('urgency_score', 0.0)
    comp_val = features.get('complexity_score', 0.0)
    sent_val = features.get('sentiment_score', 0.0)
    
    # Base risk: Urgency (50%) + Complexity (30%) + Sentiment Penalty (20%)
    raw_risk = (urg_val * 0.5) + (comp_val * 0.3)
    if sent_val < -0.4: raw_risk += 0.2
    sla_risk = min(max(raw_risk, 0.01), 1.0)

    # 5. Non-Support / Junk Detection
    has_support_intent = _has_support_intent(clean_text, features, result)
    can_route_by_signal = _can_route_by_direct_signal(result, clean_text)

    is_junk = False
    if (
        not has_support_intent
        and result['entropy'] > 1.6
        and result['confidence'] < 0.4
        and urg_val < 0.1
        and not features.get('product_entities')
    ):
        is_junk = True
    if (
        not has_support_intent
        and features.get('token_count', 0) < 10
        and urg_val < 0.1
        and not features.get('has_question')
        and result['confidence'] < 0.6
    ):
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
        'sla_breach_probability': round(sla_risk, 4),
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
    elif result.get('clarification_applied'):
        final_decision.update({
            'action': 'route',
            'queue': result['queue'],
            'reason': result['reason'],
            'clarification_applied': True,
            'clarification_choice': result.get('clarification_choice'),
            'clarification_question_id': result.get('clarification_question_id'),
        })
    elif is_multi_intent:
        final_decision.update({
            'action': 'multi_route',
            'primary_queue': unique_intents[0],
            'secondary_queue': unique_intents[1],
            'reason': f"Multiple intents detected: {', '.join(unique_intents)}",
        })
    elif _needs_clarification(result, clean_text):
        final_decision['action'] = 'clarify'
    elif result['confidence'] < 0.62 and not can_route_by_signal:
        final_decision['action'] = 'escalate'

    # Stats Tracking
    action = final_decision['action']
    if action == 'route':
        _stats['total_routed'] += 1
    elif action == 'clarify':
        _stats['total_clarified'] += 1
    elif action == 'multi_route':
        _stats['total_multi_route'] += 1
        _stats['total_routed'] += 1
    else:
        _stats['total_escalated'] += 1

    # Clarification Generation
    if action == 'clarify':
        engine = get_clarify()
        from ensemble_router import CATEGORY_MAP
        probs = np.array([result['all_probs'].get(c, 0) for c in CATEGORY_MAP.values()])
        final_decision['clarification'] = engine.generate_question(
            clean_text,
            probs,
            top_two_classes=result.get('top_two_classes'),
        )

    return final_decision

@app.post('/sla/predict')
def predict_sla(req: SLARequest):
    """Predict SLA breach risk from operational features."""
    predictor = get_sla()
    return predictor.explain(req.model_dump())

@app.post('/churn/signal')
def churn_signal(req: ThreadRequest):
    """Extract churn-risk signals from a support conversation."""
    extractor = get_churn()
    return extractor.extract(req.thread_texts)

@app.post('/clarify')
def get_clarification(req: ClarifyRequest):
    clar = get_clarify()
    if req.current_probs:
        probs = np.array(req.current_probs)
    else:
        router = get_router()
        res = router.route(req.text)
        probs = np.array(list(res['all_probs'].values()))
    return clar.generate_question(
        req.text,
        probs,
        top_two_classes=req.top_two_classes,
    )


@app.post('/explain')
def explain_decision(req: ExplainRequest):
    """Return token-level explanation data for the routed decision."""
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    target_idx = CATEGORY_INDEX.get((req.target_class or '').strip().lower())
    explainer = get_explainer()
    if explainer is not None:
        result = explainer.explain(req.text, target_idx)
        if 'error' not in result:
            result['source'] = 'shap_transformer'
            if req.target_class:
                result['target_category'] = req.target_class
            return result
        logger.warning("SHAP explanation unavailable; using heuristic fallback: %s", result['error'])

    return _heuristic_explanation(req.text, req.target_class)


@app.get('/model/status')
def model_status():
    """Expose runtime model status for demos, monitoring, and deployment checks."""
    router = get_router()
    clarify = get_clarify()
    return {
        'router': getattr(router, 'model_status', {
            'bert_available': getattr(router, '_bert_available', False),
            'mode': 'ensemble' if getattr(router, '_bert_available', False) else 'sklearn_fallback',
        }),
        'historical_memory_online': bool(
            getattr(getattr(router, '_memory_layer', None), 'is_ready', False)
        ),
        'clarification_llm_configured': bool(getattr(clarify, 'groq_client', None)),
        'explainability': 'shap_transformer' if get_explainer() is not None else 'heuristic_keywords',
    }

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
            'multi_route_pct': round(_stats.get('total_multi_route', 0) / total * 100, 1),
        },
        'model': 'Ensemble (Transformer + LR)' if bert_on else 'Sklearn fallback (LR only)',
        'bert_online': bert_on,
        'model_status': getattr(router, 'model_status', None),
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
