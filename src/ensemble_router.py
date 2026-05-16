# src/ensemble_router.py
# SupportMind — Ensemble Confidence-Gated Router
# Combines DistilBERT (MC Dropout) + TF-IDF Logistic Regression
# for best-in-class accuracy on ticket routing.
#
# Strategy: weighted soft-voting on probability distributions
#   final_probs = w_bert * bert_probs + w_sklearn * sklearn_probs
#
# Why this beats either model alone:
#   - DistilBERT: captures semantic meaning, handles paraphrases
#   - TF-IDF+LR : captures keyword/n-gram signals, very confident on clear cases
#   - Ensemble  : DistilBERT corrects LR on ambiguous tickets,
#                 LR corrects BERT on keyword-heavy ones

import os
import gc
import pickle
import logging
import numpy as np
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ── Category map ────────────────────────────────────────────────────────────
CATEGORY_MAP = {
    0: 'billing',
    1: 'technical_support',
    2: 'account_management',
    3: 'feature_request',
    4: 'compliance_legal',
    5: 'onboarding',
    6: 'general_inquiry',
    7: 'churn_risk',
}
CATEGORY_REVERSE = {v: k for k, v in CATEGORY_MAP.items()}

# ── Routing thresholds ───────────────────────────────────────────────────────
ROUTE_THRESHOLD   = 0.82   # ensemble conf >= this → auto-route
CLARIFY_THRESHOLD = 0.58   # ensemble conf >= this → ask 1 question
ENTROPY_MAX       = 0.32   # ensemble entropy <= this → low ambiguity
MC_PASSES         = int(os.getenv('SUPPORTMIND_MC_PASSES', '3'))  # CPU demo default

# ── Ensemble weights ─────────────────────────────────────────────────────────
# BERT weight is higher because it generalises better to unseen phrasing.
# These are tunable — increase SKLEARN_W if LR is more accurate on your data.
# BERT weight is significantly higher because DeBERTa-v3 is extremely robust.
BERT_W   = 0.75
SKLEARN_W = 0.25


class EnsembleRouter:
    """
    Ensemble Confidence-Gated Router.

    Combines:
      1. DistilBERT fine-tuned on support tickets (MC Dropout for uncertainty)
      2. TF-IDF + Calibrated Logistic Regression baseline

    Falls back to sklearn-only if DistilBERT model weights are absent.
    Drop-in replacement for ConfidenceGatedRouter — same .route() interface.
    """

    def __init__(self, model_dir: Optional[str] = None, device: str = 'cpu'):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ultimate_path = os.path.join(base, 'models', 'deberta_ultimate')
        standard_path = os.path.join(base, 'models', 'ticket_classifier')
        
        if model_dir is None:
            if os.path.exists(os.path.join(ultimate_path, 'config.json')):
                self.model_dir = ultimate_path
            else:
                self.model_dir = standard_path
        else:
            self.model_dir = model_dir

        self._bert_router = None
        self._sklearn_pipe = None
        self._bert_available = False
        self._bert_reason = 'not_loaded'
        self._sklearn_source = 'unknown'

        # IMPORTANT: Load BERT first and do a warmup pass.
        # On Windows, unpickling sklearn before PyTorch's first forward pass
        # causes a segfault in torch.distributed/optree DLLs.
        self._load_bert(device)
        if self._bert_available:
            self._warmup_bert()
        self._load_sklearn()

        try:
            from historical_memory import HistoricalMemoryLayer
            self._memory_layer = HistoricalMemoryLayer()
        except Exception as e:
            logger.warning(f"[EnsembleRouter] Could not load Historical Memory Layer: {e}")
            self._memory_layer = None

        self.model_status = {
            'mode': 'ensemble_transformer_lr' if self._bert_available else 'sklearn_fallback',
            'bert_available': self._bert_available,
            'bert_reason': self._bert_reason,
            'sklearn_source': self._sklearn_source,
            'model_dir': os.path.relpath(self.model_dir, base),
            'memory_available': bool(
                getattr(getattr(self, '_memory_layer', None), 'is_ready', False)
            ),
        }

        logger.info(
            f"[EnsembleRouter] BERT={'ON' if self._bert_available else 'OFF (fallback)'} | "
            f"sklearn=ON | weights=({BERT_W}/{SKLEARN_W}) | memory={'ON' if getattr(self, '_memory_layer', None) and self._memory_layer.is_ready else 'OFF'}"
        )

    def _warmup_bert(self):
        """Perform a warmup forward pass to initialize PyTorch/CUDA state."""
        try:
            self._bert_router.mc_predict("warmup", n_passes=1)
            logger.info("[EnsembleRouter] BERT warmup complete.")
        except Exception as e:
            logger.warning(f"[EnsembleRouter] BERT warmup failed: {e}")

    # ── Model loaders ────────────────────────────────────────────────────────

    def _load_sklearn(self):
        # Check model_dir first, then fall back to ticket_classifier
        pkl = os.path.join(self.model_dir, 'sklearn_router.pkl')
        if not os.path.exists(pkl):
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            pkl = os.path.join(base, 'models', 'ticket_classifier', 'sklearn_router.pkl')
        if not os.path.exists(pkl):
            logger.warning(
                "[EnsembleRouter] sklearn_router.pkl not found. "
                "Using embedded synthetic fallback model."
            )
            self._sklearn_pipe = self._build_embedded_sklearn()
            self._sklearn_source = 'embedded_synthetic'
            return
        with open(pkl, 'rb') as f:
            self._sklearn_pipe = pickle.load(f)
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._sklearn_source = os.path.relpath(pkl, base)
        logger.info(f"[EnsembleRouter] sklearn pipeline loaded from {pkl}.")

    def _build_embedded_sklearn(self):
        """Build a tiny in-memory classifier so clean clones and CI still run."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        examples = {
            'billing': [
                'invoice is wrong', 'refund request', 'payment failed',
                'billing charge incorrect', 'subscription price changed',
            ],
            'technical_support': [
                'api returns 500 error', 'export is broken', 'dashboard crash',
                'integration timeout', 'feature not working',
            ],
            'account_management': [
                'reset password', 'add user account', 'sso login issue',
                'change admin permission', 'locked out of account',
            ],
            'feature_request': [
                'please add dark mode', 'new feature request',
                'need custom dashboard', 'enhancement idea',
            ],
            'compliance_legal': [
                'gdpr data request', 'soc 2 audit report',
                'data processing agreement', 'privacy compliance',
            ],
            'onboarding': [
                'help with setup', 'new user onboarding',
                'configure integration', 'getting started guide',
            ],
            'general_inquiry': [
                'how do i use this', 'pricing question', 'where is documentation',
                'do you offer a demo',
            ],
            'churn_risk': [
                'cancel my account', 'switching to competitor',
                'very frustrated', 'not renewing contract',
            ],
        }

        texts, labels = [], []
        for category, samples in examples.items():
            for sample in samples:
                texts.append(sample)
                labels.append(CATEGORY_REVERSE[category])

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
            ('clf', LogisticRegression(class_weight='balanced', max_iter=1000)),
        ])
        pipeline.fit(texts, labels)
        return pipeline

    def _load_bert(self, device: str):
        """Load transformer router when the runtime is configured for it."""
        disable_transformer = os.getenv('SUPPORTMIND_DISABLE_TRANSFORMER', '0') == '1'
        force_transformer = os.getenv('SUPPORTMIND_FORCE_TRANSFORMER', '0') == '1'

        if disable_transformer:
            self._bert_reason = 'disabled_by_SUPPORTMIND_DISABLE_TRANSFORMER'
            logger.warning("[EnsembleRouter] Transformer loading disabled by environment.")
            return

        if os.name == 'nt' and not force_transformer:
            self._bert_reason = 'disabled_on_windows_set_SUPPORTMIND_FORCE_TRANSFORMER_to_enable'
            logger.warning(
                "[EnsembleRouter] Transformer loading disabled on Windows by default "
                "to avoid native PyTorch/safetensors crashes. Set "
                "SUPPORTMIND_FORCE_TRANSFORMER=1 to enable it."
            )
            return

        import json, traceback as tb
        model_bin  = os.path.join(self.model_dir, 'pytorch_model.bin')
        model_safe = os.path.join(self.model_dir, 'model.safetensors')
        config     = os.path.join(self.model_dir, 'config.json')

        bert_ready = os.path.exists(config) and (
            os.path.exists(model_bin) or os.path.exists(model_safe)
        )

        if not bert_ready:
            self._bert_reason = 'weights_not_found'
            logger.warning(
                "[EnsembleRouter] DistilBERT weights not found — running sklearn-only."
            )
            return

        # Check for stale baseline stub (only present before first real training run)
        try:
            with open(config) as f:
                cfg = json.load(f)
            if cfg.get('model_type') == 'baseline_sklearn':
                self._bert_reason = 'baseline_stub_config'
                logger.warning("[EnsembleRouter] config.json is baseline stub — skipping BERT.")
                return
        except Exception:
            pass

        try:
            from confidence_router import ConfidenceGatedRouter
            self._bert_router = ConfidenceGatedRouter(self.model_dir, device=device)
            self._bert_available = not getattr(self._bert_router, '_fallback_mode', False)
            fallback_reason = getattr(self._bert_router, 'fallback_reason', None)
            self._bert_reason = (
                'loaded' if self._bert_available
                else f'confidence_router_fallback: {fallback_reason or "unknown"}'
            )
            gc.collect()
            if self._bert_available:
                logger.info(f"[EnsembleRouter] {self._bert_router.model.config.model_type.upper()} loaded successfully.")
        except (Exception, OSError) as e:
            logger.error(f"[EnsembleRouter] BERT load failed (likely memory constraint): {e}")
            # Ensure we don't leave a half-initialized router
            self._bert_router = None
            self._bert_available = False
            self._bert_reason = f'load_failed: {type(e).__name__}'
            gc.collect()

    # ── Prediction ───────────────────────────────────────────────────────────

    def _sklearn_probs(self, text: str) -> np.ndarray:
        """Return calibrated probability distribution from sklearn pipeline."""
        return self._sklearn_pipe.predict_proba([text])[0]   # shape [8]

    def _bert_probs(self, text: str) -> np.ndarray:
        """Return MC-Dropout probability distribution from DistilBERT."""
        _, _, _, mean_p, _ = self._bert_router.mc_predict(text, n_passes=MC_PASSES)
        return mean_p   # shape [8]

    def _blend(self, text: str):
        """
        Compute blended probability distribution.
        Returns: (blended_probs, bert_probs_or_None, sklearn_probs, bert_std_or_None)
        """
        sk_probs = self._sklearn_probs(text)

        if self._bert_available:
            _, _, _, bert_mean, bert_std = self._bert_router.mc_predict(text, MC_PASSES)
            blended = BERT_W * bert_mean + SKLEARN_W * sk_probs
            # Re-normalise (floating point can drift slightly)
            blended = blended / blended.sum()
            return blended, bert_mean, sk_probs, bert_std
        else:
            return sk_probs, None, sk_probs, np.zeros(8)

    # ── Public API ───────────────────────────────────────────────────────────

    def route(self, ticket_text: str, n_passes: int = MC_PASSES) -> Dict:
        """
        Route a ticket through the ensemble confidence gate.
        Returns the same dict schema as ConfidenceGatedRouter.route()
        so it is a drop-in replacement in api.py.
        """
        blended, bert_p, sk_p, bert_std = self._blend(ticket_text)

        confidence  = float(blended.max())
        entropy     = float(-np.sum(blended * np.log(blended + 1e-9)))
        
        # ── Temperature Scaling (T=0.7) ──────────────────────────────────
        # Sharpen probabilities to reduce noise in unrelated classes.
        # logits_scaled = logits / T; softmax(logits_scaled)
        # Since we have probs, we can approximate with power scaling:
        # p_scaled = p^(1/T) / sum(p^(1/T))
        T = 0.7
        blended_sharp = np.power(blended + 1e-9, 1.0 / T)
        blended_sharp = blended_sharp / blended_sharp.sum()
        
        # ── Keyword Reinforcement ────────────────────────────────────────
        # If text contains specific strong keywords for a category, 
        # give that category a small 'calibration boost'.
        reinforce_map = {
            'billing': ['invoice', 'refund', 'charge', 'payment', 'billing'],
            'technical_support': ['error', 'bug', 'crash', '500', 'api', 'broken', 'not working'],
            'account_management': ['login', 'password', 'reset', 'account', 'permission', 'access', 'sso', 'user'],
            'feature_request': ['feature', 'add', 'request', 'enhancement', 'dark mode', 'new capability', 'could you add'],
            'compliance_legal': ['gdpr', 'compliance', 'legal', 'audit', 'privacy'],
            'churn_risk': ['cancel', 'leaving', 'competitor', 'terminate', 'switching'],
            'onboarding': ['setup', 'configure', 'getting started', 'new user', 'import'],
        }
        text_low = ticket_text.lower()
        for cat, kws in reinforce_map.items():
            hit_count = sum(1 for kw in kws if kw in text_low)
            if hit_count:
                idx = CATEGORY_REVERSE[cat]
                blended_sharp[idx] *= 1.0 + min(0.45, hit_count * 0.12)
                blended_sharp[idx] += min(0.12, hit_count * 0.03)
        
        # Re-normalise after boost
        blended_sharp = blended_sharp / blended_sharp.sum()
        
        confidence = float(blended_sharp.max())
        pred_class = int(blended_sharp.argmax())
        category   = CATEGORY_MAP[pred_class]
        
        # ── Visual Confidence Cap (98.5%) ────────────────────────────────
        # Probabilistic ML should rarely claim 100% certainty.
        display_confidence = min(confidence, 0.985)

        # Build ranking
        ranking = sorted(
            [(CATEGORY_MAP[i], round(float(blended_sharp[i]), 4)) for i in range(8)],
            key=lambda x: x[1], reverse=True
        )
        top_two = [ranking[0][0], ranking[1][0]]

        base = {
            'confidence':       round(display_confidence, 4),
            'raw_confidence':   round(confidence, 4),
            'entropy':          round(entropy,    4),
            'top_category':     category,
            'all_probs':        {CATEGORY_MAP[i]: round(float(blended_sharp[i]), 4) for i in range(8)},
            'std_probs':        {CATEGORY_MAP[i]: round(float(bert_std[i]), 4) for i in range(8)},
            'category_ranking': ranking,
            'top_two_classes':  top_two,
            'mc_passes':        n_passes,
            # Extra ensemble diagnostics
            'ensemble': {
                'bert_available':  self._bert_available,
                'bert_top':        CATEGORY_MAP[int(bert_p.argmax())] if bert_p is not None else None,
                'sklearn_top':     CATEGORY_MAP[int(sk_p.argmax())],
                'bert_weight':     BERT_W if self._bert_available else 0.0,
                'sklearn_weight':  SKLEARN_W if self._bert_available else 1.0,
                'agreement':       (
                    CATEGORY_MAP[int(bert_p.argmax())] == CATEGORY_MAP[int(sk_p.argmax())]
                    if bert_p is not None else True
                ),
            }
        }

        top1_score = ranking[0][1]
        top2_score = ranking[1][1]
        margin = top1_score - top2_score
        
        hist_boost = 0.0
        if getattr(self, '_memory_layer', None) and self._memory_layer.is_ready:
            hist_boost = self._memory_layer.compute_historical_boost(ticket_text, category)
            base['historical_boost'] = hist_boost

        base['margin'] = round(margin, 4)
        base['confidence'] = round(display_confidence, 4)

        critical_labels = ['compliance_legal', 'account_management']
        
        effective_conf = confidence + hist_boost

        if category in critical_labels:
            if effective_conf >= 0.90 and margin >= 0.35 and entropy < 0.60:
                action = 'route'
                reason = f'• Safe to auto-route sensitive intent<br>• Confidence: {confidence:.2%}<br>• Margin: {margin:.2f}'
                if hist_boost > 0: reason += f'<br>• <span style="color:var(--green)">Historical Match Boost: +{hist_boost:.2%}</span>'
            else:
                action = 'escalate'
                reason = f'• Escalated sensitive intent ({category})<br>• Strict confidence/margin threshold not met'
                if hist_boost > 0: reason += f'<br>• <span style="color:var(--green)">Historical Match Boost: +{hist_boost:.2%}</span> (Insufficient)'
        elif category == 'technical_support':
            # Category-specific check for technical support to catch billing misroutes
            billing_keywords = ['invoice', 'billing', 'charge', 'refund', 'payment', 'subscription', 'plan']
            has_billing_kw = any(kw in ticket_text.lower() for kw in billing_keywords)
            
            if has_billing_kw and 'billing' in [r[0] for r in ranking[:3]]:
                action = 'clarify'
                reason = f'• Billing overlap detected<br>• Clarification needed between technical_support and billing'
            elif effective_conf >= 0.88 and margin >= 0.30 and entropy < 0.65:
                # Stricter thresholds for technical_support
                action = 'route'
                reason = f'• Strong dominant intent<br>• Confidence: {confidence:.2%}<br>• Margin: {margin:.2f}<br>• Safe to auto-route'
                if hist_boost > 0: reason += f'<br>• <span style="color:var(--green)">Historical Match Boost: +{hist_boost:.2%}</span>'
            elif effective_conf >= 0.60 and entropy < 1.05:
                action = 'clarify'
                reason = f'• Medium ambiguity detected<br>• Clarification needed between {top_two[0]} and {top_two[1]}<br>• Margin: {margin:.2f}'
                if hist_boost > 0: reason += f'<br>• <span style="color:var(--green)">Historical Match Boost: +{hist_boost:.2%}</span> (Insufficient for auto-route)'
            else:
                action = 'escalate'
                reason = f'• High ambiguity / Low confidence ({confidence:.2%})<br>• Multiple overlapping intents detected<br>• Human triage needed'
        else:
            if effective_conf >= 0.85 and margin >= 0.25 and entropy < 0.70:
                action = 'route'
                reason = f'• Strong dominant intent<br>• Confidence: {confidence:.2%}<br>• Margin: {margin:.2f}<br>• Safe to auto-route'
                if hist_boost > 0: reason += f'<br>• <span style="color:var(--green)">Historical Match Boost: +{hist_boost:.2%}</span>'
            elif effective_conf >= 0.60 and entropy < 1.05:
                action = 'clarify'
                reason = f'• Medium ambiguity detected<br>• Clarification needed between {top_two[0]} and {top_two[1]}<br>• Margin: {margin:.2f}'
                if hist_boost > 0: reason += f'<br>• <span style="color:var(--green)">Historical Match Boost: +{hist_boost:.2%}</span> (Insufficient for auto-route)'
            else:
                action = 'escalate'
                reason = f'• High ambiguity / Low confidence ({confidence:.2%})<br>• Multiple overlapping intents detected<br>• Human triage needed'

        return {**base, 'action': action, 'queue': category if action == 'route' else None, 'reason': reason}

    def batch_route(self, tickets: list, n_passes: int = MC_PASSES) -> list:
        return [self.route(t, n_passes) for t in tickets]

    # Property to expose model/tokenizer for the SHAP explainer in api.py
    @property
    def model(self):
        if self._bert_available:
            return self._bert_router.model
        return None

    @property
    def tokenizer(self):
        if self._bert_available:
            return self._bert_router.tokenizer
        return None


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    router = EnsembleRouter()

    tests = [
        "My invoice from last month is incorrect, please fix the billing.",
        "The API keeps returning 500 errors since last Tuesday's update.",
        "I want to cancel — this tool has been broken for weeks.",
        "How do I add another user to our account?",
        "We need GDPR data processing agreements for our EU customers.",
        "Not happy at all, considering switching to a competitor.",
        "Can you add a dark mode to the dashboard?",
        "Just signed up — how do I import my existing data?",
        # Tricky ambiguous cases
        "Invoice is wrong AND the app keeps crashing.",
        "Not happy with service",
    ]

    print(f"\n{'='*90}")
    print(f"  SupportMind Ensemble Router — BERT={'ON' if router._bert_available else 'OFF (sklearn only)'}")
    print(f"{'='*90}\n")

    for ticket in tests:
        r = router.route(ticket)
        agree = 'AGREE' if r['ensemble']['agreement'] else 'DISAGREE'
        print(
            f"[{r['action'].upper():8s}] [{r['confidence']:.2%}] "
            f"{'H' if r['entropy'] < ENTROPY_MAX else 'L'}-certainty | "
            f"{r['top_category']:20s} | "
            f"Models: {agree} | {ticket[:60]}"
        )
