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
MC_PASSES         = 10     # MC Dropout stochastic passes (sequential for memory)

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
            raise FileNotFoundError(
                f"sklearn_router.pkl not found. "
                "Run: python train_baseline.py"
            )
        with open(pkl, 'rb') as f:
            self._sklearn_pipe = pickle.load(f)
        logger.info(f"[EnsembleRouter] sklearn pipeline loaded from {pkl}.")

    def _load_bert(self, device: str):
        """Load fine-tuned DistilBERT. Skips gracefully if weights not saved yet."""
        import json, traceback as tb
        model_bin  = os.path.join(self.model_dir, 'pytorch_model.bin')
        model_safe = os.path.join(self.model_dir, 'model.safetensors')
        config     = os.path.join(self.model_dir, 'config.json')

        bert_ready = os.path.exists(config) and (
            os.path.exists(model_bin) or os.path.exists(model_safe)
        )

        if not bert_ready:
            logger.warning(
                "[EnsembleRouter] DistilBERT weights not found — running sklearn-only."
            )
            return

        # Check for stale baseline stub (only present before first real training run)
        try:
            with open(config) as f:
                cfg = json.load(f)
            if cfg.get('model_type') == 'baseline_sklearn':
                logger.warning("[EnsembleRouter] config.json is baseline stub — skipping BERT.")
                return
        except Exception:
            pass

        try:
            from confidence_router import ConfidenceGatedRouter
            self._bert_router = ConfidenceGatedRouter(self.model_dir, device=device)
            self._bert_available = True
            gc.collect()
            logger.info(f"[EnsembleRouter] {self._bert_router.model.config.model_type.upper()} loaded successfully.")
        except (Exception, OSError) as e:
            logger.error(f"[EnsembleRouter] BERT load failed (likely memory constraint): {e}")
            # Ensure we don't leave a half-initialized router
            self._bert_router = None
            self._bert_available = False
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
        pred_class  = int(blended.argmax())
        category    = CATEGORY_MAP[pred_class]

        # Build ranking
        ranking = sorted(
            [(CATEGORY_MAP[i], round(float(blended[i]), 4)) for i in range(8)],
            key=lambda x: x[1], reverse=True
        )
        top_two = [ranking[0][0], ranking[1][0]]

        base = {
            'confidence':       round(confidence, 4),
            'entropy':          round(entropy,    4),
            'top_category':     category,
            'all_probs':        {CATEGORY_MAP[i]: round(float(blended[i]), 4) for i in range(8)},
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
        base['confidence'] = round(confidence, 4)

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
