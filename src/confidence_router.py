# src/confidence_router.py
# Core module: MC Dropout Confidence-Gated Ticket Router
# SupportMind v1.0 — Asmitha

import numpy as np
from typing import Dict, Tuple, Optional
import os
import logging

logger = logging.getLogger(__name__)

torch = None
AutoModelForSequenceClassification = None
AutoTokenizer = None

# Thresholds — tuned for DeBERTa-v3 ensemble
ROUTE_THRESHOLD = 0.85     # Higher threshold for higher quality model
CLARIFY_THRESHOLD = 0.60   
ENTROPY_MAX = 0.28         
MC_PASSES_CPU = 10         # Sequential passes for memory safety
MC_PASSES_GPU = 50         # GPU allows for much better sampling

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


class _FallbackModel:
    def eval(self):
        return None

    def modules(self):
        return []

    def parameters(self):
        return []


class _FallbackTokenizer:
    def __call__(self, *args, **kwargs):
        return {}


class ConfidenceGatedRouter:
    """
    Confidence-Gated Ticket Router using Monte Carlo Dropout.
    Supports DistilBERT and DeBERTa-v3 via AutoModel API.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        self._fallback_mode = False

        force_transformer = os.getenv('SUPPORTMIND_FORCE_TRANSFORMER', '0') == '1'
        if os.name == 'nt' and not force_transformer:
            self.device = 'cpu'
            self._init_fallback(
                "Transformer loading is disabled by default on Windows "
                "to avoid native safetensors/PyTorch access violations. "
                "Set SUPPORTMIND_FORCE_TRANSFORMER=1 to force full model loading."
            )
            return

        global torch, AutoModelForSequenceClassification, AutoTokenizer
        try:
            import torch as torch_module
            from transformers import (
                AutoModelForSequenceClassification as model_loader,
                AutoTokenizer as tokenizer_loader,
            )
            torch = torch_module
            AutoModelForSequenceClassification = model_loader
            AutoTokenizer = tokenizer_loader
        except Exception as exc:
            self.device = 'cpu'
            self._init_fallback(f"PyTorch/Transformers unavailable: {exc}")
            return

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Check for ultimate model first, then standard, then base
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ultimate_path = os.path.join(base_dir, 'models', 'deberta_ultimate')
        standard_path = os.path.join(base_dir, 'models', 'ticket_classifier')
        
        if model_path is None:
            if os.path.exists(os.path.join(ultimate_path, 'config.json')):
                model_name = ultimate_path
            elif os.path.exists(os.path.join(standard_path, 'config.json')):
                model_name = standard_path
            else:
                model_name = 'microsoft/deberta-v3-base'
        else:
            model_name = model_path
        
        logger.info(f"Loading model from: {model_name}")
        logger.info(f"Device: {self.device}")

        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=len(CATEGORY_MAP),
                low_cpu_mem_usage=True
            ).to(self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as exc:
            self._init_fallback(f"Transformer load failed: {exc}")
            return
        
        self.model.eval()
        import gc
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            self.mc_passes = MC_PASSES_GPU
        else:
            self.mc_passes = MC_PASSES_CPU

        logger.info(f"Model loaded successfully. MC Passes: {self.mc_passes} | Params: {sum(p.numel() for p in self.model.parameters()):,}")

    def _init_fallback(self, reason: str):
        self.model = _FallbackModel()
        self.tokenizer = _FallbackTokenizer()
        self.mc_passes = 1
        self._fallback_mode = True
        logger.warning(
            "ConfidenceGatedRouter using lightweight fallback. %s",
            reason,
        )

    def _activate_dropout(self):
        """Keep Dropout active at inference time for MC sampling."""
        if self._fallback_mode or torch is None:
            return
        for m in self.model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()

    def _fallback_predict(self, text: str) -> Tuple[float, float, int, np.ndarray, np.ndarray]:
        text_low = (text or '').lower()
        scores = np.ones(len(CATEGORY_MAP), dtype=float) * 0.2
        keyword_map = {
            0: ['invoice', 'billing', 'refund', 'charge', 'payment', 'subscription'],
            1: ['error', 'bug', 'crash', '500', 'api', 'broken', 'not working'],
            2: ['login', 'password', 'account', 'permission', 'sso', 'user'],
            3: ['feature', 'add', 'request', 'enhancement', 'dark mode'],
            4: ['gdpr', 'compliance', 'legal', 'audit', 'privacy'],
            5: ['setup', 'configure', 'onboard', 'getting started', 'import'],
            6: ['question', 'how do i', 'information', 'help'],
            7: ['cancel', 'churn', 'competitor', 'switching', 'terminate', 'frustrated'],
        }

        for idx, keywords in keyword_map.items():
            scores[idx] += sum(1.0 for keyword in keywords if keyword in text_low)

        probs = scores / scores.sum()
        confidence = float(probs.max())
        entropy = float(-np.sum(probs * np.log(probs + 1e-9)))
        pred_class = int(probs.argmax())
        std_p = np.zeros(len(CATEGORY_MAP), dtype=float)
        return confidence, entropy, pred_class, probs, std_p

    def mc_predict(self, text: str, n_passes: Optional[int] = None) -> Tuple[float, float, int, np.ndarray, np.ndarray]:
        """
        Run N stochastic forward passes via MC Dropout.
        Uses sequential passes to avoid OOM with large models (DeBERTa).
        
        Returns:
            confidence: max(mean_probs)
            entropy: Shannon entropy
            pred_class: predicted class index
            mean_probs: mean probability distribution [num_classes]
            std_probs: standard deviation per class [num_classes]
        """
        if n_passes is None:
            n_passes = self.mc_passes

        if self._fallback_mode:
            return self._fallback_predict(text)

        inputs = self.tokenizer(
            text, return_tensors='pt',
            truncation=True, max_length=128, padding='max_length'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        self._activate_dropout()

        all_probs = []
        with torch.no_grad():
            for _ in range(n_passes):
                logits = self.model(**inputs).logits              # [1, num_classes]
                p = torch.softmax(logits, dim=-1).cpu().numpy()   # [1, num_classes]
                all_probs.append(p[0])

        probs = np.array(all_probs)                  # [n_passes, num_classes]
        mean_p = probs.mean(axis=0)                  # [num_classes]
        std_p = probs.std(axis=0)                    # [num_classes]
        confidence = float(mean_p.max())
        entropy = float(-np.sum(mean_p * np.log(mean_p + 1e-9)))
        pred_class = int(mean_p.argmax())

        return confidence, entropy, pred_class, mean_p, std_p

    def route(self, ticket_text: str, n_passes: Optional[int] = None) -> Dict:
        if n_passes is None:
            n_passes = self.mc_passes
        """
        Route a ticket through the 3-tier confidence gate.
        
        Args:
            ticket_text: Raw ticket text from customer
            n_passes: Number of MC Dropout passes (default 20)
            
        Returns:
            Dictionary with:
                - action: 'route' | 'clarify' | 'escalate'
                - confidence: float [0, 1]
                - entropy: float [0, +inf]
                - top_category: string category name
                - all_probs: list of probabilities per class
                - std_probs: list of std deviations per class (epistemic uncertainty)
                - category_ranking: sorted list of (category, probability) tuples
        """
        conf, ent, cls, probs, std_probs = self.mc_predict(ticket_text, n_passes)
        category = CATEGORY_MAP[cls]

        # Build category ranking (sorted by probability, descending)
        ranking = sorted(
            [(CATEGORY_MAP[i], float(probs[i])) for i in range(len(CATEGORY_MAP))],
            key=lambda x: x[1], reverse=True
        )

        # Top two classes for clarification targeting
        top_two = [ranking[0][0], ranking[1][0]]

        base = {
            'confidence': round(conf, 4),
            'entropy': round(ent, 4),
            'top_category': category,
            'all_probs': {CATEGORY_MAP[i]: round(float(probs[i]), 4) for i in range(len(CATEGORY_MAP))},
            'std_probs': {CATEGORY_MAP[i]: round(float(std_probs[i]), 4) for i in range(len(CATEGORY_MAP))},
            'category_ranking': ranking,
            'top_two_classes': top_two,
            'mc_passes': n_passes,
        }

        if conf >= ROUTE_THRESHOLD and ent <= ENTROPY_MAX:
            return {**base, 'action': 'route', 'queue': category,
                    'reason': f'High confidence ({conf:.2%}) with low entropy ({ent:.3f})'}
        elif conf >= CLARIFY_THRESHOLD:
            return {**base, 'action': 'clarify',
                    'reason': f'Medium confidence ({conf:.2%}) — clarification needed between {top_two[0]} and {top_two[1]}'}
        else:
            return {**base, 'action': 'escalate',
                    'reason': f'Low confidence ({conf:.2%}) — requires human triage'}

    def batch_route(self, tickets: list, n_passes: Optional[int] = None) -> list:
        """Route multiple tickets."""
        return [self.route(t, n_passes) for t in tickets]


if __name__ == '__main__':
    # Quick test
    router = ConfidenceGatedRouter()
    
    test_tickets = [
        "My invoice from last month is incorrect, please fix the billing.",
        "Hey, we have been having issues with the export function since last Tuesday's update. Also our invoice from last month looks incorrect. Can someone help? We are considering upgrading but want this sorted first.",
        "How do I reset my password?",
        "We need to ensure our data handling complies with GDPR regulations.",
        "I want to cancel my subscription, this tool is broken.",
    ]
    
    for ticket in test_tickets:
        result = router.route(ticket)
        print(f"\n{'='*80}")
        print(f"Ticket: {ticket[:80]}...")
        print(f"Action: {result['action'].upper()}")
        print(f"Category: {result['top_category']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Entropy: {result['entropy']:.4f}")
        print(f"Reason: {result['reason']}")
        print(f"Top 3: {result['category_ranking'][:3]}")

