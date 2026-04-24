# src/confidence_router.py
# Core module: MC Dropout Confidence-Gated Ticket Router
# SupportMind v1.0 — Asmitha

import torch
import numpy as np
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from typing import Dict, Tuple, Optional
import os
import logging

logger = logging.getLogger(__name__)

# Thresholds — tunable per deployment
ROUTE_THRESHOLD = 0.80     # conf >= this → auto-route
CLARIFY_THRESHOLD = 0.55   # conf >= this → ask 1 question
ENTROPY_MAX = 0.35         # entropy <= this → low ambiguity
MC_PASSES = 20             # stochastic forward passes

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


class ConfidenceGatedRouter:
    """
    Confidence-Gated Ticket Router using Monte Carlo Dropout.
    
    Uses DistilBERT with MC Dropout inference to produce calibrated
    confidence scores and Shannon entropy, enabling a 3-tier decision:
    ROUTE (high confidence) / CLARIFY (medium) / ESCALATE (low).
    """

    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        """
        Initialize the router.
        
        Args:
            model_path: Path to fine-tuned DistilBERT model weights.
                        If None, loads base distilbert-base-uncased.
            device: 'auto', 'cpu', or 'cuda'
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        model_exists = model_path and os.path.exists(os.path.join(model_path, 'config.json'))
        model_name = model_path if model_exists else 'distilbert-base-uncased'
        
        logger.info(f"Loading model from: {model_name}")
        logger.info(f"Device: {self.device}")

        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=len(CATEGORY_MAP)
        ).to(self.device)
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            model_path if model_exists else 'distilbert-base-uncased'
        )
        
        self.model.eval()
        logger.info(f"Model loaded successfully. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def _activate_dropout(self):
        """Keep Dropout active at inference time for MC sampling."""
        for m in self.model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()

    def mc_predict(self, text: str, n_passes: int = MC_PASSES) -> Tuple[float, float, int, np.ndarray, np.ndarray]:
        """
        Run N stochastic forward passes via MC Dropout.
        
        Returns:
            confidence: max(mean_probs) — how strongly the model believes in its top prediction
            entropy: Shannon entropy — how spread the probability mass is across classes
            pred_class: predicted class index
            mean_probs: mean probability distribution [num_classes]
            std_probs: standard deviation per class [num_classes] — epistemic uncertainty
        """
        inputs = self.tokenizer(
            text, return_tensors='pt',
            truncation=True, max_length=256, padding='max_length'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Batch N copies for vectorized inference efficiency
        batch = {k: v.repeat(n_passes, 1) for k, v in inputs.items()}

        self._activate_dropout()

        with torch.no_grad():
            logits = self.model(**batch).logits                    # [N, num_classes]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()    # [N, num_classes]

        mean_p = probs.mean(axis=0)              # [num_classes]
        std_p = probs.std(axis=0)                # [num_classes] — epistemic uncertainty
        confidence = float(mean_p.max())
        entropy = float(-np.sum(mean_p * np.log(mean_p + 1e-9)))
        pred_class = int(mean_p.argmax())

        return confidence, entropy, pred_class, mean_p, std_p

    def route(self, ticket_text: str, n_passes: int = MC_PASSES) -> Dict:
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

    def batch_route(self, tickets: list, n_passes: int = MC_PASSES) -> list:
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

