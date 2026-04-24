# src/sla_predictor.py
# SLA Breach Predictor — XGBoost at T=0
# SupportMind v1.0 — Asmitha

import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not installed.")

FEATURE_NAMES = [
    'text_complexity_score', 'agent_queue_depth', 'customer_tier',
    'hour_of_day', 'day_of_week', 'similar_ticket_avg_hrs',
    'sentiment_score', 'repeat_issue', 'escalated_before',
]

class SLABreachPredictor:
    def __init__(self, model_path='models/sla_predictor/sla_xgb.json'):
        self.model = None
        self.model_path = model_path
        if HAS_XGBOOST and os.path.exists(model_path):
            self.model = xgb.Booster()
            self.model.load_model(model_path)
        elif HAS_XGBOOST:
            logger.warning(
                f"Model not found at {model_path}. "
                "Run train_sla.py to generate it. Using heuristic fallback."
            )



    def predict(self, features: dict) -> float:
        if self.model and HAS_XGBOOST:
            vec = np.array([[features.get(f, 0) for f in FEATURE_NAMES]])
            dm = xgb.DMatrix(vec, feature_names=FEATURE_NAMES)
            return round(float(self.model.predict(dm)[0]), 4)
        # Heuristic fallback
        s = 0.3
        s += features.get('agent_queue_depth', 10) * 0.01
        s += features.get('repeat_issue', 0) * 0.15
        s += features.get('escalated_before', 0) * 0.10
        s -= features.get('sentiment_score', 0) * 0.15
        s += max(0, features.get('similar_ticket_avg_hrs', 4) - 4) * 0.03
        if features.get('customer_tier', 2) >= 4: s += 0.10
        h = features.get('hour_of_day', 12)
        if h < 6 or h > 20: s += 0.08
        return round(min(max(s, 0.0), 1.0), 4)

    def explain(self, features: dict) -> dict:
        prob = self.predict(features)
        risk = 'high' if prob >= 0.7 else 'medium' if prob >= 0.4 else 'low'
        factors = []
        if features.get('agent_queue_depth', 0) > 20: factors.append('High queue depth')
        if features.get('sentiment_score', 0) < -0.5: factors.append('Negative sentiment')
        if features.get('repeat_issue', 0): factors.append('Repeat issue')
        if features.get('escalated_before', 0): factors.append('Previously escalated')
        if features.get('customer_tier', 1) >= 4: factors.append('Enterprise SLA')
        return {'breach_probability': prob, 'risk_level': risk,
                'contributing_factors': factors,
                'recommendation': 'Prioritize' if prob >= 0.6 else 'Standard'}

