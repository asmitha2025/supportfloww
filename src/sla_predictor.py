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

    @staticmethod
    def _heuristic_score(features: dict) -> float:
        """Conservative SLA risk prior used to calibrate weak/old model outputs."""
        queue_depth = max(0, float(features.get('agent_queue_depth', 10)))
        customer_tier = max(1, min(4, int(features.get('customer_tier', 2))))
        sentiment = float(features.get('sentiment_score', 0))
        similar_hours = max(0, float(features.get('similar_ticket_avg_hrs', 4)))
        complexity = max(0, float(features.get('text_complexity_score', 8)))
        hour = int(features.get('hour_of_day', 12))

        score = 0.10
        score += min(queue_depth / 50.0, 1.0) * 0.24
        score += ((customer_tier - 1) / 3.0) * 0.16
        score += min(max(-sentiment, 0.0), 1.0) * 0.16
        score += min(max(similar_hours - 4.0, 0.0) / 12.0, 1.0) * 0.12
        score += min(complexity / 20.0, 1.0) * 0.08

        if features.get('repeat_issue', 0):
            score += 0.13
        if features.get('escalated_before', 0):
            score += 0.12
        if hour < 6 or hour > 20:
            score += 0.05

        return round(min(max(score, 0.0), 0.98), 4)

    def _model_predict(self, features: dict) -> float:
        if self.model and HAS_XGBOOST:
            vec = np.array([[features.get(f, 0) for f in FEATURE_NAMES]])
            dm = xgb.DMatrix(vec, feature_names=FEATURE_NAMES)
            return round(float(self.model.predict(dm)[0]), 4)
        return self._heuristic_score(features)

    def predict(self, features: dict) -> float:
        model_prob = self._model_predict(features)
        heuristic_prob = self._heuristic_score(features)
        # SLA is an operational safety signal: if the trained artifact is
        # under-confident on obvious risk factors, use the conservative prior.
        return round(max(model_prob, heuristic_prob), 4)

    def explain(self, features: dict) -> dict:
        model_prob = self._model_predict(features)
        heuristic_prob = self._heuristic_score(features)
        prob = self.predict(features)
        risk = 'high' if prob >= 0.7 else 'medium' if prob >= 0.4 else 'low'
        factors = []
        if features.get('agent_queue_depth', 0) > 20: factors.append('High queue depth')
        if features.get('sentiment_score', 0) < -0.5: factors.append('Negative sentiment')
        if features.get('repeat_issue', 0): factors.append('Repeat issue')
        if features.get('escalated_before', 0): factors.append('Previously escalated')
        if features.get('customer_tier', 1) >= 4: factors.append('Enterprise SLA')
        source = 'model'
        if heuristic_prob > model_prob:
            source = 'heuristic_guardrail' if self.model and HAS_XGBOOST else 'heuristic_fallback'
        return {'breach_probability': prob, 'risk_level': risk,
                'model_probability': model_prob,
                'heuristic_probability': heuristic_prob,
                'calibration_source': source,
                'contributing_factors': factors,
                'recommendation': 'Prioritize' if prob >= 0.6 else 'Standard'}

