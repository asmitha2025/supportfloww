# tests/test_sla_predictor.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sla_predictor import SLABreachPredictor

def test_predictor_init():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'sla_predictor', 'sla_xgb.json')
    pred = SLABreachPredictor(model_path)
    assert pred is not None

def test_predict_returns_probability():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'sla_predictor', 'sla_xgb.json')
    pred = SLABreachPredictor(model_path)
    features = {
        'text_complexity_score': 10.0, 'agent_queue_depth': 15,
        'customer_tier': 3, 'hour_of_day': 14, 'day_of_week': 2,
        'similar_ticket_avg_hrs': 4.5, 'sentiment_score': -0.3,
        'repeat_issue': 0, 'escalated_before': 0,
    }
    prob = pred.predict(features)
    assert 0 <= prob <= 1

def test_high_risk_features():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'sla_predictor', 'sla_xgb.json')
    pred = SLABreachPredictor(model_path)
    high_risk = {
        'text_complexity_score': 16.0, 'agent_queue_depth': 30,
        'customer_tier': 4, 'hour_of_day': 23, 'day_of_week': 6,
        'similar_ticket_avg_hrs': 12.0, 'sentiment_score': -0.9,
        'repeat_issue': 1, 'escalated_before': 1,
    }
    low_risk = {
        'text_complexity_score': 5.0, 'agent_queue_depth': 3,
        'customer_tier': 1, 'hour_of_day': 10, 'day_of_week': 1,
        'similar_ticket_avg_hrs': 1.5, 'sentiment_score': 0.8,
        'repeat_issue': 0, 'escalated_before': 0,
    }
    assert pred.predict(high_risk) > pred.predict(low_risk)

def test_explain():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'sla_predictor', 'sla_xgb.json')
    pred = SLABreachPredictor(model_path)
    result = pred.explain({
        'text_complexity_score': 10.0, 'agent_queue_depth': 25,
        'customer_tier': 4, 'hour_of_day': 14, 'day_of_week': 2,
        'similar_ticket_avg_hrs': 4.5, 'sentiment_score': -0.6,
        'repeat_issue': 1, 'escalated_before': 0,
    })
    assert 'breach_probability' in result
    assert 'risk_level' in result
    assert result['risk_level'] in ('low', 'medium', 'high')

if __name__ == '__main__':
    test_predictor_init()
    print("✓ Predictor init")
    test_predict_returns_probability()
    print("✓ Returns probability")
    test_high_risk_features()
    print("✓ High > low risk")
    test_explain()
    print("✓ Explain works")
    print("\nAll tests passed! ✅")
