import os
import sys
import json
import pytest
from fastapi.testclient import TestClient

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from api import app, _order_intents_by_probability

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_route_endpoint():
    payload = {
        "text": "I can't log in to my account, it says my password is wrong",
        "customer_id": "test_123"
    }
    response = client.post("/route", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "action" in data
    assert "confidence" in data
    assert "latency_ms" in data
    assert data["customer_id"] == "test_123"
    # Ensure SLA breach probability is returned
    assert "sla_breach_probability" in data

def test_clear_ticket_does_not_always_clarify():
    payload = {
        "text": "My invoice from last month shows wrong amount please fix this billing error",
        "customer_id": "test_123"
    }
    response = client.post("/route", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["action"] in ("route", "escalate")
    assert "clarification" not in data

def test_direct_feature_request_routes_without_clarification():
    payload = {
        "text": "Could you add dark mode to the dashboard in a future release?",
        "customer_id": "test_123"
    }
    response = client.post("/route", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["action"] == "route"
    assert data["top_category"] == "feature_request"
    assert "clarification" not in data

def test_password_reset_is_support_request():
    payload = {
        "text": "I forgot my password and need help resetting access to my account.",
        "customer_id": "test_123"
    }
    response = client.post("/route", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["action"] == "route"
    assert data["top_category"] == "account_management"

def test_admin_password_access_routes_account_management():
    payload = {
        "text": "I forgot my password and cannot access the admin dashboard.",
        "customer_id": "test_123"
    }
    response = client.post("/route", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["action"] == "route"
    assert data["top_category"] == "account_management"
    assert "clarification" not in data

def test_invoice_and_sso_login_detects_multi_intent():
    payload = {
        "text": "The invoice is wrong, and also SSO login is broken for our managers.",
        "customer_id": "test_123"
    }
    response = client.post("/route", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["action"] == "multi_route"
    assert data["primary_queue"] == "billing"
    assert data["secondary_queue"] == "account_management"
    assert data["is_multi_intent"] is True

@pytest.mark.parametrize("text", [
    "Hey, we have been having issues with the export function since last Tuesday's update. Also our invoice from last month looks incorrect.",
    "Could you please help resolve this? This is becoming difficult for our onboarding team and we are disappointed with repeated delays.",
])
def test_multi_route_primary_matches_probability_chart(text):
    response = client.post("/route", json={"text": text, "customer_id": "test_123"})
    assert response.status_code == 200
    data = response.json()
    if data["action"] == "multi_route":
        chart_top = max(data["all_probs"], key=data["all_probs"].get)
        assert data["primary_queue"] == chart_top
        assert data["secondary_queue"] != data["primary_queue"]

def test_multi_route_ordering_uses_probability_chart():
    result = {
        "all_probs": {
            "billing": 0.69,
            "technical_support": 0.11,
            "churn_risk": 0.16,
        }
    }
    assert _order_intents_by_probability(
        ["technical_support", "billing"],
        result,
    ) == ["billing", "technical_support"]

def test_route_applies_clarification_answer():
    payload = {
        "text": "Export is broken and the invoice looks incorrect.",
        "customer_id": "test_123",
        "clarification_choice": "Billing or invoice issue",
        "clarification_target": "billing",
        "clarification_question_id": "Q001"
    }
    response = client.post("/route", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["action"] == "route"
    assert data["top_category"] == "billing"
    assert data["clarification_applied"] is True
    assert "clarification" not in data

def test_sla_predict_endpoint():
    payload = {
        "text_complexity_score": 10.5,
        "agent_queue_depth": 5,
        "customer_tier": 2,
        "hour_of_day": 10,
        "day_of_week": 1,
        "similar_ticket_avg_hrs": 2.0,
        "sentiment_score": 0.5,
        "repeat_issue": 0,
        "escalated_before": 0
    }
    response = client.post("/sla/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "breach_probability" in data
    assert "risk_level" in data

def test_churn_signal_endpoint():
    payload = {
        "thread_texts": [
            "This product is terrible and I'm very frustrated.",
            "I'm going to cancel my subscription and switch to a competitor."
        ]
    }
    response = client.post("/churn/signal", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_risk_score" in data
    assert "risk_level" in data
    assert data["competitor_mention"] is True
    assert data["cancellation_language"] is True

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "routing_stats" in data
    assert "model_status" in data

def test_explain_endpoint():
    response = client.post(
        "/explain",
        json={
            "text": "The invoice charge is wrong and I need a refund.",
            "target_class": "billing"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "tokens" in data
    assert "values" in data
    assert len(data["tokens"]) == len(data["values"])
    assert data["source"] in ("shap_transformer", "heuristic_keywords")

def test_model_status_endpoint():
    response = client.get("/model/status")
    assert response.status_code == 200
    data = response.json()
    assert "router" in data
    assert "explainability" in data
    assert data["explainability"] in ("shap_transformer", "heuristic_keywords")
