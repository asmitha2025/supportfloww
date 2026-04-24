import os
import sys
import json
from fastapi.testclient import TestClient

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from api import app

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
