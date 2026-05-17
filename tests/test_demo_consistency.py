import os
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from api import app


client = TestClient(app)

DEMO_TICKETS = [
    "My invoice from last month shows $299 but my plan is $199. Please fix this billing error immediately.",
    "The API endpoint /v2/export returns a 500 error when batch size exceeds 1000 records. Stack trace attached.",
    "The invoice is wrong, and also SSO login is broken for our managers.",
    "Hey, we have been having issues with the export function since last Tuesday's update. Also our invoice from last month looks incorrect.",
    "Our launch is in 30 minutes and export is not working. The client is waiting on this before signing.",
    "Could you please help resolve this? This is becoming difficult for our onboarding team and we are disappointed with repeated delays.",
    "No rush, but can you tell me how to update the invoice email before tomorrow?",
    "Could you add dark mode to the dashboard in a future release?",
    "I forgot my password and cannot access the admin dashboard.",
    "We are going to cancel and switch to a competitor next month if this issue is not resolved.",
]


def route(text):
    response = client.post('/route', json={'text': text, 'customer_id': 'demo_qa'})
    assert response.status_code == 200
    return response.json()


def test_demo_tickets_return_consistent_decision_payloads():
    for ticket in DEMO_TICKETS:
        data = route(ticket)

        assert data['action'] in {
            'route',
            'clarify',
            'escalate',
            'multi_route',
            'invalid_input',
        }
        assert 'latency_ms' in data
        assert 'features' in data

        if data['action'] != 'invalid_input':
            probs = data['all_probs']
            assert abs(sum(probs.values()) - 1.0) < 0.02
            assert data['top_category'] in probs


def test_multi_route_labels_match_probability_chart():
    for ticket in DEMO_TICKETS:
        data = route(ticket)
        if data['action'] != 'multi_route':
            continue

        sorted_probs = sorted(
            data['all_probs'].items(),
            key=lambda item: item[1],
            reverse=True,
        )
        chart_top_two = [category for category, _ in sorted_probs[:2]]

        assert chart_top_two == [data['primary_queue'], data['secondary_queue']]
        assert data['route_chart_consistent'] is True


def test_non_neutral_sentiment_has_visible_evidence():
    for ticket in DEMO_TICKETS:
        data = route(ticket)
        if data['action'] == 'invalid_input':
            continue

        features = data['features']
        sentiment_label = features.get('sentiment_label')
        if sentiment_label and sentiment_label != 'neutral':
            assert features.get('sentiment_evidence'), ticket
            assert data['sentiment_evidence_consistent'] is True

