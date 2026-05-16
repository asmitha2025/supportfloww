import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from feature_extraction import FeatureExtractor


def test_detects_indirect_urgency_without_urgent_keyword():
    extractor = FeatureExtractor()
    features = extractor.extract(
        "Our launch is in 30 minutes and export is not working. "
        "The client is waiting on this before signing."
    )

    assert features['urgency_score'] >= 0.5
    assert features['urgency_level'] in ('high', 'critical')
    assert any(e.startswith('deadline_pressure') for e in features['urgency_evidence'])
    assert any(e.startswith('business_impact') for e in features['urgency_evidence'])


def test_detects_polite_negative_sentiment():
    extractor = FeatureExtractor()
    features = extractor.extract(
        "Could you please help resolve this? This is becoming difficult "
        "for our onboarding team and we are disappointed with repeated delays."
    )

    assert features['sentiment_score'] < -0.2
    assert features['sentiment_label'] in ('concerned', 'frustrated')
    assert any(
        e.startswith(('frustration', 'polite_negative'))
        for e in features['sentiment_evidence']
    )


def test_no_rush_deescalates_urgency():
    extractor = FeatureExtractor()
    features = extractor.extract(
        "No rush, but can you tell me how to update the invoice email before tomorrow?"
    )

    assert features['urgency_score'] <= 0.35
    assert features['urgency_level'] in ('low', 'medium')
    assert any(e.startswith('deescalation') for e in features['urgency_evidence'])
