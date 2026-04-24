# src/churn_extractor.py
# Churn Signal Extractor — Sentiment + Pattern Analysis
# SupportMind v1.0 — Asmitha

import re
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except ImportError:
    HAS_VADER = False

COMPETITOR_PATTERNS = [
    r'switch(?:ing)? to', r'moving to', r'looking at \w+',
    r'competitor', r'alternative', r'another (?:tool|platform|solution)',
    r'better option', r'other providers',
]

CANCELLATION_PATTERNS = [
    r'cancel', r'stop (?:using|subscription)', r'end (?:my )?contract',
    r'not renew(?:ing)?', r'downgrad(?:e|ing)', r'close (?:my )?account',
    r'terminate', r'discontinue', r'opt out',
]

FRUSTRATION_PATTERNS = [
    r'very frustrated', r'completely broken', r'this is unacceptable',
    r'third time', r'again\b', r'still not (?:fixed|working|resolved)',
    r'waste of time', r'terrible', r'awful', r'disgusted',
    r'fed up', r'last straw', r'ridiculous',
]

URGENCY_PATTERNS = [
    r'asap', r'urgent(?:ly)?', r'immediately', r'critical',
    r'blocking', r'production (?:is )?down', r'outage',
    r'deadline', r'cannot wait',
]


class ChurnSignalExtractor:
    """
    Extracts churn risk signals from support thread history.
    
    Scans for competitor mentions, cancellation language, frustration
    patterns, and sentiment trajectory. Produces a composite churn
    risk score [0–1] for CRM health record updates.
    """

    def __init__(self):
        if HAS_VADER:
            self.analyzer = SentimentIntensityAnalyzer()
        else:
            self.analyzer = None
            logger.warning("VADER not installed. Using basic sentiment heuristic.")

    def _get_sentiment(self, text: str) -> float:
        """Get sentiment score from -1.0 (negative) to 1.0 (positive)."""
        if self.analyzer:
            return self.analyzer.polarity_scores(text)['compound']
        # Basic fallback
        neg_words = ['bad', 'terrible', 'awful', 'broken', 'frustrated',
                     'angry', 'worst', 'hate', 'useless', 'horrible']
        pos_words = ['good', 'great', 'love', 'excellent', 'amazing',
                     'helpful', 'perfect', 'thanks', 'wonderful']
        text_lower = text.lower()
        neg = sum(1 for w in neg_words if w in text_lower)
        pos = sum(1 for w in pos_words if w in text_lower)
        total = neg + pos
        if total == 0:
            return 0.0
        return (pos - neg) / total

    def extract(self, thread_texts: List[str]) -> Dict:
        """
        Extract churn signals from a support thread.
        
        Args:
            thread_texts: List of message strings in the support thread
            
        Returns:
            Dictionary with churn_risk_score, flags, and details
        """
        full_text = ' '.join(thread_texts).lower()

        # Pattern matching
        competitor = any(re.search(p, full_text) for p in COMPETITOR_PATTERNS)
        cancellation = any(re.search(p, full_text) for p in CANCELLATION_PATTERNS)
        frustration = sum(1 for p in FRUSTRATION_PATTERNS if re.search(p, full_text))
        urgency = sum(1 for p in URGENCY_PATTERNS if re.search(p, full_text))

        # Sentiment trajectory (across messages)
        sentiments = [self._get_sentiment(t) for t in thread_texts[:10]]
        neg_count = sum(1 for s in sentiments if s < -0.3)
        avg_sentiment = sum(sentiments) / max(len(sentiments), 1)

        # Sentiment trajectory: is it getting worse?
        if len(sentiments) >= 3:
            early = sum(sentiments[:len(sentiments)//2]) / max(len(sentiments)//2, 1)
            late = sum(sentiments[len(sentiments)//2:]) / max(len(sentiments) - len(sentiments)//2, 1)
            deteriorating = late < early - 0.2
        else:
            deteriorating = False

        # Composite churn risk score [0–1]
        score = min(1.0,
            (0.40 if cancellation else 0.0) +
            (0.30 if competitor else 0.0) +
            min(frustration * 0.10, 0.20) +
            (neg_count / max(len(sentiments), 1)) * 0.10 +
            (0.10 if deteriorating else 0.0)
        )

        risk_level = 'critical' if score >= 0.7 else 'high' if score >= 0.5 else 'medium' if score >= 0.3 else 'low'

        return {
            'churn_risk_score': round(score, 3),
            'risk_level': risk_level,
            'competitor_mention': competitor,
            'cancellation_language': cancellation,
            'frustration_count': frustration,
            'urgency_count': urgency,
            'negative_sentiment_ratio': round(neg_count / max(len(sentiments), 1), 3),
            'average_sentiment': round(avg_sentiment, 3),
            'sentiment_deteriorating': deteriorating,
            'message_count': len(thread_texts),
            'recommendation': self._get_recommendation(score, competitor, cancellation),
        }

    def _get_recommendation(self, score: float, competitor: bool, cancellation: bool) -> str:
        if score >= 0.7:
            return 'IMMEDIATE escalation to Customer Success Manager'
        if cancellation:
            return 'Route to retention team with priority flag'
        if competitor:
            return 'Alert Account Manager — competitive threat detected'
        if score >= 0.4:
            return 'Flag for proactive outreach within 24 hours'
        return 'Standard processing — monitor sentiment'


if __name__ == '__main__':
    extractor = ChurnSignalExtractor()
    thread = [
        "Hi, I've been having issues with the export feature for two weeks now.",
        "This is the third time I'm reporting this. Still not fixed.",
        "I'm very frustrated. We're looking at alternative solutions.",
        "If this isn't resolved by Friday, we'll need to cancel our subscription.",
    ]
    result = extractor.extract(thread)
    for k, v in result.items():
        print(f"  {k}: {v}")

