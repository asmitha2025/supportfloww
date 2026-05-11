# src/feature_extraction.py
# Feature Extraction Module — Multi-signal ticket analysis
# SupportMind v1.0 — Asmitha

import re
import logging
from typing import Dict

logger = logging.getLogger(__name__)

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except ImportError:
    HAS_VADER = False

CRITICAL_URGENCY = [
    'crash', 'blocked', 'down', 'failing', 'cannot access', 'production issue',
    'outage', 'emergency', 'critical', 'urgent', 'immediately', 'blocking',
]

GENERAL_URGENCY = [
    'asap', 'deadline', 'sla', 'escalate', 'priority', 'time-sensitive', 'showstopper',
]

COMPLEXITY_KEYWORDS = [
    'integration', 'migration', 'sso', 'bulk', 'setup', 'configure', 'synchronization',
    'permissions', 'architecture', 'implementation', 'customization',
]

MULTI_INTENT_KEYWORDS = ['also', 'and', 'additionally', 'moreover', 'furthermore', 'plus']


class FeatureExtractor:
    """
    Extracts multi-signal features from raw ticket text.
    
    Features:
        - Sentiment score (VADER or fallback)
        - Urgency score (Operational danger)
        - Complexity score (Implementation difficulty)
        - Product/feature entity recognition
        - Text complexity (Flesch-Kincaid approximation)
        - Token count
        - Named entities (basic regex-based NER)
    """

    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer() if HAS_VADER else None

    def extract(self, text: str) -> Dict:
        """Extract all features from ticket text."""
        text_lower = text.lower()
        words = text.split()
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

        urg_flags = self._urgency_flags(text_lower)
        
        return {
            'sentiment_score': self._sentiment(text),
            'urgency_flags': urg_flags,
            'urgency_score': self._calculate_urgency(text_lower),
            'complexity_score': self._calculate_complexity(text_lower),
            'product_entities': self._product_entities(text_lower),
            'text_complexity_score': self._flesch_kincaid(words, sentences),
            'token_count': len(words),
            'sentence_count': len(sentences),
            'has_question': '?' in text,
            'has_error_code': bool(re.search(r'error\s*(?:code\s*)?[\d#:]+|err[-_]\d+|HTTP\s*\d{3}', text, re.I)),
            'has_multi_intent_signal': any(kw in text_lower for kw in MULTI_INTENT_KEYWORDS),
            'email_mentions': len(re.findall(r'[\w.+-]+@[\w-]+\.[\w.]+', text)),
            'url_mentions': len(re.findall(r'https?://\S+', text)),
            'mentioned_dates': bool(re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\blast\s+(?:week|month|tuesday|monday|wednesday|thursday|friday)\b', text_lower)),
        }

    def _sentiment(self, text: str) -> float:
        if self.sentiment_analyzer:
            return self.sentiment_analyzer.polarity_scores(text)['compound']
        neg = ['bad','terrible','broken','frustrated','angry','worst','hate','useless']
        pos = ['good','great','love','excellent','amazing','helpful','thanks']
        tl = text.lower()
        n = sum(1 for w in neg if w in tl)
        p = sum(1 for w in pos if w in tl)
        return (p - n) / max(p + n, 1)

    def _urgency_flags(self, text_lower: str) -> list:
        found = [kw for kw in CRITICAL_URGENCY if kw in text_lower]
        found.extend([kw for kw in GENERAL_URGENCY if kw in text_lower])
        return list(set(found))

    def _calculate_urgency(self, text_lower: str) -> float:
        """Operational danger score."""
        crit_count = sum(1 for kw in CRITICAL_URGENCY if kw in text_lower)
        gen_count = sum(1 for kw in GENERAL_URGENCY if kw in text_lower)
        
        score = (crit_count * 0.4) + (gen_count * 0.15)
        return min(max(score, 0.0), 1.0)

    def _calculate_complexity(self, text_lower: str) -> float:
        """Implementation difficulty score."""
        comp_count = sum(1 for kw in COMPLEXITY_KEYWORDS if kw in text_lower)
        score = comp_count * 0.25
        return min(max(score, 0.0), 1.0)

    def _product_entities(self, text_lower: str) -> list:
        found = []
        for kw, label in PRODUCT_KEYWORDS.items():
            if kw in text_lower and label not in found:
                found.append(label)
        return found

    def _flesch_kincaid(self, words: list, sentences: list) -> float:
        if not words or not sentences:
            return 0.0
        avg_sentence_len = len(words) / len(sentences)
        syllables = sum(self._count_syllables(w) for w in words)
        avg_syllables = syllables / max(len(words), 1)
        grade = 0.39 * avg_sentence_len + 11.8 * avg_syllables - 15.59
        return round(max(0, grade), 2)

    def _count_syllables(self, word: str) -> int:
        word = word.lower().strip(".,!?;:'\"")
        if len(word) <= 2:
            return 1
        vowels = 'aeiouy'
        count = 0
        prev_vowel = False
        for ch in word:
            is_vowel = ch in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        if word.endswith('e') and count > 1:
            count -= 1
        return max(count, 1)


if __name__ == '__main__':
    ext = FeatureExtractor()
    ticket = "Hey, we have been having issues with the export function since last Tuesday's update. Also our invoice from last month looks incorrect. Can someone help? We are considering upgrading but want this sorted first."
    features = ext.extract(ticket)
    for k, v in features.items():
        print(f"  {k}: {v}")

