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
    'outage', 'emergency', 'critical', 'urgent', 'immediately', 'blocking', 'locked out',
]

GENERAL_URGENCY = [
    'asap', 'deadline', 'sla', 'escalate', 'priority', 'time-sensitive', 'showstopper', 'presentation',
]

CONTEXTUAL_URGENCY_SIGNALS = [
    (
        'business_impact',
        0.30,
        [
            r'\b(?:affecting|impacting|blocking)\s+(?:our\s+)?(?:customers|users|team|business|operations|sales|revenue|payroll|launch|production)\b',
            r'\b(?:customers?|clients?)\s+(?:(?:are|is)\s+)?(?:waiting|blocked|affected|unable)\b',
            r"\b(?:cannot|can't|unable to)\s+(?:process|ship|launch|serve|sell|invoice|onboard|work|access)\b",
        ],
    ),
    (
        'deadline_pressure',
        0.25,
        [
            r'\b(?:in|within)\s+\d+\s*(?:min|mins|minutes|hour|hours|hrs|days?)\b',
            r'\b(?:by|before)\s+(?:today|tomorrow|eod|end of day|tonight|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\b(?:launch|demo|go-live|renewal|payroll|board meeting|presentation)\b',
        ],
    ),
    (
        'production_outage',
        0.40,
        [
            r'\bproduction\s+(?:is\s+)?(?:down|blocked|broken|failing|impacted)\b',
            r'\b(?:all|multiple|many)\s+(?:users|customers|accounts|teams)\s+(?:are\s+)?(?:affected|blocked|down|unable)\b',
            r'\b(?:system|service|platform|dashboard|api)\s+(?:is\s+)?(?:down|unavailable|not responding)\b',
        ],
    ),
    (
        'access_loss',
        0.25,
        [
            r"\b(?:locked out|cannot access|can't access|unable to access|access is blocked)\b",
            r'\b(?:login|sso|authentication)\s+(?:is\s+)?(?:broken|failing|down|not working)\b',
        ],
    ),
    (
        'repeat_issue',
        0.20,
        [
            r'\b(?:again|still|keeps?|repeated|recurring)\b',
            r'\b(?:second|third|fourth)\s+time\b',
            r'\b(?:raised|reported|opened)\s+(?:this\s+)?(?:before|multiple times|again)\b',
        ],
    ),
]

DEESCALATION_PATTERNS = [
    r'\bnot urgent\b',
    r'\bno rush\b',
    r'\bwhenever you can\b',
    r'\bwhen you have time\b',
]

NEGATIVE_SENTIMENT_SIGNALS = [
    (
        'frustration',
        -0.30,
        [
            r'\bfrustrat(?:ed|ing|ion)\b',
            r'\bnot happy\b',
            r'\bdisappoint(?:ed|ing|ment)\b',
            r'\bthis is becoming difficult\b',
            r'\bnot ideal\b',
            r'\bunacceptable\b',
            r'\bterrible\b',
            r'\bawful\b',
        ],
    ),
    (
        'trust_risk',
        -0.25,
        [
            r'\b(?:losing|lost)\s+(?:trust|confidence)\b',
            r'\b(?:considering|thinking about)\s+(?:switching|leaving|cancelling|canceling)\b',
        ],
    ),
    (
        'polite_negative',
        -0.22,
        [
            r'\b(?:this|it)\s+is\s+(?:affecting|impacting|blocking)\b',
            r'\b(?:could you please|please)\b.*\b(?:fix|resolve|help)\b.*\b(?:blocking|affecting|stuck|broken|failing)\b',
            r'\b(?:becoming|getting)\s+(?:difficult|hard|painful)\b',
        ],
    ),
]

POSITIVE_SENTIMENT_SIGNALS = [
    (
        'appreciation',
        0.08,
        [
            r'\bthanks?\b',
            r'\bthank you\b',
            r'\bappreciate\b',
        ],
    ),
]

LEXICON_SENTIMENT_CUES = [
    'wrong', 'broken', 'error', 'failed', 'failing', 'issue', 'problem',
    'invalid', 'locked out', 'cannot access', "can't access", 'not working',
    'frustrated', 'disappointed', 'difficult', 'unacceptable',
]

COMPLEXITY_KEYWORDS = [
    'integration', 'migration', 'sso', 'bulk', 'setup', 'configure', 'synchronization',
    'permissions', 'architecture', 'implementation', 'customization',
]

MULTI_INTENT_KEYWORDS = ['also', 'and', 'additionally', 'moreover', 'furthermore', 'plus']

PRODUCT_KEYWORDS = {
    'dashboard': 'Dashboard',
    'api': 'API',
    'sso': 'SSO',
    'export': 'Export',
    'integration': 'Integration'
}


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

        urgency = self._urgency_details(text_lower)
        sentiment = self._sentiment_details(text)
        
        return {
            'sentiment_score': sentiment['score'],
            'sentiment_label': sentiment['label'],
            'sentiment_evidence': sentiment['evidence'],
            'sentiment_raw_score': sentiment['raw_score'],
            'urgency_flags': urgency['flags'],
            'urgency_score': urgency['score'],
            'urgency_level': urgency['level'],
            'urgency_evidence': urgency['evidence'],
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
        return self._sentiment_details(text)['score']

    def _sentiment_details(self, text: str) -> Dict:
        tl = text.lower()
        if self.sentiment_analyzer:
            score = self.sentiment_analyzer.polarity_scores(text)['compound']
        else:
            neg = ['bad','terrible','broken','frustrated','angry','worst','hate','useless', 'invalid', 'locked out']
            pos = ['good','great','love','excellent','amazing','helpful','thanks']
            n = sum(1 for w in neg if w in tl)
            p = sum(1 for w in pos if w in tl)
            score = (p - n) / max(p + n, 1)

        raw_score = score
        adjustment, evidence = self._score_pattern_signals(tl, NEGATIVE_SENTIMENT_SIGNALS)
        positive_adjustment, positive_evidence = self._score_pattern_signals(tl, POSITIVE_SENTIMENT_SIGNALS)

        # Polite support messages often include "thanks" while still expressing risk.
        if evidence:
            positive_adjustment *= 0.35

        if 'locked out' in tl:
            adjustment -= 0.35
            evidence.append('access_sentiment: locked out')
        if 'invalid' in tl:
            adjustment -= 0.20
            evidence.append('error_sentiment: invalid')

        score = max(min(score + adjustment + positive_adjustment, 1.0), -1.0)

        if raw_score <= -0.20 and not evidence:
            lexicon_hits = [cue for cue in LEXICON_SENTIMENT_CUES if cue in tl]
            evidence.extend([f'lexicon_sentiment: {cue}' for cue in lexicon_hits[:3]])

        if score <= -0.55 or any(e.startswith(('frustration', 'trust_risk')) for e in evidence):
            label = 'frustrated'
        elif score <= -0.20 or evidence:
            label = 'concerned'
        elif score >= 0.30:
            label = 'positive'
        else:
            label = 'neutral'

        return {
            'score': round(score, 4),
            'raw_score': round(raw_score, 4),
            'label': label,
            'evidence': evidence + positive_evidence,
        }

    def _urgency_flags(self, text_lower: str) -> list:
        return self._urgency_details(text_lower)['flags']

    def _calculate_urgency(self, text_lower: str) -> float:
        """Operational danger score."""
        return self._urgency_details(text_lower)['score']

    def _urgency_details(self, text_lower: str) -> Dict:
        critical_hits = [kw for kw in CRITICAL_URGENCY if kw in text_lower]
        general_hits = [kw for kw in GENERAL_URGENCY if kw in text_lower]
        contextual_score, contextual_evidence = self._score_pattern_signals(
            text_lower,
            CONTEXTUAL_URGENCY_SIGNALS,
        )

        evidence = []
        evidence.extend([f'explicit_critical: {kw}' for kw in critical_hits])
        evidence.extend([f'explicit_general: {kw}' for kw in general_hits])
        evidence.extend(contextual_evidence)

        score = (len(critical_hits) * 0.25) + (len(general_hits) * 0.12) + contextual_score
        if any(re.search(p, text_lower) for p in DEESCALATION_PATTERNS):
            score = min(score, 0.35)
            evidence.append('deescalation: no immediate pressure')

        score = round(min(max(score, 0.0), 1.0), 4)

        if score >= 0.75:
            level = 'critical'
        elif score >= 0.50:
            level = 'high'
        elif score >= 0.25:
            level = 'medium'
        else:
            level = 'low'

        return {
            'score': score,
            'level': level,
            'flags': sorted(set(critical_hits + general_hits + [
                e.split(':', 1)[0] for e in contextual_evidence
            ])),
            'evidence': evidence,
        }

    def _score_pattern_signals(self, text_lower: str, signal_specs: list) -> tuple:
        score = 0.0
        evidence = []
        for label, weight, patterns in signal_specs:
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    score += weight
                    evidence.append(f'{label}: {match.group(0)}')
                    break
        return score, evidence

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

