# src/ticket_validator.py
# Ticket Input Validator — Edge Case Handler
# SupportMind v1.0 — Asmitha

import re
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────
MIN_WORDS = 4
MAX_WORDS = 500
MIN_CHARS = 10
MAX_CHARS = 3000

# Non-English character detection
# Covers: Arabic, Hindi/Devanagari, Tamil, Chinese, 
#         Japanese, Korean, Thai, Russian
NON_LATIN_PATTERN = re.compile(
    r'[\u0600-\u06FF'  # Arabic
    r'\u0900-\u097F'  # Devanagari (Hindi)
    r'\u0B80-\u0BFF'  # Tamil
    r'\u4E00-\u9FFF'  # Chinese
    r'\u3040-\u30FF'  # Japanese
    r'\uAC00-\uD7AF'  # Korean
    r'\u0E00-\u0E7F'  # Thai
    r'\u0400-\u04FF]' # Russian/Cyrillic
)

# Gibberish detection
# No vowels in long sequences = likely gibberish
GIBBERISH_PATTERN = re.compile(r'\b[^aeiou\s]{6,}\b', re.IGNORECASE)


# Already resolved patterns
RESOLVED_PATTERNS = [
    r'never ?mind',
    r'problem (?:is )?(?:solved|fixed|resolved)',
    r'(?:sorted|fixed) (?:it )?(?:out)?',
    r'no longer (?:need|require)',
    r'cancel (?:this )?(?:ticket|request)',
    r'disregard',
    r'ignore (?:this|my)',
    r'thanks?,? (?:got it|all good|figured)',
]

# Greeting/test patterns
GREETING_PATTERNS = [
    r'^hi+\s*[.!?]*$',
    r'^hello+\s*[.!?]*$',
    r'^hey+\s*[.!?]*$',
    r'^test\s*[.!?]*$',
    r'^testing\s*[.!?]*$',
    r'^help\s*[.!?]*$',
    r'^\?\s*$',
    r'^\.+$',
]

# Abuse/spam patterns (basic)
SPAM_PATTERNS = [
    r'(.)\1{9,}',           # Same char repeated 10+ times
    r'(\b\w+\b)(\s+\1){4,}', # Same word repeated 5+ times
]


class TicketValidator:
    """
    Validates and pre-processes ticket text before ML inference.
    
    Catches edge cases early so the ML pipeline never receives
    bad input. Returns structured validation result with
    specific response for each edge case.
    """

    def validate(self, text: str) -> Dict:
        """
        Validate ticket text and return result.
        
        Returns:
            {
                'valid': bool,
                'cleaned_text': str,      # cleaned version if valid
                'error_type': str | None, # type of error if invalid
                'response': str,          # what to show user
                'should_route': bool,     # proceed to ML?
            }
        """

        # ── Check 1: Empty or None ──────────────────────
        if not text or not text.strip():
            return self._invalid(
                error_type='empty',
                response="It looks like your message is empty. "
                         "Please describe your issue and we'll help you right away."
            )

        # Clean whitespace
        cleaned = ' '.join(text.strip().split())

        # ── Check 2: Too short ──────────────────────────
        words = cleaned.split()
        if len(words) < MIN_WORDS or len(cleaned) < MIN_CHARS:
            # Check if it's a greeting specifically
            if any(re.match(p, cleaned.lower()) for p in GREETING_PATTERNS):
                return self._invalid(
                    error_type='greeting',
                    response="Hi there! 👋 Could you describe the issue "
                             "you're experiencing? We're here to help."
                )
            return self._invalid(
                error_type='too_short',
                response="Could you share a bit more detail about your issue? "
                         "The more context you provide, the faster we can help."
            )

        # ── Check 3: Too long ───────────────────────────
        if len(words) > MAX_WORDS or len(cleaned) > MAX_CHARS:
            # Truncate intelligently — keep first 500 words
            truncated_words = words[:MAX_WORDS]
            cleaned = ' '.join(truncated_words)
            logger.info(f"Ticket truncated from {len(words)} to {MAX_WORDS} words")
            # Still valid — just truncated
            return self._valid(
                cleaned_text=cleaned,
                warning="Your message was very long — "
                        "we've focused on the first part to route you correctly."
            )

        # ── Check 4: Non-English ────────────────────────
        non_latin_chars = len(NON_LATIN_PATTERN.findall(cleaned))
        total_chars = len(re.sub(r'\s', '', cleaned))
        non_latin_ratio = non_latin_chars / max(total_chars, 1)

        if non_latin_ratio > 0.3:
            language = self._detect_language_hint(cleaned)
            return self._invalid(
                error_type='non_english',
                response=f"We noticed your message may be in another language. "
                         f"Our routing system currently works best in English. "
                         f"Could you resend your message in English? "
                         f"We want to make sure you reach the right team quickly."
            )

        # ── Check 5: Already resolved ───────────────────
        if any(re.search(p, cleaned.lower()) for p in RESOLVED_PATTERNS):
            return self._invalid(
                error_type='resolved',
                response="Glad to hear it's sorted! 😊 "
                         "If you need anything else, don't hesitate to reach out.",
                should_route=False
            )

        # ── Check 6: Gibberish ──────────────────────────
        gibberish_matches = GIBBERISH_PATTERN.findall(cleaned)
        total_words = len(words)
        gibberish_ratio = len(gibberish_matches) / max(total_words, 1)

        if gibberish_ratio >= 0.4:
            return self._invalid(
                error_type='gibberish',
                response="We couldn't quite understand your message. "
                         "Could you describe your issue in plain language?"
            )

        # ── Check 7: Spam ───────────────────────────────
        for pattern in SPAM_PATTERNS:
            if re.search(pattern, cleaned):
                return self._invalid(
                    error_type='spam',
                    response="We weren't able to process your message. "
                             "Please describe your issue clearly."
                )

        # ── Check 8: Only numbers/symbols ───────────────
        alpha_chars = len(re.findall(r'[a-zA-Z]', cleaned))
        if alpha_chars < 5:
            return self._invalid(
                error_type='no_text',
                response="Could you describe your issue in words? "
                         "We want to make sure you reach the right team."
            )

        # ── All checks passed ────────────────────────────
        return self._valid(cleaned_text=cleaned)

    def _valid(self, cleaned_text: str, warning: str = None) -> Dict:
        return {
            'valid': True,
            'cleaned_text': cleaned_text,
            'error_type': None,
            'response': warning,
            'should_route': True,
            'warning': warning is not None,
        }

    def _invalid(self,
                 error_type: str,
                 response: str,
                 should_route: bool = False) -> Dict:
        return {
            'valid': False,
            'cleaned_text': None,
            'error_type': error_type,
            'response': response,
            'should_route': should_route,
            'warning': False,
        }

    def _detect_language_hint(self, text: str) -> str:
        """Basic language hint for logging."""
        if re.search(r'[\u0B80-\u0BFF]', text):
            return 'Tamil'
        if re.search(r'[\u0900-\u097F]', text):
            return 'Hindi'
        if re.search(r'[\u0600-\u06FF]', text):
            return 'Arabic'
        if re.search(r'[\u4E00-\u9FFF]', text):
            return 'Chinese'
        if re.search(r'[\uAC00-\uD7AF]', text):
            return 'Korean'
        return 'Unknown'


# ── Quick test ──────────────────────────────────────────
if __name__ == '__main__':
    validator = TicketValidator()

    test_cases = [
        ("hi", "greeting"),
        ("", "empty"),
        ("   ", "empty"),
        ("asdfghjkl qwerty zxcvbnm poiuytrewq", "gibberish"),
        ("எனது கணக்கில் சிக்கல் உள்ளது", "tamil"),
        ("My invoice is wrong please help me fix this billing issue", "valid"),
        ("never mind got it sorted thanks", "resolved"),
        ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "spam"),
        ("500 404 200 301 302", "no_text"),
        ("The API endpoint returns 500 error " * 200, "too_long"),
    ]

    print("=" * 60)
    print("TICKET VALIDATOR — EDGE CASE TESTS")
    print("=" * 60)

    for text, expected in test_cases:
        result = validator.validate(text)
        status = "[OK]" if not result['valid'] or result['valid'] else "[ERROR]"
        preview = text[:40] + "..." if len(text) > 40 else text
        print(f"\nInput:    '{preview}'")
        print(f"Expected: {expected}")
        print(f"Got:      {result['error_type'] or 'valid'}")
        print(f"Response: {result['response']}")
