# src/clarification_engine.py
# Clarification Engine — Maximum Information Gain Question Selector
# SupportMind v1.0 — Asmitha
# Updated: Hybrid LLM + Template Architecture

import json
import os
import logging
from dotenv import load_dotenv
load_dotenv()
import numpy as np
from scipy.stats import entropy as scipy_entropy
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Groq LLM Setup ─────────────────────────────────────
try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False
    logger.warning("Groq not installed. Using template bank only.")

GROQ_MODEL = os.getenv('GROQ_MODEL', 'llama3-8b-8192')
LLM_ENABLED = os.getenv('LLM_ENABLED', 'true').lower() == 'true'


class ClarificationEngine:
    """
    Selects the optimal clarification question to resolve routing ambiguity.

    Architecture (Hybrid):
    1. Try LLM-generated question (ticket-specific, dynamic)
    2. Fall back to template bank if LLM fails or unavailable
    3. Fall back to generic question if both fail

    When the confidence-gated router returns 'clarify' (confidence 0.55-0.80),
    this engine selects one question using either:
    - Groq LLaMA3: generates question specific to the exact ticket text
    - Template bank: 47 pre-built questions scored by expected info gain
    """

    def __init__(self, bank_path: str = 'data/clarification_bank.json'):
        # Load template bank
        if not os.path.exists(bank_path):
            logger.warning(f"Bank not found at {bank_path}, using defaults")
            self.bank = self._default_bank()
        else:
            with open(bank_path, 'r', encoding='utf-8') as f:
                self.bank = json.load(f)
        logger.info(f"Loaded {len(self.bank)} clarification templates")

        # Groq client
        self.groq_client = None
        if HAS_GROQ and LLM_ENABLED:
            api_key = os.getenv('GROQ_API_KEY')
            if api_key:
                self.groq_client = Groq(api_key=api_key)
                logger.info("Groq LLM client initialized")
            else:
                logger.warning("GROQ_API_KEY not set. Using templates only.")

    # ── LLM Question Generation ─────────────────────────

    def generate_llm_question(self,
                               ticket_text: str,
                               top_two_classes: List[str]) -> Optional[Dict]:
        """
        Use Groq LLaMA3 to generate a ticket-specific clarification question.
        Returns None if generation fails — caller falls back to templates.
        """
        if not self.groq_client:
            return None

        category_descriptions = {
            'billing': 'payment, invoice, charge, refund, subscription cost',
            'technical_support': 'software error, bug, API issue, feature not working',
            'account_management': 'user access, permissions, account settings, SSO',
            'feature_request': 'new capability, enhancement, missing feature',
            'compliance_legal': 'GDPR, audit, data privacy, regulatory',
            'onboarding': 'new user setup, getting started, configuration',
            'general_inquiry': 'general question, information request',
            'churn_risk': 'cancellation, switching, dissatisfaction',
        }

        cat_a = top_two_classes[0]
        cat_b = top_two_classes[1]
        desc_a = category_descriptions.get(cat_a, cat_a)
        desc_b = category_descriptions.get(cat_b, cat_b)

        prompt = f"""You are a B2B SaaS support triage assistant.

A customer sent this support ticket:
"{ticket_text}"

Our AI is uncertain whether this is:
- Category A: {cat_a} ({desc_a})
- Category B: {cat_b} ({desc_b})

Generate ONE short clarifying question that:
1. References specific details from this exact ticket
2. Has exactly two answer options (one pointing to each category)
3. Is friendly and professional
4. Is under 25 words

Respond with ONLY this JSON, no other text:
{{
  "question": "your question here",
  "option_a": "short answer pointing to {cat_a}",
  "option_b": "short answer pointing to {cat_b}"
}}"""

        try:
            response = self.groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3,
            )

            raw = response.choices[0].message.content.strip()

            # Clean up response
            if '```json' in raw:
                raw = raw.split('```json')[1].split('```')[0].strip()
            elif '```' in raw:
                raw = raw.split('```')[1].split('```')[0].strip()

            result = json.loads(raw)

            # Validate required fields
            if not all(k in result for k in ['question', 'option_a', 'option_b']):
                raise ValueError("Missing required fields in LLM response")

            logger.info(f"LLM question generated successfully")

            return {
                'question_id': 'LLM_DYNAMIC',
                'question_text': result['question'],
                'options': [result['option_a'], result['option_b']],
                'expected_gain': 0.75,
                'relevant_classes': top_two_classes,
                'source': 'llm_groq',
                'fallback': False,
            }

        except json.JSONDecodeError as e:
            logger.warning(f"LLM JSON parse failed: {e}. Falling back to template.")
            return None
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}. Falling back to template.")
            return None

    # ── Template Selection ──────────────────────────────

    def expected_information_gain(self,
                                   question: dict,
                                   current_probs: np.ndarray) -> float:
        """Calculate expected entropy reduction for a template question."""
        prior_entropy = scipy_entropy(current_probs + 1e-9)
        gains = []
        for answer_label, posterior in question['posteriors'].items():
            posterior_probs = np.array(posterior)
            posterior_probs = posterior_probs / (posterior_probs.sum() + 1e-9)
            posterior_entropy = scipy_entropy(posterior_probs + 1e-9)
            gain = prior_entropy - posterior_entropy
            gains.append(max(gain, 0))
        return float(np.mean(gains)) if gains else 0.0

    def select_question(self,
                        current_probs: np.ndarray,
                        top_two_classes: List[str],
                        asked_ids: Optional[List[str]] = None,
                        ticket_text: Optional[str] = None) -> Dict:
        """
        Select best clarification question.

        Priority:
        1. LLM dynamic question (if ticket_text provided and Groq available)
        2. Template bank (information gain scoring)
        3. Generic fallback

        Args:
            current_probs: probability distribution [num_classes]
            top_two_classes: top two predicted categories
            asked_ids: already asked question IDs
            ticket_text: original ticket text for LLM generation
        """

        # ── Layer 1: Try LLM ──────────────────────────
        if ticket_text and self.groq_client:
            llm_question = self.generate_llm_question(
                ticket_text, top_two_classes
            )
            if llm_question:
                return llm_question

        # ── Layer 2: Template bank ────────────────────
        asked_ids = asked_ids or []

        relevant = [
            q for q in self.bank
            if any(c in q.get('relevant_classes', []) for c in top_two_classes)
            and q['id'] not in asked_ids
        ]

        if not relevant:
            relevant = [q for q in self.bank if q['id'] not in asked_ids]

        # ── Layer 3: Generic fallback ─────────────────
        if not relevant:
            return {
                'question_id': 'FALLBACK',
                'question_text': 'Could you clarify the main issue you need resolved today?',
                'options': [],
                'expected_gain': 0.0,
                'source': 'fallback',
                'fallback': True,
            }

        # Score templates by info gain
        scored = [
            (q, self.expected_information_gain(q, current_probs))
            for q in relevant
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        best_q, best_gain = scored[0]

        return {
            'question_id': best_q['id'],
            'question_text': best_q['text'],
            'options': best_q.get('options', []),
            'expected_gain': round(best_gain, 4),
            'relevant_classes': best_q.get('relevant_classes', []),
            'source': 'template',
            'fallback': False,
        }

    def get_all_questions(self) -> List[Dict]:
        return self.bank

    def get_question_by_id(self, question_id: str) -> Optional[Dict]:
        for q in self.bank:
            if q['id'] == question_id:
                return q
        return None

    def _default_bank(self) -> list:
        return [
            {
                "id": "Q001",
                "text": "Is the main issue related to (A) a software error or (B) your billing or invoice?",
                "options": ["Software error", "Billing or invoice"],
                "relevant_classes": ["billing", "technical_support"],
                "posteriors": {
                    "technical": [0.85,0.05,0.03,0.02,0.01,0.01,0.02,0.01],
                    "billing": [0.05,0.82,0.05,0.02,0.02,0.01,0.02,0.01]
                }
            },
        ]


if __name__ == '__main__':
    engine = ClarificationEngine()

    probs = np.array([0.35, 0.30, 0.10, 0.08, 0.05, 0.04, 0.05, 0.03])
    top_two = ['billing', 'technical_support']

    # Test template
    result = engine.select_question(probs, top_two)
    print(f"Template: {result['question_text']}")
    print(f"Source: {result['source']}")

    # Test LLM
    ticket = "Hey, export has been broken since Tuesday and our invoice looks wrong too"
    result_llm = engine.select_question(probs, top_two, ticket_text=ticket)
    print(f"\nLLM: {result_llm['question_text']}")
    print(f"Source: {result_llm['source']}")
