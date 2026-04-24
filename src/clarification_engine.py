# src/clarification_engine.py
# Clarification Engine — Maximum Information Gain Question Selector
# SupportMind v1.0 — Asmitha

import json
import numpy as np
from scipy.stats import entropy as scipy_entropy
from typing import Dict, List, Optional
import os
import logging

logger = logging.getLogger(__name__)


class ClarificationEngine:
    """
    Selects the optimal clarification question to resolve routing ambiguity.
    
    When the confidence-gated router returns 'clarify' (confidence 0.55–0.80),
    this engine selects one question from a bank of 47 templates using
    maximum expected information gain — the question whose answer would
    reduce Shannon entropy the most.
    """

    def __init__(self, bank_path: str = 'data/clarification_bank.json'):
        """
        Load the clarification question bank.
        
        Args:
            bank_path: Path to the JSON file containing question templates
                       with posterior distributions per answer.
        """
        if not os.path.exists(bank_path):
            logger.warning(f"Clarification bank not found at {bank_path}, using built-in defaults")
            self.bank = self._default_bank()
        else:
            with open(bank_path, 'r', encoding='utf-8') as f:
                self.bank = json.load(f)
        
        logger.info(f"Loaded {len(self.bank)} clarification templates")

    def expected_information_gain(self, question: dict, current_probs: np.ndarray) -> float:
        """
        Estimate how much a question's answer would reduce routing entropy.
        
        For each possible answer to the question, we have a posterior probability
        distribution over categories. The expected information gain is the
        average reduction in entropy across all possible answers.
        
        Args:
            question: Question template dict with 'posteriors' mapping
            current_probs: Current probability distribution over categories [num_classes]
            
        Returns:
            Expected entropy reduction (higher = better question)
        """
        prior_entropy = scipy_entropy(current_probs + 1e-9)
        gains = []
        
        for answer_label, posterior in question['posteriors'].items():
            posterior_probs = np.array(posterior)
            # Normalize posteriors
            posterior_probs = posterior_probs / (posterior_probs.sum() + 1e-9)
            posterior_entropy = scipy_entropy(posterior_probs + 1e-9)
            gain = prior_entropy - posterior_entropy
            gains.append(max(gain, 0))  # Information gain can't be negative
        
        return float(np.mean(gains)) if gains else 0.0

    def select_question(self, current_probs: np.ndarray, top_two_classes: List[str],
                        asked_ids: Optional[List[str]] = None) -> Dict:
        """
        Select the question that maximizes expected entropy reduction.
        
        Args:
            current_probs: Current probability distribution [num_classes]
            top_two_classes: The two most likely categories from the router
            asked_ids: List of question IDs already asked (to avoid repeats)
            
        Returns:
            Dictionary with question_id, question_text, options, expected_gain
        """
        asked_ids = asked_ids or []
        
        # Filter to questions relevant to the ambiguous categories
        relevant = [
            q for q in self.bank
            if any(c in q.get('relevant_classes', []) for c in top_two_classes)
            and q['id'] not in asked_ids
        ]
        
        if not relevant:
            # Fallback to general questions
            relevant = [q for q in self.bank if q['id'] not in asked_ids]
        
        if not relevant:
            return {
                'question_id': 'NONE',
                'question_text': 'Could you provide more details about your issue?',
                'options': [],
                'expected_gain': 0.0,
                'fallback': True
            }
        
        # Score each question by expected information gain
        scored = []
        for q in relevant:
            gain = self.expected_information_gain(q, current_probs)
            scored.append((q, gain))
        
        # Select the best question
        scored.sort(key=lambda x: x[1], reverse=True)
        best_q, best_gain = scored[0]
        
        return {
            'question_id': best_q['id'],
            'question_text': best_q['text'],
            'options': best_q.get('options', []),
            'expected_gain': round(best_gain, 4),
            'relevant_classes': best_q.get('relevant_classes', []),
            'fallback': False
        }

    def get_all_questions(self) -> List[Dict]:
        """Return all questions in the bank."""
        return self.bank

    def get_question_by_id(self, question_id: str) -> Optional[Dict]:
        """Retrieve a specific question by ID."""
        for q in self.bank:
            if q['id'] == question_id:
                return q
        return None

    def _default_bank(self) -> list:
        """Built-in fallback question bank if JSON file is missing."""
        return [
            {
                "id": "Q001",
                "text": "Is the main issue you need resolved today related to (A) a software error or unexpected behaviour, or (B) your account billing or invoice?",
                "options": ["Software error / unexpected behaviour", "Billing or invoice issue"],
                "relevant_classes": ["billing", "technical_support"],
                "posteriors": {
                    "technical": [0.85, 0.05, 0.03, 0.02, 0.01, 0.01, 0.02, 0.01],
                    "billing": [0.05, 0.82, 0.05, 0.02, 0.02, 0.01, 0.02, 0.01]
                }
            },
            {
                "id": "Q012",
                "text": "Are you reporting something that is broken and needs fixing, or requesting a new capability you would like added?",
                "options": ["Something is broken", "Requesting a new feature"],
                "relevant_classes": ["feature_request", "technical_support"],
                "posteriors": {
                    "technical": [0.82, 0.03, 0.03, 0.05, 0.02, 0.02, 0.02, 0.01],
                    "feature": [0.05, 0.03, 0.03, 0.82, 0.02, 0.02, 0.02, 0.01]
                }
            },
            {
                "id": "Q024",
                "text": "Are you looking to make changes to your subscription plan, or do you have concerns about continuing with the service?",
                "options": ["Changes to subscription", "Concerns about continuing"],
                "relevant_classes": ["churn_risk", "account_management"],
                "posteriors": {
                    "account": [0.03, 0.03, 0.82, 0.02, 0.02, 0.02, 0.03, 0.03],
                    "churn": [0.03, 0.03, 0.05, 0.02, 0.02, 0.02, 0.03, 0.80]
                }
            }
        ]


if __name__ == '__main__':
    engine = ClarificationEngine()
    
    # Simulate an ambiguous routing result
    current_probs = np.array([0.35, 0.30, 0.10, 0.08, 0.05, 0.04, 0.05, 0.03])
    top_two = ['billing', 'technical_support']
    
    result = engine.select_question(current_probs, top_two)
    print(f"Selected Question: {result['question_text']}")
    print(f"Expected Information Gain: {result['expected_gain']}")
    print(f"Options: {result['options']}")

