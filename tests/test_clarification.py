# tests/test_clarification.py
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from clarification_engine import ClarificationEngine

def test_engine_loads():
    bank_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'clarification_bank.json')
    engine = ClarificationEngine(bank_path)
    assert len(engine.bank) == 47

def test_select_question():
    bank_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'clarification_bank.json')
    engine = ClarificationEngine(bank_path)
    probs = np.array([0.35, 0.30, 0.10, 0.08, 0.05, 0.04, 0.05, 0.03])
    result = engine.select_question(probs, ['billing', 'technical_support'])
    assert 'question_text' in result
    assert 'expected_gain' in result
    assert result['expected_gain'] > 0

def test_info_gain_positive():
    bank_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'clarification_bank.json')
    engine = ClarificationEngine(bank_path)
    probs = np.array([0.25, 0.25, 0.125, 0.125, 0.0625, 0.0625, 0.0625, 0.0625])
    result = engine.select_question(probs, ['billing', 'technical_support'])
    assert result['expected_gain'] >= 0

def test_no_duplicate_questions():
    bank_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'clarification_bank.json')
    engine = ClarificationEngine(bank_path)
    probs = np.array([0.35, 0.30, 0.10, 0.08, 0.05, 0.04, 0.05, 0.03])
    r1 = engine.select_question(probs, ['billing', 'technical_support'])
    r2 = engine.select_question(probs, ['billing', 'technical_support'], asked_ids=[r1['question_id']])
    assert r1['question_id'] != r2['question_id']

if __name__ == '__main__':
    test_engine_loads()
    print("✓ Engine loads 47 questions")
    test_select_question()
    print("✓ Select question works")
    test_info_gain_positive()
    print("✓ Info gain positive")
    test_no_duplicate_questions()
    print("✓ No duplicate questions")
    print("\nAll tests passed! ✅")
