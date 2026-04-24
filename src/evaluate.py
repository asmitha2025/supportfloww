# src/evaluate.py
# Evaluate SupportMind pipeline on validation set
# Produces comprehensive metrics for the results/ directory
# SupportMind v1.0 — Asmitha

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from collections import defaultdict

# Disable TF/JAX
os.environ['USE_TF'] = '0'
os.environ['USE_JAX'] = '0'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'ticket_classifier')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


def evaluate_router(val_df, n_passes=20):
    """Evaluate the confidence-gated router on validation data."""
    from confidence_router import ConfidenceGatedRouter, CATEGORY_MAP

    model_path = MODEL_DIR if os.path.exists(os.path.join(MODEL_DIR, 'config.json')) else None
    router = ConfidenceGatedRouter(model_path)

    results = []
    action_counts = defaultdict(int)
    correct_by_action = defaultdict(int)
    total_by_action = defaultdict(int)
    confidences = []
    entropies = []
    latencies = []

    logger.info(f"Evaluating {len(val_df)} samples with {n_passes} MC passes each...")

    for i, row in val_df.iterrows():
        text = row['text']
        true_label = int(row['label'])
        true_category = CATEGORY_MAP[true_label]

        start = time.time()
        result = router.route(text, n_passes=n_passes)
        elapsed_ms = (time.time() - start) * 1000

        pred_category = result['top_category']
        action = result['action']
        confidence = result['confidence']
        entropy = result['entropy']

        correct = pred_category == true_category

        results.append({
            'true_label': true_label,
            'true_category': true_category,
            'pred_category': pred_category,
            'action': action,
            'confidence': confidence,
            'entropy': entropy,
            'correct': correct,
            'latency_ms': round(elapsed_ms, 1),
        })

        action_counts[action] += 1
        total_by_action[action] += 1
        if correct:
            correct_by_action[action] += 1
        confidences.append(confidence)
        entropies.append(entropy)
        latencies.append(elapsed_ms)

        if (i + 1) % 50 == 0:
            logger.info(f"  Evaluated {i+1}/{len(val_df)} samples...")

    # ── Compute aggregate metrics ──
    total = len(results)
    correct_total = sum(1 for r in results if r['correct'])
    overall_accuracy = correct_total / total if total > 0 else 0

    # Accuracy by action
    accuracy_by_action = {}
    for action in ['route', 'clarify', 'escalate']:
        t = total_by_action.get(action, 0)
        c = correct_by_action.get(action, 0)
        accuracy_by_action[action] = {
            'count': t,
            'correct': c,
            'accuracy': round(c / t, 4) if t > 0 else 0,
            'percentage': round(t / total * 100, 1) if total > 0 else 0,
        }

    # Precision on auto-routed tickets (the key metric)
    routed = [r for r in results if r['action'] == 'route']
    precision_routed = sum(1 for r in routed if r['correct']) / len(routed) if routed else 0

    # Confusion matrix (category-level)
    categories = list(CATEGORY_MAP.values())
    confusion = {true_cat: {pred_cat: 0 for pred_cat in categories} for true_cat in categories}
    for r in results:
        confusion[r['true_category']][r['pred_category']] += 1

    # Per-category accuracy
    per_category = {}
    for cat in categories:
        cat_results = [r for r in results if r['true_category'] == cat]
        cat_correct = sum(1 for r in cat_results if r['correct'])
        per_category[cat] = {
            'total': len(cat_results),
            'correct': cat_correct,
            'accuracy': round(cat_correct / len(cat_results), 4) if cat_results else 0,
        }

    # Confidence calibration (binned)
    conf_bins = np.linspace(0, 1, 11)
    calibration = []
    for i in range(len(conf_bins) - 1):
        low, high = conf_bins[i], conf_bins[i+1]
        bin_results = [r for r in results if low <= r['confidence'] < high]
        if bin_results:
            bin_acc = sum(1 for r in bin_results if r['correct']) / len(bin_results)
            bin_conf = np.mean([r['confidence'] for r in bin_results])
            calibration.append({
                'bin': f"{low:.1f}-{high:.1f}",
                'count': len(bin_results),
                'accuracy': round(bin_acc, 4),
                'mean_confidence': round(bin_conf, 4),
            })

    report = {
        'summary': {
            'total_samples': total,
            'overall_accuracy': round(overall_accuracy, 4),
            'precision_auto_routed': round(precision_routed, 4),
            'mean_confidence': round(np.mean(confidences), 4),
            'mean_entropy': round(np.mean(entropies), 4),
            'mean_latency_ms': round(np.mean(latencies), 1),
            'p95_latency_ms': round(np.percentile(latencies, 95), 1),
            'mc_passes': n_passes,
        },
        'routing_distribution': {
            action: {
                'count': data['count'],
                'percentage': data['percentage'],
                'accuracy': data['accuracy'],
            }
            for action, data in accuracy_by_action.items()
        },
        'per_category_accuracy': per_category,
        'confidence_calibration': calibration,
        'confusion_matrix': confusion,
    }

    return report, results


def evaluate_sla():
    """Evaluate SLA breach predictor."""
    from sla_predictor import SLABreachPredictor

    sla_path = os.path.join(BASE_DIR, 'models', 'sla_predictor', 'sla_xgb.json')
    predictor = SLABreachPredictor(sla_path)

    # Test scenarios
    scenarios = [
        {'name': 'Low Risk', 'features': {
            'text_complexity_score': 5.0, 'agent_queue_depth': 3, 'customer_tier': 1,
            'hour_of_day': 10, 'day_of_week': 1, 'similar_ticket_avg_hrs': 1.5,
            'sentiment_score': 0.8, 'repeat_issue': 0, 'escalated_before': 0}},
        {'name': 'Medium Risk', 'features': {
            'text_complexity_score': 10.0, 'agent_queue_depth': 15, 'customer_tier': 3,
            'hour_of_day': 14, 'day_of_week': 2, 'similar_ticket_avg_hrs': 4.5,
            'sentiment_score': -0.3, 'repeat_issue': 0, 'escalated_before': 0}},
        {'name': 'High Risk', 'features': {
            'text_complexity_score': 16.0, 'agent_queue_depth': 30, 'customer_tier': 4,
            'hour_of_day': 23, 'day_of_week': 6, 'similar_ticket_avg_hrs': 12.0,
            'sentiment_score': -0.9, 'repeat_issue': 1, 'escalated_before': 1}},
    ]

    sla_results = []
    for scenario in scenarios:
        result = predictor.explain(scenario['features'])
        sla_results.append({
            'scenario': scenario['name'],
            'breach_probability': result['breach_probability'],
            'risk_level': result['risk_level'],
            'factors': result['contributing_factors'],
        })
        logger.info(f"  SLA {scenario['name']}: prob={result['breach_probability']:.3f}, risk={result['risk_level']}")

    # Verify monotonicity (high risk > medium > low)
    probs = [r['breach_probability'] for r in sla_results]
    monotonic = probs[0] < probs[1] < probs[2]

    return {
        'scenarios': sla_results,
        'monotonicity_check': monotonic,
        'model_type': 'XGBoost',
    }


def evaluate_clarification():
    """Evaluate clarification engine."""
    from clarification_engine import ClarificationEngine

    bank_path = os.path.join(BASE_DIR, 'data', 'clarification_bank.json')
    engine = ClarificationEngine(bank_path)

    # Test with different ambiguity profiles
    test_cases = [
        {'probs': [0.35, 0.30, 0.10, 0.08, 0.05, 0.04, 0.05, 0.03],
         'top_two': ['billing', 'technical_support'], 'label': 'billing_vs_tech'},
        {'probs': [0.25, 0.10, 0.30, 0.08, 0.05, 0.04, 0.15, 0.03],
         'top_two': ['account_management', 'billing'], 'label': 'account_vs_billing'},
        {'probs': [0.10, 0.35, 0.05, 0.30, 0.05, 0.05, 0.05, 0.05],
         'top_two': ['technical_support', 'feature_request'], 'label': 'tech_vs_feature'},
    ]

    clar_results = []
    for tc in test_cases:
        probs = np.array(tc['probs'])
        result = engine.select_question(probs, tc['top_two'])
        clar_results.append({
            'scenario': tc['label'],
            'question_id': result['question_id'],
            'question_text': result['question_text'],
            'expected_gain': result['expected_gain'],
            'fallback': result.get('fallback', False),
        })
        logger.info(f"  Clarification [{tc['label']}]: gain={result['expected_gain']:.4f}")

    return {
        'total_templates': len(engine.bank),
        'test_results': clar_results,
        'all_gains_positive': all(r['expected_gain'] > 0 for r in clar_results),
    }


def evaluate_churn():
    """Evaluate churn signal extractor."""
    from churn_extractor import ChurnSignalExtractor

    extractor = ChurnSignalExtractor()

    test_threads = [
        {'label': 'No Risk', 'thread': [
            "Hi, I need help setting up the webhook integration.",
            "Thanks for the quick response! That worked perfectly.",
        ]},
        {'label': 'Medium Risk', 'thread': [
            "The export feature has been broken for two weeks.",
            "This is the second time I've reported this issue.",
            "I'm quite frustrated with the response time.",
        ]},
        {'label': 'Critical Risk', 'thread': [
            "We've been having issues with the API for three weeks now.",
            "This is the third time I'm reporting this. Still not fixed.",
            "I'm very frustrated. We're looking at switching to a competitor.",
            "If this isn't resolved by Friday, we'll cancel our subscription.",
        ]},
    ]

    churn_results = []
    for tc in test_threads:
        result = extractor.extract(tc['thread'])
        churn_results.append({
            'scenario': tc['label'],
            'churn_risk_score': result['churn_risk_score'],
            'risk_level': result['risk_level'],
            'competitor_mention': result['competitor_mention'],
            'cancellation_language': result['cancellation_language'],
            'recommendation': result['recommendation'],
        })
        logger.info(f"  Churn [{tc['label']}]: score={result['churn_risk_score']:.3f}, level={result['risk_level']}")

    # Verify risk ordering
    scores = [r['churn_risk_score'] for r in churn_results]
    monotonic = scores[0] < scores[1] < scores[2]

    return {
        'scenarios': churn_results,
        'monotonicity_check': monotonic,
    }


def evaluate_features():
    """Evaluate feature extraction pipeline."""
    from feature_extraction import FeatureExtractor

    extractor = FeatureExtractor()

    test_texts = [
        "My invoice from last month shows $299 but my plan is $199.",
        "The API endpoint /v2/export returns a 500 error when batch size exceeds 1000. URGENT!",
        "Hey, quick question about the dashboard analytics feature.",
    ]

    feat_results = []
    for text in test_texts:
        features = extractor.extract(text)
        feat_results.append({
            'text_preview': text[:60] + '...',
            'sentiment_score': features['sentiment_score'],
            'urgency_flags': features['urgency_flags'],
            'product_entities': features['product_entities'],
            'text_complexity': features['text_complexity_score'],
            'token_count': features['token_count'],
        })

    return {'test_results': feat_results}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    logger.info("=" * 70)
    logger.info("SupportMind — Comprehensive Evaluation")
    logger.info("=" * 70)

    full_report = {}

    # 1. Router evaluation (the big one)
    logger.info("\n[1/5] Evaluating Confidence-Gated Router...")
    val_path = os.path.join(DATA_DIR, 'val.csv')
    if os.path.exists(val_path):
        val_df = pd.read_csv(val_path)
        # Use a subset for faster evaluation (100 samples × 20 MC passes)
        eval_subset = val_df.sample(n=min(100, len(val_df)), random_state=42)
        router_report, raw_results = evaluate_router(eval_subset, n_passes=20)
        full_report['router'] = router_report

        # Save raw predictions
        raw_path = os.path.join(RESULTS_DIR, 'router_predictions.json')
        with open(raw_path, 'w') as f:
            json.dump(raw_results, f, indent=2)
        logger.info(f"  Raw predictions saved to {raw_path}")
    else:
        logger.warning("  Validation data not found, skipping router evaluation")

    # 2. SLA evaluation
    logger.info("\n[2/5] Evaluating SLA Breach Predictor...")
    full_report['sla'] = evaluate_sla()

    # 3. Clarification evaluation
    logger.info("\n[3/5] Evaluating Clarification Engine...")
    full_report['clarification'] = evaluate_clarification()

    # 4. Churn evaluation
    logger.info("\n[4/5] Evaluating Churn Signal Extractor...")
    full_report['churn'] = evaluate_churn()

    # 5. Feature extraction evaluation
    logger.info("\n[5/5] Evaluating Feature Extraction Pipeline...")
    full_report['features'] = evaluate_features()

    # ── Save full report ──
    report_path = os.path.join(RESULTS_DIR, 'evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(full_report, f, indent=2)
    logger.info(f"\n{'='*70}")
    logger.info(f"Full evaluation report saved to: {report_path}")
    logger.info(f"{'='*70}")

    # ── Print summary ──
    if 'router' in full_report:
        s = full_report['router']['summary']
        rd = full_report['router']['routing_distribution']
        print(f"\n{'='*60}")
        print(f"  SUPPORTMIND EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"  Overall Accuracy:       {s['overall_accuracy']:.1%}")
        print(f"  Precision (Auto-Routed): {s['precision_auto_routed']:.1%}")
        print(f"  Mean Confidence:        {s['mean_confidence']:.4f}")
        print(f"  Mean Entropy:           {s['mean_entropy']:.4f}")
        print(f"  Mean Latency:           {s['mean_latency_ms']:.0f}ms")
        print(f"  P95 Latency:            {s['p95_latency_ms']:.0f}ms")
        print(f"\n  Routing Distribution:")
        for action in ['route', 'clarify', 'escalate']:
            if action in rd:
                d = rd[action]
                print(f"    {action.upper():10s}: {d['count']:4d} ({d['percentage']:5.1f}%) — acc {d['accuracy']:.1%}")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

