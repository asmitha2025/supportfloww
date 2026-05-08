# src/evaluate_ood.py
# Honest Out-Of-Distribution (OOD) Evaluation for SupportMind
#
# Evaluates the ensemble router on hand-crafted, template-free tickets
# to produce realistic accuracy numbers for portfolio presentation.
#
# Run AFTER:  python data/generate_ood_test.py
# Usage:      python src/evaluate_ood.py
#
# Outputs:
#   results/ood_evaluation_report.json   - full JSON report
#   results/ood_confusion_matrix.csv     - per-category confusion
#   Console: side-by-side in-dist vs OOD summary table
#
# SupportMind - Asmitha

import os
import sys
import json
import time
import logging
import csv
from collections import defaultdict

os.environ['USE_TF'] = '0'
os.environ['USE_JAX'] = '0'
os.environ['USE_TORCH'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, 'data', 'processed')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

CATEGORIES = [
    'billing', 'technical_support', 'account_management', 'feature_request',
    'compliance_legal', 'onboarding', 'general_inquiry', 'churn_risk'
]
CATEGORY_MAP = {cat: i for i, cat in enumerate(CATEGORIES)}
LABEL_MAP    = {i: cat for cat, i in CATEGORY_MAP.items()}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_csv(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ── Core evaluation loop ───────────────────────────────────────────────────────

def run_evaluation(tickets, router, n_passes=20, label='eval'):
    """Run the router over a ticket list and return detailed results."""
    results = []
    action_counts  = defaultdict(int)
    correct_by_cat = defaultdict(lambda: {'correct': 0, 'total': 0})
    latencies = []

    logger.info(f"[{label}] Evaluating {len(tickets)} samples ({n_passes} MC passes each)...")

    for i, row in enumerate(tickets):
        text          = row['text']
        true_label    = int(row['label'])
        true_category = LABEL_MAP[true_label]
        ood_type      = row.get('ood_type', 'standard')

        t0 = time.time()
        result = router.route(text, n_passes=n_passes)
        elapsed_ms = (time.time() - t0) * 1000

        pred_category = result['top_category']
        action        = result['action']
        confidence    = result['confidence']
        entropy       = result['entropy']
        correct       = (pred_category == true_category)

        results.append({
            'text':          text[:120],
            'true_category': true_category,
            'pred_category': pred_category,
            'action':        action,
            'confidence':    round(confidence, 4),
            'entropy':       round(entropy, 4),
            'correct':       correct,
            'ood_type':      ood_type,
            'latency_ms':    round(elapsed_ms, 1),
        })

        action_counts[action] += 1
        correct_by_cat[true_category]['total']   += 1
        correct_by_cat[true_category]['correct'] += int(correct)
        latencies.append(elapsed_ms)

        if (i + 1) % 20 == 0:
            running_acc = sum(1 for r in results if r['correct']) / len(results)
            logger.info(f"  [{label}] {i+1}/{len(tickets)} - running accuracy: {running_acc:.1%}")

    total   = len(results)
    n_correct = sum(1 for r in results if r['correct'])
    overall_acc = n_correct / total if total else 0

    # Precision on auto-routed only
    routed = [r for r in results if r['action'] == 'route']
    prec_routed = sum(1 for r in routed if r['correct']) / len(routed) if routed else 0

    # Per-category accuracy
    per_cat = {}
    for cat in CATEGORIES:
        d = correct_by_cat[cat]
        per_cat[cat] = {
            'total':   d['total'],
            'correct': d['correct'],
            'accuracy': round(d['correct'] / d['total'], 4) if d['total'] else 0,
        }

    # Routing distribution
    routing_dist = {
        action: {
            'count':      action_counts[action],
            'percentage': round(action_counts[action] / total * 100, 1),
        }
        for action in ['route', 'clarify', 'escalate']
    }

    # Ambiguous-only accuracy (subset)
    ambig = [r for r in results if r.get('ood_type') == 'ambiguous']
    ambig_acc = sum(1 for r in ambig if r['correct']) / len(ambig) if ambig else None

    # Confusion matrix
    confusion = {tc: {pc: 0 for pc in CATEGORIES} for tc in CATEGORIES}
    for r in results:
        confusion[r['true_category']][r['pred_category']] += 1

    import statistics
    return {
        'summary': {
            'total_samples':        total,
            'overall_accuracy':     round(overall_acc, 4),
            'precision_auto_routed': round(prec_routed, 4),
            'ambiguous_accuracy':   round(ambig_acc, 4) if ambig_acc is not None else None,
            'n_ambiguous_samples':  len(ambig),
            'mean_latency_ms':      round(statistics.mean(latencies), 1),
            'p95_latency_ms':       round(sorted(latencies)[int(0.95 * len(latencies))], 1),
            'mc_passes':            n_passes,
        },
        'routing_distribution': routing_dist,
        'per_category_accuracy': per_cat,
        'confusion_matrix':      confusion,
        'raw_results':           results,
    }


# ── Confusion matrix CSV helper ────────────────────────────────────────────────

def save_confusion_csv(confusion, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['true \\ pred'] + CATEGORIES)
        for true_cat in CATEGORIES:
            row = [true_cat] + [confusion[true_cat][pc] for pc in CATEGORIES]
            writer.writerow(row)
    logger.info(f"Confusion matrix saved -> {path}")


# ── Pretty terminal report ─────────────────────────────────────────────────────

def print_comparison_report(in_dist_report, ood_report):
    """Print a side-by-side summary: in-distribution vs OOD."""
    s_ind = in_dist_report.get('summary', {})
    s_ood = ood_report['summary']

    # Pull last known in-dist numbers from saved report if available,
    # otherwise use placeholders that clearly indicate they're missing
    ind_acc  = s_ind.get('overall_accuracy',     '?')
    ind_prec = s_ind.get('precision_auto_routed','?')
    ood_acc  = s_ood['overall_accuracy']
    ood_prec = s_ood['precision_auto_routed']
    ood_amb  = s_ood['ambiguous_accuracy']

    def fmt(v):
        return f"{v:.1%}" if isinstance(v, float) else str(v)

    bar = "=" * 68
    print(f"\n{bar}")
    print(f"  SUPPORTMIND - IN-DISTRIBUTION vs OUT-OF-DISTRIBUTION BENCHMARK")
    print(f"{bar}")
    print(f"  {'Metric':<36} {'In-Dist (synthetic)':>16}  {'OOD (hand-crafted)':>14}")
    print(f"  {'-'*36} {'-'*16}  {'-'*14}")
    print(f"  {'Overall Routing Accuracy':<36} {fmt(ind_acc):>16}  {fmt(ood_acc):>14}")
    print(f"  {'Precision on Auto-Routed':<36} {fmt(ind_prec):>16}  {fmt(ood_prec):>14}")
    if ood_amb is not None:
        print(f"  {'Accuracy on Ambiguous Tickets':<36} {'---':>16}  {fmt(ood_amb):>14}")
    print(f"{bar}")
    print()

    rd = ood_report['routing_distribution']
    print(f"  OOD Routing Gate Distribution:")
    for action in ['route', 'clarify', 'escalate']:
        d = rd.get(action, {'count': 0, 'percentage': 0.0})
        print(f"    {action.upper():10s}  {d['count']:4d} tickets  ({d['percentage']:5.1f}%)")
    print()

    print(f"  OOD Per-Category Accuracy:")
    pc = ood_report['per_category_accuracy']
    for cat in CATEGORIES:
        d = pc.get(cat, {'total': 0, 'correct': 0, 'accuracy': 0})
        filled = int(d['accuracy'] * 20)
        bar_vis = '#' * filled + '.' * (20 - filled)
        print(f"    {cat:<25s}  [{bar_vis}]  {d['accuracy']:.0%}  ({d['correct']}/{d['total']})")
    print()

    print(f"  OOD Mean Latency : {s_ood['mean_latency_ms']:.0f}ms")
    print(f"  OOD P95 Latency  : {s_ood['p95_latency_ms']:.0f}ms")
    print(f"{bar}")
    print()
    print("  NOTE: In-distribution numbers are evaluated on synthetic val set")
    print("        generated from the SAME template distribution as training.")
    print("        OOD numbers are the honest estimate of generalisation ability.")
    print(f"{bar}\n")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Load OOD test set ──
    ood_path = os.path.join(DATA_DIR, 'ood_test.csv')
    if not os.path.exists(ood_path):
        logger.error(
            f"OOD test set not found at {ood_path}\n"
            "Run:  python data/generate_ood_test.py  first."
        )
        sys.exit(1)

    ood_tickets = load_csv(ood_path)
    logger.info(f"Loaded {len(ood_tickets)} OOD test samples")

    # ── Load router ──
    logger.info("Loading EnsembleRouter (CPU)...")
    from ensemble_router import EnsembleRouter
    router = EnsembleRouter(device='cpu')
    logger.info("Router ready.")

    # ── Run OOD evaluation ──
    ood_report = run_evaluation(ood_tickets, router, n_passes=20, label='OOD')

    # ── Load previous in-dist report for comparison (if exists) ──
    in_dist_path = os.path.join(RESULTS_DIR, 'evaluation_report.json')
    in_dist_summary = {}
    if os.path.exists(in_dist_path):
        with open(in_dist_path) as f:
            prev = json.load(f)
        in_dist_summary = prev.get('router', {})
        logger.info("Loaded previous in-distribution evaluation for comparison.")
    else:
        logger.warning(
            "No previous evaluation_report.json found. "
            "Run  python src/evaluate.py  to generate in-distribution numbers."
        )

    # ── Save OOD report ──
    ood_report_out = {k: v for k, v in ood_report.items() if k != 'raw_results'}
    report_path = os.path.join(RESULTS_DIR, 'ood_evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(ood_report_out, f, indent=2)
    logger.info(f"OOD report saved -> {report_path}")

    # Save raw predictions separately
    raw_path = os.path.join(RESULTS_DIR, 'ood_predictions.json')
    with open(raw_path, 'w') as f:
        json.dump(ood_report['raw_results'], f, indent=2)

    # Save confusion matrix CSV
    conf_path = os.path.join(RESULTS_DIR, 'ood_confusion_matrix.csv')
    save_confusion_csv(ood_report['confusion_matrix'], conf_path)

    # ── Print final comparison ──
    print_comparison_report(in_dist_summary, ood_report)

    # ── Write a machine-readable summary for README update ──
    summary_path = os.path.join(RESULTS_DIR, 'benchmark_summary.json')
    benchmark = {
        'note': (
            'in_dist numbers are from synthetic val set (same template distribution as train). '
            'ood numbers are from hand-crafted, template-free test set. '
            'OOD numbers are the honest measure of generalisation.'
        ),
        'in_distribution': {
            'overall_accuracy':     in_dist_summary.get('summary', {}).get('overall_accuracy'),
            'precision_auto_routed': in_dist_summary.get('summary', {}).get('precision_auto_routed'),
            'test_set':             'synthetic (same template distribution as training)',
            'n_samples':            in_dist_summary.get('summary', {}).get('total_samples'),
        },
        'ood': {
            'overall_accuracy':     ood_report['summary']['overall_accuracy'],
            'precision_auto_routed': ood_report['summary']['precision_auto_routed'],
            'ambiguous_accuracy':   ood_report['summary']['ambiguous_accuracy'],
            'test_set':             'hand-crafted, template-free (OOD)',
            'n_samples':            ood_report['summary']['total_samples'],
        },
    }
    with open(summary_path, 'w') as f:
        json.dump(benchmark, f, indent=2)
    logger.info(f"Benchmark summary -> {summary_path}")


if __name__ == '__main__':
    main()
