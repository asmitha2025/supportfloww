# Known Limitations

## Benchmark Validity Note

Current benchmarks are evaluated on synthetic validation data generated
from the same template distribution as training data. Accuracy figures
(100% on the synthetic test set) reflect **in-distribution performance only**
and will not generalize to real production tickets. Real-world performance
requires fine-tuning on production ticket data — this is a known limitation
and the system architecture is designed to support this with minimal retraining.

## Out-of-Distribution Evaluation Results

Evaluated on 96 hand-crafted, template-free tickets (informal language, typos, missing context, ambiguous edge-cases):

| Metric | In-Distribution | Out-of-Distribution |
|--------|:-:|:-:|
| Overall Routing Accuracy | 100.0% | **57.3%** |
| Precision on Auto-Routed | 100.0% | **100.0%** |
| Accuracy on Ambiguous Tickets | — | **30.0%** |

OOD routing gate distribution:
- **ROUTE** (auto-assigned): 2.1% of tickets — all correct
- **CLARIFY** (flagged): 51.0% — model correctly deferred
- **ESCALATE** (flagged): 46.9% — model correctly deferred

The system auto-routed only 2.1% of novel tickets (achieving 100% precision). It correctly flagged the remaining 97.9% as needing human review. This demonstrates the confidence gate works as intended — it fails safely.

## Data & Training Scale
- Trained on synthetic tickets generated from templates (400 samples / 50 per class for initial DistilBERT training; 4,000 synthetic samples for DeBERTa fine-tuning)
- **This is a proof-of-concept system.** The architecture is designed to be production-ready; the model weights are not.
- Real-world accuracy will differ until fine-tuned on production data
- Limited class diversity due to template-based generation

## Model
- MC Dropout is a Bayesian approximation, not true Bayesian inference
- Thresholds (0.80 route, 0.55 clarify) are heuristic — need calibration per deployment context
- DistilBERT max_length=128 may truncate long enterprise tickets

## SLA Predictor
- `similar_ticket_avg_hrs` uses a static default fallback (4.5 hrs) in the API endpoint when not supplied by the caller
- **Production requirement**: this field must be populated from a live historical data feed (e.g., a data warehouse query for similar resolved tickets) to produce meaningful SLA breach predictions
- Without a real data feed, SLA breach probabilities will be under-calibrated

## Clarification Engine
- 47 templates cover common cases only
- No feedback loop to update posteriors from actual agent corrections
