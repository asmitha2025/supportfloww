# Known Limitations

## Data
- Trained on synthetic tickets generated from templates
- Real-world accuracy will differ until fine-tuned on production data
- 400 total training samples (50 per class) — limited class diversity

## Model
- MC Dropout is a Bayesian approximation, not true Bayesian inference
- Thresholds (0.80 route, 0.55 clarify) are heuristic — need calibration per deployment context
- DistilBERT max_length=128 may truncate long enterprise tickets

## SLA Predictor
- `similar_ticket_avg_hrs` uses default historical fallback in API endpoint if not provided — needs real historical data feed in production

## Clarification Engine
- 47 templates cover common cases only
- No feedback loop to update posteriors from actual agent corrections
