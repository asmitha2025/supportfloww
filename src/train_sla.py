# src/train_sla.py
# Train XGBoost model for SLA Breach Prediction

import os
import pandas as pd
import xgboost as xgb
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'sla_predictor')
MODEL_PATH = os.path.join(MODEL_DIR, 'sla_xgb.json')

FEATURE_NAMES = [
    'text_complexity_score', 'agent_queue_depth', 'customer_tier',
    'hour_of_day', 'day_of_week', 'similar_ticket_avg_hrs',
    'sentiment_score', 'repeat_issue', 'escalated_before'
]

def main():
    data_path = os.path.join(DATA_DIR, 'sla_train.csv')
    if not os.path.exists(data_path):
        logger.error(f"SLA training data not found at {data_path}. Run prepare_kaggle_data.py first.")
        return

    logger.info("Loading SLA training data...")
    df = pd.read_csv(data_path)
    
    X = df[FEATURE_NAMES]
    y = df['sla_breached']
    
    logger.info("Training XGBoost SLA Predictor...")
    dtrain = xgb.DMatrix(X, label=y, feature_names=FEATURE_NAMES)
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }
    
    model = xgb.train(params, dtrain, num_boost_round=100)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_model(MODEL_PATH)
    logger.info(f"SLA Model saved successfully to {MODEL_PATH}")

if __name__ == "__main__":
    main()
