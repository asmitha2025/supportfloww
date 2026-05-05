# train_baseline.py
# 2-minute TF-IDF + Logistic Regression baseline for SupportMind router.
# Saves a sklearn model that works as an immediate drop-in while
# DistilBERT fine-tuning runs overnight.
#
# Usage:  python train_baseline.py
#
# Output: models/ticket_classifier/sklearn_router.pkl  (sklearn pipeline)
#         models/ticket_classifier/baseline_meta.json  (metadata + accuracy)
#         models/ticket_classifier/config.json         (makes router think model exists)

import os, json, pickle, logging, time
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.calibration import CalibratedClassifierCV

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'ticket_classifier')

CATEGORY_MAP = {
    0: 'billing',
    1: 'technical_support',
    2: 'account_management',
    3: 'feature_request',
    4: 'compliance_legal',
    5: 'onboarding',
    6: 'general_inquiry',
    7: 'churn_risk',
}

def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("SupportMind — Baseline Router Training (TF-IDF + LR)")
    logger.info("=" * 60)

    # ── Load data ─────────────────────────────────────────────────
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    val_df   = pd.read_csv(os.path.join(DATA_DIR, 'val.csv'))
    test_df  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

    X_train, y_train = train_df['text'], train_df['label']
    X_val,   y_val   = val_df['text'],   val_df['label']
    X_test,  y_test  = test_df['text'],  test_df['label']

    logger.info(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    logger.info(f"Classes: {sorted(y_train.unique())}")

    # ── Build pipeline ────────────────────────────────────────────
    # TF-IDF captures n-gram patterns well for short support tickets
    # CalibratedClassifierCV ensures predict_proba is well-calibrated
    base_lr = LogisticRegression(
        max_iter=1000,
        C=2.0,
        class_weight='balanced',
        solver='lbfgs',
        multi_class='multinomial',
        random_state=42,
    )
    calibrated_lr = CalibratedClassifierCV(base_lr, cv=3, method='isotonic')

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 3),       # unigrams, bigrams, trigrams
            min_df=2,
            max_features=30_000,
            sublinear_tf=True,        # log(1+tf) damping
            strip_accents='unicode',
            analyzer='word',
        )),
        ('clf', calibrated_lr),
    ])

    # ── Train ─────────────────────────────────────────────────────
    logger.info("Training TF-IDF + Calibrated Logistic Regression...")
    pipeline.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────
    val_preds  = pipeline.predict(X_val)
    test_preds = pipeline.predict(X_test)
    val_acc    = accuracy_score(y_val,  val_preds)
    test_acc   = accuracy_score(y_test, test_preds)

    logger.info(f"Val  Accuracy : {val_acc:.4f}  ({val_acc*100:.1f}%)")
    logger.info(f"Test Accuracy : {test_acc:.4f}  ({test_acc*100:.1f}%)")
    logger.info("\nPer-class report (test):\n" + classification_report(
        y_test, test_preds,
        target_names=[CATEGORY_MAP[i] for i in range(8)]
    ))

    # ── Save ──────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. sklearn pipeline
    pkl_path = os.path.join(MODEL_DIR, 'sklearn_router.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(pipeline, f)
    logger.info(f"Saved sklearn pipeline → {pkl_path}")

    # 2. metadata
    meta = {
        'model_type': 'sklearn_tfidf_lr_baseline',
        'val_accuracy': round(val_acc, 4),
        'test_accuracy': round(test_acc, 4),
        'n_train': len(X_train),
        'category_map': CATEGORY_MAP,
        'trained_at': pd.Timestamp.now().isoformat(),
        'note': 'Baseline model — replace with DistilBERT once overnight training completes',
    }
    with open(os.path.join(MODEL_DIR, 'baseline_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # 3. config.json stub — makes confidence_router.py think a model exists
    #    (router falls back to distilbert-base-uncased if config.json is absent)
    #    We write a marker so the router knows to use sklearn path
    stub_config = {
        'model_type': 'baseline_sklearn',
        'num_labels': 8,
        'id2label': {str(k): v for k, v in CATEGORY_MAP.items()},
        'label2id': {v: str(k) for k, v in CATEGORY_MAP.items()},
    }
    with open(os.path.join(MODEL_DIR, 'config.json'), 'w') as f:
        json.dump(stub_config, f, indent=2)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"✅ Baseline training complete in {elapsed:.1f}s")
    logger.info(f"   Test accuracy: {test_acc*100:.1f}%")
    logger.info(f"   Model dir    : {MODEL_DIR}")
    logger.info("=" * 60)
    logger.info("Next: run keep_alive.ps1 overnight for full DistilBERT model.")

if __name__ == '__main__':
    main()
