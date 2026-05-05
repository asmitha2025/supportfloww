"""
Quick accuracy eval:
- Tests sklearn (TF-IDF+LR) accuracy on test set  
- Tests DistilBERT on 50 samples (due to RAM)
- Reports both
"""
import sys, os, gc
sys.path.insert(0, 'src')
os.environ['USE_TF'] = '0'
os.environ['USE_JAX'] = '0'

import pickle, pandas as pd, numpy as np

cats = {0:'billing',1:'technical_support',2:'account_management',3:'feature_request',
        4:'compliance_legal',5:'onboarding',6:'general_inquiry',7:'churn_risk'}

df = pd.read_csv('data/processed/test.csv')

# ── 1. sklearn accuracy (fast, 600 samples) ──────────────────────────────────
print("="*50)
print("1. TF-IDF + LogReg baseline (all 600 test samples)")
with open('models/ticket_classifier/sklearn_router.pkl', 'rb') as f:
    pipe = pickle.load(f)

preds = pipe.predict(df['text'])
truths = df['label'].values
acc_sk = (preds == truths).mean()
print(f"   Accuracy: {acc_sk*100:.2f}%")

per_class_sk = {}
for lbl, name in cats.items():
    mask = truths == lbl
    if mask.sum() > 0:
        per_class_sk[name] = (preds[mask] == truths[mask]).mean()
for cls, acc in sorted(per_class_sk.items()):
    bar = '#' * int(acc*20)
    print(f"   {cls:22s}: {acc*100:5.1f}%  {bar}")

# ── 2. DistilBERT on 50 samples (memory-safe) ────────────────────────────────
print()
print("="*50)
print("2. DistilBERT v2 (50 sample spot-check, n_passes=3)")
try:
    from confidence_router import ConfidenceGatedRouter
    router = ConfidenceGatedRouter('models/ticket_classifier')
    
    sample = df.sample(50, random_state=42).reset_index(drop=True)
    correct = 0
    for _, row in sample.iterrows():
        r = router.route(row['text'], n_passes=3)
        if r['top_category'] == cats[row['label']]:
            correct += 1
    bert_acc = correct / 50
    print(f"   Accuracy: {bert_acc*100:.1f}% on 50 samples")
    gc.collect()
except Exception as e:
    print(f"   BERT eval failed: {e}")

# ── 3. Ensemble on 50 samples ─────────────────────────────────────────────────
print()
print("="*50)
print("3. Ensemble (DistilBERT + TF-IDF, 50 samples, n_passes=3)")
try:
    from ensemble_router import EnsembleRouter
    er = EnsembleRouter()
    print(f"   BERT online: {er._bert_available}")
    
    sample = df.sample(50, random_state=42).reset_index(drop=True)
    correct = 0
    for _, row in sample.iterrows():
        r = er.route(row['text'], n_passes=3)
        if r['top_category'] == cats[row['label']]:
            correct += 1
    print(f"   Accuracy: {correct/50*100:.1f}% on 50 samples")
except Exception as e:
    print(f"   Ensemble eval failed: {e}")
    import traceback; traceback.print_exc()

print()
print("="*50)
print("SUMMARY")
print(f"  sklearn (600 samples): {acc_sk*100:.2f}%")
print(f"  DistilBERT v2 spot:    see above")
print(f"  Ensemble spot:         see above")
print("="*50)
