import sys, os
sys.path.insert(0, 'src')
os.environ['USE_TF'] = '0'
os.environ['USE_JAX'] = '0'

import traceback

print("Step 1: importing ConfidenceGatedRouter...")
try:
    from confidence_router import ConfidenceGatedRouter
    print("Step 2: import OK, loading model...")
    router = ConfidenceGatedRouter('models/ticket_classifier')
    print("Step 3: model loaded! Testing route...")
    result = router.route('My invoice is wrong and I was double charged')
    print("Step 4: route OK!")
    print(f"  Action: {result['action']}")
    print(f"  Category: {result['top_category']}")
    print(f"  Confidence: {result['confidence']}")
except Exception as e:
    print(f"FAILED: {e}")
    traceback.print_exc()

print("\nStep 5: testing EnsembleRouter...")
try:
    from ensemble_router import EnsembleRouter
    er = EnsembleRouter()
    print(f"  BERT online: {er._bert_available}")
    r2 = er.route('My invoice is wrong and I was double charged')
    print(f"  Action: {r2['action']} | Category: {r2['top_category']} | Conf: {r2['confidence']}")
except Exception as e:
    print(f"Ensemble FAILED: {e}")
    traceback.print_exc()
