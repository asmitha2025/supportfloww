# tests/test_confidence_router.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from confidence_router import ConfidenceGatedRouter, CATEGORY_MAP, ROUTE_THRESHOLD, CLARIFY_THRESHOLD

def test_router_init():
    router = ConfidenceGatedRouter()
    assert router.model is not None
    assert router.tokenizer is not None

def test_mc_predict_returns_correct_shape():
    router = ConfidenceGatedRouter()
    conf, ent, cls, mean_p, std_p = router.mc_predict("Test ticket", n_passes=5)
    assert 0 <= conf <= 1
    assert ent >= 0
    assert cls in CATEGORY_MAP
    assert len(mean_p) == len(CATEGORY_MAP)
    assert len(std_p) == len(CATEGORY_MAP)
    assert abs(sum(mean_p) - 1.0) < 0.01

def test_route_returns_valid_action():
    router = ConfidenceGatedRouter()
    result = router.route("My invoice is wrong", n_passes=5)
    assert result['action'] in ('route', 'clarify', 'escalate')
    assert 'confidence' in result
    assert 'entropy' in result
    assert 'top_category' in result
    assert result['top_category'] in CATEGORY_MAP.values()

def test_route_has_all_fields():
    router = ConfidenceGatedRouter()
    result = router.route("API returns 500 error", n_passes=5)
    required = ['action', 'confidence', 'entropy', 'top_category', 'all_probs',
                'std_probs', 'category_ranking', 'top_two_classes', 'mc_passes', 'reason']
    for field in required:
        assert field in result, f"Missing field: {field}"

def test_batch_route():
    router = ConfidenceGatedRouter()
    results = router.batch_route(["Ticket 1", "Ticket 2"], n_passes=3)
    assert len(results) == 2

if __name__ == '__main__':
    test_router_init()
    print("✓ Router init")
    test_mc_predict_returns_correct_shape()
    print("✓ MC predict shape")
    test_route_returns_valid_action()
    print("✓ Route valid action")
    test_route_has_all_fields()
    print("✓ Route all fields")
    test_batch_route()
    print("✓ Batch route")
    print("\nAll tests passed! ✅")
