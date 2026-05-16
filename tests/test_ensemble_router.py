import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import ensemble_router as er


def test_embedded_sklearn_fallback_when_model_file_missing(monkeypatch, tmp_path):
    real_exists = er.os.path.exists

    def fake_exists(path):
        path = str(path)
        if path.endswith('sklearn_router.pkl'):
            return False
        if path.endswith('config.json') or path.endswith('model.safetensors') or path.endswith('pytorch_model.bin'):
            return False
        return real_exists(path)

    monkeypatch.setattr(er.os.path, 'exists', fake_exists)

    router = er.EnsembleRouter(model_dir=str(tmp_path), device='cpu')
    result = router.route("The invoice charge is wrong and I need a refund.")

    assert router.model_status["mode"] == "sklearn_fallback"
    assert router.model_status["sklearn_source"] == "embedded_synthetic"
    assert result["top_category"] in er.CATEGORY_MAP.values()
    assert result["action"] in ("route", "clarify", "escalate")
