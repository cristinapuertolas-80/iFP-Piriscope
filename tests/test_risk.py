import numpy as np
import pytest

from src.ml import risk


def _data(seed=0, n=5000):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.1).astype(int)
    return y, y * 0.4 + rng.random(n) * 0.6


def test_umbrales_ordenados():
    y, p = _data()
    t = risk.fit_risk_thresholds(y, p, "m")
    assert t.medio <= t.alto


def test_recall_objetivo_en_validacion():
    y, p = _data()
    t = risk.fit_risk_thresholds(y, p, "m", recall_medio=0.8, recall_alto=0.5)
    assert (p[y == 1] >= t.medio).mean() >= 0.8
    assert (p[y == 1] >= t.alto).mean() >= 0.5


def test_asignacion_niveles():
    t = risk.RiskThresholds(medio=0.3, alto=0.6, recall_medio=0.8, recall_alto=0.5,
                            model_id="m", n_val=10, n_pos_val=2)
    assert list(risk.assign_levels([0.1, 0.3, 0.59, 0.6, 0.9], t)) == \
        ["bajo", "medio", "medio", "alto", "alto"]


def test_recalls_invalidos():
    y, p = _data()
    with pytest.raises(ValueError):
        risk.fit_risk_thresholds(y, p, "m", recall_medio=0.5, recall_alto=0.8)


def test_guardar_y_cargar(tmp_path):
    y, p = _data()
    t = risk.fit_risk_thresholds(y, p, "v1_xgboost_con_fwi")
    path = risk.save_thresholds(t, tmp_path)
    assert path.name == "v1_xgboost_con_fwi_umbrales.json"
    assert risk.load_thresholds(path) == t
