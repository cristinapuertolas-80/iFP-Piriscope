"""Lectura de `monitorizacion.json` en el dashboard y tablas de la pestaña «Estado del sistema»."""
import json

import pytest

from src.dashboard import datos


def _mon(version=1):
    return {
        "version_esquema": version,
        "predicciones": {"ventana": {"por_fecha": [
            {"fecha": "2026-11-15", "cobertura_pct": 92.0, "pct_respaldo": 100.0, "pct_medio_alto": 0.0,
             "niveles": {"bajo": 288, "medio": 0, "alto": 0}},
            {"fecha": "2026-11-16", "cobertura_pct": 90.0, "pct_respaldo": 100.0, "pct_medio_alto": 1.0,
             "niveles": {"bajo": 280, "medio": 2, "alto": 0}}]}},
        "deriva": {"variables": [
            {"variable": "temperatura", "psi": 0.05, "estado": "estable", "n_reciente": 300,
             "nulos_pct_reciente": 0.0, "nulos_pct_referencia": 17.0},
            {"variable": "fwi", "psi": None, "estado": "sin_datos", "n_reciente": 0,
             "nulos_pct_reciente": 100.0, "nulos_pct_referencia": 3.0}]},
    }


@pytest.fixture(autouse=True)
def _solo_local(monkeypatch):
    monkeypatch.setenv("PIRISCOPE_DASHBOARD_ORIGEN", "local")   # los tests no usan la red


def test_lee_el_fichero_local(tmp_path):
    (tmp_path / "monitorizacion.json").write_text(json.dumps(_mon()), encoding="utf-8")
    assert datos.leer_monitorizacion(gold_dir=tmp_path)["version_esquema"] == 1


def test_sin_fichero_roto_o_otra_version_devuelve_none(tmp_path):
    assert datos.leer_monitorizacion(gold_dir=tmp_path) is None
    (tmp_path / "monitorizacion.json").write_text("{roto", encoding="utf-8")
    assert datos.leer_monitorizacion(gold_dir=tmp_path) is None
    (tmp_path / "monitorizacion.json").write_text(json.dumps(_mon(version=99)), encoding="utf-8")
    assert datos.leer_monitorizacion(gold_dir=tmp_path) is None


def test_si_falla_hugging_face_usa_el_local(tmp_path, monkeypatch):
    monkeypatch.setenv("PIRISCOPE_DASHBOARD_ORIGEN", "hf")
    monkeypatch.setattr(datos, "_token", lambda: (_ for _ in ()).throw(RuntimeError("sin red")))
    (tmp_path / "monitorizacion.json").write_text(json.dumps(_mon()), encoding="utf-8")
    assert datos.leer_monitorizacion(gold_dir=tmp_path) is not None


def test_tablas():
    v = datos.tabla_ventana(_mon())
    assert list(v.columns) == ["fecha", "cobertura_pct", "pct_respaldo", "pct_medio_alto", "bajo", "medio", "alto"]
    assert len(v) == 2 and v["medio"].tolist() == [0, 2]
    d = datos.tabla_deriva(_mon())
    assert d["estado"].tolist() == ["estable", "sin_datos"] and d["psi"].isna().tolist() == [False, True]
