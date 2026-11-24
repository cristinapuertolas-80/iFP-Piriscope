"""`hf_hub._comprobar_monitorizacion`: la monitorización solo se publica si es válida y actual."""
import json

import pandas as pd

from src.ml import hf_hub


def _escribe(tmp_path, ultima="2026-11-16"):
    (tmp_path / hf_hub.FICHERO_MONITORIZACION).write_text(
        json.dumps({"version_esquema": 1, "predicciones": {"ultima_fecha_objetivo": ultima}}), encoding="utf-8")


def test_no_existe_devuelve_none(tmp_path):
    assert hf_hub._comprobar_monitorizacion(tmp_path, pd.Timestamp("2026-11-16")) is None


def test_valido_y_actual(tmp_path):
    _escribe(tmp_path)
    assert hf_hub._comprobar_monitorizacion(tmp_path, pd.Timestamp("2026-11-16")) == tmp_path / "monitorizacion.json"


def test_de_otra_ejecucion_se_omite(tmp_path):
    _escribe(tmp_path, ultima="2026-11-10")
    assert hf_hub._comprobar_monitorizacion(tmp_path, pd.Timestamp("2026-11-16")) is None


def test_roto_o_sin_claves_se_omite_sin_lanzar(tmp_path):
    (tmp_path / "monitorizacion.json").write_text("{no es json", encoding="utf-8")
    assert hf_hub._comprobar_monitorizacion(tmp_path, pd.Timestamp("2026-11-16")) is None
    (tmp_path / "monitorizacion.json").write_text(json.dumps({"version_esquema": 1}), encoding="utf-8")
    assert hf_hub._comprobar_monitorizacion(tmp_path, pd.Timestamp("2026-11-16")) is None
