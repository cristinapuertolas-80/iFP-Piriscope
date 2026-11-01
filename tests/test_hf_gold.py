"""Tests de la publicación de Gold en Hugging Face: usan ficheros Parquet pequeños y un cliente simulado, sin red."""
from types import SimpleNamespace

import pandas as pd
import pytest

from src.ml import hf_hub


def _escribir(carpeta, pred=None, expl=None):
    if pred is None:
        pred = pd.DataFrame({
            "id_prediccion": ["15001_2026-10-19_v1", "15002_2026-10-19_v1"], "id_municipio": ["15001", "15002"],
            "fecha_objetivo": pd.to_datetime(["2026-10-19", "2026-10-19"]), "version_modelo": ["v1", "v1"],
            "probabilidad": [0.02, 0.05], "nivel_riesgo": ["bajo", "alto"]})
    if expl is None:
        expl = pd.DataFrame({"id_prediccion": [i for i in pred["id_prediccion"] for _ in range(2)],
                             "variable": ["ffmc", "fwi"] * len(pred), "valor_shap": [0.1, 0.2] * len(pred)})
    pred.to_parquet(carpeta / "predicciones.parquet", index=False)
    expl.to_parquet(carpeta / "explicaciones.parquet", index=False)
    return pred, expl


class ApiFalsa:
    def __init__(self, error=None):
        self.commits, self.error = [], error

    def create_commit(self, **kwargs):
        if self.error:
            raise self.error
        self.commits.append(kwargs)
        return SimpleNamespace(commit_url="https://huggingface.co/datasets/x/commit/abc")


def subir(carpeta, api, **extra):
    return hf_hub.subir_gold(carpeta, repo_id="usuario/dataset", repo_type="dataset", token="hf_escritura",
                             api=api, **extra)


def test_comprobar_gold_resume_el_par_de_ficheros(tmp_path):
    _escribir(tmp_path)
    r = hf_hub.comprobar_gold(tmp_path)
    assert r["n_predicciones"] == 2 and r["n_explicaciones"] == 4
    assert r["fecha_max"] == pd.Timestamp("2026-10-19") and r["dec50"] is False


def test_detecta_las_columnas_de_dec50(tmp_path):
    pred, expl = _escribir(tmp_path)
    expl = expl.assign(valor_variable=1.0, valor_base=-4.0, shap_resto=0.0)
    _escribir(tmp_path, pred, expl)
    assert hf_hub.comprobar_gold(tmp_path)["dec50"] is True


def test_sube_los_dos_ficheros_en_un_solo_commit(tmp_path):
    _escribir(tmp_path)
    api = ApiFalsa()
    url = subir(tmp_path, api)
    assert url.endswith("/commit/abc") and len(api.commits) == 1
    c = api.commits[0]
    assert c["repo_id"] == "usuario/dataset" and c["repo_type"] == "dataset"
    assert sorted(op.path_in_repo for op in c["operations"]) == ["data/gold/explicaciones.parquet",
                                                                 "data/gold/predicciones.parquet"]
    assert "2 predicciones" in c["commit_message"] and "2026-10-19" in c["commit_message"]


def test_dry_run_no_sube_nada_ni_exige_token(tmp_path, monkeypatch):
    _escribir(tmp_path)
    monkeypatch.setattr(hf_hub, "_configuracion", lambda: ("usuario/dataset", "dataset", ""))
    api = ApiFalsa()
    assert hf_hub.subir_gold(tmp_path, dry_run=True, api=api) is None
    assert api.commits == []


def test_sin_token_falla_con_un_mensaje_claro(tmp_path, monkeypatch):
    _escribir(tmp_path)
    monkeypatch.setattr(hf_hub, "_configuracion", lambda: ("usuario/dataset", "dataset", ""))
    with pytest.raises(ValueError, match="HF_TOKEN"):
        hf_hub.subir_gold(tmp_path, api=ApiFalsa())


def test_falta_un_fichero(tmp_path):
    _escribir(tmp_path)
    (tmp_path / "explicaciones.parquet").unlink()
    api = ApiFalsa()
    with pytest.raises(FileNotFoundError):
        subir(tmp_path, api)
    assert api.commits == []


def test_no_publica_explicaciones_que_no_coinciden_con_las_predicciones(tmp_path):
    pred, expl = _escribir(tmp_path)
    _escribir(tmp_path, pred, expl.iloc[:2])                 # a la segunda predicción le faltan explicaciones
    api = ApiFalsa()
    with pytest.raises(ValueError, match="no coinciden"):
        subir(tmp_path, api)
    assert api.commits == []


def test_no_publica_predicciones_duplicadas(tmp_path):
    pred, _ = _escribir(tmp_path)
    duplicada = pd.concat([pred, pred.assign(id_prediccion=pred["id_prediccion"] + "-respaldo",
                                             version_modelo="v1-respaldo")], ignore_index=True)
    _escribir(tmp_path, duplicada)
    with pytest.raises(ValueError, match="duplicadas"):
        subir(tmp_path, ApiFalsa())


def test_columnas_del_contrato_obligatorias(tmp_path):
    pred, expl = _escribir(tmp_path)
    _escribir(tmp_path, pred.drop(columns="nivel_riesgo"), expl)
    with pytest.raises(ValueError, match="nivel_riesgo"):
        hf_hub.comprobar_gold(tmp_path)


def test_un_acceso_denegado_se_explica(tmp_path):
    _escribir(tmp_path)
    with pytest.raises(RuntimeError, match="permiso de escritura"):
        subir(tmp_path, ApiFalsa(error=Exception("403 Forbidden")))
