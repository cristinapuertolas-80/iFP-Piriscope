"""Tests de la descarga de modelos de Hugging Face (sin red: la descarga es un doble)."""

import pytest

from src.ingest import config as ingest_config
from src.ml import inferir, modelos_hf

CFG = ingest_config.HuggingFaceConfig(repo_id="cpuertolas/piriscope", token="t")
V1 = ["v1_xgboost_con_fwi-sin_viento-desde2009", "v1_xgboost_sin_fwi-sin_viento-desde2009"]


def cache_falsa(tmp_path, ids, extra=()):
    """Simula la carpeta que devuelve snapshot_download con models/<id>.{joblib,json,_umbrales.json}."""
    llamadas = []

    def descargar(**kwargs):
        llamadas.append(kwargs)
        d = tmp_path / "cache" / "models"
        d.mkdir(parents=True, exist_ok=True)
        for i in ids:
            for sufijo in modelos_hf.SUFIJOS:
                (d / f"{i}{sufijo}").write_text("x")
        for nombre in extra:
            (d / nombre).write_text("x")
        return str(tmp_path / "cache")

    return descargar, llamadas


def test_descarga_los_tres_ficheros_de_cada_modelo(tmp_path):
    descargar, llamadas = cache_falsa(tmp_path, V1)
    destino = tmp_path / "models"
    copiados = modelos_hf.descargar_modelos("v1", destino, CFG, descargar)
    assert len(copiados) == 6
    assert llamadas[0]["allow_patterns"] == ["models/v1_*"]
    assert llamadas[0]["repo_id"] == "cpuertolas/piriscope" and llamadas[0]["repo_type"] == "dataset"
    # Lo que se descarga es exactamente lo que la inferencia sabe cargar
    assert inferir.listar_modelos(destino) == V1


def test_ignora_otros_ficheros_y_falla_si_no_hay_modelos_completos(tmp_path):
    descargar, _ = cache_falsa(tmp_path / "a", [], extra=["v1_suelto.joblib", "v1_nota.txt"])
    with pytest.raises(FileNotFoundError, match="v1"):
        modelos_hf.descargar_modelos("v1", tmp_path / "models", CFG, descargar)


def test_main_devuelve_2_con_mensaje_si_falla(monkeypatch, tmp_path):
    def falla(*a, **k):
        raise FileNotFoundError("sin modelos")

    monkeypatch.setattr(modelos_hf, "descargar_modelos", falla)
    assert modelos_hf.main(["--version", "v9", "--models-dir", str(tmp_path)]) == 2
