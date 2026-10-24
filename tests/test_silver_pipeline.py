"""Test de extremo a extremo de Silver con un Bronze sintético."""

import json
from datetime import date

import pytest

from src.silver import almacen, pipeline
from src.silver.__main__ import main

from .silver_fixtures import CENTRO, escribir_aemet, escribir_egif, escribir_fwi, escribir_ine_cnig, parte
from .test_silver_meteo import ESTACIONES, _reg


def _bronze(raiz):
    escribir_ine_cnig(raiz / "ine_cnig" / "20260101T000000Z")
    escribir_aemet(raiz / "aemet" / "20260101T000000Z",
                   {"0001_2024-07-01_2024-07-02.json": [_reg("2024-07-01", tmed="20,0", prec="0,0", hrMedia="150"),
                                                        _reg("2024-07-02", tmed="22,0", prec="Ip")]},
                   ESTACIONES)
    escribir_fwi(raiz / "effis" / "20260101T000000Z" / "fwi_2024_07.nc", ["2024-07-01", "2024-07-02"], 7.0)
    escribir_egif(raiz / "egif" / "20260101T000000Z",
                  [parte("2024_15_0001", "ABEGONDO", "02/07/2024 15:00:00", *CENTRO["15001"]),
                   parte("2024_15_0002", "PORTUGAL", "02/07/2024 18:00:00")])


def test_pipeline_completo(tmp_path):
    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _bronze(bronze)
    pipeline.ejecutar(base_dir=silver, bronze_dir=bronze, hf_dir=tmp_path / "nada")

    municipio = almacen.leer("municipio", silver)
    meteo = almacen.leer("observacion_meteorologica", silver)
    fwi = almacen.leer("indicador_fwi", silver)
    eventos = almacen.leer("evento_incendio", silver)
    assert len(municipio) == 2
    assert len(meteo) == 4 and meteo["id_obs_meteo"].is_unique
    assert set(meteo["id_estacion_aemet"]) == {"0001"}
    assert meteo["humedad_relativa"].isna().all()  # 150 % está fuera de rango -> nulo
    assert len(fwi) == 4 and (fwi["fwi"] == 7.0).all()
    assert eventos["id_municipio"].isna().tolist() == [False, True]
    for tabla in (municipio, meteo, fwi, eventos):
        assert "fecha_insert" in tabla.columns

    informes = silver / "informes"
    resumen = json.loads((informes / "informe_calidad.json").read_text(encoding="utf-8"))
    assert resumen["evento_incendio"]["pct_con_codigo_ine"] == 50.0
    assert resumen["observacion_meteorologica"]["valores_fuera_rango"] == 1
    for nombre in ("valores_fuera_rango.csv", "asignacion_estaciones.csv", "incendios_no_asignados.csv",
                   "incendios_muestra_validacion.csv", "cobertura_meteo_municipios.csv"):
        assert (informes / nombre).exists()


def test_reproceso_parcial_no_duplica_ni_borra(tmp_path, monkeypatch):
    import pandas as pd

    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _bronze(bronze)
    monkeypatch.setattr(almacen, "ahora_utc", lambda: pd.Timestamp("2026-01-01", tz="UTC"))
    pipeline.ejecutar(base_dir=silver, bronze_dir=bronze, hf_dir=tmp_path / "nada")
    monkeypatch.setattr(almacen, "ahora_utc", lambda: pd.Timestamp("2026-01-02", tz="UTC"))
    pipeline.ejecutar(date(2024, 7, 2), date(2024, 7, 2), ["meteo", "fwi", "incendios"],
                      base_dir=silver, bronze_dir=bronze, hf_dir=tmp_path / "nada")
    meteo = almacen.leer("observacion_meteorologica", silver)
    assert len(meteo) == 4 and meteo["id_obs_meteo"].is_unique
    assert meteo["fecha_insert"].nunique() == 2
    assert len(almacen.leer("evento_incendio", silver)) == 2


def test_cli_error_si_no_hay_bronze(tmp_path, monkeypatch):
    monkeypatch.setattr("src.silver.fuentes.HF_RAW_DIR", tmp_path / "nada")
    monkeypatch.setattr("src.ingest.config.BRONZE_DIR", tmp_path / "nada")
    monkeypatch.setattr("src.ingest.config.SILVER_DIR", tmp_path / "silver")
    assert main(["--tablas", "municipios"]) == 2


def test_tablas_sin_municipios_requiere_municipio_previo(tmp_path):
    with pytest.raises(FileNotFoundError):
        pipeline.ejecutar(tablas=["meteo"], base_dir=tmp_path / "silver", bronze_dir=tmp_path,
                          hf_dir=tmp_path)
