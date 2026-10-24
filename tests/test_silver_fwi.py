"""Tests del FWI en Silver."""

from datetime import date

import pandas as pd

from src.silver import fwi

from .silver_fixtures import escribir_fwi, municipios_df

FECHAS = ["2024-07-01", "2024-07-02"]


def test_abrir_convierte_longitudes_y_recorta(tmp_path):
    escribir_fwi(tmp_path / "fwi_2024_07.nc", FECHAS, 5.0)
    ds = fwi.abrir(tmp_path / "fwi_2024_07.nc")
    assert float(ds.longitude.min()) < 0
    assert set(ds.data_vars) == set(fwi.VARIABLES)


def test_celda_de_tierra_mas_cercana_evita_el_mar(tmp_path):
    # La fila de 43.0 N es mar (NaN): aunque esté más cerca, no se elige
    escribir_fwi(tmp_path / "fwi_2024_07.nc", FECHAS, 5.0)
    municipio, _ = municipios_df()
    celdas = fwi.celdas_municipios(fwi.abrir(tmp_path / "fwi_2024_07.nc"), municipio)
    assert set(celdas["lat_celda"]) == {42.75}


def test_procesar_prioriza_consolidado_sobre_intermedio(tmp_path):
    escribir_fwi(tmp_path / "effis" / "20260101T000000Z" / "fwi_2024_07.nc", FECHAS, 5.0)
    escribir_fwi(tmp_path / "effis" / "20260201T000000Z" / "fwi_2024_07_intermedio.nc", FECHAS, 9.0)
    municipio, _ = municipios_df()
    df = fwi.procesar(municipio, bronze_dir=tmp_path, hf_dir=tmp_path / "nada")
    assert len(df) == 4 and df["id_indicador_fwi"].is_unique
    assert df["fwi"].tolist() == [5.0] * 4
    assert set(df["origen_fwi"]) == {"consolidado"}
    assert df["id_indicador_fwi"].iloc[0] == "15001_2024-07-01"


def test_procesar_usa_intermedio_si_no_hay_consolidado_y_filtra_rango(tmp_path):
    escribir_fwi(tmp_path / "effis" / "20260201T000000Z" / "fwi_2024_07_intermedio.nc", FECHAS, 9.0)
    municipio, _ = municipios_df()
    df = fwi.procesar(municipio, desde=date(2024, 7, 2), hasta=date(2024, 7, 2),
                      bronze_dir=tmp_path, hf_dir=tmp_path / "nada")
    assert df["fecha"].tolist() == [pd.Timestamp("2024-07-02")] * 2
    assert set(df["origen_fwi"]) == {"intermedio"}


def test_meses_ausentes():
    inv = pd.DataFrame({"anio": [2024, 2024], "mes": [1, 3]})
    assert fwi.meses_ausentes(inv, date(2024, 1, 1), date(2024, 3, 31)) == ["2024-02"]
