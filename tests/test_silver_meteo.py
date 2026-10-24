"""Tests de la meteorología AEMET en Silver."""

from datetime import date

import pandas as pd

from src.silver import meteo

from .silver_fixtures import escribir_aemet, municipios_df

ESTACIONES = [
    # La 0001 está junto al municipio 15001; la 0002, a unos 60 km
    {"indicativo": "0001", "nombre": "CERCA", "provincia": "A CORUÑA", "altitud": "100",
     "latitud_dec": "42.95", "longitud_dec": "-8.33"},
    {"indicativo": "0002", "nombre": "LEJOS", "provincia": "LUGO", "altitud": "500",
     "latitud_dec": "42.95", "longitud_dec": "-7.5"},
]


def _reg(fecha, indicativo="0001", **valores):
    return {"fecha": fecha, "indicativo": indicativo, **valores}


def test_a_numero_coma_decimal_ip_y_acum():
    s = pd.Series(["12,5", "Ip", "Acum", "", None, "0,0"])
    assert meteo.a_numero(s, inapreciable=0.0).tolist()[:2] == [12.5, 0.0]
    assert meteo.a_numero(s, inapreciable=0.0).isna().tolist() == [False, False, True, True, True, False]
    assert pd.isna(meteo.a_numero(pd.Series(["Ip"]))[0])  # Ip sólo vale 0 en la precipitación


def test_leer_diarios_prevalece_la_ingesta_mas_reciente(tmp_path):
    escribir_aemet(tmp_path / "aemet" / "20260101T000000Z",
                   {"0001_2026-01-01_2026-01-02.json": [_reg("2026-01-01", tmed="10,0", prec="Ip"),
                                                        _reg("2026-01-02", tmed="11,0")]}, ESTACIONES)
    escribir_aemet(tmp_path / "aemet" / "20260105T000000Z",
                   {"0001_2026-01-02_2026-01-02.json": [_reg("2026-01-02", tmed="12,0")]}, ESTACIONES)
    obs = meteo.leer_diarios(tmp_path, tmp_path / "nada")
    assert obs["temperatura"].tolist() == [10.0, 12.0]
    assert obs["precipitacion"].tolist()[0] == 0.0
    assert not obs.duplicated(["id_estacion_aemet", "fecha"]).any()


def test_leer_diarios_filtra_por_rango(tmp_path):
    escribir_aemet(tmp_path / "aemet" / "20260101T000000Z",
                   {"0001_2025-01-01_2025-01-01.json": [_reg("2025-01-01", tmed="1,0")],
                    "0001_2026-01-01_2026-01-01.json": [_reg("2026-01-01", tmed="2,0")]}, ESTACIONES)
    obs = meteo.leer_diarios(tmp_path, tmp_path / "nada", desde=date(2026, 1, 1))
    assert obs["temperatura"].tolist() == [2.0]


def test_nombre_propio_de_estacion():
    assert meteo.nombre_propio("SANTIAGO DE COMPOSTELA AEROPUERTO") == "Santiago de Compostela Aeropuerto"
    assert meteo.nombre_propio("O BARCO DE VALDEORRAS") == "O Barco de Valdeorras"
    assert meteo.nombre_propio("OURENSE,INSTITUTO") == "Ourense, Instituto"
    assert meteo.nombre_propio("ROIS, CASAS DO PORTO") == "Rois, Casas do Porto"
    assert meteo.nombre_propio("RODEIRO-VILAMAIOR") == "Rodeiro-Vilamaior"
    assert meteo.nombre_propio("A CORUÑA  BENS ") == "A Coruña Bens"
    assert pd.isna(meteo.nombre_propio(None))


def test_leer_estaciones_tipa_coordenadas(tmp_path):
    escribir_aemet(tmp_path / "aemet" / "20260101T000000Z", {}, ESTACIONES)
    est = meteo.leer_estaciones(tmp_path, tmp_path / "nada")
    assert est["id_estacion_aemet"].tolist() == ["0001", "0002"]
    assert est["altitud"].tolist() == [100.0, 500.0]


def test_asignar_estacion_mas_cercana_y_distancia():
    municipio, _ = municipios_df()
    est = pd.DataFrame({"id_estacion_aemet": ["0001", "0002"], "latitud": [42.95, 42.95],
                        "longitud": [-8.33, -7.5]})
    asig = meteo.asignar_estaciones(municipio, est)
    assert asig["id_estacion_aemet"].tolist() == ["0001", "0001"]
    assert (asig["dist_estacion_km"] < 15).all()
    # Si la cercana no tiene datos, se usa la siguiente
    asig = meteo.asignar_estaciones(municipio, est, con_datos={"0002"})
    assert asig["id_estacion_aemet"].tolist() == ["0002", "0002"]


def test_observacion_meteorologica_una_fila_por_municipio_y_dia_con_huecos_nulos():
    municipio, _ = municipios_df()
    asig = pd.DataFrame({"id_municipio": ["15001", "15002"], "id_estacion_aemet": ["0001", "0001"],
                         "dist_estacion_km": [1.0, 9.0]})
    obs = meteo.tipar_registros(pd.DataFrame([_reg("2026-01-01", tmed="10,0"), _reg("2026-01-03", tmed="9,0")])
                                .assign(timestamp_ingesta="t"))
    df = meteo.observacion_meteorologica(municipio, asig, obs, date(2026, 1, 1), date(2026, 1, 3))
    assert len(df) == 6 and df["id_obs_meteo"].is_unique
    assert df["id_obs_meteo"].iloc[0] == "15001_2026-01-01"
    assert df.loc[df["fecha"] == "2026-01-02", "temperatura"].isna().all()  # no se imputa
    assert df["dist_estacion_km"].tolist() == [1.0] * 3 + [9.0] * 3
