"""Tests de los incendios de EGIF en Silver: tipado y asignación del código INE."""

import pandas as pd

from src.silver import incendios

from .silver_fixtures import CENTRO, municipios_df, parte

EQUIVALENCIAS = pd.DataFrame({"id_provincia": ["15"], "municipio_egif": ["VIEJO NOMBRE"],
                              "id_municipio": ["15002"]})


def _asignar(partes):
    municipio, limites = municipios_df()
    df = incendios.tipar(pd.DataFrame(partes).astype("string"))
    return incendios.asignar_municipio(df, municipio, limites, EQUIVALENCIAS)


def test_tipar_fechas_numeros_y_columnas_de_la_exportacion():
    df = incendios.tipar(pd.DataFrame([parte("2020_15_0001", "ABEGONDO", "14/08/2020 16:30:00",
                                             *CENTRO["15001"])]).astype("string"))
    fila = df.iloc[0]
    assert fila["fecha_deteccion"] == pd.Timestamp("2020-08-14 16:30")
    assert fila["superficie_forestal_ha"] == 1.5
    assert fila["campana"] == 2020 and fila["huso"] == 29 and fila["datum"] == "ETRS89"
    assert fila["coordenada_x"] == CENTRO["15001"][0]
    assert fila["causa"] == "[400] Intencionado"  # formato de EGIF sin el espacio doble


def test_tipar_huecos_como_nulos():
    df = incendios.tipar(pd.DataFrame([parte("2020_15_0002", "ABEGONDO", "14/08/2020 16:30:00", datum="  ")])
                         .astype("string"))
    assert pd.isna(df["coordenada_x"].iloc[0]) and pd.isna(df["huso"].iloc[0]) and pd.isna(df["datum"].iloc[0])


def test_asignacion_por_nombre_validada_con_coordenadas():
    df = _asignar([parte("a", "ABEGONDO", "01/07/2020 10:00:00", *CENTRO["15001"]),
                   parte("b", "CAÑIZA, A", "01/07/2020 10:00:00", *CENTRO["15002"])])
    assert df["id_municipio"].tolist() == ["15001", "15002"]
    assert df["metodo_asignacion"].tolist() == ["nombre_validado"] * 2


def test_nombre_prevalece_si_las_coordenadas_discrepan():
    df = _asignar([parte("a", "ABEGONDO", "01/07/2020 10:00:00", *CENTRO["15002"])])
    assert df["id_municipio"].iloc[0] == "15001"
    assert df["id_municipio_coordenadas"].iloc[0] == "15002"
    assert df["metodo_asignacion"].iloc[0] == "nombre_coordenadas_discrepan"


def test_tabla_de_equivalencias_y_datum_ed50_por_defecto():
    df = _asignar([parte("a", "VIEJO NOMBRE", "01/07/2020 10:00:00", *CENTRO["15002"], datum="")])
    assert df["id_municipio"].iloc[0] == "15002"
    assert df["metodo_asignacion"].iloc[0] == "nombre_validado"


def test_coordenadas_si_el_nombre_no_casa_y_sin_asignar_fuera_de_galicia():
    df = _asignar([parte("a", "INDETERMINADO", "01/07/2020 10:00:00", *CENTRO["15002"]),
                   parte("b", "PORTUGAL", "01/07/2020 10:00:00", 400_000, 4_600_000),
                   parte("c", "OTRA PROVINCIA", "01/07/2020 10:00:00")])
    assert df["id_municipio"].tolist()[0] == "15002"
    assert df["id_municipio"].isna().tolist() == [False, True, True]
    assert df["metodo_asignacion"].tolist() == ["coordenadas", "sin_asignar", "sin_asignar"]


def test_nombre_de_otra_provincia_gallega():
    df = _asignar([parte("a", "ABEGONDO", "01/07/2020 10:00:00", provincia="27")])
    assert df["id_municipio"].iloc[0] == "15001"
    assert bool(df["asignado_otra_provincia"].iloc[0])
    assert df["metodo_asignacion"].iloc[0] == "nombre_sin_coordenadas"


def test_informes_de_asignacion():
    municipio, _ = municipios_df()
    df = _asignar([parte("a", "ABEGONDO", "01/07/2020 10:00:00", *CENTRO["15001"]),
                   parte("b", "PORTUGAL", "02/07/2020 10:00:00")])
    no_asig = incendios.informe_no_asignados(df)
    assert no_asig[["municipio_egif", "n_partes"]].values.tolist() == [["PORTUGAL", 1]]
    muestra = incendios.muestra_validacion(df, municipio)
    assert muestra["nombre_asignado"].tolist() == ["Abegondo"]
    resumen = incendios.resumen_asignacion(df)
    assert resumen["pct_asignados"].tolist() == [50.0]


def test_asignacion_a_revisar_solo_discrepancias_y_por_coordenadas():
    municipio, _ = municipios_df()
    df = _asignar([parte("a", "ABEGONDO", "01/07/2020 10:00:00", *CENTRO["15001"]),
                   parte("b", "ABEGONDO", "01/07/2020 10:00:00", *CENTRO["15002"]),
                   parte("c", "INDETERMINADO", "01/07/2020 10:00:00", *CENTRO["15002"]),
                   parte("d", "PORTUGAL", "02/07/2020 10:00:00")])
    revisar = incendios.asignacion_a_revisar(df, municipio)
    assert revisar["metodo_asignacion"].tolist() == ["nombre_coordenadas_discrepan", "coordenadas"]
    assert revisar["nombre_coordenadas"].tolist() == ["A Cañiza", "A Cañiza"]
