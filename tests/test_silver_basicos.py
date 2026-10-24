"""Tests de utilidades de Silver: fuentes, almacén, calidad, geo y municipios."""

from datetime import date

import pandas as pd
import pyarrow.parquet as pq

from src.silver import almacen, calidad, fuentes, geo, municipios

from .silver_fixtures import municipios_df


# --- fuentes ----------------------------------------------------------------
def test_ejecuciones_combina_bronze_y_hf_y_prioriza_bronze(tmp_path):
    bronze, hf = tmp_path / "bronze", tmp_path / "hf"
    (bronze / "aemet" / "20260102T000000Z").mkdir(parents=True)
    (hf / "AEMET" / "20260101T000000Z").mkdir(parents=True)
    (hf / "AEMET" / "20260102T000000Z").mkdir(parents=True)
    runs = fuentes.ejecuciones("aemet", bronze, hf)
    assert [r.name for r in runs] == ["20260101T000000Z", "20260102T000000Z"]
    assert runs[1].parent.parent == bronze


def test_ultimo_fichero_devuelve_la_ingesta_mas_reciente(tmp_path):
    for ts in ("20260101T000000Z", "20260201T000000Z"):
        (tmp_path / "egif" / ts).mkdir(parents=True)
        (tmp_path / "egif" / ts / "egif_galicia_unificado.parquet").write_text(ts)
    ruta = fuentes.ultimo_fichero("egif", "egif_galicia_unificado.parquet", tmp_path, tmp_path / "nada")
    assert ruta.read_text() == "20260201T000000Z"


# --- almacén ----------------------------------------------------------------
def _tabla(fechas, valor):
    return pd.DataFrame({"id": [f"15001_{f}" for f in fechas], "fecha": pd.to_datetime(fechas), "v": valor})


def test_guardar_tipa_fecha_como_date32_y_anade_fecha_insert(tmp_path):
    almacen.guardar(_tabla(["2024-01-01"], 1), "t", base_dir=tmp_path)
    esquema = pq.read_schema(tmp_path / "t.parquet")
    assert str(esquema.field("fecha").type) == "date32[day]"
    assert "UTC" in str(esquema.field("fecha_insert").type)
    assert pd.api.types.is_datetime64_any_dtype(almacen.leer("t", tmp_path)["fecha"])


def test_guardar_sobrescribe_solo_el_periodo_reprocesado(tmp_path):
    almacen.guardar(_tabla(["2024-01-01", "2024-01-02", "2024-01-03"], 1), "t", base_dir=tmp_path)
    almacen.guardar(_tabla(["2024-01-02"], 2), "t", "fecha", date(2024, 1, 2), date(2024, 1, 2),
                    ordenar_por=["fecha"], base_dir=tmp_path)
    df = almacen.leer("t", tmp_path)
    assert df["v"].tolist() == [1, 2, 1]
    assert df["id"].is_unique


def test_guardar_descarta_filas_nuevas_fuera_del_rango(tmp_path):
    almacen.guardar(_tabla(["2024-01-01", "2024-02-01"], 1), "t", "fecha", date(2024, 1, 1),
                    date(2024, 1, 31), base_dir=tmp_path)
    assert len(almacen.leer("t", tmp_path)) == 1


# --- calidad ----------------------------------------------------------------
def test_aplicar_rangos_pone_a_nulo_y_registra():
    df = pd.DataFrame({"k": ["a", "b", "c"], "humedad_relativa": pd.array([50, 120, None], dtype="Float64")})
    limpio, anomalias = calidad.aplicar_rangos(df, "obs", "k")
    assert limpio["humedad_relativa"].isna().tolist() == [False, True, True]
    assert anomalias[["clave", "columna", "valor"]].values.tolist() == [["b", "humedad_relativa", 120.0]]


def test_informe_conserva_tablas_de_ejecuciones_previas(tmp_path):
    i = calidad.Informe(tmp_path)
    i.tabla("a", filas=1)
    i.escribir()
    j = calidad.Informe(tmp_path)
    j.tabla("b", filas=2)
    import json
    datos = json.loads(j.escribir().read_text(encoding="utf-8"))
    assert datos == {"a": {"filas": 1}, "b": {"filas": 2}}


# --- geo --------------------------------------------------------------------
def test_haversine_un_grado_de_latitud():
    assert abs(float(geo.haversine_km(42, -8, 43, -8)) - 111.2) < 0.2


def test_mas_cercano():
    idx, dist = geo.mas_cercano([42.0, 43.0], [-8.0, -8.0], [43.01, 42.0], [-8.0, -8.01])
    assert idx.tolist() == [1, 0]
    assert (dist < 2).all()


# --- municipios -------------------------------------------------------------
def test_normalizar_nombre_antepone_articulo_y_quita_tildes():
    assert municipios.normalizar_nombre("Cañiza, A") == "A CANIZA"
    assert municipios.normalizar_nombre("CAÑIZA, A") == "A CANIZA"
    assert municipios.normalizar_nombre("A Cañiza") == "A CANIZA"
    assert municipios.normalizar_nombre("Pontes de García Rodríguez, As") == "AS PONTES DE GARCIA RODRIGUEZ"
    assert municipios.normalizar_nombre("Oza-Cesuras") == "OZA CESURAS"
    assert municipios.normalizar_nombre(None) == ""


def test_construir_municipios_con_centroide_y_provincia():
    municipio, limites = municipios_df()
    assert municipio["id_municipio"].tolist() == ["15001", "15002"]
    assert municipio["nombre"].tolist() == ["Abegondo", "A Cañiza"]
    assert set(municipio["provincia"]) == {"A Coruña"}
    assert municipio["reservado"].isna().all()
    # El centroide del primer cuadrado queda al oeste del segundo y dentro de su recinto
    assert municipio["lon_centroide"].iloc[0] < municipio["lon_centroide"].iloc[1]
    assert str(limites.crs) == "EPSG:4326" and len(limites) == 2


def test_construir_municipios_con_reservados():
    from .silver_fixtures import recintos

    ine = pd.DataFrame({"id_municipio": ["15001", "15002"], "id_provincia": ["15", "15"],
                        "nombre_ine": ["Abegondo", "Cañiza, A"]}).astype("string")
    reservados = pd.DataFrame({"id_municipio": ["15002"], "reservado": [1]})
    municipio, _ = municipios.construir(ine, municipios.filtrar_recintos(recintos()), reservados)
    assert municipio["reservado"].tolist() == [0, 1]


def test_leer_diccionario_ine_filtra_galicia(tmp_path):
    from .silver_fixtures import escribir_ine_cnig

    escribir_ine_cnig(tmp_path)
    ine = municipios.leer_diccionario_ine(tmp_path / "ine" / "diccionario26.xlsx")
    assert ine["id_municipio"].tolist() == ["15001", "15002"]
    assert ine["nombre_ine"].tolist() == ["Abegondo", "Cañiza, A"]
