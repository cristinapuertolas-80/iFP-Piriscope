import json

import pandas as pd
import pytest

from src.ingest import egif


def _exportacion(ruta, filas, titulo=True):
    """Simula una exportación del buscador EGIF con filas de título encima de la cabecera."""
    cabecera = ["Nº Parte", "Campaña", "Provincia", "Municipio", "Fecha de detección",
                "Fecha de extinción", "Causa", "Superficie forestal (ha)", "Coordenada X", "Coordenada Y", "Huso"]
    with pd.ExcelWriter(ruta) as w:
        df = pd.DataFrame(filas, columns=cabecera)
        inicio = 2 if titulo else 0
        if titulo:
            pd.DataFrame([["Estadística General de Incendios Forestales"]]).to_excel(
                w, index=False, header=False, startrow=0)
        df.to_excel(w, index=False, startrow=inicio)
    return ruta


FILA_CORUNA = ["101", "2020", "A Coruña", "Arteixo", "14/08/2020", "15/08/2020", "Intencionado", "1,5",
               "538000", "4795000", "29"]
FILA_LUGO = ["101", "2020", "Lugo", "Becerreá", "02/09/2020", "02/09/2020", "Rayo", "0,3",
             "640000", "4740000", "29"]


@pytest.mark.parametrize("entrada,esperado", [
    ("Fecha de Detección", "fecha_de_deteccion"),
    ("Nº Parte", "n_parte"),
    ("Superficie forestal (ha)", "superficie_forestal"),
    ("  CAMPAÑA ", "campana"),
])
def test_normalizar_nombre(entrada, esperado):
    assert egif.normalizar_nombre(entrada) == esperado


def test_mapear_columnas():
    mapeo = egif.mapear_columnas(["n_parte", "campana", "fecha_de_deteccion", "x", "y", "otra"])
    assert mapeo["n_parte"] == "numero_parte"
    assert mapeo["fecha_de_deteccion"] == "fecha_deteccion"
    assert mapeo["x"] == "coordenada_x"
    assert "otra" not in mapeo


@pytest.mark.parametrize("nombre,codigo", [
    ("egif_A_Coruna_2000_2022.xlsx", "15"), ("partes_lugo.xlsx", "27"), ("OURENSE.csv", "32"),
    ("export_36.xlsx", "36"), ("galicia.xlsx", None),
])
def test_provincia_desde_nombre(nombre, codigo):
    assert egif.provincia_desde_nombre_fichero(nombre) == codigo


def test_leer_fichero_detecta_cabecera(tmp_path):
    ruta = _exportacion(tmp_path / "egif_coruna.xlsx", [FILA_CORUNA])
    (tabla,) = egif.leer_fichero(ruta)
    assert "n_parte" in tabla.columns and "fecha_de_deteccion" in tabla.columns
    assert len(tabla) == 1
    assert tabla["fichero_origen"].iat[0] == "egif_coruna.xlsx"


def test_unificar_quita_duplicados_y_detecta_parte_repetido_entre_provincias(tmp_path):
    a = egif.leer_fichero(_exportacion(tmp_path / "coruna.xlsx", [FILA_CORUNA, FILA_CORUNA]))
    b = egif.leer_fichero(_exportacion(tmp_path / "lugo.xlsx", [FILA_LUGO]))
    df, eliminados, info = egif.unificar(a + b)
    assert len(df) == 2 and len(eliminados) == 1
    # El nº 101 se repite en A Coruña y Lugo en 2020 -> se añade la provincia al id_evento
    assert info["formato_id_evento"] == "campana_provincia_numero"
    assert set(df["id_evento"]) == {"2020_15_101", "2020_27_101"}
    assert df["id_evento"].is_unique


def test_unificar_sin_columnas_clave_avisa():
    t = pd.DataFrame({"municipio": ["Arteixo"], "fichero_origen": ["x.csv"], "hoja_origen": ["csv"]})
    df, eliminados, info = egif.unificar([t])
    assert "aviso" in info and eliminados.empty


def test_campana_desde_fecha_si_falta(tmp_path):
    t = pd.DataFrame({"n_parte": ["7"], "fecha_de_deteccion": ["03/07/2019"],
                      "fichero_origen": ["ourense.csv"], "hoja_origen": ["csv"]})
    df, _, _ = egif.unificar([t])
    assert df["campana"].iat[0] == "2019"
    assert df["id_evento"].iat[0] == "2019_7"
    assert df["id_provincia"].iat[0] == "32"


def test_recuentos_contrastados_con_oficiales(tmp_path):
    df = pd.DataFrame({"id_provincia": ["15", "15", "27"], "campana": ["2020", "2020", "2020"]})
    oficiales = tmp_path / "oficiales.csv"
    oficiales.write_text("id_provincia,campana,n_incendios_oficial\n15,2020,2\n27,2020,4\n")
    rec = egif.recuentos_provincia_anio(df, oficiales).set_index("id_provincia")
    assert rec.loc["15", "diferencia"] == 0
    assert rec.loc["27", "diferencia"] == -3
    assert rec.loc["15", "provincia"] == "A Coruña"


def test_ingestar_end_to_end(tmp_path, bronze):
    landing = tmp_path / "landing"
    landing.mkdir()
    f1 = _exportacion(landing / "coruna.xlsx", [FILA_CORUNA, FILA_CORUNA])
    f2 = _exportacion(landing / "lugo.xlsx", [FILA_LUGO])
    run = egif.ingestar(ficheros=[f1, f2], base_dir=bronze, oficiales=None)
    assert (run.dir / "originales" / "coruna.xlsx").read_bytes() == f1.read_bytes()
    unificado = pd.read_parquet(run.dir / "egif_galicia_unificado.parquet")
    assert len(unificado) == 2
    m = json.loads((run.dir / "manifest.json").read_text(encoding="utf-8"))
    rutas = {f["ruta"] for f in m["ficheros"]}
    assert {"egif_galicia_unificado.parquet", "informe_duplicados.csv",
            "recuentos_provincia_anio.csv", "mapeo_columnas.json"} <= rutas
    # Faltan Ourense y Pontevedra -> incidencias documentadas
    ausentes = [i for i in run.incidencias if i["tipo"] == "provincia_ausente"]
    assert len(ausentes) == 2


def test_ingestar_sin_ficheros_explica_que_hacer(bronze, monkeypatch):
    monkeypatch.setattr(egif, "localizar_ficheros", lambda *a, **k: [])
    with pytest.raises(FileNotFoundError, match="landing/egif"):
        egif.ingestar(base_dir=bronze)
