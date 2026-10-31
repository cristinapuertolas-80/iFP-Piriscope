"""Tests del dashboard: preparación de datos (contrato Gold) y arranque de la app con datos de juguete."""
import pandas as pd
import pytest

from src.dashboard import datos


def _catalogo():
    return pd.DataFrame({"id_municipio": ["15001", "15002", "27001"], "nombre": ["Abegondo", "Ames", "Abadín"],
                         "provincia": ["A Coruña", "A Coruña", "Lugo"], "lat": [43.2, 42.9, 43.4],
                         "lon": [-8.3, -8.6, -7.5]})


def _pred(filas):
    df = pd.DataFrame(filas, columns=["id_municipio", "fecha_objetivo", "version_modelo", "probabilidad",
                                      "nivel_riesgo", "fwi", "fecha_insert"])
    df["fecha_objetivo"] = pd.to_datetime(df["fecha_objetivo"])
    df["fecha_insert"] = pd.to_datetime(df["fecha_insert"], utc=True)
    df["id_prediccion"] = df["id_municipio"] + "_" + df["fecha_objetivo"].dt.strftime("%Y-%m-%d") + "_" + \
        df["version_modelo"]
    return df


def test_deduplicar_gana_principal_y_luego_lo_mas_reciente():
    pred = _pred([
        ("15001", "2026-10-19", "v1-respaldo", 0.02, "bajo", None, "2026-10-20"),
        ("15001", "2026-10-19", "v1", 0.05, "alto", 12.0, "2026-10-19"),
        ("15002", "2026-10-19", "v1", 0.01, "bajo", 3.0, "2026-10-19"),
        ("15002", "2026-10-19", "v1", 0.03, "medio", 3.0, "2026-10-21"),
    ])
    d = datos.deduplicar(pred).set_index("id_municipio")
    assert len(d) == 2
    assert d.loc["15001", "version_modelo"] == "v1"
    assert d.loc["15002", "nivel_riesgo"] == "medio"


def test_predicciones_dia_incluye_todos_los_municipios():
    pred = _pred([
        ("15001", "2026-10-19", "v1", 0.05, "alto", 12.0, "2026-10-19"),
        ("15002", "2026-10-19", "v1-respaldo", 0.01, "bajo", None, "2026-10-19"),
    ])
    dia = datos.predicciones_dia(pred, _catalogo(), "2026-10-19").set_index("id_municipio")
    assert len(dia) == 3
    assert dia.loc["27001", "nivel_mostrado"] == datos.SIN_DATO
    assert bool(dia.loc["15002", "respaldo"]) and not bool(dia.loc["15001", "respaldo"])
    assert not bool(dia.loc["27001", "respaldo"])
    # La prioridad se calcula por versión de modelo: cada una tiene su propio «primero»
    assert dia.loc["15001", "prioridad_dia"] == 0
    assert dia.loc["15002", "prioridad_dia"] == 0


def test_fecha_sin_predicciones():
    pred = _pred([("15001", "2026-10-19", "v1", 0.05, "alto", 12.0, "2026-10-19")])
    dia = datos.predicciones_dia(pred, _catalogo(), "2030-01-01")
    assert (dia["nivel_mostrado"] == datos.SIN_DATO).all()
    assert not datos.episodio_generalizado(dia)
    assert datos.fecha_con_datos(pred, "2030-01-01") == pd.Timestamp("2026-10-19")
    assert datos.fecha_con_datos(pred, "2026-10-19") == pd.Timestamp("2026-10-19")


def test_episodio_generalizado():
    dia = pd.DataFrame({"nivel_mostrado": ["alto"] * 9 + ["bajo", datos.SIN_DATO]})
    assert datos.episodio_generalizado(dia)          # 9 de 10 con dato
    dia = pd.DataFrame({"nivel_mostrado": ["alto"] * 8 + ["bajo"] * 2})
    assert not datos.episodio_generalizado(dia)


def test_leer_predicciones_conserva_ceros_iniciales(tmp_path):
    pred = _pred([("01001", "2026-10-19", "v1", 0.05, "alto", 12.0, "2026-10-19")])
    pred["id_municipio"] = pred["id_municipio"].str.lstrip("0")   # como si alguien lo hubiera convertido a número
    ruta = tmp_path / "predicciones.parquet"
    pred.to_parquet(ruta)
    assert datos.leer_predicciones(ruta)["id_municipio"].tolist() == ["01001"]


def test_formatos():
    assert datos.fmt_pct(0.0432) == "4,3 %"
    assert datos.fmt_num(None) == "no disponible"
    assert datos.fmt_prioridad(0.12) == "top 12 %"
    assert datos.fmt_prioridad(0) == "top 1 %"
    assert datos.clave_orden("A Coruña") == "CORUNA A"


def test_origen_local_y_respaldo_sin_conexion(tmp_path, monkeypatch):
    for nombre in datos.FICHEROS_GOLD:
        (tmp_path / nombre).write_bytes(b"")
    monkeypatch.setenv("PIRISCOPE_DASHBOARD_ORIGEN", "local")
    assert datos.localizar_gold(gold_dir=tmp_path).origen == "local"

    def sin_red(*_a, **_k):
        raise ConnectionError("sin red")
    monkeypatch.setenv("PIRISCOPE_DASHBOARD_ORIGEN", "hf")
    monkeypatch.setattr(datos, "descargar_gold_hf", sin_red)
    origen = datos.localizar_gold(gold_dir=tmp_path)
    assert origen.origen == "local" and origen.aviso
    with pytest.raises(RuntimeError):
        datos.localizar_gold(gold_dir=tmp_path / "vacia")


def test_geojson_tiene_los_313_municipios():
    cat = datos.catalogo_municipios(datos.leer_geojson())
    assert len(cat) == 313 and cat["id_municipio"].is_unique
    assert set(cat["provincia"]) == {"A Coruña", "Lugo", "Ourense", "Pontevedra"}
    assert cat[["nombre", "lat", "lon"]].notna().all().all()


def test_app_arranca_con_datos_de_juguete(tmp_path, monkeypatch):
    """Smoke test (contrato, apartado 10): la app arranca y lee predicciones.parquet sin errores."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from src.ingest import config

    cat = datos.catalogo_municipios(datos.leer_geojson())
    ids = cat["id_municipio"].tolist()
    filas = [(i, f, "v1" if f < "2026-10-20" else "v1-respaldo", p, n, 10.0 if f < "2026-10-20" else None,
              "2026-10-21")
             for f in ("2026-10-18", "2026-10-19", "2026-10-20")
             for i, p, n in zip(ids[:300], [0.01, 0.03, 0.06] * 100, ["bajo", "medio", "alto"] * 100)]
    pred = _pred(filas)
    pred.to_parquet(tmp_path / "predicciones.parquet")
    expl = pd.DataFrame({"id_prediccion": pred["id_prediccion"].repeat(5).to_numpy(),
                         "variable": ["ffmc", "fwi", "temperatura", "mes", "precipitacion"] * len(pred),
                         "valor_shap": [0.3, 0.2, -0.1, 0.05, -0.02] * len(pred),
                         "valor_variable": [90.0, 20.0, 25.0, 8.0, 0.0] * len(pred),
                         "valor_base": -3.8, "shap_resto": 0.1})
    expl.to_parquet(tmp_path / "explicaciones.parquet")

    monkeypatch.setenv("PIRISCOPE_DASHBOARD_ORIGEN", "local")
    monkeypatch.setattr(config, "GOLD_DIR", tmp_path)
    at = AppTest.from_file(str(config.REPO_ROOT / "src" / "dashboard" / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert any("20/10/2026" in m.value for m in at.markdown)       # última fecha por defecto, no la de hoy
    assert any("Estimación sin FWI" in m.value for m in at.markdown)
