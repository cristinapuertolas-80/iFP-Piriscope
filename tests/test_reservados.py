import json

import numpy as np
import pandas as pd
import pytest

from src.ml import reservados as rv


def _df(seed=0):
    """313 municipios en 4 provincias y 58 estaciones, con tasas de incendio distintas."""
    rng = np.random.default_rng(seed)
    provincias = ["A Coruña", "Lugo", "Ourense", "Pontevedra"]
    filas = []
    for i in range(313):
        est = f"E{i % 58:03d}"
        prov = provincias[(i % 58) % 4]          # una estación pertenece a una sola provincia
        filas.append({"id_municipio": f"{15000 + i:05d}", "provincia": prov, "id_estacion_aemet": est,
                      "dias_con_incendio": int(rng.poisson(3 + 6 * ((i % 58) % 5 == 0))) })
    df = pd.DataFrame(filas)
    df["tasa"] = df["dias_con_incendio"] / 7670
    return df


def test_los_bloques_se_reservan_completos():
    out, _ = rv.seleccionar_reservados(_df())
    por_estacion = out.groupby("id_estacion_aemet")["reservado"].nunique()
    assert (por_estacion == 1).all()


def test_fraccion_real_cercana_a_la_pedida():
    out, _ = rv.seleccionar_reservados(_df(), fraccion=0.20)
    assert 0.14 <= out["reservado"].mean() <= 0.26


def test_todas_las_provincias_tienen_reservados():
    out, _ = rv.seleccionar_reservados(_df())
    assert (out.groupby("provincia")["reservado"].sum() > 0).all()


def test_cada_estrato_aporta_al_menos_un_bloque():
    _, bloques = rv.seleccionar_reservados(_df())
    assert (bloques.groupby("estrato")["reservado"].sum() >= 1).all()


def test_reproducible_con_la_misma_semilla_y_distinto_con_otra():
    a, _ = rv.seleccionar_reservados(_df(), semilla=1)
    b, _ = rv.seleccionar_reservados(_df(), semilla=1)
    c, _ = rv.seleccionar_reservados(_df(), semilla=2)
    assert a["reservado"].tolist() == b["reservado"].tolist()
    assert a["reservado"].tolist() != c["reservado"].tolist()


def test_frecuencia_de_incendios_comparable_entre_reservados_y_resto():
    out, _ = rv.seleccionar_reservados(_df())
    r = out.groupby("reservado")["dias_con_incendio"].mean()
    assert 0.6 <= r[1] / r[0] <= 1.6


def test_fraccion_invalida():
    with pytest.raises(ValueError):
        rv.seleccionar_reservados(_df(), fraccion=1.5)


# --- Carga desde Silver (tablas sintéticas) ----------------------------------
def _silver(tmp_path, extra_estacion=False, sin_estacion=False):
    pd.DataFrame({"id_municipio": ["15001", "15002", "27001"],
                  "provincia": ["A Coruña", "A Coruña", "Lugo"]}).to_parquet(tmp_path / "municipio.parquet")
    obs = pd.DataFrame({"id_municipio": ["15001", "15001", "15002", "27001"],
                        "id_estacion_aemet": ["0001", "0001", "0001", "0002"]})
    if extra_estacion:
        obs.loc[1, "id_estacion_aemet"] = "0003"
    if sin_estacion:
        obs = obs[obs["id_municipio"] != "27001"]
    obs.to_parquet(tmp_path / "observacion_meteorologica.parquet")
    pd.DataFrame({
        "id_municipio": ["15001", "15001", "15001", "27001", "27001", None],
        "fecha_deteccion": pd.to_datetime(["2010-07-01 10:00", "2010-07-01 18:00", "2025-07-01 09:00",
                                           "2015-08-01 09:00", "2000-08-01 09:00", "2012-01-01 09:00"]),
    }).to_parquet(tmp_path / "evento_incendio.parquet")
    return tmp_path


def test_cargar_datos_cuenta_dias_distintos_y_filtra_anios(tmp_path):
    df = rv.cargar_datos(_silver(tmp_path), 2001, 2021).set_index("id_municipio")
    assert df.loc["15001", "dias_con_incendio"] == 1      # 2 partes el mismo día; el de 2025 queda fuera
    assert df.loc["27001", "dias_con_incendio"] == 1      # el de 2000 queda fuera
    assert df.loc["15002", "dias_con_incendio"] == 0
    assert df.loc["15001", "tasa"] == pytest.approx(1 / 7670)
    assert df.loc["15002", "id_estacion_aemet"] == "0001"


def test_cargar_datos_falla_con_varias_estaciones_o_sin_estacion(tmp_path):
    with pytest.raises(ValueError, match="más de una estación"):
        rv.cargar_datos(_silver(tmp_path, extra_estacion=True))
    with pytest.raises(ValueError, match="sin estación"):
        rv.cargar_datos(_silver(tmp_path, sin_estacion=True))


# --- CLI ---------------------------------------------------------------------
def _datos_silver_grandes(tmp_path):
    df = _df()
    df[["id_municipio", "provincia"]].to_parquet(tmp_path / "municipio.parquet")
    df[["id_municipio", "id_estacion_aemet"]].to_parquet(tmp_path / "observacion_meteorologica.parquet")
    filas = [(r.id_municipio, pd.Timestamp("2010-01-01") + pd.Timedelta(days=k))
             for r in df.itertuples() for k in range(int(r.dias_con_incendio))]
    pd.DataFrame(filas, columns=["id_municipio", "fecha_deteccion"]).to_parquet(tmp_path / "evento_incendio.parquet")


def test_cli_escribe_csv_meta_y_no_sobrescribe(tmp_path):
    _datos_silver_grandes(tmp_path)
    csv, meta = tmp_path / "res.csv", tmp_path / "meta.json"
    args = ["--silver-dir", str(tmp_path), "--salida", str(csv), "--meta", str(meta)]
    assert rv.main(args) == 0

    lista = pd.read_csv(csv, dtype={"id_municipio": str})
    assert list(lista.columns) == ["id_municipio", "reservado"]
    assert (lista["reservado"] == 1).all() and lista["id_municipio"].str.len().eq(5).all()
    assert lista["id_municipio"].is_unique
    m = json.loads(meta.read_text(encoding="utf-8"))
    assert m["semilla"] == 42 and m["resumen"]["n_reservados"] == len(lista)

    contenido = csv.read_text()
    assert rv.main(args) == 2                      # congelada: se niega a sobrescribir
    assert csv.read_text() == contenido
    assert rv.main(args + ["--forzar"]) == 0


def test_cli_error_si_faltan_tablas(tmp_path):
    assert rv.main(["--silver-dir", str(tmp_path / "nada"), "--salida", str(tmp_path / "x.csv"),
                    "--meta", str(tmp_path / "m.json")]) == 2
