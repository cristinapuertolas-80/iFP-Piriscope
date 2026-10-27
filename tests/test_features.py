import json

import numpy as np
import pandas as pd
import pytest

from src.features import dataset as ds
from src.features import derived as dv
from src.ml import config, splits

NAN = np.nan


def _serie(valores_por_mun: dict[str, list[float]], inicio="2021-01-01") -> pd.DataFrame:
    filas = []
    for mun, vals in valores_por_mun.items():
        fechas = pd.date_range(inicio, periods=len(vals), freq="D")
        filas += [{"id_municipio": mun, "fecha": f, "precipitacion": v} for f, v in zip(fechas, vals)]
    return pd.DataFrame(filas)


def _racha_lenta(vals, umbral=1.0):
    """Referencia ingenua, día a día, para una serie de un municipio."""
    out, racha, conocida = [], 0, True
    for v in vals:
        if np.isnan(v):
            conocida, racha = False, 0
            out.append(NAN)
        elif v >= umbral:
            conocida, racha = True, 0
            out.append(0.0)
        else:
            racha += 1
            out.append(float(racha) if conocida else NAN)
    return out


# --- racha_dias_secos --------------------------------------------------------
def test_racha_basica_y_cuenta_desde_el_inicio():
    df = _serie({"a": [0, 0, 5, 0, 0, 0]})
    assert dv.racha_dias_secos(df).tolist() == [1, 2, 0, 1, 2, 3]


def test_racha_con_huecos_es_desconocida_hasta_el_siguiente_dia_lluvioso():
    df = _serie({"a": [0, NAN, 0, 5, 0]})
    r = dv.racha_dias_secos(df)
    assert np.isnan(r[1]) and np.isnan(r[2])
    assert r[0] == 1 and r[3] == 0 and r[4] == 1


def test_racha_no_se_mezcla_entre_municipios():
    df = _serie({"a": [0, 0, 0, NAN], "b": [0, 0, 5]})
    r = dv.racha_dias_secos(df)
    assert r[4:].tolist() == [1, 2, 0]          # b arranca limpio aunque a acabe en NaN


def test_racha_y_acumulada_con_tipos_nullable_como_en_silver():
    """Silver guarda id_municipio como 'string' y las medidas como 'Float64' (con pd.NA)."""
    df = _serie({"a": [0, 0, 5, NAN, 0], "b": [0, 5, 0, 0, 0]})
    df["id_municipio"] = df["id_municipio"].astype("string")
    df["precipitacion"] = df["precipitacion"].astype("Float64")
    out = dv.anadir_derivadas(df)
    assert np.allclose(out["racha_dias_secos"], [1, 2, 0, NAN, NAN, 1, 0, 1, 2, 3], equal_nan=True)
    assert out["precip_acum_7d"].dtype == "float64"


def test_racha_coincide_con_la_referencia_lenta():
    rng = np.random.default_rng(3)
    series = {}
    for m in "abcd":
        v = rng.choice([0.0, 0.2, 3.0, NAN], size=300, p=[.5, .2, .2, .1])
        series[m] = list(v)
    df = _serie(series)
    esperado = np.concatenate([_racha_lenta(v) for v in series.values()])
    assert np.allclose(dv.racha_dias_secos(df), esperado, equal_nan=True)


def test_racha_umbral():
    df = _serie({"a": [0.5, 0.9, 1.0, 0.0]})
    assert dv.racha_dias_secos(df, umbral_seco_mm=1.0).tolist() == [1, 2, 0, 1]
    assert dv.racha_dias_secos(df, umbral_seco_mm=0.5).tolist() == [0, 0, 0, 1]


# --- precip_acum_7d ----------------------------------------------------------
def test_acumulada_7d_ventana_que_termina_en_d():
    df = _serie({"a": [1.0] * 10})
    acum = dv.precip_acum_7d(df)
    assert acum.iloc[:6].isna().all() and (acum.iloc[6:] == 7.0).all()


def test_acumulada_7d_falta_un_dia():
    df = _serie({"a": [1.0, 1, 1, NAN, 1, 1, 1, 1, 1, 1, 1]})
    assert dv.precip_acum_7d(df, min_dias=7).iloc[6:10].isna().all()     # ventanas que incluyen el hueco
    assert dv.precip_acum_7d(df, min_dias=6).iloc[6] == 6.0
    assert dv.precip_acum_7d(df).iloc[-1] == 7.0                         # el hueco ya salió de la ventana


def test_acumulada_7d_no_cruza_municipios():
    df = _serie({"a": [1.0] * 8, "b": [2.0] * 8})
    acum = dv.precip_acum_7d(df)
    assert acum.iloc[8:14].isna().all() and acum.iloc[14] == 14.0


def test_mes_y_columnas_derivadas():
    out = dv.anadir_derivadas(_serie({"a": [0.0] * 40}))
    assert out["mes"].iloc[0] == 1 and out["mes"].iloc[-1] == 2
    assert {"racha_dias_secos", "precip_acum_7d", "mes"} <= set(out.columns)


# --- HU-05: ninguna derivada usa información posterior al día D ---------------
def test_derivadas_no_usan_datos_posteriores_a_d():
    rng = np.random.default_rng(0)
    series = {m: list(rng.choice([0.0, 2.0, NAN], size=120, p=[.6, .3, .1])) for m in "abc"}
    base = _serie(series)
    corte = pd.Timestamp("2021-03-01")
    alterado = base.copy()
    futuro = alterado["fecha"] > corte
    alterado.loc[futuro, "precipitacion"] = rng.choice([0.0, 9.0, NAN], size=int(futuro.sum()))

    a, b = dv.anadir_derivadas(base), dv.anadir_derivadas(alterado)
    hasta_d = a["fecha"] <= corte
    for col in ("racha_dias_secos", "precip_acum_7d", "mes"):
        assert np.allclose(a.loc[hasta_d, col].astype(float), b.loc[hasta_d, col].astype(float), equal_nan=True), col


def test_serie_con_huecos_o_desordenada_falla():
    df = _serie({"a": [0.0] * 5})
    with pytest.raises(ValueError, match="saltos"):
        dv.anadir_derivadas(df.drop(index=2))
    with pytest.raises(ValueError, match="ordenado"):
        dv.anadir_derivadas(df.iloc[::-1])


# --- Etiqueta y particiones --------------------------------------------------
def _grid(muns, ini, fin):
    f = pd.date_range(ini, fin, freq="D")
    return pd.MultiIndex.from_product([muns, f], names=["id_municipio", "fecha"]).to_frame(index=False)


def test_etiqueta_es_el_dia_siguiente_y_no_el_mismo_dia():
    g = _grid(["a"], "2021-07-01", "2021-07-20")
    ev = pd.DataFrame({"id_municipio": ["a", "a"],
                       "fecha_deteccion": pd.to_datetime(["2021-07-10 08:00", "2021-07-10 22:00"])})
    out = ds.etiquetar(g, ev, pd.Timestamp("2021-12-31")).set_index("fecha")["incendio_d1"]
    assert out["2021-07-09"] == 1 and out["2021-07-10"] == 0 and out["2021-07-11"] == 0
    assert out.sum() == 1                                    # dos partes el mismo día = una sola etiqueta


def test_sin_etiqueta_fuera_de_la_cobertura_de_egif():
    g = _grid(["a"], "2023-12-25", "2024-01-05")
    ev = pd.DataFrame({"id_municipio": ["a"], "fecha_deteccion": pd.to_datetime(["2024-01-03"])})
    out = ds.etiquetar(g, ev, pd.Timestamp("2023-12-31")).set_index("fecha")["incendio_d1"]
    assert out["2023-12-30"] == 0                           # D+1 = 31/12: cubierto
    assert out["2023-12-31":].isna().all()                  # D+1 posterior a la cobertura: sin etiqueta
    assert out.loc["2024-01-02"] is pd.NA                   # aunque haya parte el 03/01, no se etiqueta


def test_partes_sin_municipio_se_ignoran():
    g = _grid(["a"], "2021-07-01", "2021-07-05")
    ev = pd.DataFrame({"id_municipio": [None], "fecha_deteccion": pd.to_datetime(["2021-07-03"])})
    assert ds.etiquetar(g, ev, pd.Timestamp("2021-12-31"))["incendio_d1"].sum() == 0


def test_particion_por_anio_y_solo_filas_etiquetadas():
    p = ds.Params()
    g = _grid(["a"], "2021-12-30", "2024-01-02")
    g = ds.asignar_particion(ds.etiquetar(g, pd.DataFrame({"id_municipio": [], "fecha_deteccion": []}),
                                          p.fin_etiquetas), p).set_index("fecha")
    assert g.loc["2021-12-31", "particion"] == "entrenamiento"
    assert g.loc["2022-06-01", "particion"] == "validacion"
    assert g.loc["2023-06-01", "particion"] == "test"
    assert g.loc["2023-12-31":, "particion"].isna().all()   # sin etiqueta => sin partición


def test_params_invalidos():
    with pytest.raises(ValueError):
        ds.Params(anio_train_hasta=2022, anio_validacion=2022, anio_test=2023)


# --- Dataset completo desde un Silver sintético ------------------------------
def _silver(tmp_path, con_csv=True):
    muns = ["15001", "15002", "27001"]
    g = _grid(muns, "2020-01-01", "2024-03-31")
    rng = np.random.default_rng(1)
    n = len(g)
    obs = g.assign(dist_estacion_km=5.0, temperatura=rng.normal(15, 5, n), humedad_relativa=rng.uniform(30, 100, n),
                   viento=rng.uniform(0, 6, n), precipitacion=rng.choice([0.0, 0.0, 4.0, NAN], n))
    obs["fecha"] = obs["fecha"].dt.date
    obs["id_municipio"] = obs["id_municipio"].astype("string")
    for c in ["dist_estacion_km", "temperatura", "humedad_relativa", "viento", "precipitacion"]:
        obs[c] = obs[c].astype("Float64")
    fwi = g[~((g["fecha"] >= "2020-03-01") & (g["fecha"] <= "2020-03-31"))].copy()
    for c in config.FWI_FEATURES:
        fwi[c] = rng.uniform(0, 30, len(fwi))
    fwi["fecha"] = fwi["fecha"].dt.date
    fwi["id_municipio"] = fwi["id_municipio"].astype("string")
    for c in config.FWI_FEATURES:
        fwi[c] = fwi[c].astype("Float64")
    ev = pd.DataFrame({"id_municipio": ["15001", "15001", "27001", "15002", None],
                       "fecha_deteccion": pd.to_datetime(["2021-07-10 12:00", "2021-07-10 20:00", "2023-12-31 23:00",
                                                          "2024-02-01 10:00", "2021-08-01 10:00"])})
    ev["id_municipio"] = ev["id_municipio"].astype("string")
    mun = pd.DataFrame({"id_municipio": pd.array(muns, dtype="string"),
                        "provincia": pd.array(["A Coruña", "A Coruña", "Lugo"], dtype="string"),
                        "reservado": pd.array([None] * 3, dtype="Int8")})
    s = tmp_path / "silver"
    s.mkdir()
    obs.to_parquet(s / "observacion_meteorologica.parquet")
    fwi.to_parquet(s / "indicador_fwi.parquet")
    ev.to_parquet(s / "evento_incendio.parquet")
    mun.to_parquet(s / "municipio.parquet")
    csv = tmp_path / "reservados.csv"
    if con_csv:
        pd.DataFrame({"id_municipio": ["27001"], "reservado": [1]}).to_csv(csv, index=False)
    return s, csv


def test_dataset_completo_cumple_el_contrato_de_ml(tmp_path):
    s, csv = _silver(tmp_path)
    silver = ds.leer_silver(s)
    df = ds.construir_dataset(silver, ds.leer_reservados(csv, silver["municipio"]))
    assert df[["id_municipio", "fecha"]].duplicated().sum() == 0
    assert len(df) == 3 * len(pd.date_range("2020-01-01", "2024-03-31"))

    train = df[df["incendio_d1"].notna()].copy()
    train["incendio_d1"] = train["incendio_d1"].astype("int8")
    config.validate_schema(train)
    splits.check_partition_integrity(train)
    assert train["particion"].notna().all()

    assert df.loc[df["id_municipio"] == "27001", "reservado"].eq(1).all()
    assert df.loc[df["id_municipio"] != "27001", "reservado"].eq(0).all()
    marzo = df[(df["fecha"] >= "2020-03-01") & (df["fecha"] <= "2020-03-31")]
    assert marzo["fwi"].isna().all() and df["fwi"].notna().any()      # el hueco de FWI queda como nulo
    assert df.loc[(df["id_municipio"] == "27001") & (df["fecha"] == "2023-12-30"), "incendio_d1"].iloc[0] == 1
    assert df.loc[df["fecha"] > "2023-12-30", "incendio_d1"].isna().all()   # el parte de 2024 no etiqueta nada
    assert df["incendio_d1"].sum() == 2                                      # 15001 (2021-07-09) y 27001 (2023-12-30)


def test_reservados_obligatorios(tmp_path):
    s, csv = _silver(tmp_path, con_csv=False)
    silver = ds.leer_silver(s)
    with pytest.raises(ValueError, match="Falta"):
        ds.leer_reservados(csv, silver["municipio"])


def test_reservado_desde_silver_si_no_hay_csv(tmp_path):
    mun = pd.DataFrame({"id_municipio": ["1", "2"], "provincia": ["x", "x"], "reservado": [0, 1]})
    assert ds.leer_reservados(tmp_path / "no.csv", mun).tolist() == [0, 1]


def test_csv_con_municipios_inexistentes(tmp_path):
    s, _ = _silver(tmp_path)
    csv = tmp_path / "malo.csv"
    pd.DataFrame({"id_municipio": ["99999"], "reservado": [1]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="no existen"):
        ds.leer_reservados(csv, ds.leer_silver(s)["municipio"])


def test_cli_escribe_los_dos_parquet_y_el_informe(tmp_path):
    s, csv = _silver(tmp_path)
    gold = tmp_path / "gold"
    assert ds.main(["--silver-dir", str(s), "--gold-dir", str(gold), "--reservados", str(csv)]) == 0

    train = pd.read_parquet(gold / "dataset_entrenamiento.parquet")
    full = pd.read_parquet(gold / "dataset_municipio_dia.parquet")
    config.validate_schema(train)
    assert len(train) < len(full) and train["incendio_d1"].dtype == "int8"
    inf = json.loads((gold / "gold_informe.json").read_text(encoding="utf-8"))
    assert inf["filas"] == len(full) and inf["filas_etiquetadas"] == len(train)
    assert inf["positivos"] == 2 and set(inf["prevalencia_pct_por_particion"]) == {"entrenamiento", "validacion", "test"}


def test_cli_error_sin_reservados(tmp_path):
    s, csv = _silver(tmp_path, con_csv=False)
    assert ds.main(["--silver-dir", str(s), "--gold-dir", str(tmp_path / "g"), "--reservados", str(csv)]) == 2
