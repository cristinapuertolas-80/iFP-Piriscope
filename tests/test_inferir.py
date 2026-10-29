import numpy as np
import pandas as pd
import pytest

from src.ml import config, inferir, risk, splits, train


@pytest.fixture(scope="module")
def entorno(toy_gold, tmp_path_factory):
    base = tmp_path_factory.mktemp("inf")
    df = toy_gold.copy()
    rng = np.random.default_rng(0)
    df["incendio_d1"] = ((df["temperatura"] + 0.3 * rng.normal(size=len(df))) > 1.8).astype(int)
    bundle = train.train_model(df, "xgboost", "con_fwi", params={"n_estimators": 20, "max_depth": 3},
                               anio_desde=2019, excluir=["viento"])
    val = train.score_dataframe(bundle, splits.select_scenario(df, "A", split="validacion"))
    thr = risk.fit_risk_thresholds(val[config.TARGET], val["proba"], bundle.model_id)
    models = base / "models"
    train.save_model(bundle, models)
    risk.save_thresholds(thr, models)
    ruta = base / "dataset_municipio_dia.parquet"
    df.to_parquet(ruta, index=False)
    return {"base": base, "models": models, "dataset": ruta, "id": bundle.model_id, "df": df}


def _args(e, salida, *extra):
    return ["--models-dir", str(e["models"]), "--dataset", str(e["dataset"]),
            "--salida-predicciones", str(salida / "p.parquet"), "--salida-explicaciones", str(salida / "e.parquet"), *extra]


def test_ultimo_dia_por_defecto_y_modelo_unico(entorno, tmp_path, capsys):
    assert inferir.main(_args(entorno, tmp_path)) == 0
    pred, expl = pd.read_parquet(tmp_path / "p.parquet"), pd.read_parquet(tmp_path / "e.parquet")
    assert len(pred) == entorno["df"]["id_municipio"].nunique()
    assert (pd.to_datetime(pred["fecha_objetivo"]) == entorno["df"]["fecha"].max() + pd.Timedelta(days=1)).all()
    assert (expl.groupby("id_prediccion").size() == 5).all() and set(expl["id_prediccion"]) == set(pred["id_prediccion"])
    assert entorno["id"] in capsys.readouterr().out


def test_rango_de_fechas_y_top_k(entorno, tmp_path):
    assert inferir.main(_args(entorno, tmp_path, "--desde", "2023-12-01", "--hasta", "2023-12-03", "--top-k", "3")) == 0
    pred, expl = pd.read_parquet(tmp_path / "p.parquet"), pd.read_parquet(tmp_path / "e.parquet")
    assert len(pred) == 3 * entorno["df"]["id_municipio"].nunique()
    assert (expl.groupby("id_prediccion").size() == 3).all()


def test_figuras_de_ejemplo(entorno, tmp_path):
    ej = tmp_path / "ej"
    assert inferir.main(_args(entorno, tmp_path, "--ejemplo-dir", str(ej))) == 0
    nombres = {p.name for p in ej.iterdir()}
    assert "shap_global.html" in nombres and "shap_global.csv" in nombres
    assert any(n.startswith("shap_local_") and n.endswith(".html") for n in nombres)
    imp = pd.read_csv(ej / "shap_global.csv")
    assert "viento" not in set(imp["variable"])                    # el modelo se entrenó sin viento
    assert imp["importancia_shap"].is_monotonic_decreasing


def test_municipio_de_ejemplo_concreto(entorno, tmp_path):
    mun = entorno["df"]["id_municipio"].iloc[0]
    assert inferir.main(_args(entorno, tmp_path, "--ejemplo-dir", str(tmp_path / "ej"), "--ejemplo-municipio", mun)) == 0
    assert any(mun in p.name for p in (tmp_path / "ej").glob("shap_local_*.html"))
    assert inferir.main(_args(entorno, tmp_path, "--ejemplo-dir", str(tmp_path / "ej"), "--ejemplo-municipio", "99999")) == 2


def test_errores_claros(entorno, tmp_path, capsys):
    assert inferir.main(["--models-dir", str(tmp_path / "vacio"), "--dataset", str(entorno["dataset"])]) == 2
    assert "Indica --modelo" in capsys.readouterr().err
    assert inferir.main(_args(entorno, tmp_path, "--modelo", "no_existe")) == 2
    assert "Disponibles" in capsys.readouterr().err
    assert inferir.main(["--models-dir", str(entorno["models"]), "--dataset", str(tmp_path / "nada.parquet")]) == 2
    assert inferir.main(_args(entorno, tmp_path, "--desde", "2030-01-01", "--hasta", "2030-01-02")) == 2


def test_varios_modelos_exige_elegir(entorno, tmp_path):
    otra = tmp_path / "models"
    b = train.train_model(entorno["df"], "xgboost", "sin_fwi", params={"n_estimators": 10})
    for carpeta in (entorno["models"], otra):
        pass
    import shutil
    shutil.copytree(entorno["models"], otra)
    train.save_model(b, otra)
    risk.save_thresholds(risk.RiskThresholds(0.1, 0.2, 0.8, 0.5, b.model_id, 10, 2), otra)
    args = ["--models-dir", str(otra), "--dataset", str(entorno["dataset"]),
            "--salida-predicciones", str(tmp_path / "p.parquet"), "--salida-explicaciones", str(tmp_path / "e.parquet")]
    assert inferir.main(args) == 2
    assert inferir.main(args + ["--modelo", entorno["id"]]) == 0


# --- Dominio del modelo: variables que nunca faltaron al entrenar -----------------------------------
FWI_NULO_DESDE = pd.Timestamp("2023-12-30")


@pytest.fixture
def entorno_fwi_nulo(entorno, tmp_path):
    """Dataset con el FWI vacío en los últimos días y un modelo de respaldo sin FWI guardado."""
    df = entorno["df"].copy()
    df.loc[df["fecha"] >= FWI_NULO_DESDE, config.FWI_FEATURES] = np.nan
    ruta = tmp_path / "dataset_municipio_dia.parquet"
    df.to_parquet(ruta, index=False)
    models = tmp_path / "models"
    import shutil
    shutil.copytree(entorno["models"], models)
    b = train.train_model(entorno["df"], "xgboost", "sin_fwi", params={"n_estimators": 20, "max_depth": 3},
                          anio_desde=2019, excluir=["viento"])
    val = train.score_dataframe(b, splits.select_scenario(entorno["df"], "A", split="validacion"))
    train.save_model(b, models)
    risk.save_thresholds(risk.fit_risk_thresholds(val[config.TARGET], val["proba"], b.model_id), models)
    return {"dataset": ruta, "models": models, "principal": entorno["id"], "respaldo": b.model_id, "df": df}


def _a(e, tmp_path, *extra):
    return ["--models-dir", str(e["models"]), "--dataset", str(e["dataset"]), "--modelo", e["principal"],
            "--salida-predicciones", str(tmp_path / "p.parquet"), "--salida-explicaciones", str(tmp_path / "e.parquet"), *extra]


def test_features_sin_nulos_detecta_las_que_nunca_faltaron(entorno):
    df = entorno["df"].copy()
    df.loc[df["particion"] == "entrenamiento", "temperatura"] = np.where(
        np.arange((df["particion"] == "entrenamiento").sum()) % 7 == 0, np.nan, 1.0)
    bundle, _ = inferir.cargar(entorno["models"], entorno["id"])
    dom = inferir.features_sin_nulos(bundle, df)
    assert "temperatura" not in dom and "fwi" in dom and "mes" in dom


def test_sin_respaldo_las_filas_sin_fwi_no_se_predicen(entorno_fwi_nulo, tmp_path, capsys):
    e = entorno_fwi_nulo
    assert inferir.main(_a(e, tmp_path, "--desde", "2023-12-28", "--hasta", "2023-12-31")) == 0
    pred = pd.read_parquet(tmp_path / "p.parquet")
    assert pd.to_datetime(pred["fecha_objetivo"]).max() == pd.Timestamp("2023-12-30")     # D=29 es el último con FWI
    assert pred["probabilidad"].notna().all() and (pred["version_modelo"] == "v1").all()
    out = capsys.readouterr().out
    assert "sin predicción" in out and "0 con el modelo de respaldo" in out


def test_con_respaldo_se_predicen_con_el_modelo_sin_fwi(entorno_fwi_nulo, tmp_path):
    e = entorno_fwi_nulo
    assert inferir.main(_a(e, tmp_path, "--desde", "2023-12-28", "--hasta", "2023-12-31",
                           "--modelo-respaldo", e["respaldo"])) == 0
    pred, expl = pd.read_parquet(tmp_path / "p.parquet"), pd.read_parquet(tmp_path / "e.parquet")
    n_mun = e["df"]["id_municipio"].nunique()
    assert len(pred) == 4 * n_mun and pred["id_prediccion"].is_unique and pred["probabilidad"].notna().all()
    resp = pred[pred["version_modelo"] == "v1-respaldo"]
    assert len(resp) == 2 * n_mun and (pd.to_datetime(resp["fecha_objetivo"]) >= FWI_NULO_DESDE + pd.Timedelta(days=1)).all()
    v_resp = expl[expl["id_prediccion"].isin(resp["id_prediccion"])]["variable"]
    assert not set(v_resp) & set(config.FWI_FEATURES)          # el respaldo no usa el FWI
    assert set(expl["id_prediccion"]) == set(pred["id_prediccion"])


def test_ultimo_dia_fuera_de_dominio_falla_con_mensaje_claro(entorno_fwi_nulo, tmp_path, capsys):
    e = entorno_fwi_nulo
    assert inferir.main(_a(e, tmp_path)) == 2                      # el último día no tiene FWI
    assert "modelo-respaldo" in capsys.readouterr().err
    assert inferir.main(_a(e, tmp_path, "--modelo-respaldo", e["respaldo"])) == 0
    assert (pd.read_parquet(tmp_path / "p.parquet")["version_modelo"] == "v1-respaldo").all()


def test_reemplazar_borra_las_salidas_anteriores(entorno, tmp_path):
    base = ["--models-dir", str(entorno["models"]), "--dataset", str(entorno["dataset"]),
            "--salida-predicciones", str(tmp_path / "p.parquet"), "--salida-explicaciones", str(tmp_path / "e.parquet")]
    assert inferir.main(base + ["--desde", "2023-12-01", "--hasta", "2023-12-02"]) == 0
    assert inferir.main(base + ["--desde", "2023-12-10", "--hasta", "2023-12-10"]) == 0
    assert len(pd.read_parquet(tmp_path / "p.parquet")) == 3 * entorno["df"]["id_municipio"].nunique()
    assert inferir.main(base + ["--desde", "2023-12-10", "--hasta", "2023-12-10", "--reemplazar"]) == 0
    assert len(pd.read_parquet(tmp_path / "p.parquet")) == entorno["df"]["id_municipio"].nunique()


def test_con_informacion_ignora_el_mes():
    class B:
        features = ["temperatura", "humedad_relativa", "precipitacion", "mes"]
    df = pd.DataFrame({"temperatura": [np.nan, 1.0, 1.0], "humedad_relativa": [np.nan, np.nan, 2.0],
                       "precipitacion": [np.nan, np.nan, 0.0], "mes": [10, 10, 10]})
    assert inferir.con_informacion(df, B(), 1).tolist() == [False, True, True]
    assert inferir.con_informacion(df, B(), 2).tolist() == [False, False, True]


def test_filas_sin_meteorologia_no_se_predicen_ni_con_respaldo(entorno_fwi_nulo, tmp_path, capsys):
    e = entorno_fwi_nulo
    df = pd.read_parquet(e["dataset"])
    df.loc[df["fecha"] == "2023-12-31", config.METEO_FEATURES + ["racha_dias_secos", "precip_acum_7d"]] = np.nan
    ruta = tmp_path / "sin_meteo.parquet"
    df.to_parquet(ruta, index=False)
    args = ["--models-dir", str(e["models"]), "--dataset", str(ruta), "--modelo", e["principal"],
            "--modelo-respaldo", e["respaldo"], "--desde", "2023-12-28", "--hasta", "2023-12-31",
            "--salida-predicciones", str(tmp_path / "p.parquet"), "--salida-explicaciones", str(tmp_path / "e.parquet")]
    assert inferir.main(args) == 0
    pred = pd.read_parquet(tmp_path / "p.parquet")
    # D=31/12 no tiene FWI ni meteorología: sin predicción. D=30/12 (sin FWI, con meteorología): respaldo
    assert pd.to_datetime(pred["fecha_objetivo"]).max() == pd.Timestamp("2023-12-31")
    n = e["df"]["id_municipio"].nunique()
    assert len(pred[pd.to_datetime(pred["fecha_objetivo"]) == "2023-12-31"]) == n
    assert pred["probabilidad"].notna().all()
    assert f"{n} sin predicción" in capsys.readouterr().out.replace(",", "").replace(".", "")


def test_min_variables_mas_estricto_omite_filas_poco_informadas(entorno, tmp_path):
    df = entorno["df"].copy()
    rng = np.random.default_rng(1)
    for c in ["temperatura", "humedad_relativa", "precipitacion"]:          # como en los datos reales: huecos al entrenar
        df.loc[rng.random(len(df)) < 0.1, c] = np.nan
    b = train.train_model(df, "xgboost", "con_fwi", params={"n_estimators": 20, "max_depth": 3},
                          anio_desde=2019, excluir=["viento"])
    val = train.score_dataframe(b, splits.select_scenario(df, "A", split="validacion"))
    models = tmp_path / "m"
    train.save_model(b, models)
    risk.save_thresholds(risk.fit_risk_thresholds(val[config.TARGET], val["proba"], b.model_id), models)
    df.loc[df["fecha"] == "2023-12-31", ["temperatura", "humedad_relativa", "precipitacion"]] = np.nan
    ruta = tmp_path / "poca_info.parquet"
    df.to_parquet(ruta, index=False)
    base = ["--models-dir", str(models), "--dataset", str(ruta), "--desde", "2023-12-31", "--hasta", "2023-12-31",
            "--salida-predicciones", str(tmp_path / "p.parquet"), "--salida-explicaciones", str(tmp_path / "e.parquet")]
    assert inferir.main(base) == 0                                   # quedan FWI y derivadas con dato
    assert inferir.main(base + ["--min-variables", "20"]) == 2       # ninguna fila llega a 20 variables

def test_reejecutar_no_duplica(entorno, tmp_path):
    a = _args(entorno, tmp_path, "--desde", "2023-12-01", "--hasta", "2023-12-02")
    assert inferir.main(a) == 0 and inferir.main(a) == 0
    pred = pd.read_parquet(tmp_path / "p.parquet")
    assert not pred.duplicated(["id_municipio", "fecha_objetivo"]).any()
    assert len(pred) == 2 * entorno["df"]["id_municipio"].nunique()


def test_v1_sustituye_a_v1_respaldo(entorno, entorno_fwi_nulo, tmp_path):
    e = entorno_fwi_nulo
    rango = ["--desde", "2023-12-29", "--hasta", "2023-12-31"]
    assert inferir.main(_a(e, tmp_path, *rango, "--modelo-respaldo", e["respaldo"])) == 0
    assert (pd.read_parquet(tmp_path / "p.parquet")["version_modelo"] == "v1-respaldo").any()
    e2 = {**e, "dataset": entorno["dataset"]}            # llega el FWI
    assert inferir.main(_a(e2, tmp_path, *rango, "--modelo-respaldo", e["respaldo"])) == 0
    pred, expl = pd.read_parquet(tmp_path / "p.parquet"), pd.read_parquet(tmp_path / "e.parquet")
    assert not pred.duplicated(["id_municipio", "fecha_objetivo"]).any()
    assert (pred["version_modelo"] == "v1").all()
    assert set(expl["id_prediccion"]) == set(pred["id_prediccion"])


def test_error_en_ejemplo_no_bloquea_salidas(entorno, tmp_path):
    assert inferir.main(_args(entorno, tmp_path, "--ejemplo-dir", str(tmp_path / "ej"),
                              "--ejemplo-municipio", "99999")) == 2
    assert (tmp_path / "p.parquet").exists() and (tmp_path / "e.parquet").exists()