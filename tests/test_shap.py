import numpy as np
import pandas as pd
import pytest

from src.explainability import plots, shap_explain as sx
from src.ml import predict, risk, splits, train

FAST = {"n_estimators": 40, "max_depth": 3}


@pytest.fixture(scope="module")
def setup(toy_gold):
    df = toy_gold.copy()
    rng = np.random.default_rng(0)
    df["incendio_d1"] = ((df["temperatura"] + 0.3 * rng.normal(size=len(df))) > 1.8).astype(int)
    bundle = train.train_model(df, "xgboost", "con_fwi", params=FAST)
    val = train.score_dataframe(bundle, splits.select_scenario(df, "A", split="validacion"))
    thr = risk.fit_risk_thresholds(val["incendio_d1"], val["proba"], bundle.model_id)
    pred = predict.build_predictions(df, bundle, thr, desde="2023-12-01", hasta="2023-12-03")
    return df, bundle, pred


def test_shap_aditivo_xgboost(setup):
    df, b, _ = setup
    X = df.sample(300, random_state=0)
    S, base = sx.compute_shap(b, X)
    margin = b.model.predict(X[b.features], output_margin=True)
    assert S.shape == (300, len(b.features))
    assert np.allclose(S.sum(axis=1) + base, margin, atol=1e-3)


def test_shap_aditivo_random_forest(setup):
    df, _, _ = setup
    rf = train.train_model(df, "random_forest", "sin_fwi", params={"n_estimators": 20, "max_depth": 5})
    X = df.sample(200, random_state=0)
    S, base = sx.compute_shap(rf, X)
    proba = rf.model.predict_proba(X[rf.features])[:, 1]
    assert np.allclose(S.sum(axis=1) + base, proba, atol=1e-3)


def test_esquema_y_top5(setup):
    df, b, pred = setup
    e = sx.build_explanations(pred, df, b, top_k=5)
    assert list(e.columns) == sx.EXPL_COLUMNS
    assert set(e["id_prediccion"]) == set(pred["id_prediccion"])
    assert (e.groupby("id_prediccion").size() == 5).all()
    assert set(e["variable"]) <= set(b.features)


def test_ordenado_por_valor_absoluto(setup):
    df, b, pred = setup
    e = sx.build_explanations(pred, df, b, top_k=5)
    for _, g in e.groupby("id_prediccion"):
        a = g["valor_shap"].abs().to_numpy()
        assert (np.diff(a) <= 1e-12).all()


def test_explica_con_la_fila_del_dia_d(setup):
    """La explicación de una predicción D+1 coincide con SHAP calculado a mano sobre la fila D."""
    df, b, pred = setup
    row = pred.iloc[0]
    fila = df[(df["id_municipio"] == row["id_municipio"])
              & (pd.to_datetime(df["fecha"]) == row["fecha_objetivo"] - pd.Timedelta(days=1))]
    S, _ = sx.compute_shap(b, fila)
    esperado = pd.Series(S[0], index=b.features).abs().idxmax()
    e = sx.build_explanations(pred.iloc[[0]], df, b, top_k=5)
    assert e.iloc[0]["variable"] == esperado
    assert e.iloc[0]["valor_shap"] == pytest.approx(dict(zip(b.features, S[0]))[esperado])


def test_top_k_none_devuelve_todas(setup):
    df, b, pred = setup
    e = sx.build_explanations(pred, df, b, top_k=None)
    assert (e.groupby("id_prediccion").size() == len(b.features)).all()


def test_falla_si_falta_fila_del_dia_d(setup):
    df, b, pred = setup
    p = pred.iloc[[0]].copy()
    p["fecha_objetivo"] = pd.Timestamp("2030-01-02")
    with pytest.raises(ValueError, match="no tienen su fila"):
        sx.build_explanations(p, df, b)


def test_falla_con_predicciones_duplicadas(setup):
    df, b, pred = setup
    with pytest.raises(ValueError, match="duplicados"):
        sx.build_explanations(pd.concat([pred.iloc[[0]], pred.iloc[[0]]]), df, b)


def test_importancia_global_detecta_variable_relevante(setup):
    df, b, _ = setup
    imp = sx.global_importance(b, splits.select_scenario(df, "A", split="validacion"))
    assert imp.iloc[0]["variable"] == "temperatura"
    assert imp["importancia_shap"].is_monotonic_decreasing
    imp2 = sx.global_importance(b, df, max_rows=1000, seed=3)
    imp3 = sx.global_importance(b, df, max_rows=1000, seed=3)
    assert imp2.equals(imp3)


def test_write_sobrescribe_y_conserva(setup, tmp_path):
    df, b, pred = setup
    path = tmp_path / "expl.parquet"
    e = sx.build_explanations(pred, df, b)
    sx.write_explanations(e, path)
    e2 = e[e["id_prediccion"] == pred["id_prediccion"].iloc[0]].copy()
    e2["valor_shap"] = 9.0
    sx.write_explanations(e2, path)
    out = pd.read_parquet(path)
    assert len(out) == len(e)
    assert (out[out["id_prediccion"] == pred["id_prediccion"].iloc[0]]["valor_shap"] == 9.0).all()


def test_grafico_local_5_barras_con_signo_en_texto(setup):
    df, b, pred = setup
    e = sx.build_explanations(pred, df, b)
    fig = plots.fig_local(e, pred["id_prediccion"].iloc[0], sx.shap_units(b))
    assert len(fig.data[0].x) == 5
    assert all(("sube" in t) or ("baja" in t) for t in fig.data[0].text)


def test_grafico_local_id_inexistente(setup):
    df, b, pred = setup
    e = sx.build_explanations(pred, df, b)
    with pytest.raises(ValueError):
        plots.fig_local(e, "nope")


def test_grafico_global(setup):
    df, b, _ = setup
    imp = sx.global_importance(b, df.head(2000))
    assert len(plots.fig_global(imp, top=5).data[0].x) == 5
