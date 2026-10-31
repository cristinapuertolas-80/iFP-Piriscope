import warnings

import numpy as np
import pandas as pd
import pytest

from src.explainability import plots, shap_explain
from src.ml import config, inferir, predict, risk, splits, train


@pytest.fixture(scope="module")
def caso(toy_gold):
    df = toy_gold.copy()
    rng = np.random.default_rng(0)
    df["incendio_d1"] = ((df["temperatura"] + 0.3 * rng.normal(size=len(df))) > 1.8).astype(int)
    bundle = train.train_model(df, "xgboost", "con_fwi", params={"n_estimators": 20, "max_depth": 3},
                               anio_desde=2019, excluir=["viento"])
    val = train.score_dataframe(bundle, splits.select_scenario(df, "A", split="validacion"))
    thr = risk.fit_risk_thresholds(val[config.TARGET], val["proba"], bundle.model_id)
    fecha = pd.to_datetime(df[config.COL_FECHA]).dt.normalize().max()
    dentro, _ = inferir.separar_por_dominio(df, bundle, fecha, fecha)
    pred = predict.build_predictions(dentro, bundle, thr, desde=fecha, hasta=fecha)
    expl = shap_explain.build_explanations(pred, df, bundle, top_k=5)
    return {"df": df, "bundle": bundle, "val": val, "pred": pred, "expl": expl}


def test_columnas_ampliadas_y_cinco_por_prediccion(caso):
    e = caso["expl"]
    assert list(e.columns) == shap_explain.EXPL_COLUMNS_EXT
    assert (e.groupby("id_prediccion").size() == 5).all()
    assert set(e["id_prediccion"]) == set(caso["pred"]["id_prediccion"])


def test_esquema_base_sin_valores(caso):
    e = shap_explain.build_explanations(caso["pred"], caso["df"], caso["bundle"], con_valores=False)
    assert list(e.columns) == shap_explain.EXPL_COLUMNS


def test_valor_variable_es_el_del_dia_d(caso):
    df, e, pred = caso["df"], caso["expl"], caso["pred"]
    fila = pred.iloc[0]
    dia_d = pd.Timestamp(fila["fecha_objetivo"]).normalize() - pd.Timedelta(days=1)
    f = pd.to_datetime(df[config.COL_FECHA]).dt.normalize()
    reg = df[(df[config.COL_MUNICIPIO].astype(str) == str(fila["id_municipio"])) & (f == dia_d)].iloc[0]
    for _, r in e[e["id_prediccion"] == fila["id_prediccion"]].iterrows():
        esperado = reg[r["variable"]]
        assert (pd.isna(esperado) and pd.isna(r["valor_variable"])) or np.isclose(esperado, r["valor_variable"])


def test_reconstruye_la_probabilidad_del_modelo(caso):
    e, pred = caso["expl"], caso["pred"]
    g = e.groupby("id_prediccion").agg(base=("valor_base", "first"), top=("valor_shap", "sum"),
                                       resto=("shap_resto", "first"))
    prob = 1 / (1 + np.exp(-(g["base"] + g["top"] + g["resto"])))
    esperado = pred.set_index("id_prediccion")["probabilidad"].loc[g.index]
    assert np.allclose(prob.to_numpy(), esperado.to_numpy(), atol=5e-4)


def test_top_k_none_deja_el_resto_a_cero(caso):
    e = shap_explain.build_explanations(caso["pred"], caso["df"], caso["bundle"], top_k=None)
    assert np.allclose(e["shap_resto"], 0, atol=1e-5)


def test_top_k_invalido(caso):
    with pytest.raises(ValueError):
        shap_explain.build_explanations(caso["pred"], caso["df"], caso["bundle"], top_k=0)


def test_verificar_no_avisa_con_el_modelo_correcto(caso):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        shap_explain.build_explanations(caso["pred"], caso["df"], caso["bundle"], verificar=True)
    assert not [x for x in w if "aditivos" in str(x.message)]


def test_importancia_por_grupos(caso):
    b, val = caso["bundle"], caso["val"]
    g = shap_explain.global_importance(b, val, por_grupo=True)
    ind = shap_explain.global_importance(b, val).set_index("variable")["importancia_shap"]
    assert "Índices FWI" in set(g["variable"]) and "ffmc" not in set(g["variable"])
    assert g["importancia_shap"].is_monotonic_decreasing and (g["importancia_shap"] >= 0).all()
    miembros = [f for f in shap_explain.GRUPOS["Índices FWI"] if f in ind.index]
    fwi = g.set_index("variable").loc["Índices FWI", "importancia_shap"]
    assert fwi <= ind.loc[miembros].sum() + 1e-9          # desigualdad triangular


def test_escribir_sobre_un_fichero_anterior_a_la_ampliacion(caso, tmp_path):
    e, ruta = caso["expl"], tmp_path / "e.parquet"
    e.iloc[:5][shap_explain.EXPL_COLUMNS].assign(id_prediccion="viejo").to_parquet(ruta, index=False)
    shap_explain.write_explanations(e, ruta)
    out = pd.read_parquet(ruta)
    assert list(out.columns) == shap_explain.EXPL_COLUMNS_EXT
    viejo = out[out["id_prediccion"] == "viejo"]
    assert len(viejo) == 5 and viejo["valor_base"].isna().all()


def test_figura_local_por_defecto_como_antes(caso):
    e = caso["expl"]
    id0 = e["id_prediccion"].iloc[0]
    f = plots.fig_local(e, id0)
    esperadas = set(e.loc[e["id_prediccion"] == id0, "variable"])
    assert len(f.data[0].y) == 5 and set(f.data[0].y) == esperadas
    assert all(("sube" in t) or ("baja" in t) for t in f.data[0].text)


def test_figura_local_con_valores_resto_y_titulo(caso):
    e = caso["expl"]
    id0 = e["id_prediccion"].iloc[0]
    f = plots.fig_local(e, id0, titulo="T", etiquetas=plots.ETIQUETAS, mostrar_valores=True, incluir_resto=True)
    assert len(f.data[0].y) == 6 and f.data[0].y[0] == "Resto de variables"
    assert f.layout.title.text == "T" and all(" = " in y for y in f.data[0].y[1:])


def test_cascada_termina_en_la_probabilidad(caso):
    e, pred = caso["expl"], caso["pred"]
    id0 = e["id_prediccion"].iloc[0]
    w = plots.fig_cascada(e, id0).data[0]
    assert list(w.measure) == ["absolute"] + ["relative"] * 6 + ["total"]
    prob = 1 / (1 + np.exp(-sum(w.y[:-1])))
    assert np.isclose(prob, pred.set_index("id_prediccion").loc[id0, "probabilidad"], atol=5e-4)


def test_cascada_exige_las_columnas_ampliadas(caso):
    e = caso["expl"]
    with pytest.raises(ValueError, match="DEC-50"):
        plots.fig_cascada(e[shap_explain.EXPL_COLUMNS], e["id_prediccion"].iloc[0])
