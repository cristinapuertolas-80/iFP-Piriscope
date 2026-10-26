import json

import numpy as np
import pytest

from src.ml import config, train
from src.ml.splits import select_train

FAST = {"n_estimators": 30, "max_depth": 3}


@pytest.fixture(scope="module")
def df(toy_gold):
    d = toy_gold.copy()
    # señal débil para que haya algo que aprender
    d["incendio_d1"] = ((d["temperatura"] + np.random.default_rng(0).normal(size=len(d))) > 2.3).astype(int)
    return d


@pytest.mark.parametrize("algo", train.ALGORITHMS)
def test_entrena_y_predice_probabilidades(df, algo):
    b = train.train_model(df, algo, "con_fwi", params=FAST)
    scored = train.score_dataframe(b, df.head(200))
    assert scored["proba"].between(0, 1).all()
    assert b.n_train == len(select_train(df))


@pytest.mark.parametrize("variant", list(config.FEATURE_SETS))
def test_variantes_usan_solo_sus_features(df, variant):
    b = train.train_model(df, "xgboost", variant, params=FAST)
    assert b.features == config.get_features(variant)
    assert not set(b.features) & set(config.FORBIDDEN_FEATURES)
    if variant != "con_fwi":
        assert not set(b.features) & set(config.FWI_FEATURES)


def test_entrena_solo_con_no_reservados_y_periodo_train(df):
    b = train.train_model(df, "random_forest", params={"n_estimators": 10})
    tr = select_train(df)
    assert (tr["reservado"] == 0).all() and (tr["particion"] == "entrenamiento").all()
    assert b.n_pos_train == int(tr["incendio_d1"].sum())


def test_reproducible_con_misma_semilla(df):
    a = train.train_model(df, "xgboost", params=FAST, seed=1)
    b = train.train_model(df, "xgboost", params=FAST, seed=1)
    X = df.head(500)
    assert np.allclose(train.score_dataframe(a, X)["proba"], train.score_dataframe(b, X)["proba"])


def test_guardar_y_cargar(df, tmp_path):
    b = train.train_model(df, "xgboost", "sin_fwi", version="v9", params=FAST)
    path = train.save_model(b, tmp_path)
    assert path.name == "v9_xgboost_sin_fwi.joblib"
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["seed"] == config.SEED and meta["version"] == "v9" and meta["features"] == b.features
    b2 = train.load_model(path)
    X = df.head(300)
    assert np.allclose(train.score_dataframe(b, X)["proba"], train.score_dataframe(b2, X)["proba"])


def test_train_all_variants(df):
    out = train.train_all_variants(df, "xgboost", params=FAST)
    assert set(out) == set(config.FEATURE_SETS)


def test_algoritmo_invalido(df):
    with pytest.raises(ValueError, match="no válido"):
        train.train_model(df, "red_neuronal")


def test_sin_positivos_falla(df):
    d = df.copy()
    d["incendio_d1"] = 0
    with pytest.raises(ValueError, match="dos clases"):
        train.train_model(d, "xgboost")


def test_no_toca_test_ni_reservados_early_stopping(df):
    """El early stopping usa validación de municipios NO reservados."""
    b = train.train_model(df, "xgboost", params={"n_estimators": 200, "max_depth": 3})
    assert b.best_iteration is not None and b.best_iteration < 200


def test_falla_si_esquema_invalido(df):
    with pytest.raises(ValueError, match="Faltan columnas"):
        train.train_model(df.drop(columns=["viento"]))


def test_anio_desde_y_excluir_quedan_en_el_nombre_y_los_metadatos(df):
    b = train.train_model(df, "xgboost", "con_fwi", params=FAST, anio_desde=2020, excluir=["viento", "fwi"])
    assert b.variant == "con_fwi-sin_fwi_viento-desde2020"
    assert "viento" not in b.features and "fwi" not in b.features and "temperatura" in b.features
    assert b.anio_desde == 2020 and set(b.excluidas) == {"viento", "fwi"}
    assert b.n_train < train.train_model(df, "xgboost", "con_fwi", params=FAST).n_train
    assert b.model_id == "v1_xgboost_con_fwi-sin_fwi_viento-desde2020"


def test_excluir_variable_que_no_esta_en_la_variante_no_cambia_el_nombre(df):
    b = train.train_model(df, "xgboost", "sin_fwi", params=FAST, excluir=["fwi"])
    assert b.variant == "sin_fwi" and b.excluidas == []


def test_excluir_desconocida_o_todo_falla(df):
    with pytest.raises(ValueError, match="desconocidas"):
        train.train_model(df, "xgboost", excluir=["inventada"])
    with pytest.raises(ValueError, match="ninguna"):
        train.train_model(df, "xgboost", "solo_meteo", excluir=list(config.METEO_FEATURES))
