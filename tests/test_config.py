import pytest

from src.ml import config


def test_ninguna_variante_usa_features_prohibidas():
    for v in config.FEATURE_SETS:
        feats = config.get_features(v)
        assert not set(feats) & set(config.FORBIDDEN_FEATURES)


def test_sin_fwi_no_contiene_fwi():
    assert not set(config.get_features("sin_fwi")) & set(config.FWI_FEATURES)
    assert not set(config.get_features("solo_meteo")) & set(config.FWI_FEATURES)


def test_variante_desconocida():
    with pytest.raises(KeyError):
        config.get_features("inventada")


def test_schema_ok(toy_gold):
    config.validate_schema(toy_gold)


def test_schema_columna_faltante(toy_gold):
    with pytest.raises(ValueError, match="Faltan columnas"):
        config.validate_schema(toy_gold.drop(columns=["fwi"]))


def test_schema_duplicados(toy_gold):
    with pytest.raises(ValueError, match="duplicadas"):
        config.validate_schema(__import__("pandas").concat([toy_gold, toy_gold.head(1)]))


def test_schema_target_invalido(toy_gold):
    bad = toy_gold.copy()
    bad.loc[bad.index[0], "incendio_d1"] = 2
    with pytest.raises(ValueError, match="0/1"):
        config.validate_schema(bad)
