import pytest

from src.ml import splits


def test_train_solo_no_reservados_y_entrenamiento(toy_gold):
    tr = splits.select_train(toy_gold)
    assert (tr["reservado"] == 0).all() and (tr["particion"] == "entrenamiento").all()


def test_escenarios_no_se_solapan_con_train(toy_gold):
    tr_idx = set(splits.select_train(toy_gold).index)
    for s in splits.SCENARIOS:
        assert not tr_idx & set(splits.select_scenario(toy_gold, s).index)


def test_escenarios_a_b_c_condiciones(toy_gold):
    a = splits.select_scenario(toy_gold, "A")
    b = splits.select_scenario(toy_gold, "B")
    c = splits.select_scenario(toy_gold, "C")
    assert (a["reservado"] == 0).all() and a["particion"].isin(["validacion", "test"]).all()
    assert (b["reservado"] == 1).all() and (b["particion"] == "entrenamiento").all()
    assert (c["reservado"] == 1).all() and c["particion"].isin(["validacion", "test"]).all()


def test_split_validacion_y_test(toy_gold):
    v = splits.select_scenario(toy_gold, "A", split="validacion")
    t = splits.select_scenario(toy_gold, "A", split="test")
    assert (v["particion"] == "validacion").all() and (t["particion"] == "test").all()


def test_b_no_admite_split(toy_gold):
    with pytest.raises(ValueError):
        splits.select_scenario(toy_gold, "B", split="test")


def test_escenario_invalido(toy_gold):
    with pytest.raises(ValueError):
        splits.select_scenario(toy_gold, "Z")


def test_integridad_ok(toy_gold):
    splits.check_partition_integrity(toy_gold)


def test_integridad_detecta_solape_temporal(toy_gold):
    bad = toy_gold.copy()
    bad.loc[bad["fecha"] == "2022-06-01", "particion"] = "entrenamiento"
    with pytest.raises(ValueError, match="solapa"):
        splits.check_partition_integrity(bad)


def test_integridad_detecta_reservado_inconsistente(toy_gold):
    bad = toy_gold.copy()
    idx = bad.index[0]
    bad.loc[idx, "reservado"] = 1  # un municipio no reservado con una fila marcada
    with pytest.raises(ValueError):
        splits.check_partition_integrity(bad)


def test_select_train_anio_desde(toy_gold):
    todo = splits.select_train(toy_gold)
    recorte = splits.select_train(toy_gold, anio_desde=2020)
    assert 0 < len(recorte) < len(todo)
    assert (recorte["fecha"].dt.year >= 2020).all()
