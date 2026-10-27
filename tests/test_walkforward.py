import numpy as np
import pandas as pd
import pytest

from src.ml import config, walkforward as wf


@pytest.fixture(scope="module")
def dataset(toy_gold, tmp_path_factory):
    df = toy_gold.copy()
    rng = np.random.default_rng(0)
    df["incendio_d1"] = ((df["temperatura"] + 0.3 * rng.normal(size=len(df))) > 1.8).astype(int)
    ruta = tmp_path_factory.mktemp("gold") / "dataset_entrenamiento.parquet"
    df.to_parquet(ruta, index=False)
    return df, ruta


def _args(ruta, tmp_path, *extra):
    return ["--dataset", str(ruta), "--salida", str(tmp_path / "wf.csv"), "--variantes", "con_fwi",
            "--n-estimators", "20", "--bootstrap", "10", *extra]


def test_pliegue_usa_solo_no_reservados_y_anios_hasta_el_pliegue(dataset):
    df, _ = dataset
    d = wf.preparar_pliegue(df, 2021)
    assert (d["reservado"] == 0).all() and d["fecha"].dt.year.max() == 2021
    por = d.groupby("particion")["fecha"].agg(lambda s: (s.dt.year.min(), s.dt.year.max()))
    assert por["entrenamiento"][1] == 2019 and por["validacion"] == (2020, 2020) and por["test"] == (2021, 2021)


def test_cada_pliegue_entrena_solo_con_el_pasado(dataset):
    df, _ = dataset
    from src.ml import splits
    d = wf.preparar_pliegue(df, 2022)
    assert splits.select_train(d)["fecha"].dt.year.max() == 2020
    splits.check_partition_integrity(d)


def test_cli_informe_por_anio_y_veredicto(dataset, tmp_path, capsys):
    _, ruta = dataset
    assert wf.main(_args(ruta, tmp_path, "--desde-fold", "2021", "--hasta-fold", "2022")) == 0
    inf = pd.read_csv(tmp_path / "wf.csv")
    assert set(inf["anio"]) == {2021, 2022} and set(inf["sistema"]) == {"modelo", "baseline_fwi"}
    res = wf.resumen_pliegues(inf)
    assert len(res) == 2 and {"auc_pr_modelo", "auc_pr_fwi", "dif_auc_pr"} <= set(res.columns)
    v = wf.veredicto(res)
    assert v["anios"].iloc[0] == 2 and v["anios_modelo_mejor"].iloc[0] <= 2
    assert "anios_modelo_mejor" in capsys.readouterr().out


def test_no_toca_el_anio_de_test(dataset, tmp_path, capsys):
    _, ruta = dataset
    assert wf.main(_args(ruta, tmp_path, "--desde-fold", "2022", "--hasta-fold", "2023")) == 2
    assert "test" in capsys.readouterr().err


def test_error_si_falta_el_dataset(tmp_path):
    assert wf.main(["--dataset", str(tmp_path / "nada.parquet")]) == 2


def test_veredicto_cuenta_anios():
    res = pd.DataFrame({"variante": ["a"] * 4, "anio": [1, 2, 3, 4], "n_pos": 10, "auc_pr_modelo": 0.1,
                        "auc_pr_fwi": 0.1, "dif_auc_pr": [0.01, -0.01, 0.02, 0.03],
                        "dif_lo": [0.001, -0.02, -0.01, 0.005], "dif_hi": 0.05})
    v = wf.veredicto(res).iloc[0]
    assert (v["anios"], v["anios_modelo_mejor"], v["anios_ic_excluye_0"]) == (4, 3, 2)
