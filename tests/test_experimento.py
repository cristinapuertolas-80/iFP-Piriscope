import numpy as np
import pandas as pd
import pytest

from src.ml import config, experimento as ex


@pytest.fixture(scope="module")
def dataset(toy_gold, tmp_path_factory):
    df = toy_gold.copy()
    rng = np.random.default_rng(0)
    df["incendio_d1"] = ((df["temperatura"] + 0.3 * rng.normal(size=len(df))) > 1.8).astype(int)
    ruta = tmp_path_factory.mktemp("gold") / "dataset_entrenamiento.parquet"
    df.to_parquet(ruta, index=False)
    return ruta


def _args(dataset, tmp_path, *extra):
    return ["--dataset", str(dataset), "--salida", str(tmp_path / "informe.csv"),
            "--variantes", "con_fwi", "sin_fwi", "--n-estimators", "30", "--bootstrap", "10", *extra]


def test_desarrollo_solo_valida_en_validacion_de_no_reservados(dataset, tmp_path, capsys):
    assert ex.main(_args(dataset, tmp_path)) == 0
    inf = pd.read_csv(tmp_path / "informe.csv")
    assert set(inf["escenario"]) == {"A-validacion"}
    assert set(inf["variante"]) == {"con_fwi", "sin_fwi"}
    assert set(inf["grupo"]) == {"global"}
    assert "DESARROLLO" in capsys.readouterr().out


def test_final_incluye_a_b_c_y_provincias(dataset, tmp_path):
    assert ex.main(_args(dataset, tmp_path, "--final")) == 0
    inf = pd.read_csv(tmp_path / "informe.csv")
    assert set(inf["escenario"]) == {"A-test", "B", "C-test"}
    assert {"A Coruña", "global"} <= set(inf["grupo"])
    assert set(inf["sistema"]) == {"modelo", "baseline_fwi"}


def test_modelo_y_baseline_usan_las_mismas_filas(dataset, tmp_path):
    ex.main(_args(dataset, tmp_path))
    g = pd.read_csv(tmp_path / "informe.csv").query("grupo == 'global'")
    assert (g.groupby(["escenario", "variante"])["n"].nunique() == 1).all()


def test_guardar_escribe_modelo_y_umbrales(dataset, tmp_path):
    ex.main(_args(dataset, tmp_path, "--guardar", "--models-dir", str(tmp_path / "models")))
    nombres = {p.name for p in (tmp_path / "models").iterdir()}
    assert {"v1_xgboost_con_fwi.joblib", "v1_xgboost_con_fwi_umbrales.json"} <= nombres


def test_resumen_global_columnas(dataset, tmp_path):
    ex.main(_args(dataset, tmp_path))
    r = ex.resumen_global(pd.read_csv(tmp_path / "informe.csv"))
    assert {"auc_pr_modelo", "auc_pr_fwi", "precision_modelo", "precision_fwi"} <= set(r.columns)
    assert len(r) == 2


def test_error_si_falta_el_dataset(tmp_path, capsys):
    assert ex.main(["--dataset", str(tmp_path / "nada.parquet")]) == 2
    assert "src.features" in capsys.readouterr().err


def test_error_si_el_dataset_no_cumple_el_contrato(dataset, tmp_path):
    mal = pd.read_parquet(dataset).drop(columns=["viento"])
    ruta = tmp_path / "mal.parquet"
    mal.to_parquet(ruta, index=False)
    assert ex.main(["--dataset", str(ruta)]) == 2


def test_anio_desde_y_excluir_en_el_informe(dataset, tmp_path):
    assert ex.main(_args(dataset, tmp_path, "--anio-desde", "2019", "--excluir", "viento")) == 0
    inf = pd.read_csv(tmp_path / "informe.csv")
    assert set(inf["variante"]) == {"con_fwi-sin_viento-desde2019", "sin_fwi-sin_viento-desde2019"}


def test_informe_incluye_la_diferencia_pareada_con_el_fwi(dataset, tmp_path):
    ex.main(_args(dataset, tmp_path))
    inf = pd.read_csv(tmp_path / "informe.csv")
    fila = inf[(inf["sistema"] == "modelo") & (inf["grupo"] == "global")]
    assert fila[["dif_auc_pr", "dif_lo", "dif_hi", "p_mejora"]].notna().all().all()
    assert (fila["dif_lo"] <= fila["dif_auc_pr"]).all() and (fila["dif_auc_pr"] <= fila["dif_hi"]).all()
    assert inf[inf["sistema"] == "baseline_fwi"]["dif_auc_pr"].isna().all()
    assert {"dif_auc_pr", "p_mejora"} <= set(ex.resumen_global(inf).columns)


def test_param_llega_al_modelo_guardado(dataset, tmp_path):
    import json
    ex.main(_args(dataset, tmp_path, "--param", "max_depth=2", "learning_rate=0.1", "--guardar",
                  "--models-dir", str(tmp_path / "m")))
    meta = json.loads((tmp_path / "m" / "v1_xgboost_con_fwi.json").read_text(encoding="utf-8"))
    assert meta["params"]["max_depth"] == 2 and meta["params"]["learning_rate"] == 0.1


def test_param_mal_formado(dataset, tmp_path):
    assert ex.main(_args(dataset, tmp_path, "--param", "max_depth")) == 2
    assert ex.parsear_params(["a=1", "b=0.5", "c=hola"]) == {"a": 1, "b": 0.5, "c": "hola"}


def test_bloque_dias_se_acepta_y_cero_es_por_filas(dataset, tmp_path):
    assert ex.main(_args(dataset, tmp_path, "--bloque-dias", "14")) == 0
    assert ex.main(_args(dataset, tmp_path, "--bloque-dias", "0")) == 0


def test_con_bloques():
    d = pd.DataFrame({"fecha": pd.to_datetime(["2022-01-01", "2022-01-07", "2022-01-08", "2022-01-20"])})
    out, col = ex.con_bloques(d, 7)
    assert out[col].tolist() == [0, 0, 1, 2]
    assert ex.con_bloques(d, 0)[1] is None
