import numpy as np
import pytest

from src.ml import evaluate as ev


def _scored(toy_gold, seed=0):
    """Escenario con un 'modelo' informativo y un FWI de ruido."""
    rng = np.random.default_rng(seed)
    d = toy_gold[toy_gold["particion"] == "test"].copy()
    d["proba"] = d["incendio_d1"] * 0.5 + rng.random(len(d))   # informativo
    d["fwi"] = rng.random(len(d))                              # ruido
    return d


def test_auc_pr_perfecto():
    assert ev.auc_pr([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


def test_auc_pr_sin_positivos_es_nan():
    assert np.isnan(ev.auc_pr([0, 0, 0], [0.1, 0.2, 0.3]))


def test_threshold_for_recall_alcanza_objetivo():
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.1).astype(int)
    s = y * 0.3 + rng.random(2000)
    thr = ev.threshold_for_recall(y, s, 0.8)
    assert ev.metrics(y, s, thr)["recall"] >= 0.8


def test_threshold_sin_positivos_falla():
    with pytest.raises(ValueError):
        ev.threshold_for_recall([0, 0], [0.1, 0.2])


def test_metrics_valores_conocidos():
    m = ev.metrics([1, 1, 0, 0], [0.9, 0.2, 0.8, 0.1], threshold=0.5)
    assert m["recall"] == 0.5 and m["precision"] == 0.5 and m["n_pos"] == 2


def test_bootstrap_intervalo_ordenado_y_reproducible(toy_gold):
    d = _scored(toy_gold)
    y, s = d["incendio_d1"].to_numpy(), d["proba"].to_numpy()
    thr = ev.threshold_for_recall(y, s, 0.8)
    a = ev.bootstrap_ci(y, s, thr, n_boot=100, seed=7)
    b = ev.bootstrap_ci(y, s, thr, n_boot=100, seed=7)
    assert a == b
    assert a["auc_pr"][0] <= a["auc_pr"][1] and a["recall"][0] <= a["recall"][1]


def test_compare_mismas_filas_y_descarta_nan_fwi(toy_gold):
    d = _scored(toy_gold)
    d.loc[d.index[:500], "fwi"] = np.nan
    thr = ev.thresholds_from_validation(d, target_recall=0.8)
    rep = ev.compare_model_vs_baseline(d, thr, n_boot=20)
    glob = rep[rep["grupo"] == "global"]
    assert glob["n"].nunique() == 1 and glob["n"].iloc[0] == len(d) - 500
    assert glob["n_pos"].nunique() == 1


def test_compare_desagrega_por_provincia(toy_gold):
    d = _scored(toy_gold)
    thr = ev.thresholds_from_validation(d)
    rep = ev.compare_model_vs_baseline(d, thr, n_boot=20)
    assert set(rep["grupo"]) == {"global", "A Coruña", "Lugo", "Ourense", "Pontevedra"}
    assert set(rep["sistema"]) == {"modelo", "baseline_fwi"}


def test_modelo_informativo_supera_baseline_ruido(toy_gold):
    d = _scored(toy_gold)
    thr = ev.thresholds_from_validation(d)
    rep = ev.compare_model_vs_baseline(d, thr, n_boot=20, by_province=False)
    auc = rep.set_index("sistema")["auc_pr"]
    assert auc["modelo"] > auc["baseline_fwi"]


def test_compare_columnas_faltantes(toy_gold):
    with pytest.raises(ValueError, match="Faltan columnas"):
        ev.compare_model_vs_baseline(toy_gold, {"modelo": 0, "baseline_fwi": 0})


def test_compare_umbrales_incompletos(toy_gold):
    d = _scored(toy_gold)
    with pytest.raises(ValueError, match="thresholds"):
        ev.compare_model_vs_baseline(d, {"modelo": 0.5})


def test_diferencia_auc_pr_identica_es_cero():
    rng = np.random.default_rng(0)
    y = (rng.random(3000) < 0.05).astype(int)
    s = rng.random(3000)
    d = ev.diferencia_auc_pr(y, s, s, n_boot=50)
    assert d["dif"] == 0 and d["lo"] == 0 and d["hi"] == 0 and d["p_mejora"] == 0


def test_diferencia_auc_pr_detecta_mejora_y_es_reproducible():
    rng = np.random.default_rng(1)
    y = (rng.random(5000) < 0.05).astype(int)
    bueno, malo = y * 0.5 + rng.random(5000), rng.random(5000)
    d = ev.diferencia_auc_pr(y, bueno, malo, n_boot=100, seed=3)
    assert d["dif"] > 0 and d["lo"] > 0 and d["p_mejora"] == 1.0
    assert d == ev.diferencia_auc_pr(y, bueno, malo, n_boot=100, seed=3)
    inv = ev.diferencia_auc_pr(y, malo, bueno, n_boot=100, seed=3)
    assert inv["hi"] < 0 and inv["p_mejora"] == 0.0


def test_diferencia_auc_pr_pareada_es_mas_estrecha_que_comparar_intervalos():
    rng = np.random.default_rng(2)
    y = (rng.random(4000) < 0.05).astype(int)
    base = y * 0.3 + rng.random(4000)
    mejor = base + y * 0.15
    d = ev.diferencia_auc_pr(y, mejor, base, n_boot=100)
    thr = ev.threshold_for_recall(y, base, 0.8)
    ci = ev.bootstrap_ci(y, base, thr, n_boot=100)["auc_pr"]
    assert (d["hi"] - d["lo"]) < (ci[1] - ci[0])


def test_diferencia_sin_positivos_es_nan():
    assert np.isnan(ev.diferencia_auc_pr([0, 0], [0.1, 0.2], [0.3, 0.4])["dif"])


def _agrupado(seed=0, n_bloques=60, filas=150):
    """Positivos concentrados en pocos bloques (olas), como los incendios reales."""
    rng = np.random.default_rng(seed)
    bloques = np.repeat(np.arange(n_bloques), filas)
    tasa = np.where(rng.random(n_bloques) < 0.1, 0.30, 0.01)[bloques]
    y = (rng.random(bloques.size) < tasa).astype(int)
    return y, y * 0.4 + rng.random(y.size), bloques


def test_bootstrap_por_bloques_da_intervalo_mas_ancho_con_datos_agrupados():
    y, s, b = _agrupado()
    thr = ev.threshold_for_recall(y, s, 0.8)
    filas = ev.bootstrap_ci(y, s, thr, n_boot=200)["auc_pr"]
    bloques = ev.bootstrap_ci(y, s, thr, n_boot=200, bloques=b)["auc_pr"]
    assert (bloques[1] - bloques[0]) > 1.5 * (filas[1] - filas[0])


def test_bootstrap_por_bloques_es_reproducible_y_acepta_etiquetas_de_texto():
    y, s, b = _agrupado()
    thr = ev.threshold_for_recall(y, s, 0.8)
    etiquetas = np.array([f"S{k}" for k in b])
    a = ev.bootstrap_ci(y, s, thr, n_boot=50, seed=5, bloques=etiquetas)
    assert a == ev.bootstrap_ci(y, s, thr, n_boot=50, seed=5, bloques=etiquetas)


def test_diferencia_por_bloques_mas_ancha_que_por_filas():
    y, s, b = _agrupado()
    otro = np.random.default_rng(9).random(y.size)
    filas = ev.diferencia_auc_pr(y, s, otro, n_boot=200)
    bloques = ev.diferencia_auc_pr(y, s, otro, n_boot=200, bloques=b)
    assert (bloques["hi"] - bloques["lo"]) > (filas["hi"] - filas["lo"])
    assert bloques["dif"] == filas["dif"]


def test_compare_con_col_bloque(toy_gold):
    d = _scored(toy_gold)
    d["_b"] = d["fecha"].dt.isocalendar().week.astype(int)
    thr = ev.thresholds_from_validation(d)
    rep = ev.compare_model_vs_baseline(d, thr, n_boot=20, by_province=False, col_bloque="_b")
    assert len(rep) == 2
    with pytest.raises(ValueError, match="Faltan columnas"):
        ev.compare_model_vs_baseline(d, thr, n_boot=5, col_bloque="no_existe")
