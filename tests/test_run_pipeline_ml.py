"""Tests del orquestador: orden y parada de etapas, etapa Gold y etapa ML (entrenar / inferir).

No ejecutan nada real: sustituyen las etapas, el entrenamiento, la inferencia y los subprocesos.
"""
from types import SimpleNamespace

import pytest

import run_pipeline as rp
from src.ml import config, experimento, inferir


@pytest.fixture
def etapas(monkeypatch):
    """Sustituye las cuatro etapas por funciones que registran la llamada y devuelven `resultado[nombre]`."""
    llamadas, resultado = [], {"ingesta": True, "silver": True, "gold": True, "ml": True}

    def falsa(nombre):
        def f(*a, **k):
            llamadas.append(nombre)
            return resultado[nombre]
        return f

    for nombre in resultado:
        monkeypatch.setattr(rp, f"etapa_{nombre}", falsa(nombre))
    return llamadas, resultado


def test_las_etapas_se_ejecutan_en_el_orden_del_pipeline(etapas):
    llamadas, _ = etapas
    assert rp.main(["--etapas", "ml", "gold"]) == 0
    assert llamadas == ["gold", "ml"]


def test_las_fechas_solo_son_obligatorias_para_ingesta_y_silver(etapas):
    assert rp.main(["--etapas", "gold", "ml"]) == 0
    for e in ("ingesta", "silver"):
        with pytest.raises(SystemExit) as exc:
            rp.main(["--etapas", e])
        assert exc.value.code == 2


def test_si_falla_gold_no_se_ejecuta_ml_salvo_con_continuar(etapas):
    llamadas, resultado = etapas
    resultado["gold"] = False
    assert rp.main(["--etapas", "gold", "ml"]) == 1
    assert llamadas == ["gold"]
    llamadas.clear()
    assert rp.main(["--etapas", "gold", "ml", "--continuar"]) == 1
    assert llamadas == ["gold", "ml"]


def test_una_ingesta_fallida_no_detiene_silver(etapas):
    llamadas, resultado = etapas
    resultado["ingesta"] = False
    assert rp.main(["--desde", "2024-01-01", "--hasta", "2024-01-31", "--etapas", "ingesta", "silver"]) == 1
    assert llamadas == ["ingesta", "silver"]


def test_rangos_de_fechas_incoherentes(etapas):
    with pytest.raises(SystemExit):
        rp.main(["--desde", "2024-02-01", "--hasta", "2024-01-01", "--etapas", "silver"])
    with pytest.raises(SystemExit):
        rp.main(["--etapas", "ml", "--inferir-desde", "2024-02-01", "--inferir-hasta", "2024-01-01"])


# --- etapa gold ------------------------------------------------------------------------------------
def test_gold_exige_que_existan_los_dos_datasets(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "GOLD_DIR", tmp_path)
    monkeypatch.setattr(config, "DATASET_ENTRENAMIENTO", tmp_path / "dataset_entrenamiento.parquet")
    monkeypatch.setattr(rp.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    assert rp.etapa_gold() is False                                   # no se ha generado nada
    (tmp_path / "dataset_entrenamiento.parquet").write_text("x")
    assert rp.etapa_gold() is False                                   # falta el dataset municipio-día
    (tmp_path / "dataset_municipio_dia.parquet").write_text("x")
    assert rp.etapa_gold() is True


def test_gold_falla_si_el_subproceso_falla(monkeypatch):
    monkeypatch.setattr(rp.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=3))
    assert rp.etapa_gold() is False


# --- etapa ml --------------------------------------------------------------------------------------
@pytest.fixture
def ml(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(config, "GOLD_DIR", tmp_path)
    reg = {"entrenar": [], "inferir": [], "verificar": []}

    def guardar_modelos(version):
        for variante in ("con_fwi", "sin_fwi"):
            mid = f"{version}_xgboost_{variante}-sin_viento-desde2009"
            (tmp_path / f"{mid}.joblib").write_text("x")
            (tmp_path / f"{mid}_umbrales.json").write_text("{}")

    def entrenar(argv):
        reg["entrenar"].append(argv)
        guardar_modelos(argv[argv.index("--version") + 1])
        return 0

    reg["guardar_modelos"] = guardar_modelos
    reg["codigo_inferir"] = 0
    monkeypatch.setattr(experimento, "main", entrenar)
    monkeypatch.setattr(inferir, "main", lambda argv: reg["inferir"].append(argv) or reg["codigo_inferir"])
    monkeypatch.setattr(rp, "verificar_salidas", lambda *a: reg["verificar"].append(a) or True)
    return reg


def valor(argv, opcion):
    return argv[argv.index(opcion) + 1]


def test_entrenar_usa_la_configuracion_de_dec47_y_no_toca_el_test(ml):
    assert rp.main(["--etapas", "ml", "--ml-acciones", "entrenar", "--version-modelo", "v2"]) == 0
    argv = ml["entrenar"][0]
    assert "--final" not in argv and "--guardar" in argv
    assert valor(argv, "--version") == "v2" and valor(argv, "--anio-desde") == "2009"
    assert "viento" in argv and "scale_pos_weight=1" in argv
    assert argv[argv.index("--variantes") + 1: argv.index("--variantes") + 3] == ["con_fwi", "sin_fwi"]
    assert valor(argv, "--salida").endswith("informe_desarrollo_v2.csv")      # no pisa el informe de desarrollo actual
    assert ml["inferir"] == []                                               # por defecto solo entrena lo pedido


def test_no_sobrescribe_modelos_existentes_salvo_que_se_pida(ml):
    ml["guardar_modelos"]("v2")
    assert rp.main(["--etapas", "ml", "--ml-acciones", "entrenar", "--version-modelo", "v2"]) == 1
    assert ml["entrenar"] == []
    assert rp.main(["--etapas", "ml", "--ml-acciones", "entrenar", "--version-modelo", "v2",
                    "--sobrescribir-modelo"]) == 0
    assert len(ml["entrenar"]) == 1


def test_inferir_resuelve_el_principal_y_el_respaldo(ml):
    ml["guardar_modelos"]("v1")
    assert rp.main(["--etapas", "ml"]) == 0
    argv = ml["inferir"][0]
    assert valor(argv, "--modelo") == "v1_xgboost_con_fwi-sin_viento-desde2009"
    assert valor(argv, "--modelo-respaldo") == "v1_xgboost_sin_fwi-sin_viento-desde2009"
    assert "--desde" not in argv and "--hasta" not in argv and "--reemplazar" not in argv   # último día


def test_inferir_pasa_el_rango_y_las_opciones(ml, tmp_path):
    ml["guardar_modelos"]("v1")
    assert rp.main(["--etapas", "ml", "--inferir-desde", "2026-09-01", "--inferir-hasta", "2026-10-20",
                    "--reemplazar-predicciones", "--ejemplo-dir", str(tmp_path / "ej")]) == 0
    argv = ml["inferir"][0]
    assert valor(argv, "--desde") == "2026-09-01" and valor(argv, "--hasta") == "2026-10-20"
    assert "--reemplazar" in argv and valor(argv, "--ejemplo-dir") == str(tmp_path / "ej")


def test_inferir_sin_modelos_falla_con_mensaje_claro(ml):
    assert rp.main(["--etapas", "ml"]) == 1
    assert ml["inferir"] == []
    with pytest.raises(ValueError, match="No encuentro un único modelo"):
        rp.resolver_modelos(config.MODELS_DIR, "v1", None, None)


def test_modelo_explicito_tiene_prioridad(ml):
    ml["guardar_modelos"]("v1")
    assert rp.main(["--etapas", "ml", "--modelo", "mio", "--modelo-respaldo", "suyo"]) == 0
    assert valor(ml["inferir"][0], "--modelo") == "mio" and valor(ml["inferir"][0], "--modelo-respaldo") == "suyo"


def test_entrenar_e_inferir_usan_el_modelo_recien_guardado(ml):
    assert rp.main(["--etapas", "ml", "--ml-acciones", "entrenar", "inferir", "--version-modelo", "v3"]) == 0
    assert valor(ml["inferir"][0], "--modelo").startswith("v3_")


def test_una_inferencia_con_codigo_distinto_de_cero_falla_y_no_verifica(ml):
    ml["guardar_modelos"]("v1")
    ml["codigo_inferir"] = 2
    assert rp.main(["--etapas", "ml"]) == 1
    assert ml["verificar"] == []


def test_la_verificacion_se_puede_omitir(ml):
    ml["guardar_modelos"]("v1")
    assert rp.main(["--etapas", "ml"]) == 0 and len(ml["verificar"]) == 1
    assert rp.main(["--etapas", "ml", "--sin-verificar"]) == 0 and len(ml["verificar"]) == 1
