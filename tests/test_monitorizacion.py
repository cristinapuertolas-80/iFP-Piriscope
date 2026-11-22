"""Tests de src/ml/monitorizar.py con datos sintéticos (sin red ni datos reales)."""
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.ml import monitorizar as m


def test_psi_misma_distribucion_es_baja():
    rng = np.random.default_rng(0)
    assert m.psi(rng.normal(size=5000), rng.normal(size=2000)) < 0.05


def test_psi_distribucion_desplazada_es_deriva():
    rng = np.random.default_rng(0)
    valor = m.psi(rng.normal(size=5000), rng.normal(loc=2, size=2000))
    assert valor > m.PSI_DERIVA and m.estado_psi(valor) == "deriva"


def test_psi_pocos_datos_o_constante_es_none():
    assert m.psi([1.0] * 100, [1.0] * 100) is None
    assert m.psi(np.arange(10), np.arange(10)) is None
    assert m.estado_psi(None) == "sin_datos"


def test_psi_ignora_nulos_y_trata_categorias():
    rng = np.random.default_rng(1)
    ref = pd.Series(rng.integers(0, 4, 3000).astype(float))
    act = pd.Series(np.r_[rng.integers(0, 4, 500).astype(float), [np.nan] * 50])
    assert m.psi(ref, act) < 0.05
    assert m.psi(ref, pd.Series([9.0] * 100)) > m.PSI_DERIVA   # valor nunca visto


def _datos(desplazar_fwi=0.0):
    rng = np.random.default_rng(2)
    municipios = [f"{27000 + i:05d}" for i in range(10)]
    # entrenamiento: 2010-2012, enero-diciembre; actual: noviembre de 2026
    fechas_ent = pd.date_range("2010-01-01", "2012-12-31")
    ent = pd.DataFrame({"fecha": np.repeat(fechas_ent, 3)})
    ent["particion"] = "entrenamiento"
    ent["temperatura"] = 10 + 8 * np.sin((ent["fecha"].dt.dayofyear - 110) / 365 * 2 * np.pi) + rng.normal(size=len(ent))
    ent["fwi"] = rng.gamma(2, 2, size=len(ent))
    fechas_act = pd.date_range("2026-10-01", "2026-11-15")
    ds = pd.DataFrame({"fecha": np.repeat(fechas_act, 10), "id_municipio": municipios * len(fechas_act)})
    ds["temperatura"] = 10 + 8 * np.sin((ds["fecha"].dt.dayofyear - 110) / 365 * 2 * np.pi) + rng.normal(size=len(ds))
    ds["fwi"] = rng.gamma(2, 2, size=len(ds)) + desplazar_fwi
    ds.loc[ds["fecha"] > "2026-11-10", "fwi"] = np.nan
    filas = []
    for f in pd.date_range("2026-10-17", "2026-11-16"):
        for i, mun in enumerate(municipios):
            if f == pd.Timestamp("2026-11-16") and i >= 8:
                continue   # 2 municipios sin predicción en la última fecha
            resp = f > pd.Timestamp("2026-11-11")
            filas.append({"id_municipio": mun, "fecha_objetivo": f,
                          "version_modelo": "v1-respaldo" if resp else "v1",
                          "probabilidad": 0.02 if resp else 0.01 + 0.001 * i,
                          "nivel_riesgo": "bajo"})
    calidad = {"municipio": {"filas": 10},
               "observacion_meteorologica": {"hasta": "2026-11-15"},
               "indicador_fwi": {"hasta": "2026-11-10", "dias_intermedio": 40, "meses_intermedio": ["2026-11"]}}
    return pd.DataFrame(filas), ds, ent, calidad


def test_informe_completo_sin_deriva():
    pred, ds, ent, calidad = _datos()
    i = m.construir_informe(pred, ds, ent, calidad, ["temperatura", "fwi"], date(2026, 11, 19),
                            ventana=30, anio_desde=2009)
    u = i["predicciones"]["ultima_fecha"]
    assert u["fecha"] == "2026-11-16" and u["municipios"] == 8 and u["cobertura_pct"] == 80.0
    assert u["pct_respaldo"] == 100.0 and u["niveles"] == {"bajo": 8, "medio": 0, "alto": 0}
    assert i["predicciones"]["dias_finales_con_respaldo"] == 5
    assert i["fuentes"]["aemet"]["retraso_dias"] == 4 and i["fuentes"]["fwi"]["retraso_dias"] == 9
    assert i["fuentes"]["fwi"]["retraso_vs_meteo_dias"] == 5
    estados = {f["variable"]: f["estado"] for f in i["deriva"]["variables"]}
    assert estados["temperatura"] != "deriva"   # la referencia es estacional: no hay deriva falsa
    # sin emparejar por época del año, noviembre contra todo el año sí daría deriva
    sin_estacion = m.psi(ent["temperatura"], ds["temperatura"])
    assert sin_estacion > m.PSI_DERIVA
    assert any("Cobertura" in a for a in i["avisos"]) and i["estado"] == "revisar"
    json.dumps(i, default=m._json_default)


def test_deriva_detectada_y_retrasos():
    pred, ds, ent, calidad = _datos(desplazar_fwi=8.0)
    calidad["indicador_fwi"]["hasta"] = "2026-10-20"
    i = m.construir_informe(pred, ds, ent, calidad, ["temperatura", "fwi"], date(2026, 11, 19))
    assert i["deriva"]["variable_psi_max"] == "fwi"
    assert any("Deriva" in a for a in i["avisos"]) and any("FWI lleva" in a for a in i["avisos"])


def test_main_escribe_json(tmp_path):
    pytest.importorskip("pyarrow")
    pred, ds, ent, calidad = _datos()
    pred.to_parquet(tmp_path / "p.parquet")
    ds.to_parquet(tmp_path / "d.parquet")
    ent.to_parquet(tmp_path / "e.parquet")
    (tmp_path / "c.json").write_text(json.dumps(calidad), encoding="utf-8")
    md = tmp_path / "resumen.md"
    salida = tmp_path / "mon.json"
    codigo = m.main(["--pred", str(tmp_path / "p.parquet"), "--dataset", str(tmp_path / "d.parquet"),
                     "--entrenamiento", str(tmp_path / "e.parquet"), "--calidad", str(tmp_path / "c.json"),
                     "--validacion", str(tmp_path / "no_existe.json"), "--models-dir", str(tmp_path),
                     "--hoy", "2026-11-19", "--salida", str(salida), "--resumen-md", str(md)])
    # las variables por defecto que no están en los datos sintéticos se omiten sin fallar
    assert codigo == 0 and json.loads(salida.read_text(encoding="utf-8"))["version_esquema"] == 1
    assert "Monitorización" in md.read_text(encoding="utf-8")


def test_main_falta_entrada(tmp_path):
    pytest.importorskip("pyarrow")
    assert m.main(["--pred", str(tmp_path / "no.parquet"), "--salida", str(tmp_path / "x.json")]) == 1
