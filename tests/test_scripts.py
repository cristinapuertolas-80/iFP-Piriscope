"""Comprobación de humo de los scripts auxiliares de `scripts/`.

Cada script debe poder ejecutarse directamente (``python scripts/<nombre>.py --help``) desde cualquier
carpeta. Detecta imports relativos o rutas que se rompen al mover un script (como pasó al sacar los
análisis de `src/ml/`).
"""
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = sorted((Path(__file__).resolve().parents[1] / "scripts").glob("*.py"))


def test_hay_scripts():
    assert SCRIPTS


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_el_script_muestra_su_ayuda(script, tmp_path):
    r = subprocess.run([sys.executable, str(script), "--help"], cwd=tmp_path, capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "usage" in r.stdout.lower()


def test_comparar_predicciones_detecta_reproduccion_y_diferencias(tmp_path):
    import importlib.util

    import pandas as pd

    ruta = Path(__file__).resolve().parents[1] / "scripts" / "comparar_predicciones.py"
    spec = importlib.util.spec_from_file_location("comparar_predicciones", ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    base = pd.DataFrame({"id_municipio": ["15001", "15002", "15001"],
                         "fecha_objetivo": pd.to_datetime(["2026-10-01", "2026-10-01", "2026-10-02"]),
                         "version_modelo": ["v1", "v1", "v1-respaldo"],
                         "probabilidad": [0.01, 0.05, 0.02], "nivel_riesgo": ["bajo", "medio", "bajo"]})
    base.to_parquet(tmp_path / "ref.parquet")
    base.to_parquet(tmp_path / "igual.parquet")
    distinto = base.copy()
    distinto.loc[1, ["probabilidad", "nivel_riesgo"]] = [0.2, "alto"]
    distinto.to_parquet(tmp_path / "distinto.parquet")

    args = ["--referencia", str(tmp_path / "ref.parquet"), "--nuevo"]
    assert mod.main(args + [str(tmp_path / "igual.parquet")]) == 0
    assert mod.main(args + [str(tmp_path / "distinto.parquet")]) == 1
    tabla = mod.comparar(mod.leer(tmp_path / "ref.parquet", None, None),
                         mod.leer(tmp_path / "distinto.parquet", None, None))
    total = tabla.iloc[0]
    assert total["en_ambos"] == 3 and total["cambios_de_nivel"] == 1
    assert abs(total["dif_max_probabilidad"] - 0.15) < 1e-12
    # El filtro de fechas deja fuera la fila cambiada del día 1
    assert mod.main(args + [str(tmp_path / "distinto.parquet"), "--desde", "2026-10-02"]) == 0
