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
