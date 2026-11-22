"""Compara dos versiones de modelo (p. ej. v1 y v1-check): metadatos y umbrales.

Sirve para comprobar la reproducibilidad del reentrenamiento (NF4, propuesta 8):
si se reentrena con la misma semilla, los mismos datos y las mismas dependencias,
los umbrales (y el resto de campos) deben salir iguales.

    python scripts/comparar_modelos.py v1 v1-check [--models-dir models] [--tol 1e-6]

Compara, por variante, el ``.json`` del modelo y su ``_umbrales.json``. Ignora la
fecha de entrenamiento, la versión y el identificador. Devuelve 0 si todo coincide
dentro de la tolerancia y 1 si hay diferencias o faltan ficheros.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

IGNORAR = {"trained_at", "version", "model_id"}


def aplanar(obj, prefijo=""):
    """Convierte un JSON anidado en {ruta: valor} con las hojas."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from aplanar(v, f"{prefijo}.{k}" if prefijo else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from aplanar(v, f"{prefijo}[{i}]")
    else:
        yield prefijo, obj


def _normaliza(valor, versiones):
    if isinstance(valor, str):
        for v in versiones:
            valor = valor.replace(v, "<version>")
    return valor


def comparar_json(a: dict, b: dict, versiones, tol: float):
    """Devuelve (diferencias, mayor_diferencia_numerica)."""
    fa = {k: v for k, v in aplanar(a) if k.split(".")[-1] not in IGNORAR}
    fb = {k: v for k, v in aplanar(b) if k.split(".")[-1] not in IGNORAR}
    difs, mayor = [], 0.0
    for clave in sorted(set(fa) | set(fb)):
        if clave not in fa or clave not in fb:
            difs.append((clave, fa.get(clave, "—"), fb.get(clave, "—")))
            continue
        x, y = _normaliza(fa[clave], versiones), _normaliza(fb[clave], versiones)
        es_num = all(isinstance(t, (int, float)) and not isinstance(t, bool) for t in (x, y))
        if es_num:
            d = abs(x - y)
            mayor = max(mayor, d)
            if d > tol * max(1.0, abs(x)):
                difs.append((clave, x, y))
        elif x != y:
            difs.append((clave, x, y))
    return difs, mayor


def _ficheros(models_dir: Path, version: str) -> dict[str, Path]:
    """{nombre sin la versión: ruta} de los .json de una versión."""
    return {p.name[len(version) + 1:]: p for p in models_dir.glob(f"{version}_*.json")}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("version_a")
    p.add_argument("version_b")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--tol", type=float, default=1e-6, help="tolerancia relativa en números")
    a = p.parse_args(argv)

    md = Path(a.models_dir)
    fa, fb = _ficheros(md, a.version_a), _ficheros(md, a.version_b)
    if not fa or not fb:
        print(f"No hay ficheros .json de {a.version_a if not fa else a.version_b} en {md}/")
        return 1

    versiones = [a.version_a, a.version_b]
    hubo_diferencias = False
    for nombre in sorted(set(fa) | set(fb)):
        if nombre not in fa or nombre not in fb:
            print(f"[FALTA] {nombre}: solo está en {a.version_a if nombre in fa else a.version_b}")
            hubo_diferencias = True
            continue
        difs, mayor = comparar_json(json.loads(fa[nombre].read_text(encoding="utf-8")),
                                    json.loads(fb[nombre].read_text(encoding="utf-8")), versiones, a.tol)
        if difs:
            hubo_diferencias = True
            print(f"[DIFIERE] {nombre} ({len(difs)} campos)")
            for clave, x, y in difs:
                print(f"    {clave}: {x}  ->  {y}")
        else:
            print(f"[IGUAL]   {nombre} · mayor diferencia numérica {mayor:.3g}")
    print("\nResultado:", "hay diferencias" if hubo_diferencias else "reproducible (todo coincide)")
    return 1 if hubo_diferencias else 0


if __name__ == "__main__":
    sys.exit(main())
