"""Genera el GeoJSON simplificado de los 313 municipios gallegos para el mapa del dashboard.

El dashboard no lee Bronze ni Silver (contrato Gold, apartado 8.1): los límites,
nombres y provincias de los municipios se extraen una sola vez de los recintos
del CNIG y se guardan en ``src/dashboard/assets/municipios_galicia.geojson``,
que sí se versiona y sirve también de catálogo para el buscador.

El GeoPackage del CNIG (~230 MB) se toma de ``data/landing/ine_cnig/`` si está;
si no, se descarga del dataset de Hugging Face (requiere ``HF_TOKEN`` de lectura).

    python scripts/generar_mapa_municipios.py
    python scripts/generar_mapa_municipios.py --gpkg ruta/BDDAE.gpkg --tolerancia 150
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.ingest import config  # noqa: E402
from src.silver.municipios import CRS_GEOGRAFICO, CRS_METRICO, leer_recintos_cnig  # noqa: E402

GPKG_EN_HF = "data/landing/ine_cnig/BDDAE20240101_conhisto.gpkg"
SALIDA = REPO_ROOT / "src" / "dashboard" / "assets" / "municipios_galicia.geojson"


def obtener_gpkg(ruta: Path | None) -> Path:
    if ruta:
        return ruta
    locales = sorted((config.LANDING_DIR / "ine_cnig").glob("*.gpkg"))
    if locales:
        return locales[-1]
    from huggingface_hub import hf_hub_download

    cfg = config.HuggingFaceConfig()
    print(f"Descargando {GPKG_EN_HF} de {cfg.repo_id} (~230 MB, solo la primera vez)…")
    return Path(hf_hub_download(cfg.repo_id, GPKG_EN_HF, repo_type=cfg.repo_type, token=cfg.token,
                                local_dir=str(config.REPO_ROOT)))


def generar(gpkg: Path, salida: Path = SALIDA, tolerancia_m: float = 150, decimales: int = 5) -> int:
    recintos = leer_recintos_cnig(gpkg).dissolve(by="id_municipio", as_index=False, aggfunc="first")
    metrico = recintos.to_crs(CRS_METRICO)
    # Punto interior (no el centroide): ahí se dibuja el icono de «sin dato» del mapa
    interior = metrico.representative_point().to_crs(CRS_GEOGRAFICO)
    recintos["provincia"] = recintos["id_municipio"].str[:2].map(config.PROVINCIAS_GALICIA)
    recintos["lat"], recintos["lon"] = interior.y.round(decimales), interior.x.round(decimales)
    geom = metrico.simplify(tolerancia_m, preserve_topology=True).to_crs(CRS_GEOGRAFICO)
    recintos = recintos.set_geometry(geom)[["id_municipio", "nombre", "provincia", "lat", "lon", "geometry"]]
    salida.parent.mkdir(parents=True, exist_ok=True)
    datos = json.loads(recintos.to_json(drop_id=True))
    # Redondear coordenadas (5 decimales ≈ 1 m) reduce el fichero a una fracción
    def redondear(c):
        return [redondear(x) for x in c] if isinstance(c[0], list) else [round(v, decimales) for v in c]
    for f in datos["features"]:
        f["geometry"]["coordinates"] = redondear(f["geometry"]["coordinates"])
    salida.write_text(json.dumps(datos, separators=(",", ":")), encoding="utf-8")
    return len(datos["features"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gpkg", type=Path, help="GeoPackage del CNIG (por defecto, landing o Hugging Face)")
    ap.add_argument("--tolerancia", type=float, default=150, help="simplificación en metros (por defecto 150)")
    args = ap.parse_args()
    n = generar(obtener_gpkg(args.gpkg), tolerancia_m=args.tolerancia)
    print(f"{n} municipios -> {SALIDA.relative_to(REPO_ROOT)} ({SALIDA.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
