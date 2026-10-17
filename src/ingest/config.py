"""Configuración centralizada del módulo de ingesta de Piriscope.

Todos los valores se pueden sobrescribir con variables de entorno (o un
fichero ``.env`` en la raíz del repositorio, que nunca se versiona). Así el
pipeline se reejecuta con otro rango, otro dataset o otros parámetros sin
editar el código (HU-11).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv es opcional: en GitHub Actions las variables llegan por env
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

# Raíz del repositorio: src/ingest/config.py -> ../../
REPO_ROOT = Path(__file__).resolve().parents[2]

if load_dotenv is not None:
    load_dotenv(REPO_ROOT / ".env", override=False)


def _env(nombre: str, defecto: str | None = None) -> str | None:
    valor = os.getenv(nombre)
    return valor if valor not in (None, "") else defecto


# ---------------------------------------------------------------------------
# Región piloto: Galicia (DEC-01)
# ---------------------------------------------------------------------------
#: Código INE de provincia -> nombre oficial
PROVINCIAS_GALICIA: dict[str, str] = {
    "15": "A Coruña",
    "27": "Lugo",
    "32": "Ourense",
    "36": "Pontevedra",
}

#: Caja envolvente de Galicia con margen [Norte, Oeste, Sur, Este] (grados)
#: Se usa para recortar la rejilla del FWI en origen.
BBOX_GALICIA: list[float] = [44.0, -9.5, 41.7, -6.6]


# ---------------------------------------------------------------------------
# Rutas locales (arquitectura medallón, DEC-10)
# ---------------------------------------------------------------------------
DATA_DIR = Path(_env("PIRISCOPE_DATA_DIR", str(REPO_ROOT / "data")))
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
#: Carpeta donde se dejan los ficheros que se descargan a mano
#: (exportaciones del buscador EGIF, límites del CNIG...).
LANDING_DIR = DATA_DIR / "landing"


# ---------------------------------------------------------------------------
# Fuentes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AemetConfig:
    base_url: str = "https://opendata.aemet.es/opendata/api"
    api_key: str | None = field(default_factory=lambda: _env("AEMET_API_KEY"))
    #: AEMET limita a 6 meses el rango de cada petición de valores diarios.
    max_dias_por_peticion: int = int(_env("AEMET_MAX_DIAS", "180"))
    #: Pausa entre peticiones (AEMET aplica límite por minuto, HTTP 429).
    pausa_segundos: float = float(_env("AEMET_PAUSA_SEGUNDOS", "1.5"))
    #: Espera tras un 429 antes de reintentar.
    espera_429_segundos: float = float(_env("AEMET_ESPERA_429", "60"))
    max_reintentos: int = int(_env("AEMET_MAX_REINTENTOS", "5"))
    timeout: int = int(_env("AEMET_TIMEOUT", "60"))
    #: Un tramo cuyo fin sea anterior a hoy - N días se considera consolidado
    #: y se reutiliza desde la caché de Bronze en lugar de volver a pedirlo.
    dias_consolidacion: int = int(_env("AEMET_DIAS_CONSOLIDACION", "30"))
    #: Codificación de los JSON de datos de AEMET.
    encoding: str = "ISO-8859-15"


@dataclass(frozen=True)
class EffisConfig:
    #: Early Warning Data Store (EWDS) de Copernicus.
    api_url: str = field(
        default_factory=lambda: _env("CDSAPI_URL", "https://ewds.climate.copernicus.eu/api")
    )
    api_key: str | None = field(default_factory=lambda: _env("CDSAPI_KEY"))
    dataset: str = "cems-fire-historical-v1"
    #: FWI y sus componentes (Anexo A).
    variables: tuple[str, ...] = (
        "fire_weather_index",
        "fine_fuel_moisture_code",
        "duff_moisture_code",
        "drought_code",
        "initial_fire_spread_index",
        "build_up_index",
    )
    product_type: str = "reanalysis"
    #: consolidated_dataset para el histórico; intermediate_dataset para los
    #: meses más recientes aún no consolidados.
    dataset_type: str = field(
        default_factory=lambda: _env("EFFIS_DATASET_TYPE", "consolidated_dataset")
    )
    system_version: str = field(default_factory=lambda: _env("EFFIS_SYSTEM_VERSION", "4_1"))
    grid: str = "0.25/0.25"
    data_format: str = field(default_factory=lambda: _env("EFFIS_FORMATO", "netcdf"))


@dataclass(frozen=True)
class IneConfig:
    #: Relación de municipios y códigos del INE (diccionario anual).
    #: {aa} = dos últimas cifras del año.
    url_diccionario: str = "https://www.ine.es/daco/daco42/codmun/diccionario{aa}.xlsx"


@dataclass(frozen=True)
class HuggingFaceConfig:
    #: Dataset de Hugging Face donde se archivan los datos (DEC-11).
    repo_id: str = field(default_factory=lambda: _env("HF_REPO_ID", "cpuertolas/piriscope"))
    token: str | None = field(default_factory=lambda: _env("HF_TOKEN"))
    repo_type: str = "dataset"
    #: Prefijo de los datos originales dentro del dataset.
    prefijo_raw: str = "raw"


#: Nombre de cada fuente en local (data/bronze/<fuente>) y en Hugging Face
#: (raw/<FUENTE>), según el diagrama de ingesta automática (apartado 5.3).
FUENTES: dict[str, str] = {
    "aemet": "AEMET",
    "egif": "EGIF",
    "effis": "EFFIS",
    "ine_cnig": "INE_CNIG",
}

#: Atribuciones obligatorias que se guardan en cada manifiesto (apartado 8).
ATRIBUCIONES: dict[str, str] = {
    "aemet": "Fuente: AEMET. AEMET no respalda este trabajo.",
    "egif": (
        "Origen de los datos: Ministerio para la Transición Ecológica y el Reto "
        "Demográfico (EGIF)."
    ),
    "effis": (
        "Copernicus Emergency Management Service (CEMS/JRC), CC BY 4.0. "
        "FWI histórico DOI 10.24381/cds.0e89c522 (ECMWF)."
    ),
    "ine_cnig": (
        "Elaboración propia con datos extraídos del sitio web del INE: www.ine.es. "
        "© Instituto Geográfico Nacional."
    ),
}
