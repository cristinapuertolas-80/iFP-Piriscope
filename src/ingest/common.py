"""Utilidades comunes de la ingesta: sesión HTTP con reintentos, carpetas de
Bronze por fuente y fecha de ingesta, y manifiesto de trazabilidad.

Bronze guarda los datos **tal como se descargan**, sin modificarlos. Cada
ejecución crea ``data/bronze/<fuente>/<AAAAMMDDTHHMMSSZ>/`` con los ficheros
originales y un ``manifest.json`` que registra origen, tamaño y huella SHA-256
de cada fichero, los parámetros de la ejecución y la atribución de la fuente.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import config

FORMATO_TIMESTAMP = "%Y%m%dT%H%M%SZ"

log = logging.getLogger("piriscope.ingest")


def configurar_logging(nivel: int = logging.INFO) -> None:
    logging.basicConfig(
        level=nivel,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ahora_utc() -> datetime:
    return datetime.now(timezone.utc)


def timestamp_ingesta(momento: datetime | None = None) -> str:
    return (momento or ahora_utc()).strftime(FORMATO_TIMESTAMP)


def crear_sesion(reintentos: int = 5, backoff: float = 2.0) -> requests.Session:
    """Sesión HTTP con reintentos exponenciales ante errores transitorios."""
    sesion = requests.Session()
    retry = Retry(
        total=reintentos,
        connect=reintentos,
        read=reintentos,
        backoff_factor=backoff,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        raise_on_status=False,
    )
    adaptador = HTTPAdapter(max_retries=retry)
    sesion.mount("https://", adaptador)
    sesion.mount("http://", adaptador)
    sesion.headers.update({"User-Agent": "piriscope-ingest/0.1 (+github.com/cristinapuertolas-80/iFP-Piriscope)"})
    return sesion


def sha256_fichero(ruta: Path, bloque: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for trozo in iter(lambda: f.read(bloque), b""):
            h.update(trozo)
    return h.hexdigest()


def version_codigo() -> str | None:
    """SHA del commit actual, si el código se ejecuta dentro de un repo git."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=config.REPO_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001 - fuera de git no es un error
        return None


def tramos_fechas(desde: date, hasta: date, max_dias: int) -> Iterator[tuple[date, date]]:
    """Divide [desde, hasta] (ambos incluidos) en tramos de como máximo ``max_dias`` días."""
    if hasta < desde:
        raise ValueError(f"Rango de fechas inválido: {desde} > {hasta}")
    if max_dias < 1:
        raise ValueError("max_dias debe ser >= 1")
    inicio = desde
    while inicio <= hasta:
        fin = min(inicio + timedelta(days=max_dias - 1), hasta)
        yield inicio, fin
        inicio = fin + timedelta(days=1)


def meses_en_rango(desde: date, hasta: date) -> list[tuple[int, int]]:
    """Lista de (año, mes) que tocan el rango [desde, hasta]."""
    if hasta < desde:
        raise ValueError(f"Rango de fechas inválido: {desde} > {hasta}")
    meses = []
    y, m = desde.year, desde.month
    while (y, m) <= (hasta.year, hasta.month):
        meses.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return meses


@dataclass
class EjecucionBronze:
    """Una ejecución de ingesta de una fuente: carpeta + manifiesto.

    Uso::

        with EjecucionBronze("aemet", parametros={...}) as run:
            run.guardar_bytes("diarios/1387_2020-01-01_2020-06-28.json", contenido, url)
    """

    fuente: str
    parametros: dict[str, Any] = field(default_factory=dict)
    base_dir: Path | None = None
    momento: datetime = field(default_factory=ahora_utc)
    ficheros: list[dict[str, Any]] = field(default_factory=list)
    incidencias: list[dict[str, Any]] = field(default_factory=list)
    #: Ficheros que no se descargan de nuevo porque ya están en una ejecución
    #: previa de Bronze (caché). Se referencian, no se duplican.
    reutilizados: list[dict[str, Any]] = field(default_factory=list)
    estado: str = "en_curso"

    def __post_init__(self) -> None:
        if self.fuente not in config.FUENTES:
            raise ValueError(f"Fuente desconocida: {self.fuente!r}. Válidas: {list(config.FUENTES)}")
        base = self.base_dir or config.BRONZE_DIR
        # Nunca se reutiliza la carpeta de otra ejecución (dos en el mismo segundo)
        while (Path(base) / self.fuente / timestamp_ingesta(self.momento)).exists():
            self.momento += timedelta(seconds=1)
        self.timestamp = timestamp_ingesta(self.momento)
        self.dir = Path(base) / self.fuente / self.timestamp
        self.dir.mkdir(parents=True, exist_ok=True)
        log.info("Bronze %s -> %s", self.fuente, self.dir)

    # -- registro de ficheros ------------------------------------------------
    def _registrar(self, ruta: Path, origen: str | None, **extra: Any) -> Path:
        entrada = {
            "ruta": ruta.relative_to(self.dir).as_posix(),
            "bytes": ruta.stat().st_size,
            "sha256": sha256_fichero(ruta),
            "origen": origen,
        }
        entrada.update({k: v for k, v in extra.items() if v is not None})
        # Si se vuelve a escribir el mismo fichero, se sustituye la entrada.
        self.ficheros = [f for f in self.ficheros if f["ruta"] != entrada["ruta"]]
        self.ficheros.append(entrada)
        return ruta

    def ruta(self, relativa: str | Path) -> Path:
        destino = self.dir / relativa
        destino.parent.mkdir(parents=True, exist_ok=True)
        return destino

    def guardar_bytes(self, relativa: str | Path, contenido: bytes, origen: str | None = None, **extra: Any) -> Path:
        destino = self.ruta(relativa)
        destino.write_bytes(contenido)
        return self._registrar(destino, origen, **extra)

    def copiar_fichero(self, fuente: Path, relativa: str | Path | None = None, **extra: Any) -> Path:
        """Copia un fichero (descargado a mano) a Bronze sin modificarlo."""
        fuente = Path(fuente)
        destino = self.ruta(relativa or fuente.name)
        shutil.copy2(fuente, destino)
        return self._registrar(destino, f"file://{fuente.name}", **extra)

    def registrar_existente(self, relativa: str | Path, origen: str | None = None, **extra: Any) -> Path:
        """Registra en el manifiesto un fichero ya escrito dentro de la carpeta."""
        return self._registrar(self.dir / relativa, origen, **extra)

    def reutilizar(self, relativa: str | Path, cache: Path) -> None:
        self.reutilizados.append({
            "ruta": Path(relativa).as_posix(),
            "ejecucion_origen": _run_de(cache, self.fuente),
        })

    def incidencia(self, tipo: str, detalle: str, **extra: Any) -> None:
        self.incidencias.append({"tipo": tipo, "detalle": detalle, **extra})
        log.warning("[%s] %s: %s", self.fuente, tipo, detalle)

    # -- manifiesto ---------------------------------------------------------
    def manifiesto(self) -> dict[str, Any]:
        return {
            "fuente": self.fuente,
            "fuente_hf": config.FUENTES[self.fuente],
            "timestamp_ingesta": self.timestamp,
            "fecha_ingesta_utc": self.momento.isoformat(),
            "estado": self.estado,
            "parametros": self.parametros,
            "version_codigo": version_codigo(),
            "atribucion": config.ATRIBUCIONES[self.fuente],
            "n_ficheros": len(self.ficheros),
            "bytes_totales": sum(f["bytes"] for f in self.ficheros),
            "ficheros": sorted(self.ficheros, key=lambda f: f["ruta"]),
            "reutilizados_de_cache": self.reutilizados,
            "incidencias": self.incidencias,
        }

    def escribir_manifiesto(self) -> Path:
        ruta = self.dir / "manifest.json"
        ruta.write_text(json.dumps(self.manifiesto(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return ruta

    def __enter__(self) -> "EjecucionBronze":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.estado = "completada_con_incidencias" if self.incidencias else "completada"
        else:
            self.estado = "fallida"
            self.incidencias.append({"tipo": "excepcion", "detalle": f"{exc_type.__name__}: {exc}"})
        self.escribir_manifiesto()
        return False  # no se silencian las excepciones


def _run_de(ruta: Path, fuente: str) -> str:
    """Devuelve el timestamp de la ejecución a la que pertenece ``ruta``."""
    partes = Path(ruta).parts
    if fuente in partes:
        i = len(partes) - 1 - partes[::-1].index(fuente)
        if i + 1 < len(partes):
            return partes[i + 1]
    return str(ruta)


def ejecuciones_previas(fuente: str, base_dir: Path | None = None) -> list[Path]:
    """Carpetas de ejecuciones anteriores de una fuente, de la más reciente a la más antigua."""
    raiz = Path(base_dir or config.BRONZE_DIR) / fuente
    if not raiz.exists():
        return []
    return sorted((p for p in raiz.iterdir() if p.is_dir()), reverse=True)


def buscar_en_cache(fuente: str, relativa: str | Path, excluir: Path | None = None,
                    base_dir: Path | None = None) -> Path | None:
    """Busca un fichero con la misma ruta relativa en ejecuciones previas de Bronze."""
    for run_dir in ejecuciones_previas(fuente, base_dir):
        if excluir is not None and run_dir.resolve() == Path(excluir).resolve():
            continue
        candidato = run_dir / relativa
        if candidato.is_file() and candidato.stat().st_size > 0:
            return candidato
    return None
