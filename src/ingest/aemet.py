"""Ingesta de la meteorología diaria de AEMET OpenData (HU-01).

Flujo:

1. Se descarga el inventario de estaciones climatológicas y se filtran las de
   las cuatro provincias gallegas (o las indicadas con ``estaciones``).
2. Para cada estación, el rango pedido se divide en tramos de como máximo
   ``AEMET_MAX_DIAS`` días (AEMET limita cada petición a 6 meses).
3. Cada petición sigue el protocolo en dos pasos de AEMET OpenData: la API
   devuelve un JSON con la URL temporal de ``datos``, que se descarga después.
4. El JSON de datos se guarda **sin modificar** en
   ``data/bronze/aemet/<timestamp>/diarios/<estacion>_<ini>_<fin>.json``.
5. Se generan ``cobertura_estaciones.csv`` (informe de cobertura por estación)
   y ``manifest.json``.

Los tramos ya consolidados que existan en ejecuciones previas de Bronze se
reutilizan (caché): el histórico se descarga una sola vez.

Uso::

    python -m src.ingest aemet --desde 2010-01-01 --hasta 2024-12-31
"""

from __future__ import annotations

import json
import logging
import time
import unicodedata
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests

from . import config
from .common import (EjecucionBronze, buscar_en_cache, crear_sesion, error_red, es_error_ssl,
                     tramos_fechas)

log = logging.getLogger("piriscope.ingest.aemet")

RUTA_INVENTARIO = "/valores/climatologicos/inventarioestaciones/todasestaciones"
RUTA_DIARIOS = (
    "/valores/climatologicos/diarios/datos/fechaini/{ini}T00:00:00UTC"
    "/fechafin/{fin}T23:59:59UTC/estacion/{idema}"
)

#: Nombres con que AEMET identifica las provincias gallegas en el inventario.
_PROVINCIAS_AEMET = {"A CORUNA", "LA CORUNA", "CORUNA", "CORUNA A", "LUGO", "OURENSE", "ORENSE", "PONTEVEDRA"}


class AemetError(RuntimeError):
    pass


class AemetSinDatos(AemetError):
    """AEMET responde 404: no hay datos para esa estación y periodo."""


def _normalizar(texto: str) -> str:
    sin_tildes = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return " ".join(sin_tildes.upper().replace(",", " ").replace("(", " ").replace(")", " ").split())


def es_provincia_gallega(provincia: str | None) -> bool:
    return bool(provincia) and _normalizar(provincia) in _PROVINCIAS_AEMET


def dms_a_decimal(valor: str | None) -> float | None:
    """Convierte coordenadas de AEMET (``'432157N'``, ``'0082517W'``) a grados decimales."""
    if not valor:
        return None
    valor = valor.strip().upper()
    hemisferio, cifras = valor[-1], valor[:-1]
    if hemisferio not in "NSEW" or not cifras.isdigit() or len(cifras) < 6:
        return None
    grados, minutos, segundos = int(cifras[:-4]), int(cifras[-4:-2]), int(cifras[-2:])
    decimal = grados + minutos / 60 + segundos / 3600
    return round(-decimal if hemisferio in "SW" else decimal, 6)


@dataclass
class ClienteAemet:
    """Cliente mínimo de AEMET OpenData con reintentos, pausas y control del 429."""

    cfg: config.AemetConfig = None  # type: ignore[assignment]
    sesion: requests.Session | None = None

    def __post_init__(self) -> None:
        self.cfg = self.cfg or config.AemetConfig()
        if not self.cfg.api_key:
            raise AemetError(
                "Falta la API key de AEMET. Defínela en AEMET_API_KEY (fichero .env o Secrets)."
            )
        self.sesion = self.sesion or crear_sesion()
        self._ultima_peticion = 0.0

    # -- bajo nivel ---------------------------------------------------------
    def _esperar_turno(self) -> None:
        transcurrido = time.monotonic() - self._ultima_peticion
        if transcurrido < self.cfg.pausa_segundos:
            time.sleep(self.cfg.pausa_segundos - transcurrido)
        self._ultima_peticion = time.monotonic()

    def _get(self, url: str, **kwargs: Any) -> requests.Response:
        for intento in range(1, self.cfg.max_reintentos + 1):
            self._esperar_turno()
            try:
                r = self.sesion.get(url, timeout=self.cfg.timeout, **kwargs)
            except requests.RequestException as exc:
                if es_error_ssl(exc):
                    # Un certificado rechazado no se arregla reintentando
                    raise error_red("AEMET", exc) from exc
                log.warning("Error de red (%s/%s) en %s: %s", intento, self.cfg.max_reintentos, url, exc)
                time.sleep(min(2**intento, 60))
                continue
            if r.status_code == 429:
                log.warning("AEMET 429 (límite por minuto). Esperando %ss…", self.cfg.espera_429_segundos)
                time.sleep(self.cfg.espera_429_segundos)
                continue
            return r
        raise AemetError(f"Agotados {self.cfg.max_reintentos} reintentos para {url}")

    def peticion(self, ruta: str) -> tuple[bytes, str]:
        """Ejecuta el protocolo en dos pasos y devuelve (contenido bruto, url de datos)."""
        url = self.cfg.base_url + ruta
        for intento in range(1, self.cfg.max_reintentos + 1):
            r = self._get(url, headers={"api_key": self.cfg.api_key, "Accept": "application/json"})
            if r.status_code == 404:
                raise AemetSinDatos(url)
            if r.status_code == 401:
                raise AemetError("API key de AEMET no válida (401).")
            try:
                meta = r.json()
            except ValueError as exc:
                raise AemetError(f"Respuesta no JSON de AEMET ({r.status_code}) en {url}") from exc

            estado = int(meta.get("estado", r.status_code))
            if estado == 404:
                raise AemetSinDatos(meta.get("descripcion", url))
            if estado == 429:
                log.warning("AEMET estado 429. Esperando %ss…", self.cfg.espera_429_segundos)
                time.sleep(self.cfg.espera_429_segundos)
                continue
            if estado != 200 or "datos" not in meta:
                raise AemetError(f"AEMET estado {estado}: {meta.get('descripcion')} ({url})")

            datos = self._get(meta["datos"])
            if datos.status_code == 404:
                # La URL temporal de datos ha caducado: se repite el primer paso.
                log.warning("URL de datos caducada; reintentando (%s)", intento)
                continue
            datos.raise_for_status()
            return datos.content, meta["datos"]
        raise AemetError(f"No se pudo completar la petición {url}")

    # -- alto nivel ---------------------------------------------------------
    def inventario(self) -> tuple[bytes, list[dict[str, Any]]]:
        contenido, _ = self.peticion(RUTA_INVENTARIO)
        return contenido, decodificar_json(contenido, self.cfg.encoding)

    def diarios(self, idema: str, ini: date, fin: date) -> tuple[bytes, str]:
        ruta = RUTA_DIARIOS.format(ini=ini.isoformat(), fin=fin.isoformat(), idema=idema)
        return self.peticion(ruta)


def decodificar_json(contenido: bytes, encoding: str = "ISO-8859-15") -> Any:
    for enc in (encoding, "utf-8", "latin-1"):
        try:
            return json.loads(contenido.decode(enc))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise AemetError("No se pudo decodificar el JSON de AEMET")


def filtrar_estaciones_galicia(inventario: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Devuelve las estaciones gallegas del inventario con coordenadas en decimal."""
    df = pd.DataFrame(list(inventario))
    if df.empty:
        return df
    df = df[df["provincia"].map(es_provincia_gallega)].copy()
    df["latitud_dec"] = df["latitud"].map(dms_a_decimal)
    df["longitud_dec"] = df["longitud"].map(dms_a_decimal)
    return df.sort_values("indicativo").reset_index(drop=True)


def nombre_tramo(idema: str, ini: date, fin: date) -> str:
    seguro = "".join(c if c.isalnum() else "-" for c in idema)
    return f"diarios/{seguro}_{ini.isoformat()}_{fin.isoformat()}.json"


def tramo_consolidado(fin: date, dias_consolidacion: int, hoy: date | None = None) -> bool:
    return fin < (hoy or date.today()) - timedelta(days=dias_consolidacion)


def fechas_en_json(contenido: bytes, encoding: str) -> set[str]:
    try:
        registros = decodificar_json(contenido, encoding)
    except AemetError:
        return set()
    if not isinstance(registros, list):
        return set()
    return {r["fecha"] for r in registros if isinstance(r, dict) and r.get("fecha")}


def informe_cobertura(fechas_por_estacion: dict[str, set[str]], desde: date, hasta: date,
                      estaciones: pd.DataFrame | None = None,
                      errores: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """Informe de cobertura por estación (criterio de aceptación de HU-01)."""
    esperados = (hasta - desde).days + 1
    errores = errores or {}
    filas = []
    for idema, fechas in sorted(fechas_por_estacion.items()):
        validas = sorted(f for f in fechas if desde.isoformat() <= f[:10] <= hasta.isoformat())
        filas.append({
            "indicativo": idema,
            "dias_esperados": esperados,
            "dias_con_datos": len(validas),
            "cobertura_pct": round(100 * len(validas) / esperados, 2) if esperados else 0.0,
            "primera_fecha": validas[0][:10] if validas else None,
            "ultima_fecha": validas[-1][:10] if validas else None,
            "n_errores": len(errores.get(idema, [])),
        })
    df = pd.DataFrame(filas)
    if estaciones is not None and not estaciones.empty and not df.empty:
        cols = [c for c in ("indicativo", "nombre", "provincia", "altitud", "latitud_dec", "longitud_dec")
                if c in estaciones.columns]
        df = estaciones[cols].merge(df, on="indicativo", how="right")
    return df


def ingestar(desde: date, hasta: date, estaciones: list[str] | None = None,
             forzar: bool = False, base_dir: Path | None = None,
             cliente: ClienteAemet | None = None) -> EjecucionBronze:
    """Descarga los valores climatológicos diarios de AEMET para ``[desde, hasta]``.

    :param estaciones: indicativos concretos; si es ``None`` se usan todas las
        estaciones gallegas del inventario.
    :param forzar: ignora la caché de Bronze y vuelve a descargarlo todo.
    """
    cliente = cliente or ClienteAemet()
    cfg = cliente.cfg
    parametros = {"desde": desde.isoformat(), "hasta": hasta.isoformat(),
                  "estaciones": estaciones or "galicia", "forzar": forzar,
                  "max_dias_por_peticion": cfg.max_dias_por_peticion}

    with EjecucionBronze("aemet", parametros=parametros, base_dir=base_dir) as run:
        # 1. Inventario (siempre se archiva: es pequeño y puede cambiar)
        bruto_inv, inventario = cliente.inventario()
        run.guardar_bytes("inventario_estaciones.json", bruto_inv, origen=RUTA_INVENTARIO)
        gallegas = filtrar_estaciones_galicia(inventario)
        if estaciones:
            pedidas = set(estaciones)
            gallegas = filtrar_estaciones_galicia(
                [e for e in inventario if e.get("indicativo") in pedidas]
            )
            for idema in sorted(pedidas - set(gallegas.get("indicativo", []))):
                run.incidencia("estacion_desconocida", f"{idema} no está en el inventario o no es gallega")
        if gallegas.empty:
            gallegas = pd.DataFrame(columns=["indicativo"])
        gallegas.to_csv(run.ruta("estaciones_galicia.csv"), index=False)
        run.registrar_existente("estaciones_galicia.csv", origen="derivado:inventario_estaciones.json")
        log.info("%d estaciones a procesar", len(gallegas))

        # 2. Valores diarios por estación y tramo
        fechas: dict[str, set[str]] = {}
        errores: dict[str, list[str]] = {}
        tramos = list(tramos_fechas(desde, hasta, cfg.max_dias_por_peticion))
        total = len(gallegas) * len(tramos)
        n = 0
        for idema in gallegas["indicativo"]:
            fechas.setdefault(idema, set())
            for ini, fin in tramos:
                n += 1
                relativa = nombre_tramo(idema, ini, fin)
                if not forzar and tramo_consolidado(fin, cfg.dias_consolidacion):
                    cache = buscar_en_cache("aemet", relativa, excluir=run.dir, base_dir=base_dir)
                    if cache is not None:
                        run.reutilizar(relativa, cache)
                        fechas[idema] |= fechas_en_json(cache.read_bytes(), cfg.encoding)
                        continue
                try:
                    contenido, url_datos = cliente.diarios(idema, ini, fin)
                except AemetSinDatos:
                    log.debug("Sin datos %s %s–%s", idema, ini, fin)
                    continue
                except AemetError as exc:
                    errores.setdefault(idema, []).append(str(exc))
                    run.incidencia("error_descarga", str(exc), estacion=idema,
                                   desde=ini.isoformat(), hasta=fin.isoformat())
                    continue
                run.guardar_bytes(relativa, contenido, origen=url_datos.split("?")[0],
                                  estacion=idema, desde=ini.isoformat(), hasta=fin.isoformat())
                fechas[idema] |= fechas_en_json(contenido, cfg.encoding)
                if n % 25 == 0:
                    log.info("AEMET %d/%d peticiones", n, total)

        # 3. Informe de cobertura por estación
        cobertura = informe_cobertura(fechas, desde, hasta, gallegas, errores)
        cobertura.to_csv(run.ruta("cobertura_estaciones.csv"), index=False)
        run.registrar_existente("cobertura_estaciones.csv", origen="derivado:diarios")
        if not cobertura.empty:
            log.info("Cobertura media: %.1f %% (%d estaciones)",
                     cobertura["cobertura_pct"].mean(), len(cobertura))
    return run
