"""Controles de calidad de Silver: duplicados, cobertura y valores fuera de rango.

Los valores físicamente imposibles **no se corrigen ni se imputan**: se dejan
como nulos y se registran en ``informes/valores_fuera_rango.csv`` para que el
EDA decida. La imputación de faltantes queda para Gold/ML.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from . import almacen

log = logging.getLogger("piriscope.silver.calidad")

#: Rangos admisibles (mínimo, máximo), ambos incluidos. ``None`` = sin límite.
RANGOS: dict[str, tuple[float | None, float | None]] = {
    # AEMET (valores diarios)
    "temperatura": (-30, 50),
    "humedad_relativa": (0, 100),
    "precipitacion": (0, 500),
    "viento": (0, 60),
    # FWI y componentes (Canadian Forest Fire Weather Index System)
    "fwi": (0, None),
    "ffmc": (0, 101),
    "dmc": (0, None),
    "dc": (0, None),
    "isi": (0, None),
    "bui": (0, None),
    # EGIF
    "superficie_forestal_ha": (0, None),
}
COLUMNAS_ANOMALIAS = ["tabla", "clave", "fecha", "columna", "valor", "minimo", "maximo"]


def aplicar_rangos(df: pd.DataFrame, tabla: str, clave: str, columna_fecha: str | None = None,
                   rangos: dict[str, tuple[float | None, float | None]] = RANGOS
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pone a nulo los valores fuera de rango y devuelve (df, registro de anomalías).

    ``columna_fecha`` da la fecha de cada anomalía, con la que una ejecución
    parcial sustituye sólo las del periodo reprocesado (:meth:`Informe.escribir`).
    """
    df = df.copy()
    registros = []
    for col, (minimo, maximo) in rangos.items():
        if col not in df.columns:
            continue
        valores = df[col]
        fuera = pd.Series(False, index=df.index)
        if minimo is not None:
            fuera |= valores < minimo
        if maximo is not None:
            fuera |= valores > maximo
        fuera = fuera.fillna(False).astype(bool)
        if fuera.any():
            registros.append(pd.DataFrame({
                "tabla": tabla, "clave": df.loc[fuera, clave].astype("string"),
                "fecha": pd.to_datetime(df.loc[fuera, columna_fecha]).dt.date if columna_fecha else None,
                "columna": col, "valor": valores[fuera].astype(float),
                "minimo": minimo, "maximo": maximo,
            }))
            df.loc[fuera, col] = pd.NA
    anomalias = pd.concat(registros, ignore_index=True) if registros else pd.DataFrame(
        columns=COLUMNAS_ANOMALIAS)
    if len(anomalias):
        log.warning("%s: %d valores fuera de rango puestos a nulo", tabla, len(anomalias))
    return df, anomalias


def duplicados(df: pd.DataFrame, clave: str | list[str]) -> int:
    return int(df.duplicated(subset=clave).sum())


def porcentaje_nulos(df: pd.DataFrame, columnas: list[str]) -> dict[str, float]:
    return {c: round(100 * float(df[c].isna().mean()), 2) for c in columnas if c in df.columns} \
        if len(df) else {}


class Informe:
    """Acumula los resultados de los controles y los escribe en ``informes/``."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir
        self.resumen: dict[str, Any] = {}
        self.fuera_rango: list[pd.DataFrame] = []
        #: (tabla, desde, hasta) cuyas anomalías sustituyen a las de ejecuciones previas
        self.reprocesado: list[tuple[str, date | None, date | None]] = []

    def tabla(self, nombre: str, **datos: Any) -> None:
        self.resumen.setdefault(nombre, {}).update(datos)

    def anomalias(self, df: pd.DataFrame, tabla: str, desde: date | None = None,
                  hasta: date | None = None) -> None:
        """Registra las anomalías de ``tabla`` en ``[desde, hasta]`` (sin límites: todas)."""
        self.reprocesado.append((tabla, desde, hasta))
        if len(df):
            self.fuera_rango.append(df)

    def csv(self, nombre: str, df: pd.DataFrame) -> Path:
        ruta = almacen.informes_dir(self.base_dir) / nombre
        df.to_csv(ruta, index=False)
        return ruta

    def escribir(self) -> Path:
        d = almacen.informes_dir(self.base_dir)
        self._escribir_fuera_rango(d / "valores_fuera_rango.csv")
        ruta = d / "informe_calidad.json"
        previo = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
        previo.update(self.resumen)  # una ejecución parcial no borra el resumen del resto de tablas
        ruta.write_text(json.dumps(previo, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return ruta

    def _escribir_fuera_rango(self, ruta: Path) -> None:
        # Se escribe siempre (aunque quede vacío) para no dejar el de una ejecución
        # anterior; de éste sólo se conservan las tablas y periodos no reprocesados.
        previo = pd.read_csv(ruta, dtype={"clave": "string"}) if ruta.exists() else None
        if previo is None or "fecha" not in previo:  # sin fichero o de antes de la columna fecha
            previo = pd.DataFrame(columns=COLUMNAS_ANOMALIAS)
        fecha = pd.to_datetime(previo["fecha"])
        for tabla, desde, hasta in self.reprocesado:
            sustituir = previo["tabla"] == tabla
            if desde:
                sustituir &= fecha >= pd.Timestamp(desde)
            if hasta:
                sustituir &= fecha <= pd.Timestamp(hasta)
            previo, fecha = previo[~sustituir], fecha[~sustituir]
        partes = [df for df in (previo, *self.fuera_rango) if len(df)]
        todo = pd.concat(partes, ignore_index=True) if partes else pd.DataFrame(columns=COLUMNAS_ANOMALIAS)
        todo["fecha"] = pd.to_datetime(todo["fecha"]).dt.date
        todo.sort_values(["tabla", "fecha", "clave", "columna"]).to_csv(ruta, index=False)
