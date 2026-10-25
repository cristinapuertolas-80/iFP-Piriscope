"""Validación de las tablas generadas en ``data/silver`` contra el modelo de datos.

Comprueba el contrato de cada entidad Silver (apartado 7 y Anexo A del diseño
técnico) sobre las tablas **ya generadas**, no sobre el código:

* esquema: exactamente las columnas del modelo de datos (``almacen.COLUMNAS``
  + ``fecha_insert``) y sus tipos (ids texto, ``fecha`` como fecha,
  ``fecha_insert`` en UTC);
* textos limpios: sin espacios en los bordes, dobles ni duros, y sin cadenas
  vacías (los vacíos son nulos);
* claves primarias únicas y no nulas, con el formato documentado
  (``id_municipio`` de 5 cifras, ``id_obs_meteo = id_municipio_fecha``…);
* claves foráneas: todo ``id_municipio`` existe en Municipio y todo
  ``id_estacion_aemet`` en EstacionAemet;
* reglas de negocio: 313 municipios de las 4 provincias, una fila por
  municipio y día sin huecos, estación fija por municipio, valores dentro de
  rango, ≥ 95 % de partes con código INE (HU-04)…

Cada comprobación es ``error`` (la tabla no cumple el contrato) o ``aviso``
(dato a revisar en el EDA). Uso::

    python -m src.silver.validacion          # código de salida 1 si hay errores
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.ingest import config

from . import almacen
from .calidad import RANGOS
from .fwi import VARIABLES as VARIABLES_FWI

log = logging.getLogger("piriscope.silver.validacion")

N_MUNICIPIOS = 313
MUNICIPIOS_POR_PROVINCIA = {"15": 93, "27": 67, "32": 92, "36": 61}
#: Caja de Galicia sin margen [N, O, S, E] para comprobar centroides y estaciones.
CAJA_GALICIA = (43.8, -9.35, 41.8, -6.7)
#: HU-04: porcentaje mínimo de partes con código INE desde este año.
PCT_MIN_INE, ANIO_MIN_INE = 95.0, 2001
DIST_MAX_ESTACION_KM = 50.0

ESQUEMAS: dict[str, list[str]] = {t: [*cols, "fecha_insert"] for t, cols in almacen.COLUMNAS.items()}
CLAVES = {"municipio": "id_municipio", "estacion_aemet": "id_estacion_aemet",
          "observacion_meteorologica": "id_obs_meteo", "indicador_fwi": "id_indicador_fwi",
          "evento_incendio": "id_evento"}
COLUMNAS_ID = {"id_municipio", "id_estacion_aemet", "id_obs_meteo", "id_indicador_fwi", "id_evento"}


@dataclass
class Resultado:
    tabla: str
    comprobacion: str
    nivel: str  # "error" | "aviso"
    ok: bool
    detalle: str = ""


class Validador:
    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir
        self.resultados: list[Resultado] = []

    # -- registro -------------------------------------------------------------
    def comprobar(self, tabla: str, nombre: str, ok: bool, detalle: str = "", nivel: str = "error") -> bool:
        ok = bool(ok)
        self.resultados.append(Resultado(tabla, nombre, nivel, ok, detalle))
        if not ok:
            (log.error if nivel == "error" else log.warning)("[%s] %s: %s", tabla, nombre, detalle)
        return ok

    @property
    def errores(self) -> list[Resultado]:
        return [r for r in self.resultados if not r.ok and r.nivel == "error"]

    @property
    def avisos(self) -> list[Resultado]:
        return [r for r in self.resultados if not r.ok and r.nivel == "aviso"]

    # -- comprobaciones genéricas ---------------------------------------------
    def esquema(self, tabla: str) -> pd.DataFrame | None:
        ruta = almacen.ruta_tabla(tabla, self.base_dir)
        if not self.comprobar(tabla, "existe", ruta.exists(), str(ruta)):
            return None
        tipos = {f.name: str(f.type) for f in pq.read_schema(ruta)}
        faltan = [c for c in ESQUEMAS[tabla] if c not in tipos]
        self.comprobar(tabla, "columnas_obligatorias", not faltan, f"faltan {faltan}")
        sobran = [c for c in tipos if c not in ESQUEMAS[tabla]]
        self.comprobar(tabla, "sin_columnas_fuera_del_modelo", not sobran, f"sobran {sobran}")
        ids_no_texto = [c for c in COLUMNAS_ID & set(tipos) if tipos[c] not in ("string", "large_string")]
        self.comprobar(tabla, "ids_como_texto", not ids_no_texto, f"no son texto: {ids_no_texto}")
        if "fecha" in tipos:
            self.comprobar(tabla, "fecha_es_date32", tipos["fecha"] == "date32[day]", tipos["fecha"])
        if "fecha_insert" in tipos:
            self.comprobar(tabla, "fecha_insert_utc", "UTC" in tipos["fecha_insert"], tipos["fecha_insert"])
        df = almacen.leer(tabla, self.base_dir)
        self.comprobar(tabla, "no_vacia", len(df) > 0, "la tabla no tiene filas")
        self.textos_limpios(tabla, df, [c for c, t in tipos.items() if t in ("string", "large_string")])
        if "fecha_insert" in df:
            self.comprobar(tabla, "fecha_insert_sin_nulos", df["fecha_insert"].notna().all(),
                           f"{int(df['fecha_insert'].isna().sum())} nulos")
        return df

    def textos_limpios(self, tabla: str, df: pd.DataFrame, columnas: list[str]) -> None:
        sucias = {}
        for col in columnas:
            valores = pd.Series(df[col].dropna().astype(str).unique())
            mal = valores[(valores != valores.str.strip()) | valores.str.contains(r"\s\s|\xa0", regex=True)
                          | (valores == "")]
            if len(mal):
                sucias[col] = mal.head(3).tolist()
        self.comprobar(tabla, "textos_limpios", not sucias, f"espacios sobrantes o vacíos en {sucias}")

    def clave_primaria(self, tabla: str, df: pd.DataFrame) -> None:
        clave = CLAVES[tabla]
        self.comprobar(tabla, "clave_sin_nulos", df[clave].notna().all(), f"{int(df[clave].isna().sum())} nulos")
        dup = int(df[clave].duplicated().sum())
        self.comprobar(tabla, "clave_unica", dup == 0, f"{dup} duplicados de {clave}")

    def clave_natural(self, tabla: str, df: pd.DataFrame, id_col: str) -> None:
        esperado = df["id_municipio"] + "_" + df["fecha"].dt.strftime("%Y-%m-%d")
        mal = int((df[id_col] != esperado).sum())
        self.comprobar(tabla, f"{id_col}_formato", mal == 0, f"{mal} filas con id distinto de id_municipio_fecha")
        dup = int(df.duplicated(["id_municipio", "fecha"]).sum())
        self.comprobar(tabla, "una_fila_por_municipio_y_dia", dup == 0, f"{dup} repetidas")

    def foranea(self, tabla: str, df: pd.DataFrame, col: str, validos: set[str], nulos_ok: bool = False) -> None:
        valores = df[col].dropna() if nulos_ok else df[col]
        huerfanos = sorted(set(valores.astype(str)) - validos)
        self.comprobar(tabla, f"fk_{col}", not huerfanos, f"{len(huerfanos)} valores sin referencia: {huerfanos[:5]}")

    def rangos(self, tabla: str, df: pd.DataFrame) -> None:
        for col, (minimo, maximo) in RANGOS.items():
            if col not in df.columns:
                continue
            fuera = pd.Series(False, index=df.index)
            if minimo is not None:
                fuera |= (df[col] < minimo).fillna(False)
            if maximo is not None:
                fuera |= (df[col] > maximo).fillna(False)
            self.comprobar(tabla, f"rango_{col}", not fuera.any(), f"{int(fuera.sum())} valores fuera de [{minimo}, {maximo}]")

    def en_galicia(self, tabla: str, lat: pd.Series, lon: pd.Series, nombre: str) -> None:
        n, o, s, e = CAJA_GALICIA
        fuera = int((~(lat.between(s, n) & lon.between(o, e))).sum())
        self.comprobar(tabla, nombre, fuera == 0, f"{fuera} puntos fuera de Galicia")

    # -- comprobaciones por tabla -----------------------------------------------
    def municipio(self) -> set[str]:
        df = self.esquema("municipio")
        if df is None:
            return set()
        self.clave_primaria("municipio", df)
        self.comprobar("municipio", "n_municipios", len(df) == N_MUNICIPIOS, f"{len(df)} (esperados {N_MUNICIPIOS})")
        self.comprobar("municipio", "id_municipio_5_cifras", df["id_municipio"].str.fullmatch(r"\d{5}").all(),
                       "hay códigos que no son de 5 cifras")
        self.comprobar("municipio", "provincia_coherente_con_codigo",
                       (df["id_municipio"].str[:2].map(config.PROVINCIAS_GALICIA) == df["provincia"]).all(),
                       "provincia no coincide con las 2 primeras cifras de id_municipio")
        por_prov = df["id_municipio"].str[:2].value_counts().to_dict()
        self.comprobar("municipio", "municipios_por_provincia", por_prov == MUNICIPIOS_POR_PROVINCIA, str(por_prov))
        self.comprobar("municipio", "nombre_sin_nulos", df["nombre"].notna().all(), "municipios sin nombre")
        self.en_galicia("municipio", df["lat_centroide"], df["lon_centroide"], "centroides_en_galicia")
        self.comprobar("municipio", "reservado_definido", df["reservado"].notna().all(),
                       "reservado nulo: falta docs/municipios_reservados.csv (ML)", nivel="aviso")
        return set(df["id_municipio"])

    def estacion_aemet(self) -> set[str]:
        df = self.esquema("estacion_aemet")
        if df is None:
            return set()
        self.clave_primaria("estacion_aemet", df)
        con = df.dropna(subset=["latitud", "longitud"])
        self.comprobar("estacion_aemet", "coordenadas_sin_nulos", len(con) == len(df),
                       f"{len(df) - len(con)} estaciones sin coordenadas", nivel="aviso")
        self.en_galicia("estacion_aemet", con["latitud"], con["longitud"], "estaciones_en_galicia")
        return set(df["id_estacion_aemet"])

    def observacion_meteorologica(self, municipios: set[str], estaciones: set[str]) -> None:
        t = "observacion_meteorologica"
        df = self.esquema(t)
        if df is None:
            return
        self.clave_primaria(t, df)
        self.clave_natural(t, df, "id_obs_meteo")
        self.foranea(t, df, "id_municipio", municipios)
        self.foranea(t, df, "id_estacion_aemet", estaciones)
        self.comprobar(t, "todos_los_municipios", set(df["id_municipio"]) == municipios,
                       f"{len(municipios - set(df['id_municipio']))} municipios sin filas")
        dias = (df["fecha"].max() - df["fecha"].min()).days + 1
        incompletos = int((df.groupby("id_municipio").size() != dias).sum())
        self.comprobar(t, "serie_diaria_completa", incompletos == 0,
                       f"{incompletos} municipios con días sin fila en {df['fecha'].min().date()}–{df['fecha'].max().date()}")
        varias = int((df.groupby("id_municipio")["id_estacion_aemet"].nunique() > 1).sum())
        self.comprobar(t, "estacion_fija_por_municipio", varias == 0, f"{varias} municipios con varias estaciones")
        self.comprobar(t, "dist_estacion_km_valida",
                       df["dist_estacion_km"].between(0, DIST_MAX_ESTACION_KM).all(),
                       f"máx {df['dist_estacion_km'].max():.1f} km (límite {DIST_MAX_ESTACION_KM})")
        self.rangos(t, df)
        for col in ("temperatura", "precipitacion", "humedad_relativa", "viento"):
            pct = 100 * df[col].isna().mean()
            self.comprobar(t, f"cobertura_{col}", pct <= 30, f"{pct:.1f} % nulos", nivel="aviso")

    def indicador_fwi(self, municipios: set[str]) -> None:
        t = "indicador_fwi"
        df = self.esquema(t)
        if df is None:
            return
        self.clave_primaria(t, df)
        self.clave_natural(t, df, "id_indicador_fwi")
        self.foranea(t, df, "id_municipio", municipios)
        self.comprobar(t, "todos_los_municipios", set(df["id_municipio"]) == municipios,
                       f"{len(municipios - set(df['id_municipio']))} municipios sin FWI")
        por_dia = df.groupby("fecha")["id_municipio"].nunique()
        self.comprobar(t, "todos_los_municipios_cada_dia", (por_dia == len(municipios)).all(),
                       f"{int((por_dia != len(municipios)).sum())} días incompletos")
        dias = pd.date_range(df["fecha"].min(), df["fecha"].max())
        sin = dias.difference(pd.DatetimeIndex(por_dia.index))
        self.comprobar(t, "dias_sin_fwi", len(sin) == 0,
                       f"{len(sin)} días sin FWI (meses: {sorted({d.strftime('%Y-%m') for d in sin})[:12]})",
                       nivel="aviso")
        self.comprobar(t, "fwi_sin_nulos", df[VARIABLES_FWI].notna().all().all(),
                       f"{int(df[VARIABLES_FWI].isna().any(axis=1).sum())} filas con nulos", nivel="aviso")
        self.rangos(t, df)

    def evento_incendio(self, municipios: set[str]) -> None:
        t = "evento_incendio"
        df = self.esquema(t)
        if df is None:
            return
        self.clave_primaria(t, df)
        self.foranea(t, df, "id_municipio", municipios, nulos_ok=True)
        self.comprobar(t, "fecha_deteccion_sin_nulos", df["fecha_deteccion"].notna().all(),
                       f"{int(df['fecha_deteccion'].isna().sum())} partes sin fecha de detección")
        recientes = df[df["fecha_deteccion"].dt.year >= ANIO_MIN_INE]
        pct = 100 * recientes["id_municipio"].notna().mean() if len(recientes) else 0.0
        self.comprobar(t, "hu04_pct_codigo_ine", pct >= PCT_MIN_INE,
                       f"{pct:.2f} % de partes desde {ANIO_MIN_INE} con código INE (mínimo {PCT_MIN_INE} %)")
        al_reves = int((df["fecha_extincion"] < df["fecha_deteccion"]).fillna(False).sum())
        self.comprobar(t, "extincion_posterior_a_deteccion", al_reves == 0,
                       f"{al_reves} partes con extinción anterior a la detección", nivel="aviso")
        self.comprobar(t, "huso_valido", df["huso"].dropna().isin([29, 30]).all(), "husos distintos de 29/30",
                       nivel="aviso")
        self.rangos(t, df)

    # -- conjunto ---------------------------------------------------------------
    def ejecutar(self) -> "Validador":
        municipios = self.municipio()
        estaciones = self.estacion_aemet()
        self.observacion_meteorologica(municipios, estaciones)
        self.indicador_fwi(municipios)
        self.evento_incendio(municipios)
        return self

    def escribir(self) -> Path:
        ruta = almacen.informes_dir(self.base_dir) / "validacion.json"
        ruta.write_text(json.dumps({
            "ok": not self.errores, "n_comprobaciones": len(self.resultados),
            "n_errores": len(self.errores), "n_avisos": len(self.avisos),
            "resultados": [asdict(r) for r in self.resultados],
        }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return ruta


def validar(base_dir: Path | None = None) -> Validador:
    v = Validador(base_dir).ejecutar()
    v.escribir()
    return v


def main(argv: list[str] | None = None) -> int:
    from src.ingest.common import configurar_logging

    configurar_logging()
    v = validar()
    for r in v.resultados:
        marca = "OK   " if r.ok else ("ERROR" if r.nivel == "error" else "AVISO")
        print(f"{marca} {r.tabla:27s} {r.comprobacion:35s} {'' if r.ok else r.detalle}")
    print(f"\n{len(v.resultados)} comprobaciones: {len(v.errores)} errores, {len(v.avisos)} avisos "
          f"-> {almacen.informes_dir() / 'validacion.json'}")
    return 1 if v.errores else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
