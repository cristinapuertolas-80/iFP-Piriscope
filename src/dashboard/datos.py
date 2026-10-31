"""Lectura y preparación de datos del dashboard, sin dependencias de Streamlit.

Sigue el contrato de datos con ML (``docs/contrato_dashboard_gold.md``):

* Solo se leen ``predicciones.parquet`` y ``explicaciones.parquet`` de Gold. Por
  defecto se descargan del dataset de Hugging Face (``HF_TOKEN`` de lectura); si
  no hay conexión, se usan los de ``data/gold`` (``PIRISCOPE_DASHBOARD_ORIGEN=local``
  los usa siempre).
* El catálogo y los límites de los 313 municipios salen del GeoJSON versionado en
  ``assets/`` (lo genera ``scripts/generar_mapa_municipios.py``).
* Los niveles se toman siempre de ``nivel_riesgo``: nunca se recalculan con umbrales.
"""

from __future__ import annotations

import json
import logging
import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.ingest import config

log = logging.getLogger("piriscope.dashboard")
# En Windows sin modo desarrollador la caché de Hugging Face funciona sin enlaces simbólicos: no es un error
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
GEOJSON_MUNICIPIOS = ASSETS_DIR / "municipios_galicia.geojson"
RESULTADOS_DIR = config.REPO_ROOT / "docs" / "resultados"

FICHEROS_GOLD = ("predicciones.parquet", "explicaciones.parquet")
RUTA_GOLD_EN_HF = "data/gold"

NIVELES = ("bajo", "medio", "alto")
SIN_DATO = "sin dato"
#: Umbral de municipios en «medio» o «alto» para el aviso de episodio generalizado (DEC-49)
UMBRAL_EPISODIO = 0.90
COLUMNAS_PRED = ["id_prediccion", "id_municipio", "fecha_objetivo", "version_modelo",
                 "probabilidad", "nivel_riesgo", "fwi", "fecha_insert"]


# ---------------------------------------------------------------------------
# Origen de los ficheros de Gold
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OrigenGold:
    predicciones: Path
    explicaciones: Path
    origen: str            # "huggingface" o "local"
    aviso: str | None = None


def _token() -> str | None:
    token = (config.HuggingFaceConfig().token or "").strip()
    return token or None


def descargar_gold_hf(forzar: bool = False) -> dict[str, Path]:
    """Descarga los ficheros de Gold del dataset (caché de Hugging Face; solo baja lo que cambió)."""
    from huggingface_hub import hf_hub_download

    cfg = config.HuggingFaceConfig()
    return {
        nombre: Path(hf_hub_download(cfg.repo_id, f"{RUTA_GOLD_EN_HF}/{nombre}", repo_type=cfg.repo_type,
                                     token=_token(), force_download=forzar))
        for nombre in FICHEROS_GOLD
    }


def localizar_gold(forzar_descarga: bool = False, gold_dir: Path | None = None) -> OrigenGold:
    """Rutas de los ficheros de Gold: Hugging Face por defecto, ``data/gold`` como respaldo."""
    gold_dir = Path(gold_dir or config.GOLD_DIR)
    locales = {n: gold_dir / n for n in FICHEROS_GOLD}
    hay_locales = all(p.exists() for p in locales.values())

    if os.getenv("PIRISCOPE_DASHBOARD_ORIGEN", "hf").strip().lower() == "local":
        if not hay_locales:
            raise FileNotFoundError(f"No están {', '.join(FICHEROS_GOLD)} en {gold_dir}")
        return OrigenGold(locales["predicciones.parquet"], locales["explicaciones.parquet"], "local")

    try:
        rutas = descargar_gold_hf(forzar_descarga)
        return OrigenGold(rutas["predicciones.parquet"], rutas["explicaciones.parquet"], "huggingface")
    except Exception as exc:  # noqa: BLE001
        motivo = _motivo_error_hf(exc)
        if not hay_locales:
            raise RuntimeError(f"No se pudieron obtener los datos de Hugging Face: {motivo}") from exc
        log.warning("Hugging Face no disponible (%s); se usan los ficheros de %s", motivo, gold_dir)
        return OrigenGold(locales["predicciones.parquet"], locales["explicaciones.parquet"], "local",
                          aviso=f"No se pudo conectar con Hugging Face ({motivo}). Se muestran los datos locales.")


def _motivo_error_hf(exc: Exception) -> str:
    texto = str(exc)
    if "401" in texto or "403" in texto or "gated" in texto.lower():
        return ("acceso denegado: revisa que HF_TOKEN esté en .env y que tu cuenta haya aceptado "
                "las condiciones del dataset")
    return type(exc).__name__


# ---------------------------------------------------------------------------
# Lectura
# ---------------------------------------------------------------------------
def deduplicar(pred: pd.DataFrame) -> pd.DataFrame:
    """Red de seguridad del contrato (apartado 2): una predicción por municipio y fecha.

    Gana el modelo principal sobre el respaldo y, entre iguales, la ``fecha_insert`` más reciente.
    """
    pred = pred.assign(_resp=es_respaldo(pred["version_modelo"]))
    pred = (pred.sort_values(["id_municipio", "fecha_objetivo", "_resp", "fecha_insert"],
                             ascending=[True, True, True, False])
                .drop_duplicates(["id_municipio", "fecha_objetivo"]))
    return pred.drop(columns="_resp").reset_index(drop=True)


def leer_predicciones(ruta: Path) -> pd.DataFrame:
    columnas = [c for c in COLUMNAS_PRED if c in _columnas_parquet(ruta)]
    pred = pd.read_parquet(ruta, columns=columnas)
    pred["id_municipio"] = pred["id_municipio"].astype(str).str.zfill(5)
    pred["fecha_objetivo"] = pd.to_datetime(pred["fecha_objetivo"]).dt.normalize()
    if "fecha_insert" not in pred.columns:
        pred["fecha_insert"] = pd.NaT
    return deduplicar(pred)


def leer_explicacion(ruta: Path, id_prediccion: str) -> pd.DataFrame:
    """Las 5 filas SHAP de una predicción, sin cargar el fichero entero (más de 1,5 M filas)."""
    return pd.read_parquet(ruta, filters=[("id_prediccion", "==", id_prediccion)])


def _columnas_parquet(ruta: Path) -> list[str]:
    import pyarrow.parquet as pq

    return pq.read_schema(ruta).names


def leer_geojson(ruta: Path = GEOJSON_MUNICIPIOS) -> dict:
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def catalogo_municipios(geojson: dict) -> pd.DataFrame:
    """Los 313 municipios (id, nombre, provincia y punto interior) a partir del GeoJSON."""
    cat = pd.DataFrame([f["properties"] for f in geojson["features"]])
    cat["id_municipio"] = cat["id_municipio"].astype(str).str.zfill(5)
    return cat.sort_values("nombre", key=lambda s: s.map(clave_orden)).reset_index(drop=True)


def clave_orden(nombre: str) -> str:
    """Ordena «A Coruña» junto a «Coruña» y sin tener en cuenta tildes."""
    clave = unicodedata.normalize("NFKD", str(nombre)).encode("ascii", "ignore").decode().upper().strip()
    for art in ("O ", "A ", "OS ", "AS "):
        if clave.startswith(art):
            return clave[len(art):] + " " + art.strip()
    return clave


# ---------------------------------------------------------------------------
# Preparación de lo que se muestra
# ---------------------------------------------------------------------------
def es_respaldo(version: pd.Series) -> pd.Series:
    """Todo lo que termina en ``-respaldo`` (no se supone que solo exista ``v1``)."""
    return version.astype(str).str.endswith("-respaldo")


def predicciones_dia(pred: pd.DataFrame, catalogo: pd.DataFrame, fecha) -> pd.DataFrame:
    """Los 313 municipios con la predicción de ``fecha`` (o «sin dato») y su prioridad relativa."""
    fecha = pd.Timestamp(fecha).normalize()
    dia = pred[pred["fecha_objetivo"] == fecha].copy()
    if not dia.empty:
        # 0 = el municipio de mayor riesgo del día, por versión de modelo (contrato 4.4)
        dia["prioridad_dia"] = 1 - dia.groupby("version_modelo")["probabilidad"].rank(pct=True, method="max")
    dia = catalogo.merge(dia, on="id_municipio", how="left")
    dia["nivel_mostrado"] = dia["nivel_riesgo"].fillna(SIN_DATO)
    dia["respaldo"] = dia["version_modelo"].notna() & es_respaldo(dia["version_modelo"].fillna(""))
    if "prioridad_dia" not in dia.columns:
        dia["prioridad_dia"] = float("nan")
    return dia


def episodio_generalizado(dia: pd.DataFrame, umbral: float = UMBRAL_EPISODIO) -> bool:
    """≥ 90 % de los municipios con predicción en «medio» o «alto» (DEC-49)."""
    con_dato = dia[dia["nivel_mostrado"] != SIN_DATO]
    return bool(len(con_dato)) and con_dato["nivel_mostrado"].isin(["medio", "alto"]).mean() >= umbral


def fecha_con_datos(pred: pd.DataFrame, fecha) -> pd.Timestamp | None:
    """La fecha pedida si tiene predicciones; si no, la última disponible (contrato 3)."""
    fecha = pd.Timestamp(fecha).normalize()
    fechas = pred["fecha_objetivo"]
    if fechas.empty:
        return None
    return fecha if (fechas == fecha).any() else fechas.max()


def historico_municipio(pred: pd.DataFrame, id_municipio: str, desde=None, hasta=None) -> pd.DataFrame:
    h = pred[pred["id_municipio"] == id_municipio]
    if desde is not None:
        h = h[h["fecha_objetivo"] >= pd.Timestamp(desde)]
    if hasta is not None:
        h = h[h["fecha_objetivo"] <= pd.Timestamp(hasta)]
    return h.sort_values("fecha_objetivo")


def tiene_columnas_dec50(expl: pd.DataFrame) -> bool:
    """Si las explicaciones traen ``valor_variable``, ``valor_base`` y ``shap_resto`` con datos."""
    columnas = ("valor_variable", "valor_base", "shap_resto")
    return all(c in expl.columns for c in columnas) and expl["valor_base"].notna().any()


# ---------------------------------------------------------------------------
# Resultados de evaluación (ficheros estáticos de docs/resultados, contrato 11)
# ---------------------------------------------------------------------------
def leer_resultados(directorio: Path = RESULTADOS_DIR) -> dict[str, pd.DataFrame | None]:
    ficheros = {"final": "informe_final.csv", "walkforward": "informe_walkforward.csv",
                "shap_global": "shap_global.csv", "shap_grupos": "shap_global_grupos.csv"}
    salida: dict[str, pd.DataFrame | None] = {}
    for clave, nombre in ficheros.items():
        ruta = Path(directorio) / nombre
        salida[clave] = pd.read_csv(ruta, encoding="utf-8") if ruta.exists() else None
    return salida


# ---------------------------------------------------------------------------
# Formato
# ---------------------------------------------------------------------------
def fmt_pct(p, decimales: int = 1) -> str:
    if p is None or pd.isna(p):
        return "—"
    return f"{100 * float(p):.{decimales}f} %".replace(".", ",")


def fmt_num(x, decimales: int = 1) -> str:
    if x is None or pd.isna(x):
        return "no disponible"
    return f"{float(x):.{decimales}f}".replace(".", ",")


def fmt_fecha(f) -> str:
    return pd.Timestamp(f).strftime("%d/%m/%Y")


def fmt_prioridad(p) -> str:
    """«top 12 %» a partir de la prioridad relativa (0 = mayor riesgo del día)."""
    if p is None or pd.isna(p):
        return "—"
    return f"top {max(1, round(100 * float(p)))} %"
