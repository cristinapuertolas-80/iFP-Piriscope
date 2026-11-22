"""Monitorización diaria del sistema (NF4, propuesta 8).

Genera ``data/gold/monitorizacion.json`` con el estado de la última ejecución:
cobertura de municipios, uso del modelo de respaldo, reparto de niveles, retraso
de las fuentes (AEMET y FWI) y deriva (PSI) de las variables frente al
entrenamiento. No mide el rendimiento real del modelo: las etiquetas de EGIF
llegan con más de un año de retraso.

Uso (desde la raíz del repositorio, tras la inferencia):

    python -m src.ml.monitorizar [--resumen-md RUTA]

Es informativo: termina con código 0 aunque haya avisos. Solo falla si falta
alguna entrada o no se puede escribir la salida.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("piriscope.ml.monitorizar")

VERSION_ESQUEMA = 1
NIVELES = ("bajo", "medio", "alto")

# Variables sobre las que se mide la deriva. Si el .json del modelo trae su lista
# de variables se usa esa; "mes" (circular) y "viento" (excluido del modelo) no entran.
VARIABLES_POR_DEFECTO = [
    "temperatura", "humedad_relativa", "precipitacion", "racha_dias_secos",
    "precip_acum_7d", "fwi", "ffmc", "dmc", "dc", "isi", "bui",
]
EXCLUIDAS_DERIVA = {"mes", "viento"}

# Umbrales PROVISIONALES: ajustar tras las primeras semanas de ejecuciones reales.
PSI_VIGILAR = 0.10
PSI_DERIVA = 0.25
COBERTURA_MIN_PCT = 90.0      # municipios con predicción en la última fecha
RETRASO_AEMET_MAX = 7         # días entre hoy y el último dato de AEMET
RETRASO_FWI_MAX = 14          # el FWI llega con ~5 días de retraso y se ingiere cada lunes
EPISODIO_PCT = 90.0           # % de municipios en medio/alto: episodio generalizado (DEC-49)
MARGEN_ESTACIONAL_DIAS = 3    # la referencia son los mismos días del año de los años de entrenamiento


# --------------------------------------------------------------------------- PSI
def psi(referencia, actual, bins: int = 10, eps: float = 1e-4, min_ref: int = 50, min_act: int = 30):
    """Índice de estabilidad de la población (PSI) de ``actual`` frente a ``referencia``.

    Los cortes salen de los cuantiles de la referencia. Si la variable tiene pocos
    valores distintos se usa cada valor como categoría (y un hueco «otro»).
    Devuelve ``None`` si no hay datos suficientes. Los nulos se ignoran (se
    informan aparte).
    """
    ref = pd.Series(referencia).dropna().to_numpy(dtype=float)
    act = pd.Series(actual).dropna().to_numpy(dtype=float)
    if len(ref) < min_ref or len(act) < min_act:
        return None

    uniq = np.unique(ref)
    if len(uniq) < 2:
        return None
    if len(uniq) <= bins:
        idx_ref = np.searchsorted(uniq, ref)
        pos = np.searchsorted(uniq, act)
        pos_c = np.clip(pos, 0, len(uniq) - 1)
        idx_act = np.where(uniq[pos_c] == act, pos_c, len(uniq))   # no visto en la referencia -> «otro»
        n_bins = len(uniq) + 1
    else:
        cortes = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))[1:-1]
        idx_ref = np.searchsorted(cortes, ref, side="right")
        idx_act = np.searchsorted(cortes, act, side="right")
        n_bins = len(cortes) + 1

    e = np.clip(np.bincount(idx_ref, minlength=n_bins) / len(ref), eps, None)
    a = np.clip(np.bincount(idx_act, minlength=n_bins) / len(act), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def estado_psi(valor) -> str:
    if valor is None:
        return "sin_datos"
    if valor >= PSI_DERIVA:
        return "deriva"
    if valor >= PSI_VIGILAR:
        return "vigilar"
    return "estable"


def _mascara_estacional(fechas_ref: pd.Series, fechas_act: pd.Series, margen: int = MARGEN_ESTACIONAL_DIAS):
    """Filas de la referencia cuyo día del año cae cerca de alguno de la ventana actual.

    Sin esto, noviembre se compararía con el año entero y toda la meteorología
    «derivaría» solo por la estación.
    """
    permitido = np.zeros(367, dtype=bool)
    for d in fechas_act.dt.dayofyear.unique():
        for k in range(-margen, margen + 1):
            permitido[(int(d) - 1 + k) % 366 + 1] = True
    return permitido[fechas_ref.dt.dayofyear.to_numpy()]


# ------------------------------------------------------------------- bloques
def _bloque_predicciones(pred: pd.DataFrame, total_municipios: int, ventana: int) -> dict:
    pred = pred.copy()
    pred["fecha_objetivo"] = pd.to_datetime(pred["fecha_objetivo"])
    ultima = pred["fecha_objetivo"].max()
    v = pred[pred["fecha_objetivo"] >= ultima - pd.Timedelta(days=ventana - 1)]

    filas = []
    for fecha, d in v.groupby("fecha_objetivo"):
        n = int(d["id_municipio"].nunique())
        niveles = d["nivel_riesgo"].astype(str).value_counts()
        n_resp = int(d["version_modelo"].astype(str).str.endswith("-respaldo").sum())
        medio_alto = int(niveles.get("medio", 0) + niveles.get("alto", 0))
        filas.append({
            "fecha": fecha.strftime("%Y-%m-%d"),
            "municipios": n,
            "cobertura_pct": round(100 * n / total_municipios, 1),
            "pct_respaldo": round(100 * n_resp / len(d), 1),
            "niveles": {k: int(niveles.get(k, 0)) for k in NIVELES},
            "pct_medio_alto": round(100 * medio_alto / len(d), 1),
            "probabilidades_distintas": int(d["probabilidad"].nunique()),
        })

    dias_resp = 0
    for f in reversed(filas):
        if f["pct_respaldo"] > 50:
            dias_resp += 1
        else:
            break

    return {
        "ultima_fecha_objetivo": ultima.strftime("%Y-%m-%d"),
        "n_predicciones_total": int(len(pred)),
        "ultima_fecha": filas[-1],
        "dias_finales_con_respaldo": dias_resp,
        "ventana": {
            "dias": int(ventana),
            "cobertura_min_pct": min(f["cobertura_pct"] for f in filas),
            "por_fecha": filas,
        },
    }


def _bloque_fuentes(calidad: dict, hoy: date) -> dict:
    def _dias(hasta):
        return (hoy - date.fromisoformat(hasta)).days if hasta else None

    meteo = calidad.get("observacion_meteorologica", {})
    fwi = calidad.get("indicador_fwi", {})
    h_meteo, h_fwi = meteo.get("hasta"), fwi.get("hasta")
    return {
        "aemet": {"hasta": h_meteo, "retraso_dias": _dias(h_meteo)},
        "fwi": {
            "hasta": h_fwi,
            "retraso_dias": _dias(h_fwi),
            "retraso_vs_meteo_dias": (date.fromisoformat(h_meteo) - date.fromisoformat(h_fwi)).days
            if h_meteo and h_fwi else None,
            "dias_intermedio": fwi.get("dias_intermedio"),
            "meses_intermedio": fwi.get("meses_intermedio", []),
        },
    }


def _bloque_deriva(dataset, entrenamiento, variables, ventana, anio_desde) -> dict:
    ds = dataset.copy()
    ds["fecha"] = pd.to_datetime(ds["fecha"])
    fmax = ds["fecha"].max()
    desde = fmax - pd.Timedelta(days=ventana - 1)
    act = ds[ds["fecha"] >= desde]

    ent = entrenamiento.copy()
    ent["fecha"] = pd.to_datetime(ent["fecha"])
    ent = ent[(ent["particion"] == "entrenamiento") & (ent["fecha"].dt.year >= anio_desde)]
    ent = ent[_mascara_estacional(ent["fecha"], act["fecha"])]

    filas = []
    for var in variables:
        if var not in act.columns or var not in ent.columns:
            continue
        valor = psi(ent[var], act[var])
        filas.append({
            "variable": var,
            "psi": None if valor is None else round(valor, 4),
            "estado": estado_psi(valor),
            "n_reciente": int(act[var].notna().sum()),
            "nulos_pct_reciente": round(100 * float(act[var].isna().mean()), 1),
            "nulos_pct_referencia": round(100 * float(ent[var].isna().mean()), 1),
        })
    con_psi = [f for f in filas if f["psi"] is not None]
    peor = max(con_psi, key=lambda f: f["psi"]) if con_psi else None
    return {
        "metodo": f"PSI con deciles de la referencia; referencia = entrenamiento desde {anio_desde}, "
                  f"mismos días del año (±{MARGEN_ESTACIONAL_DIAS}) que la ventana reciente",
        "umbrales": {"vigilar": PSI_VIGILAR, "deriva": PSI_DERIVA},
        "ventana": {"desde": desde.strftime("%Y-%m-%d"), "hasta": fmax.strftime("%Y-%m-%d")},
        "n_filas_referencia": int(len(ent)),
        "n_filas_ventana": int(len(act)),
        "psi_max": None if peor is None else peor["psi"],
        "variable_psi_max": None if peor is None else peor["variable"],
        "variables": filas,
    }


def _avisos(pred_b, fuentes, deriva) -> tuple[list[str], list[str]]:
    avisos, info = [], []
    u = pred_b["ultima_fecha"]
    if u["cobertura_pct"] < COBERTURA_MIN_PCT:
        avisos.append(f"Cobertura de la última fecha: {u['cobertura_pct']} % de los municipios "
                      f"(mínimo {COBERTURA_MIN_PCT:.0f} %).")
    r = fuentes["aemet"]["retraso_dias"]
    if r is not None and r > RETRASO_AEMET_MAX:
        avisos.append(f"AEMET lleva {r} días de retraso (máximo {RETRASO_AEMET_MAX}).")
    r = fuentes["fwi"]["retraso_dias"]
    if r is not None and r > RETRASO_FWI_MAX:
        avisos.append(f"El FWI lleva {r} días de retraso (máximo {RETRASO_FWI_MAX}).")
    en_deriva = [f["variable"] for f in deriva["variables"] if f["estado"] == "deriva"]
    if en_deriva:
        avisos.append("Deriva (PSI ≥ %.2f): %s." % (PSI_DERIVA, ", ".join(en_deriva)))
    if pred_b["dias_finales_con_respaldo"]:
        info.append(f"Las últimas {pred_b['dias_finales_con_respaldo']} fechas se predicen con el modelo de "
                    "respaldo (sin FWI): discrimina poco.")
    if u["pct_medio_alto"] >= EPISODIO_PCT:
        info.append(f"Episodio generalizado: {u['pct_medio_alto']} % de los municipios en medio o alto (DEC-49).")
    return avisos, info


# ---------------------------------------------------------------- orquestación
def construir_informe(pred, dataset, entrenamiento, calidad, variables, hoy: date, ventana: int = 30,
                      anio_desde: int = 2009, modelo: str | None = None, respaldo: str | None = None,
                      validacion: dict | None = None) -> dict:
    total = int(calidad.get("municipio", {}).get("filas") or dataset["id_municipio"].nunique())
    pred_b = _bloque_predicciones(pred, total, ventana)
    fuentes = _bloque_fuentes(calidad, hoy)
    deriva = _bloque_deriva(dataset, entrenamiento, variables, ventana, anio_desde)
    avisos, info = _avisos(pred_b, fuentes, deriva)
    informe = {
        "version_esquema": VERSION_ESQUEMA,
        "generado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "revisar" if avisos else "ok",
        "avisos": avisos,
        "info": info,
        "modelo": {"principal": modelo, "respaldo": respaldo},
        "predicciones": pred_b,
        "fuentes": fuentes,
        "deriva": deriva,
    }
    if validacion:
        informe["silver"] = {k: validacion.get(k) for k in ("n_comprobaciones", "n_errores", "n_avisos")}
    return informe


def resumen_markdown(i: dict) -> str:
    u = i["predicciones"]["ultima_fecha"]
    f = i["fuentes"]
    l = [f"### Monitorización · estado: **{i['estado']}**",
         f"- Última fecha: {u['fecha']} · {u['municipios']} municipios ({u['cobertura_pct']} %) · "
         f"respaldo {u['pct_respaldo']} % · niveles {u['niveles']}",
         f"- Retraso: AEMET {f['aemet']['retraso_dias']} d · FWI {f['fwi']['retraso_dias']} d "
         f"({f['fwi']['dias_intermedio']} días de FWI intermedio)",
         f"- Deriva: PSI máx. {i['deriva']['psi_max']} ({i['deriva']['variable_psi_max']})"]
    l += [f"- ⚠️ {a}" for a in i["avisos"]] + [f"- ℹ️ {a}" for a in i["info"]]
    return "\n".join(l) + "\n"


def _variables_modelo(ruta: Path) -> tuple[list[str], int | None]:
    try:
        meta = json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.warning("No se pudo leer %s: se usan las variables por defecto.", ruta)
        return VARIABLES_POR_DEFECTO, None
    variables = None
    for clave in ("variables", "features", "columnas"):
        v = meta.get(clave)
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            variables = [x for x in v if x not in EXCLUIDAS_DERIVA]
            break
    anio = next((meta[k] for k in ("anio_desde", "año_desde", "desde_anio") if isinstance(meta.get(k), int)), None)
    return variables or VARIABLES_POR_DEFECTO, anio


def _leer(ruta: str, columnas: list[str]) -> pd.DataFrame:
    """Lee solo las columnas pedidas que existan en el parquet (ahorra memoria con 3 M de filas)."""
    try:
        import pyarrow.parquet as pq
        disponibles = set(pq.read_schema(ruta).names)
        columnas = [c for c in columnas if c in disponibles]
    except ImportError:
        pass
    return pd.read_parquet(ruta, columns=columnas)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (pd.Timestamp, datetime, date)):
        return o.isoformat()
    raise TypeError(f"{type(o)} no serializable")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--pred", default="data/gold/predicciones.parquet")
    p.add_argument("--dataset", default="data/gold/dataset_municipio_dia.parquet")
    p.add_argument("--entrenamiento", default="data/gold/dataset_entrenamiento.parquet")
    p.add_argument("--calidad", default="data/silver/informes/informe_calidad.json")
    p.add_argument("--validacion", default="data/silver/informes/validacion.json")
    p.add_argument("--modelo", default="v1_xgboost_con_fwi-sin_viento-desde2009")
    p.add_argument("--respaldo", default="v1_xgboost_sin_fwi-sin_viento-desde2009")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--ventana-dias", type=int, default=30)
    p.add_argument("--salida", default="data/gold/monitorizacion.json")
    p.add_argument("--resumen-md", default=None, help="Fichero al que añadir un resumen (p. ej. $GITHUB_STEP_SUMMARY)")
    p.add_argument("--hoy", default=None, help="AAAA-MM-DD (por defecto, hoy en UTC; para pruebas)")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")

    hoy = date.fromisoformat(a.hoy) if a.hoy else datetime.now(timezone.utc).date()
    variables, anio = _variables_modelo(Path(a.models_dir) / f"{a.modelo}.json")
    columnas = ["fecha", *variables]
    try:
        pred = pd.read_parquet(a.pred)
        dataset = _leer(a.dataset, ["id_municipio", *columnas])
        entrenamiento = _leer(a.entrenamiento, [*columnas, "particion"])
        calidad = json.loads(Path(a.calidad).read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError) as e:
        log.error("Falta alguna entrada de la monitorización: %s", e)
        return 1
    try:
        validacion = json.loads(Path(a.validacion).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        validacion = None

    informe = construir_informe(pred, dataset, entrenamiento, calidad, variables, hoy,
                                ventana=a.ventana_dias, anio_desde=anio or 2009,
                                modelo=a.modelo, respaldo=a.respaldo, validacion=validacion)
    salida = Path(a.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(informe, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    log.info("Escrito %s · estado %s · %d avisos", salida, informe["estado"], len(informe["avisos"]))
    for aviso in informe["avisos"]:
        log.warning(aviso)
    texto = resumen_markdown(informe)
    print(texto)
    if a.resumen_md:
        with open(a.resumen_md, "a", encoding="utf-8") as fh:
            fh.write(texto)
    return 0


if __name__ == "__main__":
    sys.exit(main())
