"""Gráficos SHAP en Plotly a partir de las tablas de Gold (sin cargar el modelo).

Accesibilidad: colores de la paleta Okabe-Ito (aptos para daltonismo) y el
sentido de cada barra también va en texto (+/-), no solo en color.

Las opciones nuevas (etiquetas legibles, valores de las variables, resto, cascada) son opcionales: sin ellas,
los gráficos se comportan como antes. Los valores y el resto requieren las columnas ampliadas de DEC-50.
"""
from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go

COLOR_SUBE = "#D55E00"   # bermellón: aumenta el riesgo
COLOR_BAJA = "#0072B2"   # azul: reduce el riesgo
COLOR_RESTO = "#999999"  # gris: resto de variables y totales

ETIQUETAS = {
    "fwi": "FWI (índice meteorológico de incendio)",
    "ffmc": "FFMC (humedad del combustible fino)",
    "dmc": "DMC (humedad de la capa superficial)",
    "dc": "DC (sequía de capas profundas)",
    "isi": "ISI (propagación inicial)",
    "bui": "BUI (combustible disponible)",
    "temperatura": "Temperatura",
    "humedad_relativa": "Humedad relativa",
    "precipitacion": "Precipitación del día",
    "viento": "Viento",
    "racha_dias_secos": "Días seguidos sin lluvia",
    "precip_acum_7d": "Lluvia acumulada en 7 días",
    "mes": "Mes del año",
}
ENTEROS = {"mes", "racha_dias_secos"}


def _etiqueta(variable: str, etiquetas: dict | None) -> str:
    return (etiquetas or {}).get(variable, variable)


def _valor(variable: str, v) -> str:
    if v is None or pd.isna(v):
        return "sin dato"
    return f"{v:.0f}" if variable in ENTEROS else f"{v:.1f}".replace(".", ",")


def _pct(p: float) -> str:
    return f"{100 * p:.1f} %".replace(".", ",")


def _texto(v: float) -> str:
    return f"{v:+.2f} ({'sube' if v > 0 else 'baja'} el riesgo)"


def _filas(expl: pd.DataFrame, id_prediccion: str) -> pd.DataFrame:
    d = expl[expl["id_prediccion"] == id_prediccion]
    if d.empty:
        raise ValueError(f"No hay explicación para {id_prediccion!r}")
    return d


def fig_local(expl: pd.DataFrame, id_prediccion: str, unidades: str = "log-odds", *, titulo: str | None = None,
              etiquetas: dict | None = None, mostrar_valores: bool = False,
              incluir_resto: bool = False) -> go.Figure:
    """Barras con las variables que más contribuyen a UNA predicción.

    titulo           sustituye al título por defecto (p. ej. municipio, fecha e índice de riesgo).
    etiquetas        {variable: texto legible}; por ejemplo `ETIQUETAS`.
    mostrar_valores  añade «= valor» a cada variable (requiere la columna `valor_variable`).
    incluir_resto    añade una barra gris con el resto de variables (requiere `shap_resto`).
    """
    d = _filas(expl, id_prediccion)
    d = d.assign(_abs=d["valor_shap"].abs()).sort_values("_abs", kind="stable")   # la mayor, arriba
    nombres = [_etiqueta(v, etiquetas) for v in d["variable"]]
    if mostrar_valores and "valor_variable" in d.columns:
        nombres = [f"{n} = {_valor(v, x)}" for n, v, x in zip(nombres, d["variable"], d["valor_variable"])]
    valores = [float(v) for v in d["valor_shap"]]
    colores = [COLOR_SUBE if v > 0 else COLOR_BAJA for v in valores]
    textos = [_texto(v) for v in valores]
    if incluir_resto and "shap_resto" in d.columns and d["shap_resto"].notna().any():
        resto = float(d["shap_resto"].iloc[0])
        nombres.insert(0, "Resto de variables")
        valores.insert(0, resto)
        colores.insert(0, COLOR_RESTO)
        textos.insert(0, _texto(resto))
    fig = go.Figure(go.Bar(x=valores, y=nombres, orientation="h", marker_color=colores, text=textos,
                           textposition="outside"))
    fig.update_layout(
        title=titulo or "Variables que más influyen en esta predicción",
        xaxis_title=f"Contribución SHAP ({unidades})", yaxis_title="",
        font={"size": 14}, margin={"l": 10, "r": 10},
    )
    return fig


def fig_cascada(expl: pd.DataFrame, id_prediccion: str, unidades: str = "log-odds", *, titulo: str | None = None,
                etiquetas: dict | None = None) -> go.Figure:
    """Cascada desde el valor base hasta la predicción (requiere `valor_base`, DEC-50)."""
    d = _filas(expl, id_prediccion)
    if "valor_base" not in d.columns or d["valor_base"].isna().all():
        raise ValueError("Faltan valor_base/shap_resto: regenera las explicaciones con la versión ampliada (DEC-50)")
    d = d.assign(_abs=d["valor_shap"].abs()).sort_values("_abs", ascending=False, kind="stable")
    base = float(d["valor_base"].iloc[0])
    pasos = [(_etiqueta(v, etiquetas), float(s)) for v, s in zip(d["variable"], d["valor_shap"])]
    if "shap_resto" in d.columns and d["shap_resto"].notna().any():
        pasos.append(("Resto de variables", float(d["shap_resto"].iloc[0])))
    total = base + sum(s for _, s in pasos)
    a_prob = (lambda z: 1 / (1 + math.exp(-z))) if unidades == "log-odds" else (lambda z: z)
    p0, p1 = a_prob(base), a_prob(total)
    fig = go.Figure(go.Waterfall(
        x=["Valor base"] + [n for n, _ in pasos] + ["Predicción"],
        y=[base] + [s for _, s in pasos] + [0],
        measure=["absolute"] + ["relative"] * len(pasos) + ["total"],
        text=[_pct(p0)] + [f"{s:+.2f}" for _, s in pasos] + [_pct(p1)],
        textposition="outside",
        increasing={"marker": {"color": COLOR_SUBE}}, decreasing={"marker": {"color": COLOR_BAJA}},
        totals={"marker": {"color": COLOR_RESTO}},
    ))
    fig.update_layout(
        title=f"{titulo or 'Cómo se llega a esta predicción'} · de {_pct(p0)} (valor base) a {_pct(p1)}",
        yaxis_title=f"Escala del modelo ({unidades})", xaxis_title="",
        font={"size": 14}, margin={"l": 10, "r": 10}, showlegend=False,
    )
    fig.update_xaxes(tickangle=-30)
    return fig


def fig_global(importance: pd.DataFrame, top: int = 10, etiquetas: dict | None = None) -> go.Figure:
    """Importancia global de variables (media de |SHAP|). Admite también la tabla por grupos."""
    d = importance.head(top).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=d["importancia_shap"], y=[_etiqueta(v, etiquetas) for v in d["variable"]], orientation="h",
        marker_color=COLOR_BAJA, text=[f"{v:.3f}" for v in d["importancia_shap"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Importancia global de las variables",
        xaxis_title="Media de |SHAP|", yaxis_title="",
        font={"size": 14}, margin={"l": 10, "r": 10},
    )
    return fig
