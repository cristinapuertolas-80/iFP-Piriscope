"""Gráficos SHAP en Plotly a partir de las tablas de Gold (sin cargar el modelo).

Accesibilidad: colores de la paleta Okabe-Ito (aptos para daltonismo) y el
sentido de cada barra también va en texto (+/-), no solo en color.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

COLOR_SUBE = "#D55E00"   # bermellón: aumenta el riesgo
COLOR_BAJA = "#0072B2"   # azul: reduce el riesgo


def fig_local(expl: pd.DataFrame, id_prediccion: str, unidades: str = "log-odds") -> go.Figure:
    """Barras con las variables que más contribuyen a UNA predicción."""
    d = expl[expl["id_prediccion"] == id_prediccion]
    if d.empty:
        raise ValueError(f"No hay explicación para {id_prediccion!r}")
    d = d.reindex(d["valor_shap"].abs().sort_values().index)  # la mayor, arriba
    fig = go.Figure(go.Bar(
        x=d["valor_shap"], y=d["variable"], orientation="h",
        marker_color=[COLOR_SUBE if v > 0 else COLOR_BAJA for v in d["valor_shap"]],
        text=[f"{v:+.2f} ({'sube' if v > 0 else 'baja'} el riesgo)" for v in d["valor_shap"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Variables que más influyen en esta predicción",
        xaxis_title=f"Contribución SHAP ({unidades})", yaxis_title="",
        font={"size": 14}, margin={"l": 10, "r": 10},
    )
    return fig


def fig_global(importance: pd.DataFrame, top: int = 10) -> go.Figure:
    """Importancia global de variables (media de |SHAP|)."""
    d = importance.head(top).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=d["importancia_shap"], y=d["variable"], orientation="h",
        marker_color=COLOR_BAJA, text=[f"{v:.3f}" for v in d["importancia_shap"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Importancia global de las variables",
        xaxis_title="Media de |SHAP|", yaxis_title="",
        font={"size": 14}, margin={"l": 10, "r": 10},
    )
    return fig
