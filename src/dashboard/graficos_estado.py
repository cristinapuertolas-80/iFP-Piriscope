"""Gráficos de la pestaña «Estado del sistema» (Plotly).

Siguen el criterio de accesibilidad del dashboard (DEC-28): el significado no va solo en el color
(líneas con marcadores y trazos distintos; el estado del PSI se escribe en cada barra).
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from src.dashboard import datos
from src.dashboard.graficos import COLORES_NIVEL, ETIQUETA_NIVEL

_MARGEN = {"l": 10, "r": 10, "t": 10, "b": 10}


def fig_niveles(ventana: pd.DataFrame) -> go.Figure:
    """Municipios por nivel de riesgo y fecha (barras apiladas), con los colores del resto del dashboard."""
    fig = go.Figure()
    for nivel in datos.NIVELES:
        fig.add_bar(x=ventana["fecha"], y=ventana[nivel], name=ETIQUETA_NIVEL[nivel],
                    marker={"color": COLORES_NIVEL[nivel], "line": {"color": "rgba(0,0,0,0.35)", "width": 1}},
                    hovertemplate="%{x|%d/%m/%Y}: %{y} municipios<extra>" + ETIQUETA_NIVEL[nivel] + "</extra>")
    fig.update_layout(barmode="stack", height=320, margin=_MARGEN, yaxis_title="Municipios",
                      legend={"orientation": "h", "y": -0.2}, xaxis={"tickformat": "%d/%m"})
    return fig


def fig_cobertura(ventana: pd.DataFrame) -> go.Figure:
    """% de municipios con predicción y % de predicciones con el modelo de respaldo, por fecha."""
    fig = go.Figure()
    fig.add_scatter(x=ventana["fecha"], y=ventana["cobertura_pct"], name="Municipios con predicción",
                    mode="lines+markers", line={"color": "#0072B2"}, marker={"symbol": "circle", "size": 7},
                    hovertemplate="%{x|%d/%m/%Y}: %{y:.1f} %<extra>Cobertura</extra>")
    fig.add_scatter(x=ventana["fecha"], y=ventana["pct_respaldo"], name="Con modelo de respaldo",
                    mode="lines+markers", line={"color": "#D55E00", "dash": "dash"},
                    marker={"symbol": "diamond", "size": 7},
                    hovertemplate="%{x|%d/%m/%Y}: %{y:.1f} %<extra>Respaldo</extra>")
    fig.update_layout(height=320, margin=_MARGEN, yaxis={"title": "%", "range": [0, 105]},
                      legend={"orientation": "h", "y": -0.2}, xaxis={"tickformat": "%d/%m"})
    return fig


def fig_deriva(tabla: pd.DataFrame, umbrales: dict, etiquetas: dict | None = None) -> go.Figure:
    """Desviación (z) de cada variable frente a los mismos días de los años de entrenamiento.

    Barras horizontales con los umbrales de «vigilar» y «fuera de lo habitual» a ambos lados y el estado escrito
    en cada barra.
    Las variables sin historial suficiente (``z`` nulo) salen con valor 0 y el texto «sin historial».
    """
    etiquetas = etiquetas or {}
    t = tabla.assign(_z=tabla["z"].fillna(0.0)).sort_values("_z")
    nombres = [etiquetas.get(v, v) for v in t["variable"]]
    textos = [("sin historial" if pd.isna(z) else f"{datos.fmt_num(z, 1)} · {datos.ESTADOS_DERIVA.get(e, e).lower()}")
              for z, e in zip(t["z"], t["estado"])]
    fig = go.Figure(go.Bar(x=t["_z"], y=nombres, orientation="h", marker={"color": "#56B4E9"},
                           text=textos, textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{text}<extra></extra>"))
    for clave, trazo in (("vigilar", "dot"), ("deriva", "dash")):
        for signo in (-1, 1):
            fig.add_vline(x=signo * umbrales[clave], line={"color": "#1A1A1A", "dash": trazo, "width": 1.5},
                          annotation_text=datos.ESTADOS_DERIVA[clave].lower() if signo > 0 else None,
                          annotation_position="top")
    limite = max(float(t["_z"].abs().max()), umbrales["deriva"]) * 1.4
    fig.update_layout(height=max(220, 34 * len(t) + 70), margin={"l": 10, "r": 60, "t": 30, "b": 10},
                      xaxis={"title": "Desviaciones típicas respecto a los años de entrenamiento",
                             "range": [-limite, limite]}, showlegend=False)
    return fig
