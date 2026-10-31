"""Figuras Plotly del dashboard (mapa, histórico de un municipio y evaluación frente al FWI).

Accesibilidad (NF2, apartado 9): paleta Okabe-Ito y el nivel siempre también en
texto (leyenda, etiqueta emergente y tabla), nunca solo en color. Los municipios
sin dato llevan además un punto oscuro y los del modelo de respaldo, borde grueso.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .datos import SIN_DATO, fmt_fecha, fmt_num, fmt_pct, fmt_prioridad

COLORES_NIVEL = {"bajo": "#0072B2", "medio": "#F0E442", "alto": "#D55E00", SIN_DATO: "#999999"}
#: Texto legible encima de cada color (el amarillo necesita texto oscuro)
TEXTO_SOBRE_NIVEL = {"bajo": "#FFFFFF", "medio": "#1A1A1A", "alto": "#FFFFFF", SIN_DATO: "#1A1A1A"}
ETIQUETA_NIVEL = {"bajo": "Bajo", "medio": "Medio", "alto": "Alto", SIN_DATO: "Sin dato"}
COLOR_MODELO = "#009E73"
COLOR_FWI = "#E69F00"
COLOR_AZAR = "#666666"

CENTRO_GALICIA = {"lat": 42.78, "lon": -7.95}
FUENTE = {"family": "Source Sans Pro, sans-serif", "size": 14}


def _hover_dia(dia: pd.DataFrame) -> list[str]:
    textos = []
    for r in dia.itertuples():
        cab = f"<b>{r.nombre}</b> ({r.provincia})<br>"
        if r.nivel_mostrado == SIN_DATO:
            textos.append(cab + "Sin dato: sin datos suficientes ese día")
            continue
        modelo = "Estimación sin FWI" if r.respaldo else "Modelo principal (con FWI)"
        textos.append(cab + f"Nivel: <b>{ETIQUETA_NIVEL[r.nivel_mostrado]}</b><br>"
                      f"Índice de riesgo: {fmt_pct(r.probabilidad)}<br>"
                      f"Prioridad relativa hoy: {fmt_prioridad(r.prioridad_dia)}<br>"
                      f"FWI: {fmt_num(r.fwi)}<br>{modelo}")
    return textos


def fig_mapa(dia: pd.DataFrame, geojson: dict, alto: int = 620) -> go.Figure:
    """Coropletas de los 313 municipios, una capa por nivel para que la leyenda sea discreta."""
    dia = dia.assign(_hover=_hover_dia(dia))
    recintos = {f["properties"]["id_municipio"]: f for f in geojson["features"]}
    fig = go.Figure()
    for nivel in ("alto", "medio", "bajo", SIN_DATO):
        for respaldo in (False, True):
            d = dia[(dia["nivel_mostrado"] == nivel) & (dia["respaldo"] == respaldo)]
            if d.empty:
                continue
            nombre = ETIQUETA_NIVEL[nivel] + (" · estimación sin FWI" if respaldo else "")
            # Cada capa lleva solo sus recintos: el GeoJSON completo repetido en cada una ralentiza la carga
            subconjunto = {"type": "FeatureCollection", "features": [recintos[i] for i in d["id_municipio"]]}
            fig.add_trace(go.Choroplethmap(
                geojson=subconjunto, featureidkey="properties.id_municipio",
                locations=d["id_municipio"], z=[1] * len(d),
                colorscale=[[0, COLORES_NIVEL[nivel]], [1, COLORES_NIVEL[nivel]]], showscale=False,
                marker={"opacity": 0.85,
                        "line": {"width": 2.2 if respaldo else 0.5, "color": "#1A1A1A" if respaldo else "#FFFFFF"}},
                unselected={"marker": {"opacity": 0.85}},
                name=nombre, legendgroup=nombre, showlegend=True,
                customdata=d["id_municipio"], text=d["_hover"], hovertemplate="%{text}<extra></extra>",
            ))
    # Streamlit no recibe el clic en las coropletas: un punto por municipio permite seleccionarlo.
    # Los «sin dato» llevan un punto oscuro grande, para que no se distingan solo por el gris.
    sin = dia["nivel_mostrado"] == SIN_DATO
    for d, nombre, estilo in ((dia[~sin], "Municipios", {"size": 8, "color": "#FFFFFF", "opacity": 0.75}),
                              (dia[sin], "● Sin dato", {"size": 11, "color": "#1A1A1A", "opacity": 0.9})):
        if d.empty:
            continue
        fig.add_trace(go.Scattermap(
            lat=d["lat"], lon=d["lon"], mode="markers", name=nombre, showlegend=nombre != "Municipios",
            marker=estilo, selected={"marker": {"size": 16, "color": "#CC0000", "opacity": 1}},
            unselected={"marker": {"opacity": estilo["opacity"]}},
            customdata=d["id_municipio"], text=d["_hover"], hovertemplate="%{text}<extra></extra>",
        ))
    fig.update_layout(
        map={"style": "carto-positron", "center": CENTRO_GALICIA, "zoom": 6.6},
        height=alto, margin={"l": 0, "r": 0, "t": 0, "b": 0}, font=FUENTE,
        legend={"title": {"text": "Nivel de riesgo"}, "x": 0.01, "y": 0.99, "bgcolor": "rgba(255,255,255,0.88)",
                "bordercolor": "#CCCCCC", "borderwidth": 1, "font": {"color": "#1A1A1A"}},
        hoverlabel={"bgcolor": "white", "font": {"color": "#1A1A1A"}},
        clickmode="event+select",
    )
    return fig


def fig_historico(hist: pd.DataFrame, fecha=None, alto: int = 340) -> go.Figure:
    """Evolución del índice de riesgo de un municipio, con el nivel en el color y en la etiqueta."""
    fig = go.Figure()
    if hist.empty:
        return fig
    fig.add_trace(go.Scatter(x=hist["fecha_objetivo"], y=100 * hist["probabilidad"], mode="lines",
                             line={"color": "#888888", "width": 1}, hoverinfo="skip", showlegend=False))
    respaldo = hist["version_modelo"].astype(str).str.endswith("-respaldo")
    for nivel in ("bajo", "medio", "alto"):
        for es_resp in (False, True):
            d = hist[(hist["nivel_riesgo"] == nivel) & (respaldo == es_resp)]
            if d.empty:
                continue
            fig.add_trace(go.Scatter(
                x=d["fecha_objetivo"], y=100 * d["probabilidad"], mode="markers",
                marker={"color": COLORES_NIVEL[nivel], "size": 6 if len(hist) > 120 else 8,
                        "symbol": "circle-open" if es_resp else "circle",
                        "line": {"width": 1.5 if es_resp else 0.5,
                                 "color": COLORES_NIVEL[nivel] if es_resp else "#333333"}},
                name=ETIQUETA_NIVEL[nivel] + (" · sin FWI" if es_resp else ""),
                customdata=[[ETIQUETA_NIVEL[nivel], fmt_pct(p)] for p in d["probabilidad"]],
                hovertemplate="%{x|%d/%m/%Y}<br>Nivel: <b>%{customdata[0]}</b><br>"
                              "Índice de riesgo: %{customdata[1]}<extra></extra>",
            ))
    if fecha is not None:
        x = pd.Timestamp(fecha)
        fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, yref="paper",
                      line={"color": "#1A1A1A", "dash": "dot", "width": 1})
        fig.add_annotation(x=x, y=1, yref="paper", text=fmt_fecha(x), showarrow=False, yanchor="bottom",
                           font={"size": 12})
    fig.update_layout(height=alto, margin={"l": 10, "r": 10, "t": 30, "b": 10}, font=FUENTE,
                      yaxis={"title": "Índice de riesgo (%)", "rangemode": "tozero"}, xaxis={"title": ""},
                      legend={"orientation": "h", "y": -0.15}, hovermode="closest")
    return fig


def fig_evaluacion_grupos(final: pd.DataFrame, escenario: str, variante: str) -> go.Figure:
    """AUC-PR del modelo y del FWI por grupo (global y provincias), con intervalos y el azar."""
    d = final[(final["escenario"] == escenario) & (final["variante"] == variante)]
    orden = ["global"] + sorted(g for g in d["grupo"].unique() if g != "global")
    etiquetas = ["Galicia" if g == "global" else g for g in orden]
    fig = go.Figure()
    for sistema, nombre, color in (("modelo", "Modelo Piriscope", COLOR_MODELO), ("baseline_fwi", "FWI", COLOR_FWI)):
        s = d[d["sistema"] == sistema].set_index("grupo").reindex(orden)
        fig.add_trace(go.Bar(
            x=etiquetas, y=s["auc_pr"], name=nombre, marker_color=color,
            error_y={"type": "data", "symmetric": False, "array": s["auc_pr_hi"] - s["auc_pr"],
                     "arrayminus": s["auc_pr"] - s["auc_pr_lo"], "color": "#444444", "thickness": 1.2},
            customdata=list(zip(s["auc_pr_lo"].round(4), s["auc_pr_hi"].round(4))),
            hovertemplate=f"<b>{nombre}</b><br>%{{x}}: AUC-PR %{{y:.4f}}<br>"
                          "Intervalo: %{customdata[0]} a %{customdata[1]}<extra></extra>",
        ))
    prev = d[d["sistema"] == "modelo"].set_index("grupo").reindex(orden)["prevalencia"]
    fig.add_trace(go.Scatter(x=etiquetas, y=prev, mode="markers", name="Azar (prevalencia)",
                             marker={"symbol": "line-ew-open", "size": 34, "line": {"width": 3, "color": COLOR_AZAR}},
                             hovertemplate="%{x}: azar = %{y:.4f}<extra></extra>"))
    fig.update_layout(barmode="group", height=380, font=FUENTE, margin={"l": 10, "r": 10, "t": 10, "b": 10},
                      yaxis={"title": "AUC-PR (más alto es mejor)", "rangemode": "tozero"},
                      legend={"orientation": "h", "y": -0.15})
    return fig


def fig_walkforward(wf: pd.DataFrame, variante: str) -> go.Figure:
    """AUC-PR por año (walk-forward 2016-2022) del modelo y del FWI con su intervalo."""
    d = wf[wf["variante"] == variante].sort_values("anio")
    fig = go.Figure()
    for sistema, nombre, color in (("modelo", "Modelo Piriscope", COLOR_MODELO), ("baseline_fwi", "FWI", COLOR_FWI)):
        s = d[d["sistema"] == sistema]
        fig.add_trace(go.Scatter(
            x=s["anio"], y=s["auc_pr"], mode="lines+markers", name=nombre, line={"color": color, "width": 2.5},
            marker={"symbol": "circle" if sistema == "modelo" else "diamond", "size": 9},
            error_y={"type": "data", "symmetric": False, "array": s["auc_pr_hi"] - s["auc_pr"],
                     "arrayminus": s["auc_pr"] - s["auc_pr_lo"], "color": color, "thickness": 1},
            hovertemplate=f"<b>{nombre}</b><br>%{{x}}: AUC-PR %{{y:.4f}}<extra></extra>",
        ))
    s = d[d["sistema"] == "modelo"]
    fig.add_trace(go.Scatter(x=s["anio"], y=s["prevalencia"], mode="lines", name="Azar (prevalencia)",
                             line={"color": COLOR_AZAR, "dash": "dash"},
                             hovertemplate="%{x}: azar = %{y:.4f}<extra></extra>"))
    fig.update_layout(height=360, font=FUENTE, margin={"l": 10, "r": 10, "t": 10, "b": 10},
                      xaxis={"title": "", "dtick": 1}, yaxis={"title": "AUC-PR", "rangemode": "tozero"},
                      legend={"orientation": "h", "y": -0.15})
    return fig
