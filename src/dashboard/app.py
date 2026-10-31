"""Dashboard de Piriscope: riesgo de incendio forestal D+1 por municipio en Galicia.

    streamlit run src/dashboard/app.py

Solo lee la capa Gold (por defecto, del dataset de Hugging Face; ver ``datos.py``)
y los ficheros estáticos de evaluación de ``docs/resultados``. No entrena, no
carga modelos y no accede a las fuentes externas (contrato Gold, NF2 5.6).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # streamlit solo añade src/dashboard al path
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.dashboard import datos  # noqa: E402
from src.dashboard.graficos import (  # noqa: E402
    COLORES_NIVEL, ETIQUETA_NIVEL, TEXTO_SOBRE_NIVEL, fig_evaluacion_grupos, fig_historico, fig_mapa,
    fig_walkforward,
)
from src.explainability.plots import ETIQUETAS, fig_cascada, fig_global, fig_local  # noqa: E402

VARIABLES_FWI = {"fwi", "ffmc", "dmc", "dc", "isi", "bui"}
PIE = ("Herramienta experimental de apoyo. No sustituye a los sistemas oficiales de predicción ni de gestión "
       "de emergencias. Las estimaciones son probabilísticas y no garantizan que vaya a producirse un incendio.")
FUENTES = ("Fuente: AEMET. Origen de los datos: Ministerio para la Transición Ecológica y el Reto Demográfico. "
           "Copernicus/EFFIS. Fuente: INE. © Instituto Geográfico Nacional. Los organismos citados no respaldan "
           "este proyecto.")
LEYENDA = ("«Medio» o «alto» significa *no descartar*, no *habrá un incendio*. Entre 1 de cada 16 y 1 de cada 50 "
           "alarmas coincide con un incendio, según el año.")
ESCENARIOS = {
    "A-test": "A · Municipios vistos en el entrenamiento, año 2023",
    "B": "B · Municipios no vistos, 2009-2021",
    "C-test": "C · Municipios no vistos, año 2023",
    "B-todos-los-anios": "B · Municipios no vistos, 2001-2021",
}
VARIANTES = {
    "con_fwi-sin_viento-desde2009": "Modelo principal (con FWI)",
    "sin_fwi-sin_viento-desde2009": "Modelo de respaldo (sin FWI)",
}

st.set_page_config(page_title="Piriscope · Riesgo de incendio en Galicia", page_icon="🔥", layout="wide")
st.markdown("""
<style>
  .nivel { display:inline-block; padding:0.25rem 0.8rem; border-radius:0.4rem; font-weight:700;
           font-size:1.15rem; border:2px solid rgba(0,0,0,0.25); }
  .nivel.respaldo { border-style:dashed; border-color:#1A1A1A; }
  .etiqueta-respaldo { display:inline-block; margin-left:0.5rem; padding:0.15rem 0.55rem; border-radius:0.4rem;
                       border:1px dashed currentColor; font-size:0.85rem; }
  .ficha h3 { margin-bottom:0.1rem; }
  .ficha .sub { opacity:0.75; margin-bottom:0.6rem; }
  div[data-testid="stMetricValue"] { font-size:1.45rem; }
  .kpis { display:flex; flex-wrap:wrap; gap:0.6rem; margin:0.9rem 0 0.6rem; }
  .kpi { flex:1 1 7.5rem; padding:0.5rem 0.75rem; border:1px solid rgba(128,128,128,0.35); border-radius:0.5rem;
         cursor:help; }
  .k-lab { font-size:0.8rem; opacity:0.75; }
  .k-val { font-size:1.35rem; font-weight:600; white-space:nowrap; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Carga de datos (cacheada: HU-09 pide menos de 5 s por consulta)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Descargando predicciones de Hugging Face…")
def cargar_gold() -> tuple[pd.DataFrame, datos.OrigenGold]:
    origen = datos.localizar_gold()
    return datos.leer_predicciones(origen.predicciones), origen


@st.cache_data(show_spinner=False)
def cargar_mapa() -> tuple[dict, pd.DataFrame]:
    geojson = datos.leer_geojson()
    return geojson, datos.catalogo_municipios(geojson)


@st.cache_data(show_spinner=False, max_entries=500)
def cargar_explicacion(ruta: str, id_prediccion: str) -> pd.DataFrame:
    return datos.leer_explicacion(Path(ruta), id_prediccion)


@st.cache_data(show_spinner=False)
def cargar_resultados() -> dict:
    return datos.leer_resultados()


@st.cache_data(show_spinner=False)
def preparar_dia(_pred: pd.DataFrame, _catalogo: pd.DataFrame, fecha: pd.Timestamp, version: str) -> pd.DataFrame:
    # `version` invalida la caché cuando se recargan los datos (los argumentos con _ no se hashean)
    return datos.predicciones_dia(_pred, _catalogo, fecha)


try:
    pred, origen = cargar_gold()
except Exception as exc:  # noqa: BLE001
    st.error(f"No se pudieron cargar las predicciones. {exc}")
    st.info("Comprueba que el fichero `.env` de la raíz tiene `HF_TOKEN` (token de lectura de Hugging Face) "
            "o que existen `data/gold/predicciones.parquet` y `data/gold/explicaciones.parquet`.")
    st.stop()

geojson, catalogo = cargar_mapa()
if pred.empty:
    st.warning("El fichero de predicciones está vacío.")
    st.stop()

ultima = pred["fecha_objetivo"].max()
primera = pred["fecha_objetivo"].min()
version_datos = str(pred["fecha_insert"].max())
nombres = dict(zip(catalogo["id_municipio"], catalogo["nombre"]))
provincia_de = dict(zip(catalogo["id_municipio"], catalogo["provincia"]))


# ---------------------------------------------------------------------------
# Barra lateral: fecha, origen de los datos y leyenda
# ---------------------------------------------------------------------------
if "fecha" not in st.session_state:
    # Un enlace puede abrir una fecha y un municipio concretos: ?fecha=2025-08-14&municipio=32054
    pedida = pd.to_datetime(st.query_params.get("fecha"), errors="coerce")
    st.session_state.fecha = (pedida if pd.notna(pedida) and primera <= pedida <= ultima else ultima).date()
    if st.query_params.get("municipio") in nombres:
        st.session_state.municipio = st.query_params["municipio"]

with st.sidebar:
    logo = REPO_ROOT / "docs" / "assets" / "logo-piriscope.svg"
    if logo.exists():
        st.image(str(logo), width=90)
    st.markdown("## Piriscope")
    st.caption("Riesgo de incendio forestal del día siguiente (D+1) por municipio · Galicia")

    st.date_input("Fecha de la predicción", key="fecha", min_value=primera.date(), max_value=ultima.date(),
                  format="DD/MM/YYYY", help="Día al que se refiere la predicción. Usa información hasta el día "
                                             "anterior: no es una previsión meteorológica.")
    st.markdown(f"**Última fecha con predicción:** {datos.fmt_fecha(ultima)}")
    if pd.Timestamp(st.session_state.fecha) != ultima:
        st.button("Ir a la última fecha", on_click=lambda: st.session_state.update(fecha=ultima.date()),
                  width="stretch")

    st.divider()
    st.markdown("**Niveles de riesgo**")
    for nivel in (*datos.NIVELES, datos.SIN_DATO):
        st.markdown(f"<span class='nivel' style='background:{COLORES_NIVEL[nivel]};color:{TEXTO_SOBRE_NIVEL[nivel]};"
                    f"font-size:0.9rem;padding:0.1rem 0.6rem'>{ETIQUETA_NIVEL[nivel]}</span>",
                    unsafe_allow_html=True)
    st.caption(LEYENDA)
    st.caption("Borde grueso o discontinuo: **estimación sin FWI** (modelo de respaldo, menos fiable).")

    st.divider()
    if origen.origen == "huggingface":
        st.caption("📦 Datos: dataset de Hugging Face [cpuertolas/piriscope]"
                   "(https://huggingface.co/datasets/cpuertolas/piriscope)")
    else:
        st.caption("📦 Datos: ficheros locales de `data/gold`")
    st.caption(f"Predicciones generadas el {pd.Timestamp(pred['fecha_insert'].max()).strftime('%d/%m/%Y %H:%M')} UTC")
    if st.button("🔄 Comprobar si hay datos nuevos", width="stretch"):
        st.cache_data.clear()
        st.rerun()

if origen.aviso:
    st.warning(origen.aviso)

fecha = pd.Timestamp(st.session_state.fecha).normalize()
dia = preparar_dia(pred, catalogo, fecha, version_datos)
hay_dia = dia["version_modelo"].notna().any()


# ---------------------------------------------------------------------------
# Piezas de la ficha de un municipio
# ---------------------------------------------------------------------------
def insignia_nivel(nivel: str, respaldo: bool) -> str:
    clase = "nivel respaldo" if respaldo else "nivel"
    html = (f"<span class='{clase}' style='background:{COLORES_NIVEL[nivel]};color:{TEXTO_SOBRE_NIVEL[nivel]}'>"
            f"Riesgo {ETIQUETA_NIVEL[nivel].lower()}</span>")
    if respaldo:
        html += "<span class='etiqueta-respaldo'>Estimación sin FWI</span>"
    return html


def tarjetas(items: list[tuple[str, str, str]]) -> str:
    """Indicadores en HTML que se reparten el ancho y bajan de línea si no caben (no se cortan)."""
    celdas = "".join(f"<div class='kpi' title='{ayuda}'><div class='k-lab'>{etiqueta} ⓘ</div>"
                     f"<div class='k-val'>{valor}</div></div>" for etiqueta, valor, ayuda in items)
    return f"<div class='kpis'>{celdas}</div>"


def ficha_municipio(fila: pd.Series) -> None:
    st.markdown(f"<div class='ficha'><h3>{fila['nombre']}</h3>"
                f"<div class='sub'>{fila['provincia']} · predicción para el {datos.fmt_fecha(fecha)}</div></div>",
                unsafe_allow_html=True)

    if fila["nivel_mostrado"] == datos.SIN_DATO:
        st.markdown(insignia_nivel(datos.SIN_DATO, False), unsafe_allow_html=True)
        st.info("Sin datos suficientes ese día para estimar el riesgo de este municipio.")
        hist = pred[pred["id_municipio"] == fila["id_municipio"]]
        if not hist.empty:
            st.caption(f"Su última predicción disponible es del {datos.fmt_fecha(hist['fecha_objetivo'].max())}.")
        return

    st.markdown(insignia_nivel(fila["nivel_mostrado"], bool(fila["respaldo"])), unsafe_allow_html=True)
    st.markdown(tarjetas([
        ("Índice de riesgo", datos.fmt_pct(fila["probabilidad"]),
         "Salida del modelo. Sirve para ordenar el riesgo, no es una probabilidad exacta: el modelo sobrestima "
         "la tasa real (entre 1,5 y 2,4 veces según el año). No compares índices entre años ni entre el modelo "
         "principal y el de respaldo."),
        ("Prioridad relativa hoy", datos.fmt_prioridad(fila["prioridad_dia"]),
         "Posición del municipio entre los de ese día (top 10 % = entre el 10 % de mayor índice). Es "
         "complementaria: no define el nivel."),
        ("FWI", datos.fmt_num(fila["fwi"]),
         "Fire Weather Index de EFFIS/Copernicus del día anterior, solo como referencia. No está disponible en "
         "las estimaciones sin FWI."),
    ]), unsafe_allow_html=True)
    if fila["respaldo"]:
        st.caption("⚠️ Estimación con el modelo de respaldo, que se usa cuando aún no ha llegado el FWI. "
                   "Discrimina menos que el modelo principal y no se compara con el FWI.")
    st.caption(LEYENDA)


def explicacion_municipio(fila: pd.Series) -> None:
    st.markdown(f"#### ¿Qué influye en el riesgo de {fila['nombre']}?")
    if fila["nivel_mostrado"] == datos.SIN_DATO:
        st.info("No hay predicción ese día, así que no hay explicación.")
        return
    expl = cargar_explicacion(str(origen.explicaciones), fila["id_prediccion"])
    if expl.empty:
        st.info("Esta predicción no tiene explicación disponible.")
        return
    completo = datos.tiene_columnas_dec50(expl)
    fig = fig_local(expl, fila["id_prediccion"], etiquetas=ETIQUETAS, mostrar_valores=completo,
                    incluir_resto=completo, titulo=" ")
    # Texto corto junto a cada barra (el sentido sigue en texto, no solo en color) y margen para que quepa
    valores = list(fig.data[0].x)
    fig.update_traces(text=[f"{v:+.2f}".replace(".", ",") + (" sube" if v > 0 else " baja") for v in valores],
                      cliponaxis=False, textfont={"size": 12})
    tope = max(abs(v) for v in valores) or 1
    fig.update_layout(title=None, xaxis_title="Contribución al índice de riesgo", height=330,
                      xaxis={"range": [-2.1 * tope if min(valores) < 0 else 0, 2.1 * tope if max(valores) > 0 else 0],
                             "zeroline": True},
                      margin={"l": 10, "r": 10, "t": 10, "b": 10}, font={"size": 13})
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption("Barras naranjas: suben el riesgo · barras azules: lo bajan · gris: el resto de variables. "
               "Se muestran las 5 variables con más peso y su valor ese día.")
    if VARIABLES_FWI & set(expl["variable"]):
        st.caption("Los índices FWI (FWI, FFMC, DMC, DC, ISI, BUI) miden juntos las condiciones de sequía y "
                   "están relacionados entre sí: no son causas independientes.")
    if completo:
        with st.expander("Cómo se llega al índice de riesgo"):
            cascada = fig_cascada(expl, fila["id_prediccion"], etiquetas=ETIQUETAS, titulo="Del valor base al índice")
            cascada.update_layout(yaxis_title="Escala interna del modelo", height=380)
            st.plotly_chart(cascada, width="stretch", config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------
st.title("🔥 Riesgo de incendio forestal en Galicia")
st.caption(f"Predicción para el **{datos.fmt_fecha(fecha)}** a partir de los datos hasta el día anterior · "
           "313 municipios")

tab_mapa, tab_tabla, tab_eval, tab_info = st.tabs(
    ["🗺️ Mapa y municipio", "📋 Tabla del día", "📊 Modelo frente a FWI", "ℹ️ Cómo leer el dashboard"])

with tab_mapa:
    if not hay_dia:
        st.warning(f"No hay predicciones para el {datos.fmt_fecha(fecha)}. "
                   f"La última fecha con predicción es el {datos.fmt_fecha(ultima)}.")
    else:
        con_dato = dia[dia["nivel_mostrado"] != datos.SIN_DATO]
        if datos.episodio_generalizado(dia):
            st.info("🔶 **Episodio de riesgo elevado en toda la región.** Más del 90 % de los municipios están en "
                    "nivel medio o alto, algo habitual en épocas de sequía. Usa la prioridad relativa para ver "
                    "dónde es mayor el riesgo dentro del día.")
        k1, k2, k3 = st.columns(3)
        k1.metric("Municipios con predicción", f"{len(con_dato)} de {len(dia)}",
                  help="Los municipios sin datos meteorológicos suficientes ese día aparecen como «Sin dato».")
        n_resp = int(dia["respaldo"].sum())
        k2.metric("Modelo utilizado", "Sin FWI (respaldo)" if n_resp == len(con_dato) else
                  ("Principal (con FWI)" if n_resp == 0 else "Mixto"),
                  help="Cuando el FWI aún no ha llegado se usa el modelo de respaldo, que discrimina menos.")
        k3.metric("Fecha", datos.fmt_fecha(fecha))
        if n_resp:
            st.caption(f"⚠️ {n_resp} municipios con **estimación sin FWI** (borde grueso en el mapa).")

    col_mapa, col_ficha = st.columns([3, 2], gap="large")
    with col_mapa:
        evento = st.plotly_chart(fig_mapa(dia, geojson), width="stretch", key="mapa",
                                 on_select="rerun", selection_mode="points",
                                 config={"displayModeBar": False, "scrollZoom": True})
        st.caption("Pulsa el punto de un municipio para ver su ficha, o búscalo por nombre. Se puede ampliar con "
                   "la rueda del ratón.")

    # Clic en el mapa -> se selecciona ese municipio (solo si la selección ha cambiado)
    puntos = (evento or {}).get("selection", {}).get("points", []) if evento else []
    clic = next((p.get("location") or p.get("customdata") for p in puntos
                 if (p.get("location") or p.get("customdata")) in nombres), None)
    if clic and clic != st.session_state.get("_ultimo_clic"):
        st.session_state["_ultimo_clic"] = clic
        st.session_state["municipio"] = clic
    if "municipio" not in st.session_state:
        top = dia.dropna(subset=["probabilidad"]).sort_values("probabilidad", ascending=False)
        st.session_state["municipio"] = top["id_municipio"].iloc[0] if len(top) else catalogo["id_municipio"].iloc[0]

    with col_ficha:
        st.selectbox("Buscar municipio", options=list(catalogo["id_municipio"]), key="municipio",
                     format_func=lambda i: f"{nombres[i]} ({provincia_de[i]})",
                     placeholder="Escribe el nombre…")
        fila_mun = dia[dia["id_municipio"] == st.session_state["municipio"]].iloc[0]
        ficha_municipio(fila_mun)
    # La URL refleja lo que se ve, para poder compartir el enlace
    st.query_params.update({"fecha": f"{fecha:%Y-%m-%d}", "municipio": st.session_state["municipio"]})

    st.divider()
    col_expl, col_hist = st.columns(2, gap="large")
    with col_expl:
        explicacion_municipio(fila_mun)
    with col_hist:
        st.markdown(f"#### Evolución del índice · {fila_mun['nombre']}")
        rango = st.segmented_control("Periodo", ["30 días", "90 días", "1 año", "Todo"], default="90 días",
                                     key="rango", label_visibility="collapsed")
        dias = {"30 días": 30, "90 días": 90, "1 año": 365}.get(rango or "90 días")
        desde = fecha - pd.Timedelta(days=dias) if dias else None
        hist = datos.historico_municipio(pred, st.session_state["municipio"], desde, fecha)
        if hist.empty:
            st.info("No hay predicciones de este municipio en ese periodo.")
        else:
            st.plotly_chart(fig_historico(hist, fecha, alto=300), width="stretch", config={"displayModeBar": False})
            st.caption("Los círculos huecos son estimaciones sin FWI: su índice no es comparable con el del "
                       "modelo principal.")

with tab_tabla:
    st.markdown(f"#### Todos los municipios · {datos.fmt_fecha(fecha)}")
    st.caption("Alternativa accesible al mapa. Cada fila muestra el nivel, el índice de riesgo y la prioridad "
               "relativa; la explicación de cada municipio está en su ficha.")
    f1, f2, f3 = st.columns([2, 2, 3])
    provincias = f1.multiselect("Provincia", sorted(catalogo["provincia"].unique()), placeholder="Todas")
    niveles = f2.multiselect("Nivel", [ETIQUETA_NIVEL[n] for n in (*datos.NIVELES, datos.SIN_DATO)],
                             placeholder="Todos")
    texto = f3.text_input("Buscar por nombre", placeholder="p. ej. Ourense")
    tabla = dia.copy()
    if provincias:
        tabla = tabla[tabla["provincia"].isin(provincias)]
    if niveles:
        tabla = tabla[tabla["nivel_mostrado"].map(ETIQUETA_NIVEL).isin(niveles)]
    if texto:
        clave = datos.clave_orden(texto)
        tabla = tabla[tabla["nombre"].map(datos.clave_orden).str.contains(clave, regex=False)]
    iconos = {"bajo": "🔵", "medio": "🟡", "alto": "🟠", datos.SIN_DATO: "⚪"}
    vista = pd.DataFrame({
        "Municipio": tabla["nombre"],
        "Provincia": tabla["provincia"],
        "Nivel": [f"{iconos[n]} {ETIQUETA_NIVEL[n]}" for n in tabla["nivel_mostrado"]],
        "Índice de riesgo (%)": (100 * tabla["probabilidad"]).round(2),
        "Prioridad relativa hoy": tabla["prioridad_dia"].map(datos.fmt_prioridad),
        "FWI": tabla["fwi"].round(1),
        "Modelo": tabla["version_modelo"].map(
            lambda v: "—" if pd.isna(v) else ("Estimación sin FWI" if str(v).endswith("-respaldo") else "Principal")),
        "Código INE": tabla["id_municipio"],
    })
    st.dataframe(vista, hide_index=True, width="stretch", height=520,
                 column_config={"Índice de riesgo (%)": st.column_config.NumberColumn(format="%.2f %%"),
                                "FWI": st.column_config.NumberColumn(format="%.1f")})
    st.caption(f"{len(vista)} municipios. Ordena pulsando en la cabecera de cada columna.")
    st.download_button("⬇️ Descargar CSV", vista.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"piriscope_{fecha:%Y-%m-%d}.csv", mime="text/csv")

with tab_eval:
    res = cargar_resultados()
    final, wf = res["final"], res["walkforward"]
    st.info("Estos son **resultados de evaluación sobre años pasados**, no de las predicciones actuales. Comparan "
            "el modelo con el FWI de EFFIS/Copernicus, el índice que se usa hoy.")
    if final is None:
        st.warning("No se encuentra `docs/resultados/informe_final.csv`.")
    else:
        e1, e2 = st.columns(2)
        escenario = e1.selectbox("Escenario de evaluación", [e for e in ESCENARIOS if e in set(final["escenario"])],
                                 format_func=ESCENARIOS.get)
        variante = e2.radio("Modelo", [v for v in VARIANTES if v in set(final["variante"])],
                            format_func=VARIANTES.get, horizontal=True)
        g = final[(final["escenario"] == escenario) & (final["variante"] == variante) & (final["grupo"] == "global")]
        mod = g[g["sistema"] == "modelo"].iloc[0]
        fwi = g[g["sistema"] == "baseline_fwi"].iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("AUC-PR del modelo", f"{mod['auc_pr']:.4f}".replace(".", ","),
                  f"{mod['auc_pr'] / mod['prevalencia']:.1f} veces el azar".replace(".", ","), delta_color="off")
        m2.metric("AUC-PR del FWI", f"{fwi['auc_pr']:.4f}".replace(".", ","),
                  f"{fwi['auc_pr'] / fwi['prevalencia']:.1f} veces el azar".replace(".", ","), delta_color="off")
        m3.metric("Diferencia modelo − FWI", f"{mod['dif_auc_pr']:+.4f}".replace(".", ","),
                  f"intervalo {mod['dif_lo']:+.4f} a {mod['dif_hi']:+.4f}".replace(".", ","),
                  delta_color="off")
        m4.metric("Días con incendio (azar)", datos.fmt_pct(mod["prevalencia"], 2),
                  f"{int(mod['n_pos']):,} de {int(mod['n']):,} municipio-días".replace(",", "."), delta_color="off")
        if mod["dif_lo"] > 0:
            st.success("El modelo supera al FWI en este escenario: el intervalo de la diferencia no incluye el cero. "
                       "La ventaja es modesta.")
        else:
            st.warning("En este escenario el modelo **no mejora** al FWI de forma clara: el intervalo de la "
                       "diferencia incluye el cero.")
        st.caption("El AUC-PR mide cómo ordena el sistema los días con incendio por encima del resto (más alto, "
                   "mejor). Con incendios tan raros, un valor bajo puede ser varias veces mejor que el azar.")

        st.markdown("##### Por provincia")
        st.plotly_chart(fig_evaluacion_grupos(final, escenario, variante), width="stretch",
                        config={"displayModeBar": False})
        prov = final[(final["escenario"] == escenario) & (final["variante"] == variante)
                     & (final["grupo"] != "global")].pivot(index="grupo", columns="sistema", values="auc_pr")
        # Contrato 11.3: no ocultar las provincias donde el modelo no mejora (A Coruña). Se marca si no
        # supera al FWI al menos en un 10 %, porque los intervalos provinciales son muy amplios.
        peores = [p for p, r in prov.iterrows() if r["modelo"] < 1.10 * r["baseline_fwi"]]
        if peores:
            st.warning(f"El modelo **no mejora claramente al FWI** en: **{', '.join(peores)}** (queda por debajo "
                       "o a menos de un 10 % del FWI).")
        st.caption("Las líneas negras son intervalos de confianza (bootstrap por bloques de 7 días). En las "
                   "provincias hay pocos incendios y los intervalos son amplios: diferencias pequeñas entre "
                   "provincias no son concluyentes.")

    if wf is not None:
        st.markdown("##### Año a año (walk-forward 2016-2022)")
        st.caption("Cada año se predice con un modelo entrenado solo con los años anteriores.")
        st.plotly_chart(fig_walkforward(wf, variante if final is not None else wf["variante"].iloc[0]),
                        width="stretch", config={"displayModeBar": False})

    if res["shap_global"] is not None:
        st.markdown("##### Variables más importantes en general")
        por_grupos = st.toggle("Agrupar variables por tipo", value=False)
        imp = res["shap_grupos"] if por_grupos and res["shap_grupos"] is not None else res["shap_global"]
        fig = fig_global(imp, top=12, etiquetas=ETIQUETAS)
        fig.update_layout(title=None, height=360 if not por_grupos else 240, xaxis_title="Peso medio en el modelo",
                          margin={"l": 10, "r": 10, "t": 10, "b": 10})
        fig.update_traces(cliponaxis=False)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("Calculado sobre la validación de 2022 con el modelo principal. No cambia con cada predicción.")

with tab_info:
    st.markdown(f"""
#### Qué es Piriscope
Estima, para cada municipio gallego, el **riesgo relativo de incendio forestal del día siguiente** combinando
meteorología de AEMET, el histórico de incendios de EGIF-MITECO y el FWI de EFFIS/Copernicus en un modelo XGBoost.

#### Cómo leer los niveles
- **Bajo, medio y alto** salen del modelo con umbrales fijados en la validación. {LEYENDA}
- El **índice de riesgo** ordena los municipios, pero no es una probabilidad exacta: el modelo sobrestima la tasa
  real de incendios.
- La **prioridad relativa** («top 12 %») indica la posición del municipio entre los de ese día. Es útil en días de
  riesgo generalizado, cuando muchos municipios están en «medio» o «alto».
- **Sin dato**: no había información meteorológica suficiente ese día. No significa riesgo bajo.
- **Estimación sin FWI**: cuando el FWI aún no ha llegado, se usa un modelo de respaldo que discrimina menos.

#### Qué no hace
No detecta incendios activos, no predice su propagación, no es una previsión meteorológica y no sustituye a los
sistemas oficiales de protección civil.

#### Datos
- Predicciones y explicaciones: capa Gold del proyecto, dataset de Hugging Face
  [cpuertolas/piriscope](https://huggingface.co/datasets/cpuertolas/piriscope). Cubren del
  {datos.fmt_fecha(primera)} al {datos.fmt_fecha(ultima)}.
- Límites municipales: recintos del CNIG simplificados.
- Explicaciones: valores SHAP de las 5 variables con más peso en cada predicción.

#### Fuentes y licencias
{FUENTES}
""")

st.divider()
st.caption(f"**{PIE}**")
st.caption(FUENTES)
