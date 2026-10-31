# Dashboard (Streamlit)

Presenta las predicciones D+1 de Piriscope: mapa de Galicia, ficha y buscador de municipios, explicación SHAP, evolución del índice, tabla accesible y comparación del modelo frente al FWI. Sigue el [contrato de datos con ML](../../docs/contrato_dashboard_gold.md).

```bash
python run_dashboard.py
```

Es lo único que hay que ejecutar, desde la raíz del repositorio. La primera vez prepara el entorno, instala `environment/requirements-dashboard.txt`, pide el token de Hugging Face y comprueba el acceso; después abre el dashboard en el navegador (normalmente http://localhost:8501). Para cerrarlo, Ctrl+C en la terminal.

| Opción | Para qué |
|---|---|
| `--cambiar-token` | Volver a pedir el token (si caduca o te equivocaste al pegarlo) |
| `--local` | Usar `data/gold/` en lugar de Hugging Face, sin token (por ejemplo, tras ejecutar el pipeline) |
| `--solo-preparar` | Instalar y comprobar, sin abrir el dashboard |
| `--puerto`, `--sin-navegador` | Elegir el puerto o no abrir el navegador |

Con el entorno ya preparado también vale `streamlit run src/dashboard/app.py`. Una vista concreta se puede compartir con la URL: `?fecha=2025-08-14&municipio=32054`.

## De dónde salen los datos

| Dato | Origen |
|---|---|
| Predicciones y explicaciones SHAP | `data/gold/predicciones.parquet` y `explicaciones.parquet` del dataset de Hugging Face [cpuertolas/piriscope](https://huggingface.co/datasets/cpuertolas/piriscope). Se descargan solos al arrancar (caché de Hugging Face; solo se baja lo que ha cambiado) |
| Límites, nombres y provincias de los 313 municipios | `assets/municipios_galicia.geojson`, versionado en el repositorio |
| Evaluación frente al FWI e importancia global | `docs/resultados/informe_final.csv`, `informe_walkforward.csv`, `shap_global*.csv` |

El dataset es de acceso restringido: hace falta un **token de lectura** de Hugging Face en el `.env` de la raíz (`HF_TOKEN=...`) y haber aceptado las condiciones del dataset con esa cuenta. `run_dashboard.py` pide el token y comprueba las dos cosas.

- Sin conexión, el dashboard usa los ficheros de `data/gold/` si existen y lo avisa.
- `PIRISCOPE_DASHBOARD_ORIGEN=local` fuerza a usar siempre `data/gold/` (por ejemplo, tras ejecutar el pipeline en local).
- El botón «Comprobar si hay datos nuevos» de la barra lateral vacía la caché y vuelve a consultar Hugging Face.

El dashboard **no** entrena, no carga modelos y no lee Bronze ni Silver.

## Regenerar el mapa

`assets/municipios_galicia.geojson` se genera a partir de los recintos del CNIG (simplificados a 150 m). Solo hay que regenerarlo si cambian los límites municipales:

```bash
python scripts/generar_mapa_municipios.py   # descarga el GeoPackage del CNIG (~230 MB) si no está en data/landing
```

## Módulos

| Fichero | Qué hace |
|---|---|
| `app.py` | Interfaz Streamlit (pestañas, barra lateral, textos obligatorios) |
| `datos.py` | Descarga y lectura de Gold, red de seguridad de duplicados, predicciones del día, casos límite y formatos. Sin dependencias de Streamlit |
| `graficos.py` | Mapa, evolución del índice y gráficos de evaluación (Plotly) |

Los gráficos SHAP reutilizan `src/explainability/plots.py`.

## Cómo cumple el contrato

| Requisito | Implementación |
|---|---|
| Fecha por defecto = última con predicción, no hoy | Barra lateral, con «Última fecha con predicción» siempre visible |
| Municipio sin fila → «Sin dato», distinto de «bajo» | Gris en el mapa con punto oscuro (no solo color) y mensaje en la ficha |
| Filas `-respaldo` | Etiqueta «Estimación sin FWI», FWI «no disponible», borde grueso en el mapa y círculo hueco en la evolución |
| Niveles | Siempre de `nivel_riesgo`, nunca recalculados; paleta Okabe-Ito y nivel en texto |
| Probabilidad | Como «índice de riesgo», junto al nivel |
| Prioridad relativa | Por fecha y versión de modelo, como «top N %» |
| Episodio generalizado (≥ 90 % medio/alto) | Aviso informativo |
| Explicaciones sin columnas de DEC-50 | Se muestran solo las barras |
| Rendimiento (HU-09, < 5 s) | Lectura cacheada; `explicaciones` se filtra por `id_prediccion` (~0,05 s) |
| Evaluación | Prevalencia y «veces el azar» junto al AUC-PR, intervalos, desglose por provincia con aviso donde el modelo no mejora al FWI |
| Textos obligatorios | Pie y fuentes en todas las páginas |
