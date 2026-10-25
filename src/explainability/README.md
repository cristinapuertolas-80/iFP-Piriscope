# `src/explainability` · Explicaciones SHAP

Calcula qué variables han empujado cada predicción hacia arriba o hacia abajo (HU-08, apartado 5.4 del diseño técnico). Se ejecuta en el proceso batch junto a la inferencia de [`src/ml`](../ml/README.md); el dashboard solo lee los resultados de Gold y **no carga el modelo**.

> **Estado:** implementado y probado con datos de juguete; aún no conectado a `run_pipeline.py`.

## Módulos

| Módulo | Qué hace |
|---|---|
| `shap_explain.py` | `build_explanations`: tabla `ExplicacionPrediccion` con las 5 variables de mayor |SHAP| por predicción. `global_importance`: media de \|SHAP\| por variable. `write_explanations`: escribe `data/gold/explicaciones.parquet` |
| `plots.py` | Gráficos Plotly a partir de las tablas de Gold: `fig_local` (una predicción) y `fig_global` (importancia global) |

## Uso

```python
from src.explainability import shap_explain, plots

expl = shap_explain.build_explanations(pred, df, bundle)        # pred: salida de predict.build_predictions
shap_explain.write_explanations(expl)                           # data/gold/explicaciones.parquet

fig = plots.fig_local(expl, pred["id_prediccion"].iloc[0], shap_explain.shap_units(bundle))
imp = shap_explain.global_importance(bundle, df_validacion)
fig_global = plots.fig_global(imp)
```

Cada predicción de D+1 se explica con la fila del día D, la misma que vio el modelo.

## Esquema de salida

| Columna | Descripción |
|---|---|
| `id_prediccion` | Clave foránea a `predicciones.parquet` |
| `variable` | Variable analizada |
| `valor_shap` | Contribución de la variable. Positivo = sube el riesgo |

La clave es `id_prediccion` + `variable`. Las escrituras sobrescriben las filas del mismo `id_prediccion`.

## Qué hay que saber al usar los valores

- **Unidades:** log-odds en XGBoost y probabilidad en Random Forest (`shap_units(bundle)` lo indica). No son comparables entre algoritmos.
- **Solo la contribución:** la tabla no guarda el valor de la variable (el esquema del Anexo A no tiene esa columna). Para mostrar «temperatura = 31 °C» hay que cruzar con la tabla de Gold o ampliar el esquema.
- **Accesibilidad (DEC-28):** los gráficos usan una paleta apta para daltonismo y el sentido de cada barra (sube o baja el riesgo) va también en texto, no solo en color.

## Decisiones

[docs/decisiones.md](../../docs/decisiones.md): DEC-43 (SHAP y gráficos desde Gold).

## Tests

```bash
pytest tests/test_shap.py
```

Comprueban, entre otras cosas, que los valores SHAP son aditivos (suma de contribuciones más valor base igual a la salida del modelo), que se usa la fila del día D y que los gráficos tienen 5 barras con el sentido en texto.
