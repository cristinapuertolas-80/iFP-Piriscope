# `src/explainability` · Explicaciones SHAP

Calcula qué variables han empujado cada predicción hacia arriba o hacia abajo (HU-08, apartado 5.4 del diseño técnico). Se ejecuta en el proceso batch junto a la inferencia de [`src/ml`](../ml/README.md); el dashboard solo lee los resultados de Gold y **no carga el modelo**.

> **Estado:** implementado y conectado a `inferir.py`; aún no conectado a `run_pipeline.py`.

## Módulos

| Módulo | Qué hace |
|---|---|
| `shap_explain.py` | `build_explanations`: tabla `ExplicacionPrediccion` con las 5 variables de mayor \|SHAP\| por predicción, su valor, el valor base y el resto. `global_importance`: media de \|SHAP\| por variable o por grupo. `write_explanations`: escribe `data/gold/explicaciones.parquet` |
| `plots.py` | Gráficos Plotly a partir de las tablas de Gold: `fig_local` (una predicción), `fig_cascada` (del valor base a la predicción) y `fig_global` (importancia global). `ETIQUETAS`: nombres legibles de las variables |

## Uso

```python
from src.explainability import shap_explain, plots

expl = shap_explain.build_explanations(pred, df, bundle, verificar=True)   # pred: salida de predict.build_predictions
shap_explain.write_explanations(expl)                                      # data/gold/explicaciones.parquet

unidades = shap_explain.shap_units(bundle)
fig = plots.fig_local(expl, pred["id_prediccion"].iloc[0], unidades, titulo="Municipio 32031 · 19/10/2026",
                      etiquetas=plots.ETIQUETAS, mostrar_valores=True, incluir_resto=True)
cascada = plots.fig_cascada(expl, pred["id_prediccion"].iloc[0], unidades, etiquetas=plots.ETIQUETAS)

imp = shap_explain.global_importance(bundle, df_validacion)                       # por variable
grupos = shap_explain.global_importance(bundle, df_validacion, por_grupo=True)    # por grupo
fig_global = plots.fig_global(grupos)
```

Cada predicción de D+1 se explica con la fila del día D, la misma que vio el modelo.

## Esquema de salida

| Columna | Descripción |
|---|---|
| `id_prediccion` | Clave foránea a `predicciones.parquet` |
| `variable` | Variable analizada |
| `valor_shap` | Contribución de la variable. Positivo = sube el riesgo |
| `valor_variable` | Valor de la variable el día D (nulo si faltaba) |
| `valor_base` | Valor esperado del modelo, en las mismas unidades que `valor_shap` |
| `shap_resto` | Suma de las contribuciones de las variables que no están en el top-k |

La clave es `id_prediccion` + `variable`. Las tres últimas columnas se repiten en las cinco filas de cada predicción (DEC-50). `build_explanations(..., con_valores=False)` devuelve solo las tres primeras.

**Reconstruir la salida del modelo** con todas las columnas:

```python
g = expl.groupby("id_prediccion").agg(base=("valor_base", "first"), top=("valor_shap", "sum"), resto=("shap_resto", "first"))
probabilidad = 1 / (1 + np.exp(-(g.base + g.top + g.resto)))      # XGBoost (log-odds)
# Random Forest: probabilidad = g.base + g.top + g.resto
```

## Qué hay que saber al usar los valores

- **Unidades:** log-odds en XGBoost y probabilidad en Random Forest (`shap_units(bundle)` lo indica). No son comparables entre algoritmos.
- **Variables correlacionadas:** los índices FWI (`fwi`, `ffmc`, `dmc`, `dc`, `isi`, `bui`) se calculan a partir de la meteorología y entre sí. SHAP reparte su efecto entre ellos, así que una temperatura con poca importancia no significa que no influya: su información ya está en `ffmc`. `global_importance(..., por_grupo=True)` suma las contribuciones por grupo (`GRUPOS`) antes del valor absoluto.
- **Ficheros anteriores a la ampliación:** las filas generadas sin las tres columnas nuevas las conservan como nulas. Para tenerlas completas, regenera con `inferir --reemplazar`. El dashboard debe tolerar columnas ausentes o nulas.
- **`verificar=True`:** comprueba que `valor_base` más la suma de SHAP reproduce la salida del modelo y avisa (no falla) si no se cumple; suele indicar que se explica con otro modelo del que predijo.
- **Accesibilidad (DEC-28):** los gráficos usan una paleta apta para daltonismo y el sentido de cada barra (sube o baja el riesgo) va también en texto, no solo en color.

## Decisiones

[docs/decisiones.md](../../docs/decisiones.md): DEC-43 (SHAP y gráficos desde Gold) y DEC-50 (explicaciones ampliadas).

## Tests

```bash
pytest tests/test_shap.py tests/test_shap_valores.py
```

`test_shap.py` comprueba, entre otras cosas, que los valores SHAP son aditivos, que se usa la fila del día D y que los gráficos tienen 5 barras con el sentido en texto. `test_shap_valores.py` cubre las columnas ampliadas (valor de la variable del día D, reconstrucción de la probabilidad, `top_k=None`), la importancia por grupos, la escritura sobre ficheros antiguos y los gráficos nuevos.
