# Resultados: modelo frente al baseline FWI

Resultados de la evaluación del modelo de Piriscope (HU-07), generados el 2026-10-27 con la configuración fijada en
[DEC-47](../decisiones.md). Sustituyen a cualquier cifra provisional de los experimentos de desarrollo.

## Resumen

- **El modelo con FWI (`con_fwi`) mejora al FWI en los tres escenarios de test**, con el intervalo de confianza de la diferencia por encima de cero en todos: A (+0,008), B (+0,024) y C (+0,017 de AUC-PR). También mejora en 6 de 7 años en la validación walk-forward (2016-2022). La mejora se mantiene en municipios que el modelo no vio en el entrenamiento (escenarios B y C).
- **La ventaja es modesta en términos absolutos.** En 2023 el AUC-PR del modelo es 0,028, unas 3,7 veces el de un modelo al azar (el FWI, 2,6 veces).
- **El modelo sin FWI (`sin_fwi`) es equivalente al FWI, no mejor**: las diferencias son prácticamente nulas y sus intervalos incluyen el cero en los tres escenarios.
- **Es un único año de test (2023)**, con pocos días con incendio. Las conclusiones son consistentes pero no deben leerse como una demostración estadística definitiva (ver limitaciones).

## Cómo se generó

| | |
|---|---|
| Datos | Dataset de Gold ([DEC-46](../decisiones.md)): 2.950.025 filas municipio-día, 2.628.887 etiquetadas, 78.706 días con incendio (2,99 %) |
| Particiones | Entrenamiento hasta 2021, validación 2022, test 2023. Prevalencia 3,18 % / 1,37 % / 0,75 % |
| Municipios reservados | 81 de 313 ([DEC-45](../decisiones.md)), nunca usados en el entrenamiento |
| Modelo | XGBoost, entrenado desde 2009, sin la variable viento, sin ponderar la clase positiva (DEC-47) |
| Baseline | FWI de EFFIS/Copernicus, usado como puntuación continua sobre exactamente las mismas filas |
| Intervalos | Bootstrap por bloques de 7 días (500 remuestras en el informe final), con diferencia de AUC-PR pareada |

```bash
python -m src.silver
python -m src.features          # usa docs/municipios_reservados.csv (lista congelada, DEC-45)
python -m src.ml.walkforward --anio-desde 2009 --excluir viento --desde-fold 2016
python -m src.ml.experimento --final --variantes con_fwi sin_fwi --anio-desde 2009 --excluir viento --guardar --version v1 --salida data/gold/informe_final.csv
```

## Escenarios de evaluación

| Escenario | Municipios | Periodo | Pregunta |
|---|---|---|---|
| A | Vistos en el entrenamiento | 2023 | ¿Funciona como sistema operativo? |
| B | Reservados (no vistos) | 2009-2021 (el del entrenamiento) | ¿Generaliza a otros municipios? |
| C | Reservados (no vistos) | 2023 | Caso más exigente |

## Modelo con FWI (`con_fwi`)

| Escenario | Filas | Días con incendio | AUC-PR modelo | AUC-PR FWI | Diferencia (IC 95 %) | Veces el azar (modelo / FWI) |
|---|---|---|---|---|---|---|
| A (2023, municipios vistos) | 84.448 | 637 | 0,028 | 0,019 | **+0,008** (+0,001 a +0,019) | 3,7 / 2,6 |
| B (reservados, 2009-2021) | 384.588 | 6.957 | 0,093 | 0,069 | **+0,024** (+0,015 a +0,034) | 5,2 / 3,8 |
| C (reservados, 2023) | 29.484 | 218 | 0,038 | 0,021 | **+0,017** (+0,003 a +0,037) | 5,2 / 2,9 |

## Modelo sin FWI (`sin_fwi`)

| Escenario | Filas | Días con incendio | AUC-PR modelo | AUC-PR FWI | Diferencia (IC 95 %) | Veces el azar (modelo / FWI) |
|---|---|---|---|---|---|---|
| A (2023, municipios vistos) | 84.448 | 637 | 0,022 | 0,019 | **+0,003** (−0,003 a +0,008) | 2,9 / 2,6 |
| B (reservados, 2009-2021) | 384.588 | 6.957 | 0,069 | 0,069 | **+0,000** (−0,005 a +0,006) | 3,8 / 3,8 |
| C (reservados, 2023) | 29.484 | 218 | 0,022 | 0,021 | **+0,001** (−0,005 a +0,010) | 3,0 / 2,9 |

## Desagregado por provincia (`con_fwi`)

| Escenario | Provincia | Días con incendio | AUC-PR modelo | AUC-PR FWI |
|---|---|---|---|---|
| A-test | A Coruña | 151 | 0,016 | 0,015 |
| A-test | Lugo | 108 | 0,029 | 0,022 |
| A-test | Ourense | 196 | 0,045 | 0,022 |
| A-test | Pontevedra | 182 | 0,029 | 0,025 |
| B | A Coruña | 1.298 | 0,066 | 0,067 |
| B | Lugo | 1.229 | 0,102 | 0,058 |
| B | Ourense | 2.864 | 0,103 | 0,073 |
| B | Pontevedra | 1.566 | 0,097 | 0,075 |
| C-test | A Coruña | 35 | 0,018 | 0,024 |
| C-test | Lugo | 52 | 0,039 | 0,032 |
| C-test | Ourense | 80 | 0,043 | 0,019 |
| C-test | Pontevedra | 51 | 0,066 | 0,024 |

Lugo, Ourense y Pontevedra mejoran al FWI en los tres escenarios. **A Coruña no mejora**: empata en A y B y queda por debajo en C (con solo 35 días con incendio, así que es poco concluyente). Es la provincia donde menos aporta el modelo.

## Validación walk-forward (`con_fwi`)

Para cada año Y se entrena solo con años anteriores, se usa Y−1 para la parada temprana y los umbrales, y se evalúa en Y. Solo municipios no reservados; no se toca el año de test.

| Año evaluado | Días con incendio | AUC-PR modelo | AUC-PR FWI | Diferencia (IC 95 %) |
|---|---|---|---|---|
| 2016 | 1.544 | 0,094 | 0,087 | +0,008 (−0,008 a +0,028) |
| 2017 | 2.041 | 0,096 | 0,080 | +0,016 (+0,011 a +0,023) |
| 2018 | 899 | 0,061 | 0,042 | +0,019 (+0,008 a +0,032) |
| 2019 | 1.125 | 0,055 | 0,042 | +0,012 (+0,000 a +0,026) |
| 2020 | 1.023 | 0,060 | 0,061 | −0,000 (−0,005 a +0,005) |
| 2021 | 693 | 0,033 | 0,025 | +0,007 (+0,003 a +0,011) |
| 2022 | 1.194 | 0,057 | 0,049 | +0,008 (−0,008 a +0,020) |

El modelo con FWI supera al FWI en **6 de 7 años**, con el intervalo por encima de cero en 4. Diferencia media +0,010 (mediana +0,008); AUC-PR medio 0,065 frente a 0,055. Sin FWI: mejor en 5 de 7 años, con diferencia media +0,004 y el intervalo por encima de cero en 2.

## Cómo leer las métricas

- **AUC-PR:** área bajo la curva precisión-recall. No depende de un umbral y es la métrica principal con clases tan desbalanceadas. Un modelo al azar obtiene un valor igual a la prevalencia.
- **Veces el azar:** AUC-PR dividido por la prevalencia del escenario. Permite comparar escenarios con prevalencias distintas.
- **Diferencia (IC 95 %):** AUC-PR del modelo menos AUC-PR del FWI, con intervalo por bootstrap *pareado* y por bloques de 7 días: se remuestrean semanas completas, porque los incendios se agrupan en olas. Sigue siendo una aproximación (con bloques de 14 días los intervalos son aún más anchos).

## Lo que cuesta en alertas

Los niveles de riesgo usan umbrales fijados en validación (2022) para un Recall del 80 %. En test, el Recall real fue **0,73 en A, 0,81 en B y 0,76 en C**: el umbral no se reproduce con exactitud de un año a otro, sobre todo porque la prevalencia baja (1,37 % en 2022, 0,75 % en 2023).

En A (2023), con Recall ≈ 0,73 el modelo marca como «medio o superior» cerca del **27 % de los municipio-días** y acierta unas 2 de cada 100 alertas; el FWI, con Recall ≈ 0,77, marca cerca del 37 % y acierta unas 1,6 de cada 100. Los dos puntos no son del todo comparables (distinto Recall); sirven para dimensionar el volumen de alertas.

## Qué se puede afirmar y qué no

**Se puede afirmar**
- El modelo con FWI mejora de forma consistente (aunque modesta) al FWI, tanto en municipios vistos como no vistos.
- Combinar las variables meteorológicas con el FWI funciona mejor que el FWI solo.
- Las variables meteorológicas sin el FWI no superan al FWI.

**No se puede afirmar**
- Que la mejora sea estadísticamente definitiva: es un solo año de test, y en A el límite inferior del intervalo está casi en cero (+0,001).
- Que el modelo sirva para sustituir a los sistemas oficiales. Es una herramienta experimental de apoyo.
- Que las probabilidades estén calibradas para 2023: el modelo se entrenó con una prevalencia del ~2 % y la de 2023 fue del 0,75 %, así que tenderá a sobrestimarlas. Hay que calibrarlas o presentarlas como índice de riesgo antes de mostrarlas como probabilidad.

## Limitaciones

- **Un único año de test (2023)** con 637 días con incendio en A y 218 en C. Los resultados por provincia en C (35 a 80 días) son muy ruidosos.
- **Etiquetas de 2022 y 2023:** [DEC-31](../decisiones.md) indica que EGIF solo es definitivo hasta 2021. Los partes por año de Silver no sugieren que estén incompletos (2021: 950, 2022: 1.713, 2023: 908), pero falta contrastarlos con las estadísticas oficiales.
- **Rendimiento absoluto bajo:** el AUC-PR de test es de 0,028 (A). Solo se usan variables meteorológicas y el FWI; no hay vegetación, terreno ni causas de los incendios.
- **A Coruña** no mejora al FWI.
- **Reserva por estación:** un municipio vecino atendido por otra estación puede quedar en entrenamiento, así que B y C pueden ser algo optimistas.
- **Cobertura de datos:** antes de 2009 la meteorología tiene muchos huecos (por eso el modelo se entrena desde 2009), y el viento, ausente en ~50 % de las filas, se excluyó por identificar estaciones.

## Corrección del escenario B

La primera ejecución del informe final incluyó en el escenario B los años 2001-2008, que el modelo no vio (se entrenó desde 2009): un error del script, ya corregido. Con esos años, `sin_fwi` salía peor que el FWI en B (diferencia −0,018, intervalo −0,033 a −0,008): un artefacto, porque en esos años la meteorología tiene muchísimos huecos y el FWI está casi completo. Con B recortado a 2009-2021 la diferencia es +0,000 (−0,005 a +0,006). El informe conserva ambas versiones: `B` (recortado) y `B-todos-los-anios`. Las cifras de A y C no cambiaron. No se modificó el modelo ni ningún parámetro.

## Ficheros de esta carpeta

| Fichero | Contenido |
|---|---|
| `informe_final.csv` | Informe final completo: A, B, C y `B-todos-los-anios`, global y por provincia, para `con_fwi` y `sin_fwi` |
| `informe_walkforward.csv` | Validación walk-forward 2016-2022 |
| `gold_informe.json` | Tamaños, prevalencia y nulos del dataset de Gold |

Columnas de los informes: `escenario`, `variante` (p. ej. `con_fwi-sin_viento-desde2009`), `grupo` (`global` o provincia), `sistema` (`modelo` o `baseline_fwi`), `n` y `n_pos` (filas y días con incendio), `prevalencia`, `auc_pr` con su intervalo (`_lo`, `_hi`), `recall` con su intervalo, `precision`, `umbral` (el fijado en validación), y, solo en la fila global del modelo, `dif_auc_pr` con `dif_lo`, `dif_hi` y `p_mejora` (fracción de remuestras en que el modelo supera al FWI).
