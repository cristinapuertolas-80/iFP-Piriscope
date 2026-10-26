# `src/features` · Capa Gold (variables derivadas y dataset municipio-día)

Lee las tablas de Silver y construye el dataset de Gold: una fila por municipio y día con las variables meteorológicas, el FWI, las **variables derivadas** (HU-05), la variable objetivo `incendio_d1` y las particiones (apartado 5.2 del diseño técnico).

> **Estado:** implementado y probado con tablas de Silver sintéticas (con los mismos tipos que las reales) y ejecutado sobre el Silver completo.

## Módulos

| Módulo | Qué hace |
|---|---|
| `derived.py` | `racha_dias_secos`, `precip_acum_7d` y `mes`, por municipio y usando solo datos hasta el día D |
| `dataset.py` | Lectura de Silver, etiqueta, particiones, `reservado`, escritura de Gold e informe |

## Cómo ejecutarlo

Desde la raíz del repositorio, con las tablas de Silver generadas (`python -m src.silver`) y la lista `docs/municipios_reservados.csv`:

```bash
python -m src.features
```

| Opción | Por defecto | Para qué |
|---|---|---|
| `--silver-dir` | `data/silver` | Tablas de entrada |
| `--gold-dir` | `data/gold` | Carpeta de salida |
| `--reservados` | `docs/municipios_reservados.csv` | Lista de municipios reservados (fuente de verdad de `reservado`) |
| `--umbral-seco-mm` | `1.0` | Un día es seco si llueve menos de este valor |
| `--min-dias-acum` | `7` | Días con dato exigidos en la ventana de la precipitación acumulada |
| `--anio-train-hasta` | `2021` | Último año de la partición de entrenamiento |
| `--anio-validacion` | `2022` | Año de validación |
| `--anio-test` | `2023` | Año de test (último con etiquetas) |

## Variables derivadas

Todas usan **solo datos de los días ≤ D** (un test lo comprueba: alterar datos posteriores a D no cambia ninguna derivada de D). No se imputa ningún día sin dato.

| Variable | Definición |
|---|---|
| `racha_dias_secos` | Días consecutivos, hasta D incluido, con precipitación < `--umbral-seco-mm`. Un día lluvioso la reinicia a 0. Un día sin dato la deja **desconocida (NaN)** hasta el siguiente día lluvioso. Al inicio de la serie cuenta desde 1 |
| `precip_acum_7d` | Precipitación (mm) de los 7 días que terminan en D (D incluido). NaN si falta algún día de la ventana |
| `mes` | Mes del día D (1-12) |

## Etiqueta y particiones

- **`incendio_d1`** = 1 si el municipio tiene al menos un parte de EGIF con fecha de detección **D+1**; 0 en caso contrario. Los partes sin municipio asignado se ignoran.
- **Sin etiqueta fuera de cobertura:** las filas cuyo D+1 cae después del final del año de test no llevan etiqueta ni partición (nunca se marcan como 0, porque no hay cobertura de EGIF). Solo aparecen en el fichero completo.
- **`particion`** por año de `fecha`: `entrenamiento` (≤ 2021), `validacion` (2022) y `test` (2023). Los años están pendientes de cerrar (DEC-21).
- **`reservado`** se lee de `docs/municipios_reservados.csv`; si no existe, se usa `municipio.reservado` de Silver, y si también es nulo el comando se detiene.

## Salidas (`data/gold/`, no versionadas)

| Fichero | Contenido |
|---|---|
| `dataset_entrenamiento.parquet` | Solo filas con etiqueta y partición. Cumple el contrato de `src/ml/config.py` (se valida antes de escribir) |
| `dataset_municipio_dia.parquet` | Todas las filas, incluidos los días recientes sin etiqueta: sirve para la inferencia D → D+1 y el dashboard |
| `gold_informe.json` | Filas, positivos, prevalencia por partición, por reservado y por provincia, y % de nulos por variable |

Columnas: `id_municipio`, `provincia`, `fecha`, `reservado`, `particion`, `incendio_d1`, las cuatro variables meteorológicas, las tres derivadas, el FWI con sus cinco componentes y `dist_estacion_km` (solo para análisis, no es variable del modelo).

## Resultado sobre el Silver actual

| | |
|---|---|
| Filas | 2.950.025 (2.628.887 etiquetadas), de 2001-01-01 a 2026-10-21 |
| Días con incendio | 78.706 = 2,99 % de las filas etiquetadas |
| Prevalencia por partición | entrenamiento 3,18 %, validación 1,37 %, test 0,75 % |
| Nulos (filas etiquetadas) | temperatura 18,5 %, humedad 27,2 %, viento 57,1 %, precipitación 19,7 %, racha 23,1 %, acumulada 24,8 %, FWI 3,25 % |

## Qué hay que saber

- La prevalencia cae de ~6 % (2001-2005) a ~1 % (desde 2018), y la cobertura meteorológica mejora de golpe en 2009. Esa mezcla puede hacer que los nulos actúen como señal de época: se decide en los experimentos de validación (`src/ml/experimento.py`).
- El viento falta en torno al 50 % desde 2012 (muchas estaciones no lo miden), así que su ausencia identifica estaciones concretas.
- Los nulos que añaden las derivadas respecto a la precipitación original son pocos (3-5 puntos): no hace falta relajar `--min-dias-acum`.

## Decisiones

[docs/decisiones.md](../../docs/decisiones.md): DEC-46 (variables derivadas, etiqueta, particiones y salidas de Gold).

## Tests

```bash
pytest tests/test_features.py
```

Comprueban, entre otras cosas, que las derivadas coinciden con una implementación lenta de referencia, que no usan datos posteriores a D, que la etiqueta es el día siguiente y no el mismo día, que no hay etiqueta fuera de cobertura y que el dataset cumple el contrato de ML con los tipos reales de Silver (`string`, `Float64`, `Int8`).
