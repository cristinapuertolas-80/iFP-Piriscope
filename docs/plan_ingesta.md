# Plan de ingesta de datos

Frecuencia, periodo y procedimiento para cargar las fuentes de Piriscope en la capa Bronze y archivarlas en Hugging Face (`raw/<FUENTE>/<timestamp>/`). El código está en [`src/ingest`](../src/ingest/README.md) y la decisión se registra en [DEC-31](decisiones.md#dec-31-periodo-y-frecuencias-de-ingesta).

Hay dos fases:

- **Fase A · Carga inicial**: una sola vez, desde local, con `scripts/carga_inicial.sh` (o `.ps1`).
- **Fase B · Actualización periódica**: automática en GitHub Actions; se puede lanzar también desde local.

---

## 1. Resumen por fuente

| Fuente | Carga inicial | Desde | Actualización | Automática | Responsable |
|---|---|---|---|---|---|
| **AEMET** | Sí | 2001-01-01 | **Diaria** (últimos 10 días) | Sí, cron diario | Cristina |
| **FWI** (EFFIS/Copernicus) | Sí | 2001-01-01 | **Semanal** (últimos 14 días disponibles) | Sí, cron los lunes | Cristina |
| **EGIF** | Sí | Todo lo disponible (mínimo 2001) | **Anual**, más una recarga antes del dataset de entrenamiento (S7) | No (exportación manual) | Cristina |
| **INE** (municipios) | Sí | Año en curso | **Anual** | No | Cristina |
| **CNIG** (límites) | Sí | Versión vigente | **Una vez** (salvo cambio oficial) | No | Cristina |

## 2. Periodo y frecuencias: por qué

**Periodo de carga: 2001–actualidad (25 años).**

- **EGIF limita el periodo.** Empieza en 1968, pero los años **definitivos** publicados por MITECO llegan a 2021. El buscador ofrece los datos «consolidados a nivel provincial en el momento de la consulta», por lo que los años posteriores pueden estar incompletos.
- **FWI** (`cems-fire-historical-v1`) cubre de 1940 a la actualidad y se actualiza a diario; no limita.
- **AEMET** tiene series largas en la mayoría de estaciones; no limita, aunque la cobertura varía por estación.
- 25 años coinciden con la estimación del diseño técnico (~3 M filas municipio-día).
- **El periodo común definitivo y la partición temporal (DEC-21) se fijan en el EDA**: se carga ancho y se recorta después (por ejemplo, a 2001–2021 si EGIF no tiene completos los años siguientes).

**Frecuencias:**

- **AEMET diaria**: el modelo predice D+1 con la meteorología observada hasta D. Se piden los últimos 10 días porque los valores diarios se publican con retraso y pueden corregirse; los duplicados se resuelven en Silver.
- **FWI semanal**: se calcula a partir del reanálisis ERA5, que llega ~5 días por detrás del tiempo real, así que bajarlo a diario no aporta. La orden usa `--retraso 5` por defecto. Los meses que aún no están en el `consolidated_dataset` se descargan automáticamente del `intermediate_dataset` (`fwi_AAAA_MM_intermedio.nc`); en ejecuciones posteriores se vuelve a intentar el consolidado y, cuando existe, lo sustituye. En Silver, si hay ambos para un mes, prevalece el consolidado.
- **EGIF anual**: es una estadística por campañas, sin tiempo real; sólo se usa para entrenar y evaluar.
- **INE anual**: el diccionario de municipios se publica una vez al año. Las variaciones de municipios alimentan `docs/tabla_equivalencias.csv`.

---

## 3. Fase A · Carga inicial (una vez, desde local)

### 3.0 Requisitos previos

- [ ] Repositorio actualizado: `git checkout main && git pull`
- [ ] Entorno activado: `source .venv/bin/activate` (Windows: `.venv\Scripts\Activate.ps1`)
- [ ] `.env` con `AEMET_API_KEY`, `CDSAPI_KEY` y `HF_TOKEN` (ver `.env.example`)
- [ ] Licencia del dataset `cems-fire-historical-v1` aceptada en la web del EWDS
- [ ] Portátil enchufado y sin suspensión (AEMET puede tardar un día entero)

### 3.1 Preparar los ficheros manuales

**CNIG · límites municipales**

1. [Centro de Descargas del CNIG](https://centrodedescargas.cnig.es) → *Información geográfica de referencia* → **recintos municipales**, formato **GeoPackage**. Seleccionar sólo las provincias gallegas si la web lo permite; si no, el fichero nacional también vale.
2. Guardar el fichero en `data/landing/ine_cnig/`.

**EGIF · partes de incendio**

1. [Buscador EGIF](https://servicio.mapa.gob.es/incendios/Search/Publico) → *Comunidad Autónoma*: Galicia → *Provincia*: A Coruña.
2. Dejar *Año Parte* vacío si el buscador lo permite; si no, exportar año a año (como mínimo desde 2001).
3. **Exportar el resultado de la Búsqueda (resumen) a archivo Excel**. Si hay más de 100.000 registros, en varios bloques.
4. Repetir para **Lugo, Ourense y Pontevedra**.
5. Guardar los Excel en `data/landing/egif/` con la provincia en el nombre (por ejemplo, `egif_lugo_2001-2025.xlsx`).

> Comprobar con la primera exportación si el «resumen» trae coordenadas UTM y superficie. Si no, valorar la exportación XML completa, porque esas columnas validan la asignación de municipios (HU-04).

### 3.2 Lanzar la carga

```bash
# macOS / Linux / Git Bash
scripts/carga_inicial.sh
```

```powershell
# Windows PowerShell
.\scripts\carga_inicial.ps1
```

El script, en este orden:

| Paso | Fuente | Qué hace | Duración orientativa |
|---|---|---|---|
| A1 | INE/CNIG | Diccionario del INE (descarga automática) + límites de `data/landing/ine_cnig/` | Minutos |
| A2 | EGIF | Unifica las exportaciones de `data/landing/egif/` (se omite si no hay ficheros) | Minutos |
| A3 | FWI | Un año por ejecución, 2001 → año anterior, y después el año en curso | Horas (cola del EWDS) |
| A4 | AEMET | Un año por ejecución, 2001 → año anterior, y después el año en curso | Hasta un día |

Cada paso sube su resultado a Hugging Face. Un paso que falla no detiene los demás; al final se listan los fallidos y el log completo queda en `logs/carga_inicial_<fecha>.log`.

**Si se interrumpe o algo falla**, se relanza el mismo script: los años y meses ya descargados se reutilizan desde la caché de `data/bronze`. Los años siempre van de 1 de enero a 31 de diciembre, de modo que los tramos coinciden entre ejecuciones.

**Opciones útiles:**

| Bash | PowerShell | Efecto |
|---|---|---|
| `--fuentes "effis aemet"` | `-Fuentes effis,aemet` | Sólo esas fuentes |
| `--desde-anio 2010 --hasta-anio 2012` | `-DesdeAnio 2010 -HastaAnio 2012` | Otro rango de años |
| `--sin-subir` | `-SinSubir` | No subir a Hugging Face (pruebas) |
| `--sin-anio-en-curso` | `-SinAnioEnCurso` | Sólo años completos |

### 3.3 Comprobaciones al terminar

- [ ] Todos los `data/bronze/<fuente>/<timestamp>/manifest.json` en estado `completada` o `completada_con_incidencias`, con incidencias entendidas.
- [ ] Hugging Face: existen `raw/AEMET`, `raw/EFFIS`, `raw/EGIF` y `raw/INE_CNIG`.
- [ ] **AEMET**: revisar `cobertura_estaciones.csv` de cada año y anotar las estaciones con cobertura baja para el EDA.
- [ ] **EGIF**: revisar `mapeo_columnas.json` (columnas reconocidas), `informe_duplicados.csv` y `recuentos_provincia_anio.csv`; anotar **el último año completo** de cada provincia.
- [ ] **FWI**: anotar los meses descargados como `_intermedio` (incidencias `usa_intermedio`).
- [ ] Registrar en `docs/decisiones.md` el periodo común real y cerrar el punto pendiente de DEC-21.

---

## 4. Fase B · Actualización periódica

### 4.1 Calendario

| Frecuencia | Fuente | Automático (GitHub Actions) | Comando equivalente en local |
|---|---|---|---|
| **Diaria** (06:17 hora peninsular) | AEMET | Cron `17 4 * * *` | `python -m src.ingest aemet --dias 10 --subir-hf` |
| **Semanal** (lunes, 06:41) | FWI | Cron `41 4 * * 1` | `python -m src.ingest effis --dias 14 --subir-hf` |
| **Trimestral** | FWI consolidado | — | `python -m src.ingest effis --desde <inicio del trimestre> --hasta <fin> --subir-hf` (sustituye los `_intermedio` que ya estén consolidados) |
| **Anual** y en **S7** | EGIF | — | Nueva exportación (3.1) → `python -m src.ingest egif --subir-hf` |
| **Anual** | INE | — | `python -m src.ingest ine_cnig --anio <año> --subir-hf` |
| **Nunca** (salvo cambio oficial) | CNIG | — | — |

Si el cron ya está funcionando, **no hace falta lanzar nada en local** para AEMET ni FWI. En local sólo se lanzan si el cron ha fallado, si faltan días o si se quieren los datos al momento.

### 4.2 Lanzarlo a mano desde GitHub

**Actions → Ingesta automática → Run workflow**, elegir la fuente (`aemet`, `effis` o `ine_cnig`) y el rango (fechas o últimos N días). EGIF no aparece porque sus ficheros se exportan a mano.

### 4.3 Vigilancia

- Revisar cada lunes el *Summary* de las últimas ejecuciones del workflow (estado, ficheros e incidencias).
- GitHub avisa por correo cuando un workflow programado falla.
- Si AEMET acumula días sin datos, relanzar con un rango mayor: `python -m src.ingest aemet --dias 30 --subir-hf`.

---

## 5. Encaje en el cronograma

| Semana | Tarea de ingesta |
|---|---|
| **S5** (16–22/10) | Fase A completa y periodo común fijado |
| **S5–S12** | Fase B: AEMET diaria y FWI semanal (automáticos) |
| **S7** (30/10–05/11) | Nueva exportación de EGIF antes de construir el dataset de entrenamiento |
| **S11** (27/11–03/12) | Última actualización antes de la entrega; recarga del FWI consolidado |

---

## 6. Problemas frecuentes

| Síntoma | Causa probable | Qué hacer |
|---|---|---|
| `CERTIFICATE_VERIFY_FAILED` / `self signed certificate in certificate chain` (el script se detiene con «FALLO DE CONEXIÓN») | El antivirus (análisis HTTPS) o la red del centro/empresa interceptan las conexiones y Python no reconoce su certificado | Ver [6.1](#61-error-de-certificado-certificate_verify_failed) |
| `Falta la API key de AEMET` | `.env` o secret sin `AEMET_API_KEY` | Revisar nombre y valor |
| `API key de AEMET no válida (401)` | Clave incorrecta o caducada | Solicitar una nueva en AEMET OpenData |
| Muchos `AEMET 429` en el log | Límite de peticiones por minuto | Nada: el código espera y reintenta. Si persiste, subir `AEMET_PAUSA_SEGUNDOS` en `.env` |
| `error_descarga` en el FWI | Licencia no aceptada, clave incorrecta o mes no disponible | Revisar la licencia en el EWDS y `CDSAPI_KEY`; relanzar |
| `recorte_rechazado` en el FWI | El servidor no acepta el recorte de Galicia | Nada: se descarga sin recorte y se recorta en Silver |
| EGIF: `columnas_no_reconocidas` | La exportación usa otros nombres de columna | Ampliar `MAPEO_COLUMNAS` en `src/ingest/egif.py` |
| EGIF: `provincia_ausente` | Falta la exportación de alguna provincia | Exportarla y relanzar `egif` |
| Error 401/403 de Hugging Face | Token sin permiso de escritura o dataset con otro nombre | Revisar `HF_TOKEN` y `HF_REPO_ID` |

### 6.1 Error de certificado (`CERTIFICATE_VERIFY_FAILED`)

No es un fallo de la fuente ni de las claves: Python no confía en el certificado que recibe al conectar. La ingesta se detiene en el primer intento y el script indica «FALLO DE CONEXIÓN».

1. **Comprobar quién emite el certificado.** Abrir https://ewds.climate.copernicus.eu en el navegador → candado → *La conexión es segura* → *Certificado*. Si el emisor es un antivirus (Avast, AVG, Kaspersky, ESET, Bitdefender…) o un proxy (Fortinet, Zscaler, Sophos…), la conexión está siendo interceptada.
2. **Solución recomendada:** que Python use los certificados del sistema operativo, que ya confían en ese antivirus o proxy. Con el `.venv` activado:
   ```bash
   pip install pip-system-certs
   ```
   Comprobar:
   ```bash
   python -c "import requests; print(requests.get('https://ewds.climate.copernicus.eu/api', timeout=30).status_code)"
   ```
   Cualquier código HTTP (200, 404…) significa que el certificado ya se acepta; si vuelve a salir `CERTIFICATE_VERIFY_FAILED`, pasar al punto 3.
3. **Alternativas:**
   - Lanzar la carga desde otra red (casa, datos del móvil).
   - Desactivar el análisis HTTPS del antivirus o añadir una excepción para `ewds.climate.copernicus.eu`, `opendata.aemet.es` y `huggingface.co`.
   - En una red corporativa, pedir a sistemas el certificado raíz del proxy.
4. **Relanzar** el script: lo ya descargado se reutiliza.

**No desactivar nunca la verificación de certificados** (`verify=False`, `PYTHONHTTPSVERIFY=0`…): dejaría las claves y los datos expuestos.

## Fuentes

- [MITECO · Estadística General de Incendios Forestales](https://www.miteco.gob.es/es/biodiversidad/temas/incendios-forestales/estadisticas-datos.html)
- [EWDS · Fire danger indices historical data (cems-fire-historical-v1)](https://ewds.climate.copernicus.eu/datasets/cems-fire-historical-v1)
- [Copernicus · ERA5 a cinco días del tiempo real](https://climate.copernicus.eu/key-update-climate-dataset-brings-data-five-days-behind-real-time)
- [CNIG · Base de Datos de Divisiones Administrativas de España](https://datos.gob.es/es/catalogo/e00125901-spaignllm)
