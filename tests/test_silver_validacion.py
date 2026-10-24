"""Tests del validador de tablas Silver.

1. Con un Silver sintético generado por el pipeline, el validador no da errores
   de contrato y detecta los defectos que se le inyectan.
2. Si existen las tablas reales en ``data/silver`` (en local, no en GitHub
   Actions), se validan contra el modelo de datos completo.
"""

import pandas as pd
import pyarrow as pa
import pytest

from src.ingest import config
from src.silver import almacen, pipeline, validacion

from .test_silver_pipeline import _bronze

#: Comprobaciones que dependen de tener los 313 municipios reales
DEPENDEN_DE_GALICIA_COMPLETA = {"n_municipios", "municipios_por_provincia", "hu04_pct_codigo_ine"}


@pytest.fixture
def silver(tmp_path):
    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _bronze(bronze)
    pipeline.ejecutar(base_dir=silver, bronze_dir=bronze, hf_dir=tmp_path / "nada")
    return silver


def _fallidas(v: validacion.Validador) -> set[str]:
    return {r.comprobacion for r in v.errores}


def test_silver_sintetico_cumple_el_contrato(silver):
    v = validacion.validar(silver)
    assert _fallidas(v) - DEPENDEN_DE_GALICIA_COMPLETA == set()
    assert (silver / "informes" / "validacion.json").exists()


def _corromper(silver, tabla, f):
    df = almacen.leer(tabla, silver)
    almacen.guardar(f(df).drop(columns="fecha_insert"), tabla, base_dir=silver)


def test_detecta_clave_duplicada(silver):
    _corromper(silver, "indicador_fwi", lambda df: pd.concat([df, df.head(1)]))
    assert {"clave_unica", "una_fila_por_municipio_y_dia"} <= _fallidas(validacion.validar(silver))


def test_detecta_valores_fuera_de_rango(silver):
    _corromper(silver, "indicador_fwi", lambda df: df.assign(ffmc=150.0))
    assert "rango_ffmc" in _fallidas(validacion.validar(silver))


def test_detecta_clave_foranea_huerfana(silver):
    _corromper(silver, "observacion_meteorologica", lambda df: df.assign(id_estacion_aemet="9999X"))
    assert "fk_id_estacion_aemet" in _fallidas(validacion.validar(silver))


def test_detecta_huecos_en_la_serie_diaria(silver):
    _corromper(silver, "observacion_meteorologica", lambda df: df.iloc[1:])
    assert "serie_diaria_completa" in _fallidas(validacion.validar(silver))


def test_detecta_columnas_fuera_del_modelo(silver):
    import pyarrow.parquet as pq

    ruta = almacen.ruta_tabla("indicador_fwi", silver)
    pq.write_table(pq.read_table(ruta).append_column("dist_celda_km", pa.array([1.0] * 4)), ruta)
    assert "sin_columnas_fuera_del_modelo" in _fallidas(validacion.validar(silver))


def test_detecta_tabla_ausente(silver):
    (silver / "estacion_aemet.parquet").unlink()
    assert "existe" in _fallidas(validacion.validar(silver))


@pytest.mark.skipif(not (config.SILVER_DIR / "municipio.parquet").exists(),
                    reason="No hay tablas Silver reales en data/silver (ejecuta python -m src.silver)")
def test_tablas_silver_reales_cumplen_el_contrato():
    v = validacion.validar()
    assert not v.errores, "\n".join(f"{r.tabla}.{r.comprobacion}: {r.detalle}" for r in v.errores)
