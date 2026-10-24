"""Capa Silver de Piriscope: limpieza, normalización, tipado y cruce espacio-temporal.

Lee los datos crudos de Bronze (``data/bronze`` o su copia de Hugging Face en
``data/hf/raw``) y genera en ``data/silver`` las cinco entidades Silver del
modelo de datos (apartado 7 del diseño técnico):

* ``municipio.parquet`` (+ ``municipio_limites.parquet`` con los polígonos)
* ``estacion_aemet.parquet``
* ``observacion_meteorologica.parquet``
* ``evento_incendio.parquet``
* ``indicador_fwi.parquet``

y los informes de calidad en ``data/silver/informes``.
"""
