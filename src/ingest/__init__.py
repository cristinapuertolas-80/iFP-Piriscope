"""Ingesta de las fuentes de Piriscope en la capa Bronze.

Módulos:

* :mod:`.aemet`     – meteorología diaria de AEMET OpenData (HU-01)
* :mod:`.egif`      – histórico de incendios EGIF-MITECO (HU-02)
* :mod:`.effis`     – FWI y componentes de EFFIS/Copernicus (HU-03)
* :mod:`.ine_cnig`  – catálogo de municipios (INE) y límites (CNIG)
* :mod:`.hf_hub`    – archivo de los originales en Hugging Face
* :mod:`.cli`       – línea de comandos: ``python -m src.ingest --help``
"""

__version__ = "0.1.0"
