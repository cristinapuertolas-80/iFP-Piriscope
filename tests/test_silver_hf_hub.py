"""Tests de la subida de Silver a Hugging Face (sin red: la API es un doble)."""

import pytest
from huggingface_hub.utils import filter_repo_objects

from src.ingest import config
from src.silver import hf_hub

CFG = config.HuggingFaceConfig(repo_id="cpuertolas/piriscope", token="t")


class ApiFalsa:
    def __init__(self, error: Exception | None = None):
        self.subidas = []
        self.error = error

    def upload_folder(self, **kwargs):
        if self.error:
            raise self.error
        self.subidas.append(kwargs)


def test_subir_silver_sube_la_carpeta_y_borra_lo_que_ya_no_esta(tmp_path):
    api = ApiFalsa()
    assert hf_hub.subir_silver(tmp_path, CFG, api) == "data/silver"
    llamada = api.subidas[0]
    assert llamada["folder_path"] == str(tmp_path) and llamada["path_in_repo"] == "data/silver"
    assert llamada["repo_id"] == "cpuertolas/piriscope" and llamada["repo_type"] == "dataset"
    assert llamada["delete_patterns"] == hf_hub.BORRAR_SI_NO_ESTAN


def test_patrones_de_borrado_solo_afectan_a_tablas_e_informes():
    # Rutas remotas relativas a data/silver, como las compara huggingface_hub con delete_patterns
    remotas = ["municipio_limites.parquet", "indicador_fwi.parquet", "informes/validacion.json",
               "informes/incendios_asignacion_no_validada.csv", "202610041546", ".gitattributes"]
    borrables = set(filter_repo_objects(remotas, allow_patterns=hf_hub.BORRAR_SI_NO_ESTAN))
    assert borrables == {"municipio_limites.parquet", "indicador_fwi.parquet", "informes/validacion.json",
                         "informes/incendios_asignacion_no_validada.csv"}


def test_subir_silver_sin_permiso_de_escritura(tmp_path):
    api = ApiFalsa(error=RuntimeError("403 Forbidden"))
    with pytest.raises(RuntimeError, match="no tiene permiso de escritura"):
        hf_hub.subir_silver(tmp_path, CFG, api)
