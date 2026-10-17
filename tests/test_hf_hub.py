import json

from src.ingest import config, hf_hub
from src.ingest.common import EjecucionBronze


class ApiFalsa:
    def __init__(self):
        self.subidas = []

    def upload_folder(self, **kwargs):
        self.subidas.append(kwargs)


def test_ruta_en_repo_sigue_el_diagrama():
    cfg = config.HuggingFaceConfig(repo_id="x/y", token="t")
    assert hf_hub.ruta_en_repo("aemet", "20261017T060000Z", cfg) == "raw/AEMET/20261017T060000Z"
    assert hf_hub.ruta_en_repo("ine_cnig", "T", cfg) == "raw/INE_CNIG/T"


def test_subir_ejecucion_y_no_repetir(bronze):
    with EjecucionBronze("effis", base_dir=bronze) as run:
        run.guardar_bytes("fwi_2020_01.nc", b"x")
    api = ApiFalsa()
    cfg = config.HuggingFaceConfig(repo_id="cpuertolas/piriscope", token="t")
    destino = hf_hub.subir_ejecucion(run.dir, cfg, api)
    assert destino.startswith("raw/EFFIS/")
    llamada = api.subidas[0]
    assert llamada["repo_type"] == "dataset" and llamada["repo_id"] == "cpuertolas/piriscope"
    assert hf_hub.MARCA_SUBIDO in llamada["ignore_patterns"]
    assert json.loads((run.dir / hf_hub.MARCA_SUBIDO).read_text())["ruta"] == destino
    # Ya marcada: subir_pendientes no la vuelve a subir
    assert hf_hub.subir_pendientes("effis", base_dir=bronze, cfg=cfg, api=api) == []
    assert len(api.subidas) == 1
