"""Los fallos de conexión (certificados, DNS, proxy) detienen la ingesta con un
mensaje claro, en lugar de reintentar cada mes/tramo o confundirse con otros errores."""
import json
from datetime import date

import pytest
import requests

from src.ingest import aemet, config, effis, hf_hub
from src.ingest.common import EjecucionBronze, ErrorRed, es_error_red, es_error_ssl

MENSAJE_SSL = ("HTTPSConnectionPool(host='ewds.climate.copernicus.eu', port=443): Max retries exceeded "
               "(Caused by SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] "
               "certificate verify failed: self signed certificate in certificate chain')))")


def test_clasificacion_de_errores():
    assert es_error_ssl(requests.exceptions.SSLError(MENSAJE_SSL))
    assert es_error_ssl(RuntimeError(MENSAJE_SSL))  # envuelto por otra librería
    assert es_error_red(requests.exceptions.ConnectionError("Failed to establish a new connection"))
    assert not es_error_red(RuntimeError("invalid request: area"))
    assert not es_error_ssl(requests.exceptions.ConnectionError("timeout"))


class DescargaSinCertificado:
    def __init__(self):
        self.llamadas = 0

    def __call__(self, dataset, peticion, destino):
        self.llamadas += 1
        raise requests.exceptions.SSLError(MENSAJE_SSL)


def test_effis_para_al_primer_error_ssl(bronze):
    desc = DescargaSinCertificado()
    with pytest.raises(ErrorRed, match="certificado"):
        effis.ingestar(date(2001, 1, 1), date(2001, 12, 31), base_dir=bronze, descargar=desc)
    assert desc.llamadas == 1  # ni reintento sin 'area', ni intermedio, ni resto de meses
    m = json.loads(next((bronze / "effis").glob("*/manifest.json")).read_text(encoding="utf-8"))
    assert m["estado"] == "fallida"
    tipos = [i["tipo"] for i in m["incidencias"]]
    assert "error_red" in tipos
    assert "recorte_rechazado" not in tipos and "usa_intermedio" not in tipos


class SesionSinCertificado:
    def __init__(self):
        self.llamadas = 0

    def get(self, *a, **k):
        self.llamadas += 1
        raise requests.exceptions.SSLError(MENSAJE_SSL)


def test_aemet_para_al_primer_error_ssl(bronze):
    sesion = SesionSinCertificado()
    cfg = config.AemetConfig(api_key="x", pausa_segundos=0, espera_429_segundos=0)
    with pytest.raises(ErrorRed, match="certificado"):
        aemet.ingestar(date(2020, 1, 1), date(2020, 1, 5), base_dir=bronze,
                       cliente=aemet.ClienteAemet(cfg=cfg, sesion=sesion))
    assert sesion.llamadas == 1


def test_hf_error_ssl_se_explica(bronze):
    class ApiSinCertificado:
        def upload_folder(self, **kw):
            raise requests.exceptions.SSLError(MENSAJE_SSL)

    with EjecucionBronze("effis", base_dir=bronze) as run:
        run.guardar_bytes("fwi_2020_01.nc", b"x")
    with pytest.raises(ErrorRed, match="pip-system-certs"):
        hf_hub.subir_ejecucion(run.dir, config.HuggingFaceConfig(repo_id="a/b", token="t"),
                               ApiSinCertificado())
    assert not (run.dir / hf_hub.MARCA_SUBIDO).exists()
