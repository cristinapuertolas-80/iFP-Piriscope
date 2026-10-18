import json
from datetime import date
from pathlib import Path

from src.ingest import config, effis


def test_construir_peticion_mes_completo():
    cfg = config.EffisConfig()
    p = effis.construir_peticion(2020, 2, cfg)
    assert p["year"] == ["2020"] and p["month"] == ["02"]
    assert p["day"][-1] == "29"  # año bisiesto
    assert "fire_weather_index" in p["variable"] and len(p["variable"]) == 6
    assert p["area"] == config.BBOX_GALICIA
    assert p["system_version"] == ["4_1"]


def test_construir_peticion_mes_parcial_sin_area():
    p = effis.construir_peticion(2021, 7, config.EffisConfig(), area=None, dia_desde=10, dia_hasta=12)
    assert p["day"] == ["10", "11", "12"]
    assert "area" not in p


class DescargaFalsa:
    def __init__(self, rechazar_area=False):
        self.peticiones = []
        self.rechazar_area = rechazar_area

    def __call__(self, dataset, peticion, destino):
        self.peticiones.append(dict(peticion))
        if self.rechazar_area and "area" in peticion:
            raise RuntimeError("invalid request: area")
        Path(destino).parent.mkdir(parents=True, exist_ok=True)
        Path(destino).write_bytes(b"CDF\x01" + json.dumps(peticion).encode())


def test_ingestar_un_fichero_por_mes(bronze):
    desc = DescargaFalsa()
    run = effis.ingestar(date(2020, 11, 20), date(2021, 1, 5), base_dir=bronze, descargar=desc)
    assert sorted(p.name for p in run.dir.glob("fwi_*")) == ["fwi_2020_11.nc", "fwi_2020_12.nc", "fwi_2021_01.nc"]
    assert desc.peticiones[0]["day"][0] == "20"
    assert desc.peticiones[-1]["day"] == ["01", "02", "03", "04", "05"]
    assert run.estado == "completada"


def test_si_rechazan_el_recorte_se_reintenta_sin_area(bronze):
    desc = DescargaFalsa(rechazar_area=True)
    run = effis.ingestar(date(2020, 1, 1), date(2020, 1, 31), base_dir=bronze, descargar=desc)
    assert (run.dir / "fwi_2020_01.nc").exists()
    assert "area" not in desc.peticiones[-1]
    assert any(i["tipo"] == "recorte_rechazado" for i in run.incidencias)


def test_meses_completos_se_reutilizan_de_cache(bronze):
    effis.ingestar(date(2020, 1, 1), date(2020, 2, 29), base_dir=bronze, descargar=DescargaFalsa())
    desc = DescargaFalsa()
    run = effis.ingestar(date(2020, 1, 1), date(2020, 2, 29), base_dir=bronze, descargar=desc)
    assert desc.peticiones == []
    assert len(run.reutilizados) == 2


class DescargaConsolidadoHasta:
    """Simula el EWDS: el consolidado sólo existe hasta (anio, mes) incluido."""

    def __init__(self, ultimo_consolidado):
        self.ultimo = ultimo_consolidado
        self.peticiones = []

    def __call__(self, dataset, peticion, destino):
        self.peticiones.append(dict(peticion))
        mes = (int(peticion["year"][0]), int(peticion["month"][0]))
        if peticion["dataset_type"] == "consolidated_dataset" and mes > self.ultimo:
            raise RuntimeError("no data available for the requested period")
        Path(destino).write_bytes(b"CDF\x01")


def test_meses_no_consolidados_usan_dataset_intermedio(bronze):
    desc = DescargaConsolidadoHasta((2026, 6))
    run = effis.ingestar(date(2026, 5, 1), date(2026, 8, 10), base_dir=bronze, descargar=desc)
    ficheros = sorted(p.name for p in run.dir.glob("fwi_*"))
    assert ficheros == ["fwi_2026_05.nc", "fwi_2026_06.nc",
                        "fwi_2026_07_intermedio.nc", "fwi_2026_08_intermedio.nc"]
    assert sum(i["tipo"] == "usa_intermedio" for i in run.incidencias) == 2
    assert not any(i["tipo"] == "error_descarga" for i in run.incidencias)


def test_mes_intermedio_se_sustituye_cuando_ya_esta_consolidado(bronze):
    effis.ingestar(date(2026, 7, 1), date(2026, 7, 31), base_dir=bronze,
                   descargar=DescargaConsolidadoHasta((2026, 6)))
    # Más adelante julio ya está consolidado: la caché no lo da por descargado
    desc = DescargaConsolidadoHasta((2026, 7))
    run = effis.ingestar(date(2026, 7, 1), date(2026, 7, 31), base_dir=bronze, descargar=desc)
    assert (run.dir / "fwi_2026_07.nc").exists()
    assert desc.peticiones[0]["dataset_type"] == "consolidated_dataset"


def test_solo_consolidado_no_recurre_al_intermedio(bronze):
    run = effis.ingestar(date(2026, 7, 1), date(2026, 7, 31), base_dir=bronze,
                         descargar=DescargaConsolidadoHasta((2026, 6)), alternativa_intermedia=False)
    assert not list(run.dir.glob("fwi_*"))
    assert any(i["tipo"] == "error_descarga" for i in run.incidencias)
