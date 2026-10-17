import json
from datetime import date, datetime, timezone

import pytest

from src.ingest.common import EjecucionBronze, buscar_en_cache, meses_en_rango, tramos_fechas


def test_tramos_cubren_el_rango_sin_huecos_ni_solapes():
    tramos = list(tramos_fechas(date(2020, 1, 1), date(2020, 12, 31), 180))
    assert tramos[0][0] == date(2020, 1, 1)
    assert tramos[-1][1] == date(2020, 12, 31)
    for (_, fin), (ini, _) in zip(tramos, tramos[1:]):
        assert (ini - fin).days == 1
    assert all((fin - ini).days + 1 <= 180 for ini, fin in tramos)


def test_tramo_de_un_dia():
    assert list(tramos_fechas(date(2021, 5, 1), date(2021, 5, 1), 180)) == [(date(2021, 5, 1), date(2021, 5, 1))]


def test_rango_invertido_falla():
    with pytest.raises(ValueError):
        list(tramos_fechas(date(2021, 5, 2), date(2021, 5, 1), 10))


def test_meses_en_rango_cruza_anio():
    assert meses_en_rango(date(2020, 11, 15), date(2021, 2, 1)) == [(2020, 11), (2020, 12), (2021, 1), (2021, 2)]


def test_ejecucion_escribe_manifiesto_con_hash(bronze):
    momento = datetime(2026, 10, 17, 6, 0, tzinfo=timezone.utc)
    with EjecucionBronze("aemet", parametros={"x": 1}, base_dir=bronze, momento=momento) as run:
        run.guardar_bytes("diarios/a.json", b"[]", origen="https://ejemplo")
    assert run.dir == bronze / "aemet" / "20261017T060000Z"
    m = json.loads((run.dir / "manifest.json").read_text(encoding="utf-8"))
    assert m["estado"] == "completada"
    assert m["fuente_hf"] == "AEMET"
    assert m["ficheros"][0]["ruta"] == "diarios/a.json"
    assert len(m["ficheros"][0]["sha256"]) == 64
    assert "AEMET" in m["atribucion"]


def test_ejecucion_fallida_queda_registrada(bronze):
    with pytest.raises(RuntimeError):
        with EjecucionBronze("egif", base_dir=bronze) as run:
            raise RuntimeError("boom")
    m = json.loads((run.dir / "manifest.json").read_text(encoding="utf-8"))
    assert m["estado"] == "fallida"
    assert m["incidencias"][0]["tipo"] == "excepcion"


def test_fuente_desconocida(bronze):
    with pytest.raises(ValueError):
        EjecucionBronze("nasa", base_dir=bronze)


def test_cache_encuentra_ejecuciones_previas(bronze):
    previo = EjecucionBronze("aemet", base_dir=bronze, momento=datetime(2026, 1, 1, tzinfo=timezone.utc))
    previo.guardar_bytes("diarios/x.json", b"[1]")
    actual = EjecucionBronze("aemet", base_dir=bronze, momento=datetime(2026, 2, 1, tzinfo=timezone.utc))
    encontrado = buscar_en_cache("aemet", "diarios/x.json", excluir=actual.dir, base_dir=bronze)
    assert encontrado == previo.dir / "diarios/x.json"
    assert buscar_en_cache("aemet", "diarios/y.json", base_dir=bronze) is None
