from datetime import date

import pytest

from src.ingest.cli import construir_parser, resolver_rango


def test_resolver_rango_por_dias():
    assert resolver_rango(None, None, 10, hoy=date(2026, 10, 17)) == (date(2026, 10, 7), date(2026, 10, 16))


def test_resolver_rango_explicito():
    assert resolver_rango(date(2020, 1, 1), date(2020, 2, 1), None) == (date(2020, 1, 1), date(2020, 2, 1))


def test_resolver_rango_sin_datos():
    with pytest.raises(SystemExit):
        resolver_rango(None, None, None)


def test_parser_aemet():
    args = construir_parser().parse_args(["aemet", "--desde", "2020-01-01", "--hasta", "2020-12-31",
                                          "--estaciones", "1387", "--subir-hf"])
    assert args.fuente == "aemet" and args.estaciones == ["1387"] and args.subir_hf


def test_parser_subir_solo_una_fuente():
    args = construir_parser().parse_args(["subir", "--solo", "egif"])
    assert args.fuente == "subir" and args.fuente_filtro == "egif"


def test_resolver_rango_con_retraso_de_publicacion():
    # FWI: ~5 días de retraso -> el último día disponible es hoy - 6
    hoy = date(2026, 10, 18)
    assert resolver_rango(None, None, 14, hoy=hoy, retraso=5) == (date(2026, 9, 29), date(2026, 10, 12))
    assert resolver_rango(date(2026, 1, 1), None, None, hoy=hoy, retraso=5) == (date(2026, 1, 1), date(2026, 10, 12))


def test_resolver_rango_vacio_falla():
    with pytest.raises(SystemExit):
        resolver_rango(date(2026, 10, 15), None, None, hoy=date(2026, 10, 18), retraso=5)


def test_parser_effis_retraso_por_defecto():
    args = construir_parser().parse_args(["effis", "--dias", "14"])
    assert args.retraso == 5 and not args.solo_consolidado
    assert construir_parser().parse_args(["aemet", "--dias", "3"]).retraso == 0


def test_error_de_conexion_devuelve_codigo_3(monkeypatch):
    from src.ingest import cli, effis
    from src.ingest.common import ErrorRed

    def falla(*a, **k):
        raise ErrorRed("EFFIS/EWDS: certificado")

    monkeypatch.setattr(effis, "ingestar", falla)
    assert cli.main(["effis", "--desde", "2001-01-01", "--hasta", "2001-01-31"]) == cli.CODIGO_ERROR_RED == 3
