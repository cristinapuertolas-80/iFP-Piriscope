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
