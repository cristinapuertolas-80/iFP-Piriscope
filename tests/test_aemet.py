import json
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from src.ingest import aemet, config
from src.ingest.common import EjecucionBronze

INVENTARIO = [
    {"indicativo": "1387", "nombre": "A CORUÑA", "provincia": "A CORUÑA", "latitud": "432157N",
     "longitud": "082517W", "altitud": "58"},
    {"indicativo": "1505", "nombre": "LUGO AEROPUERTO", "provincia": "LUGO", "latitud": "430641N",
     "longitud": "073323W", "altitud": "444"},
    {"indicativo": "3195", "nombre": "MADRID, RETIRO", "provincia": "MADRID", "latitud": "402443N",
     "longitud": "034041W", "altitud": "667"},
]


class Respuesta:
    def __init__(self, status=200, cuerpo=None, contenido=None):
        self.status_code = status
        self._cuerpo = cuerpo
        self.content = contenido if contenido is not None else json.dumps(cuerpo or {}).encode()

    def json(self):
        if self._cuerpo is None:
            raise ValueError
        return self._cuerpo

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class SesionFalsa:
    """Simula AEMET OpenData: primer paso -> URL de datos; segundo paso -> JSON."""

    def __init__(self, sin_datos=(), errores_429=0):
        self.llamadas = []
        self.sin_datos = set(sin_datos)
        self.errores_429 = errores_429

    def get(self, url, timeout=None, headers=None, **_):
        self.llamadas.append(url)
        if url.startswith("https://datos/"):
            clave = url.removeprefix("https://datos/")
            if clave == "inventario":
                return Respuesta(contenido=json.dumps(INVENTARIO, ensure_ascii=False).encode("iso-8859-15"))
            idema, ini, fin = clave.split("|")
            dias = pd.date_range(ini, fin, freq="D")
            datos = [{"fecha": d.strftime("%Y-%m-%d"), "indicativo": idema, "tmed": "12,3", "prec": "Ip"}
                     for d in dias]
            return Respuesta(contenido=json.dumps(datos).encode("iso-8859-15"))
        if self.errores_429:
            self.errores_429 -= 1
            return Respuesta(429, {"estado": 429})
        if "inventarioestaciones" in url:
            return Respuesta(200, {"estado": 200, "datos": "https://datos/inventario"})
        partes = url.split("/")
        idema = partes[-1]
        ini = partes[partes.index("fechaini") + 1][:10]
        fin = partes[partes.index("fechafin") + 1][:10]
        if idema in self.sin_datos:
            return Respuesta(200, {"estado": 404, "descripcion": "No hay datos que satisfagan esos criterios"})
        return Respuesta(200, {"estado": 200, "datos": f"https://datos/{idema}|{ini}|{fin}"})


def cliente(sesion, max_dias=180):
    cfg = config.AemetConfig(api_key="clave-test", pausa_segundos=0, espera_429_segundos=0,
                             max_dias_por_peticion=max_dias)
    return aemet.ClienteAemet(cfg=cfg, sesion=sesion)


def test_dms_a_decimal():
    assert aemet.dms_a_decimal("432157N") == pytest.approx(43.365833, abs=1e-6)
    assert aemet.dms_a_decimal("082517W") == pytest.approx(-8.421389, abs=1e-6)
    assert aemet.dms_a_decimal("") is None
    assert aemet.dms_a_decimal("abc") is None


def test_provincias_gallegas():
    assert aemet.es_provincia_gallega("A CORUÑA")
    assert aemet.es_provincia_gallega("Ourense")
    assert not aemet.es_provincia_gallega("MADRID")
    assert not aemet.es_provincia_gallega(None)


def test_filtrar_estaciones_galicia():
    df = aemet.filtrar_estaciones_galicia(INVENTARIO)
    assert list(df["indicativo"]) == ["1387", "1505"]
    assert df["latitud_dec"].notna().all()


def test_sin_api_key_falla():
    with pytest.raises(aemet.AemetError):
        aemet.ClienteAemet(cfg=config.AemetConfig(api_key=None))


def test_ingesta_completa_genera_tramos_y_cobertura(bronze):
    sesion = SesionFalsa(sin_datos={"1505"})
    run = aemet.ingestar(date(2020, 1, 1), date(2020, 12, 31), base_dir=bronze, cliente=cliente(sesion))
    diarios = sorted(p.name for p in (run.dir / "diarios").iterdir())
    assert diarios == ["1387_2020-01-01_2020-06-28.json", "1387_2020-06-29_2020-12-25.json",
                       "1387_2020-12-26_2020-12-31.json"]
    cobertura = pd.read_csv(run.dir / "cobertura_estaciones.csv", dtype={"indicativo": str})
    fila = cobertura.set_index("indicativo")
    assert fila.loc["1387", "cobertura_pct"] == 100.0
    assert fila.loc["1387", "dias_con_datos"] == 366
    assert fila.loc["1505", "dias_con_datos"] == 0
    assert "3195" not in fila.index  # Madrid queda fuera
    m = json.loads((run.dir / "manifest.json").read_text(encoding="utf-8"))
    assert m["estado"] == "completada"
    # El fichero se guarda tal como llega (sin modificar)
    bruto = (run.dir / "diarios" / diarios[0]).read_bytes()
    assert json.loads(bruto.decode("iso-8859-15"))[0]["prec"] == "Ip"


def test_reintenta_tras_429(bronze):
    sesion = SesionFalsa(errores_429=2)
    run = aemet.ingestar(date(2020, 1, 1), date(2020, 1, 10), estaciones=["1387"], base_dir=bronze,
                         cliente=cliente(sesion))
    assert (run.dir / "diarios" / "1387_2020-01-01_2020-01-10.json").exists()


def test_estacion_desconocida_es_incidencia(bronze):
    run = aemet.ingestar(date(2020, 1, 1), date(2020, 1, 5), estaciones=["1387", "9999"], base_dir=bronze,
                         cliente=cliente(SesionFalsa()))
    assert any(i["tipo"] == "estacion_desconocida" for i in run.incidencias)
    assert run.estado == "completada_con_incidencias"


def test_reutiliza_cache_de_tramos_consolidados(bronze):
    previo = EjecucionBronze("aemet", base_dir=bronze, momento=datetime(2025, 1, 1, tzinfo=timezone.utc))
    previo.guardar_bytes(aemet.nombre_tramo("1387", date(2020, 1, 1), date(2020, 1, 10)),
                         json.dumps([{"fecha": "2020-01-01"}]).encode())
    sesion = SesionFalsa()
    run = aemet.ingestar(date(2020, 1, 1), date(2020, 1, 10), estaciones=["1387"], base_dir=bronze,
                         cliente=cliente(sesion))
    assert not any("/estacion/1387" in u for u in sesion.llamadas)
    assert run.reutilizados[0]["ejecucion_origen"] == "20250101T000000Z"


def test_tramo_reciente_no_se_considera_consolidado():
    hoy = date(2026, 10, 17)
    assert not aemet.tramo_consolidado(date(2026, 10, 10), 30, hoy)
    assert aemet.tramo_consolidado(date(2026, 8, 1), 30, hoy)
