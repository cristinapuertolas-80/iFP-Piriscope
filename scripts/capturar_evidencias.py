"""Capturas del dashboard en funcionamiento (evidencias de la entrega NF3).

Abre el dashboard ya arrancado (``streamlit run src/dashboard/app.py``) con un navegador sin
interfaz (Playwright + Chromium) y guarda una captura de cada vista. No modifica datos.

    pip install playwright && python -m playwright install --with-deps chromium
    python scripts/capturar_evidencias.py --url http://localhost:8501 --salida evidencias/

Opciones: ``--fecha`` y ``--municipio`` fijan la vista de la ficha (por defecto, la última fecha
y el municipio con mayor índice de riesgo, que es lo que el dashboard muestra al abrirse).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ESPERA_CARGA_MS = 120_000


def esperar_app(page, texto: str = "Riesgo de incendio forestal en Galicia") -> None:
    """Espera a que Streamlit termine de ejecutar el script y pinte el título."""
    page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=ESPERA_CARGA_MS)
    page.get_by_text(texto).first.wait_for(timeout=ESPERA_CARGA_MS)
    # Streamlit marca el estado de ejecución; esperamos a que deje de estar "running"
    for _ in range(120):
        if page.locator('[data-testid="stStatusWidget"]').count() == 0:
            break
        time.sleep(0.5)
    time.sleep(4)  # margen para que Plotly dibuje mapa y gráficos


def captura(page, ruta: Path) -> None:
    page.screenshot(path=str(ruta), full_page=True)
    print(f"  captura: {ruta}")


def abrir_pestana(page, nombre: str) -> None:
    page.get_by_role("tab", name=nombre).click()
    time.sleep(3)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--salida", type=Path, default=Path("evidencias"))
    ap.add_argument("--fecha", default=None, help="AAAA-MM-DD para la vista de la ficha")
    ap.add_argument("--municipio", default=None, help="código INE para la vista de la ficha")
    ap.add_argument("--ancho", type=int, default=1440)
    ap.add_argument("--alto", type=int, default=2600,
                    help="alto de la ventana: Streamlit desplaza un contenedor interno, así que la captura es la ventana")
    a = ap.parse_args(argv)

    from playwright.sync_api import sync_playwright

    a.salida.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        navegador = pw.chromium.launch()
        page = navegador.new_page(viewport={"width": a.ancho, "height": a.alto}, locale="es-ES")

        print("Vista inicial (última fecha con predicción)")
        page.goto(a.url, wait_until="networkidle")
        esperar_app(page)
        captura(page, a.salida / "01_mapa_y_ficha.png")

        abrir_pestana(page, "📋 Tabla del día")
        captura(page, a.salida / "02_tabla_del_dia.png")

        abrir_pestana(page, "📊 Modelo frente a FWI")
        captura(page, a.salida / "03_modelo_frente_a_fwi.png")

        abrir_pestana(page, "ℹ️ Cómo leer el dashboard")
        captura(page, a.salida / "04_como_leer.png")

        if a.fecha or a.municipio:
            params = "&".join(f"{k}={v}" for k, v in (("fecha", a.fecha), ("municipio", a.municipio)) if v)
            print(f"Vista compartida por URL (?{params})")
            page.goto(f"{a.url}/?{params}", wait_until="networkidle")
            esperar_app(page)
            captura(page, a.salida / "05_municipio_por_url.png")

        navegador.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
