"""Lanzador del dashboard de Piriscope: prepara todo lo necesario y lo abre con un solo comando.

    python run_dashboard.py

La primera vez:

1. usa el entorno virtual activo o, si no hay ninguno, crea ``.venv`` en la raíz del repositorio;
2. instala solo lo que necesita el dashboard (``environment/requirements-dashboard.txt``);
3. pide el token de lectura de Hugging Face si no está en ``.env`` y lo guarda allí (``.env`` no se versiona);
4. comprueba que el token tiene acceso al dataset y, si no, explica cómo conseguirlo.

Después abre el dashboard en el navegador. Las siguientes veces solo comprueba y abre.

Solo usa la biblioteca estándar, para poder ejecutarse antes de instalar nada.

Ejemplos::

    python run_dashboard.py                    # preparar (si hace falta) y abrir
    python run_dashboard.py --cambiar-token    # volver a pedir el token de Hugging Face
    python run_dashboard.py --local            # usar data/gold en lugar de Hugging Face
    python run_dashboard.py --solo-preparar    # instalar y comprobar, sin abrir el dashboard
"""

from __future__ import annotations

import argparse
import getpass
import os
import re
import socket
import subprocess
import sys
import time
import venv
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
APP = REPO_ROOT / "src" / "dashboard" / "app.py"
REQUISITOS = REPO_ROOT / "environment" / "requirements-dashboard.txt"
ENV = REPO_ROOT / ".env"
ENV_EJEMPLO = REPO_ROOT / ".env.example"
GOLD_DIR = REPO_ROOT / "data" / "gold"
REPO_HF_DEFECTO = "cpuertolas/piriscope"
URL_TOKENS = "https://huggingface.co/settings/tokens"

#: Comprueba en el entorno destino que están las librerías y versiones que usa el dashboard
COMPROBAR_LIBRERIAS = """
import streamlit, pyarrow, huggingface_hub, dotenv, pandas
from plotly.graph_objects import Choroplethmap
assert tuple(int(x) for x in streamlit.__version__.split(".")[:2]) >= (1, 50)
"""

#: Comprueba primero el token (whoami) y después pide la cabecera de predicciones.parquet. Hugging Face
#: responde igual a un token falso y a una cuenta sin acceso al dataset, así que hay que separarlos.
COMPROBAR_ACCESO = """
import os, sys
from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
try:
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
except ImportError:  # huggingface_hub < 0.25
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
try:
    HfApi().whoami(token=os.environ["PIRISCOPE_TOKEN_HF"])
except HfHubHTTPError as exc:
    sys.exit(3 if getattr(exc.response, "status_code", None) in (401, 403) else 4)
except Exception:
    sys.exit(4)
try:
    get_hf_file_metadata(hf_hub_url(os.environ["PIRISCOPE_REPO_HF"], "data/gold/predicciones.parquet",
                                    repo_type="dataset"), token=os.environ["PIRISCOPE_TOKEN_HF"])
except (GatedRepoError, RepositoryNotFoundError):   # el token es válido: falta acceso al dataset
    sys.exit(2)
except HfHubHTTPError as exc:
    sys.exit(2 if getattr(exc.response, "status_code", None) in (401, 403) else 4)
except Exception:
    sys.exit(4)
"""


class ErrorPreparacion(Exception):
    """Un paso de la preparación no se puede completar; el mensaje dice qué hacer."""


def paso(texto: str) -> None:
    print(f"\n> {texto}", flush=True)


# ---------------------------------------------------------------------------
# Entorno de Python
# ---------------------------------------------------------------------------
def en_entorno_virtual() -> bool:
    return sys.prefix != sys.base_prefix


def python_de(entorno: Path) -> Path:
    return entorno / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def preparar_entorno(raiz: Path = REPO_ROOT) -> Path:
    """Python con el que se ejecuta el dashboard: el del entorno activo o el de ``.venv``."""
    if en_entorno_virtual():
        print(f"  Se usa el entorno virtual activo ({sys.prefix}).")
        return Path(sys.executable)
    entorno = raiz / ".venv"
    python = python_de(entorno)
    if not python.exists():
        print(f"  Creando el entorno virtual en {entorno} (solo la primera vez)…", flush=True)
        venv.create(entorno, with_pip=True)
    else:
        print(f"  Se usa el entorno de {entorno}.")
    return python


def faltan_librerias(python: Path) -> bool:
    return subprocess.run([str(python), "-c", COMPROBAR_LIBRERIAS], capture_output=True).returncode != 0


def instalar_librerias(python: Path) -> None:
    if not faltan_librerias(python):
        print("  Las librerías ya están instaladas.")
        return
    print("  Instalando las librerías del dashboard (1-2 minutos la primera vez)…", flush=True)
    r = subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check", "-q",
                        "-r", str(REQUISITOS)])
    if r.returncode != 0 or faltan_librerias(python):
        raise ErrorPreparacion("No se pudieron instalar las librerías. Revisa la conexión a internet y el "
                               "mensaje de pip de arriba, y vuelve a ejecutar run_dashboard.py.")


# ---------------------------------------------------------------------------
# Token de Hugging Face (.env)
# ---------------------------------------------------------------------------
def leer_env(ruta: Path | None = None) -> dict[str, str]:
    """``CLAVE=valor`` de un fichero .env (sin depender de python-dotenv, que aún puede no estar)."""
    ruta = ruta or ENV
    valores: dict[str, str] = {}
    if not ruta.exists():
        return valores
    for linea in ruta.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", linea)
        if m and not linea.lstrip().startswith("#"):
            valores[m.group(1)] = m.group(2).strip().strip("'\"")
    return valores


def guardar_en_env(clave: str, valor: str, ruta: Path | None = None, plantilla: Path | None = None) -> None:
    """Escribe ``clave=valor`` en .env (lo crea desde .env.example si no existe) sin tocar el resto."""
    ruta, plantilla = ruta or ENV, plantilla or ENV_EJEMPLO
    if ruta.exists():
        texto = ruta.read_text(encoding="utf-8", errors="surrogateescape")
    elif plantilla.exists():
        texto = plantilla.read_text(encoding="utf-8", errors="surrogateescape")
    else:
        texto = ""
    patron = re.compile(rf"^[ \t]*{re.escape(clave)}[ \t]*=.*$", re.MULTILINE)
    if patron.search(texto):
        texto = patron.sub(lambda _m: f"{clave}={valor}", texto, count=1)
    else:
        texto = texto + ("" if texto.endswith("\n") or not texto else "\n") + f"{clave}={valor}\n"
    ruta.write_text(texto, encoding="utf-8", errors="surrogateescape")


def consola_real() -> bool:
    """Si se puede leer de teclado sin que se vea lo escrito.

    En Windows, ``isatty()`` también es cierto para NUL, y ``getpass`` se quedaría esperando al teclado;
    en Git Bash (mintty) la entrada es una tubería y ``getpass`` no puede ocultar el texto.
    """
    if sys.stdin is None or not sys.stdin.isatty():
        return False
    if os.name != "nt":
        return True
    import ctypes
    import msvcrt

    modo = ctypes.c_uint32()
    return bool(ctypes.windll.kernel32.GetConsoleMode(msvcrt.get_osfhandle(sys.stdin.fileno()), ctypes.byref(modo)))


def leer_teclado(texto: str) -> str:
    try:
        return getpass.getpass(texto) if consola_real() else input(texto.replace("no se verá al pegarlo; ", ""))
    except (EOFError, OSError):
        raise ErrorPreparacion(f"Falta HF_TOKEN en {ENV} y no se puede pedir aquí. Ejecuta run_dashboard.py en una "
                               "terminal, o añade la línea HF_TOKEN=... a ese fichero.") from None


def pedir_token() -> str:
    print(f"""
  El dashboard descarga las predicciones del dataset de Hugging Face, y para eso necesita
  un token de lectura de tu cuenta (solo se pide una vez y se guarda en .env, que no se sube a GitHub):

    1. Entra en {URL_TOKENS} (crea una cuenta gratuita si no tienes).
    2. Pulsa «Create new token», elige el tipo «Read», ponle un nombre y créalo.
    3. Copia el token (empieza por hf_) y pégalo aquí.
""")
    token = leer_teclado("  Token (no se verá al pegarlo; pulsa Enter al terminar): ").strip()
    if not token:
        raise ErrorPreparacion("No se ha introducido ningún token.")
    return token


def obtener_token(cambiar: bool = False) -> str:
    token = "" if cambiar else (os.getenv("HF_TOKEN") or leer_env().get("HF_TOKEN", "")).strip()
    if token:
        print("  Token de Hugging Face encontrado.")
        return token
    token = pedir_token()
    guardar_en_env("HF_TOKEN", token)
    print(f"  Token guardado en {ENV}.")
    return token


def comprobar_acceso(python: Path, token: str, repo: str) -> int:
    """0 = acceso; 2 = la cuenta no ha aceptado el dataset; 3 = token no válido; 4 = sin conexión."""
    entorno = {**os.environ, "PIRISCOPE_TOKEN_HF": token, "PIRISCOPE_REPO_HF": repo}
    return subprocess.run([str(python), "-c", COMPROBAR_ACCESO], env=entorno, capture_output=True).returncode


def asegurar_acceso(python: Path, cambiar_token: bool) -> None:
    repo = os.getenv("HF_REPO_ID") or leer_env().get("HF_REPO_ID") or REPO_HF_DEFECTO
    token = obtener_token(cambiar_token)
    for intento in range(2):
        resultado = comprobar_acceso(python, token, repo)
        if resultado == 0:
            print("  El token tiene acceso a los datos.")
            return
        if resultado == 4:
            print("  AVISO: no se pudo conectar con Hugging Face. Se abre igualmente; si hay ficheros en "
                  "data/gold se usarán esos.")
            return
        if resultado == 2:
            raise ErrorPreparacion(
                f"Tu cuenta de Hugging Face todavía no tiene acceso al dataset {repo}.\n"
                f"  1. Abre https://huggingface.co/datasets/{repo} con tu cuenta iniciada.\n"
                f"  2. Pulsa el botón para aceptar las condiciones de acceso (se concede al momento).\n"
                f"  3. Vuelve a ejecutar: python run_dashboard.py")
        # 3: token no válido (caducado, mal copiado…): se pide otra vez una sola vez
        if intento == 0:
            print("  El token no es válido (¿caducado o mal copiado?). Vamos a pedirlo de nuevo.")
            try:
                token = obtener_token(cambiar=True)
            except ErrorPreparacion:
                break
    raise ErrorPreparacion(f"El token de Hugging Face no es válido. Crea uno de tipo «Read» en {URL_TOKENS} "
                           "y ejecuta: python run_dashboard.py --cambiar-token")


# ---------------------------------------------------------------------------
# Lanzamiento
# ---------------------------------------------------------------------------
def puerto_libre(desde: int = 8501, hasta: int = 8530) -> int:
    for puerto in range(desde, hasta):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", puerto)) != 0:
                return puerto
    raise ErrorPreparacion(f"No hay ningún puerto libre entre {desde} y {hasta - 1}.")


def esperar_servidor(puerto: int, proceso: subprocess.Popen, espera: float = 90) -> bool:
    limite = time.monotonic() + espera
    while time.monotonic() < limite and proceso.poll() is None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", puerto)) == 0:
                return True
        time.sleep(0.5)
    return False


def lanzar(python: Path, puerto: int, abrir: bool, local: bool) -> int:
    entorno = {**os.environ, **({"PIRISCOPE_DASHBOARD_ORIGEN": "local"} if local else {})}
    # headless: evita la pregunta de Streamlit por el correo la primera vez; el navegador se abre desde aquí
    orden = [str(python), "-m", "streamlit", "run", str(APP), "--server.port", str(puerto),
             "--server.headless", "true", "--browser.gatherUsageStats", "false"]
    proceso = subprocess.Popen(orden, cwd=REPO_ROOT, env=entorno)
    url = f"http://localhost:{puerto}"
    try:
        if esperar_servidor(puerto, proceso):
            print(f"\n  Dashboard abierto en {url}\n  Para cerrarlo, pulsa Ctrl+C en esta ventana.\n", flush=True)
            if abrir:
                webbrowser.open(url)
        return proceso.wait()
    except KeyboardInterrupt:
        proceso.terminate()
        proceso.wait()
        print("\n  Dashboard cerrado.")
        return 0


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--puerto", type=int, help="puerto (por defecto, el primero libre desde 8501)")
    ap.add_argument("--sin-navegador", action="store_true", help="no abrir el navegador automáticamente")
    ap.add_argument("--local", action="store_true", help="usar data/gold en lugar de Hugging Face (sin token)")
    ap.add_argument("--cambiar-token", action="store_true", help="volver a pedir el token de Hugging Face")
    ap.add_argument("--solo-preparar", action="store_true", help="instalar y comprobar, sin abrir el dashboard")
    args = ap.parse_args(argv)

    if sys.version_info < (3, 10):
        print(f"Hace falta Python 3.10 o superior (tienes {sys.version.split()[0]}).")
        return 1
    try:
        paso("1/3 Entorno de Python")
        python = preparar_entorno()
        paso("2/3 Librerías")
        instalar_librerias(python)
        paso("3/3 Acceso a los datos")
        if args.local:
            if not all((GOLD_DIR / f).exists() for f in ("predicciones.parquet", "explicaciones.parquet")):
                raise ErrorPreparacion(f"--local necesita predicciones.parquet y explicaciones.parquet en {GOLD_DIR}.")
            print(f"  Se usan los ficheros de {GOLD_DIR}.")
        else:
            asegurar_acceso(python, args.cambiar_token)
    except ErrorPreparacion as exc:
        print(f"\nERROR: {exc}")
        return 1
    if args.solo_preparar:
        print("\nTodo listo. Para abrir el dashboard: python run_dashboard.py")
        return 0
    return lanzar(python, args.puerto or puerto_libre(), not args.sin_navegador, args.local)


if __name__ == "__main__":
    sys.exit(main())
