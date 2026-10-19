#!/usr/bin/env bash
# =============================================================================
# Piriscope · Carga inicial de datos (fase A de docs/plan_ingesta.md)
#
# Orden: INE/CNIG -> EGIF -> FWI (EFFIS) -> AEMET, por años completos y con
# subida a Hugging Face. Se puede relanzar cuantas veces haga falta: lo ya
# descargado (años cerrados) se reutiliza desde la caché de data/bronze.
#
# Uso (desde cualquier carpeta, con el entorno .venv activado y el .env relleno):
#   scripts/carga_inicial.sh                              # todo, 2001..año anterior + año en curso
#   scripts/carga_inicial.sh --fuentes "effis aemet"      # sólo algunas fuentes
#   scripts/carga_inicial.sh --desde-anio 2010 --hasta-anio 2012 --sin-subir
#   scripts/carga_inicial.sh --sin-anio-en-curso          # sólo años completos
#
# El log completo se guarda en logs/carga_inicial_<fecha>.log (no se versiona).
# =============================================================================
set -uo pipefail

ANIO_ACTUAL=$(date +%Y)
DESDE_ANIO=2001
HASTA_ANIO=$((ANIO_ACTUAL - 1))
FUENTES="ine_cnig egif effis aemet"
SUBIR="--subir-hf"
ANIO_EN_CURSO=1

ayuda() { sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

while [ $# -gt 0 ]; do
  case "$1" in
    --desde-anio) DESDE_ANIO="$2"; shift 2 ;;
    --hasta-anio) HASTA_ANIO="$2"; shift 2 ;;
    --fuentes) FUENTES="$2"; shift 2 ;;
    --sin-subir) SUBIR=""; shift ;;
    --sin-anio-en-curso) ANIO_EN_CURSO=0; shift ;;
    -h|--help) ayuda ;;
    *) echo "Opción desconocida: $1 (usa --help)"; exit 2 ;;
  esac
done

# Raíz del repositorio (el script vive en scripts/)
cd "$(dirname "$0")/.." || exit 1

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if command -v python >/dev/null 2>&1; then PY=python; else PY=python3; fi
fi

mkdir -p logs
LOG="logs/carga_inicial_$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

FALLOS=()
resumen_y_salir() {
  echo ""
  echo "================================================================"
  if [ ${#FALLOS[@]} -eq 0 ]; then
    echo "Carga inicial terminada sin fallos. Revisa las incidencias de cada manifest.json."
    exit 0
  fi
  echo "Terminado con ${#FALLOS[@]} paso(s) fallido(s):"
  printf '  - %s\n' "${FALLOS[@]}"
  echo "Vuelve a lanzar el script: lo ya descargado se reutiliza."
  exit 1
}

ejecutar() {
  local nombre="$1"; shift
  echo ""
  echo "==> [$(date +%H:%M:%S)] $nombre"
  echo "    $PY -m src.ingest $*"
  "$PY" -m src.ingest "$@"
  local codigo=$?
  if [ $codigo -eq 0 ]; then
    echo "    OK: $nombre"
  elif [ $codigo -eq 3 ]; then
    echo "    FALLO DE CONEXIÓN: $nombre. Se detiene la carga: arregla la conexión"
    echo "    (ver el mensaje anterior y docs/plan_ingesta.md, apartado 6) y vuelve a lanzar el script."
    FALLOS+=("$nombre (conexión)")
    resumen_y_salir
  else
    echo "    FALLO: $nombre (código $codigo)"
    FALLOS+=("$nombre")
  fi
}

hay_ficheros() { [ -d "$1" ] && [ -n "$(find "$1" -type f ! -name '.gitkeep' 2>/dev/null | head -1)" ]; }

echo "Piriscope · carga inicial"
echo "  Años completos : $DESDE_ANIO..$HASTA_ANIO $( [ $ANIO_EN_CURSO = 1 ] && echo "+ $ANIO_ACTUAL en curso")"
echo "  Fuentes        : $FUENTES"
echo "  Hugging Face   : $( [ -n "$SUBIR" ] && echo sí || echo no)"
echo "  Python         : $($PY --version 2>&1)"
echo "  Log            : $LOG"

"$PY" -c "import src.ingest" 2>/dev/null || {
  echo "No se puede importar src.ingest. ¿Has activado .venv e instalado environment/requirements.txt?"; exit 1; }

for fuente in $FUENTES; do
  case "$fuente" in
    ine_cnig)
      hay_ficheros data/landing/ine_cnig || \
        echo "AVISO: no hay límites del CNIG en data/landing/ine_cnig/; sólo se descargará el diccionario del INE."
      ejecutar "INE/CNIG" ine_cnig $SUBIR
      ;;
    egif)
      if hay_ficheros data/landing/egif; then
        ejecutar "EGIF" egif $SUBIR
      else
        echo ""; echo "AVISO: se omite EGIF: no hay exportaciones en data/landing/egif/ (ver docs/plan_ingesta.md, paso A2)."
        FALLOS+=("EGIF (sin ficheros)")
      fi
      ;;
    effis)
      for y in $(seq "$DESDE_ANIO" "$HASTA_ANIO"); do
        ejecutar "FWI $y" effis --desde "$y-01-01" --hasta "$y-12-31" $SUBIR
      done
      if [ $ANIO_EN_CURSO = 1 ] && [ "$HASTA_ANIO" -lt "$ANIO_ACTUAL" ]; then
        ejecutar "FWI $ANIO_ACTUAL (en curso)" effis --desde "$ANIO_ACTUAL-01-01" $SUBIR
      fi
      ;;
    aemet)
      for y in $(seq "$DESDE_ANIO" "$HASTA_ANIO"); do
        ejecutar "AEMET $y" aemet --desde "$y-01-01" --hasta "$y-12-31" $SUBIR
      done
      if [ $ANIO_EN_CURSO = 1 ] && [ "$HASTA_ANIO" -lt "$ANIO_ACTUAL" ]; then
        ejecutar "AEMET $ANIO_ACTUAL (en curso)" aemet --desde "$ANIO_ACTUAL-01-01" $SUBIR
      fi
      ;;
    *) echo "Fuente desconocida: $fuente"; FALLOS+=("$fuente (desconocida)") ;;
  esac
done

resumen_y_salir
