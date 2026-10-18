<#
.SYNOPSIS
    Piriscope · Carga inicial de datos (fase A de docs/plan_ingesta.md).

.DESCRIPTION
    Orden: INE/CNIG -> EGIF -> FWI (EFFIS) -> AEMET, por años completos y con
    subida a Hugging Face. Se puede relanzar cuantas veces haga falta: lo ya
    descargado (años cerrados) se reutiliza desde la caché de data/bronze.
    El log completo se guarda en logs/carga_inicial_<fecha>.log (no se versiona).

.EXAMPLE
    .\scripts\carga_inicial.ps1
    Todo: 2001..año anterior + año en curso.

.EXAMPLE
    .\scripts\carga_inicial.ps1 -Fuentes effis,aemet

.EXAMPLE
    .\scripts\carga_inicial.ps1 -DesdeAnio 2010 -HastaAnio 2012 -SinSubir

.NOTES
    Antes: activar .venv (.venv\Scripts\Activate.ps1) y rellenar .env.
    Si PowerShell bloquea el script: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
#>
[CmdletBinding()]
param(
    [int]$DesdeAnio = 2001,
    [int]$HastaAnio = ((Get-Date).Year - 1),
    [ValidateSet("ine_cnig", "egif", "effis", "aemet")]
    [string[]]$Fuentes = @("ine_cnig", "egif", "effis", "aemet"),
    [switch]$SinSubir,
    [switch]$SinAnioEnCurso
)

$ErrorActionPreference = "Continue"
$AnioActual = (Get-Date).Year
$Subir = if ($SinSubir) { @() } else { @("--subir-hf") }

# Raíz del repositorio (el script vive en scripts\)
Set-Location (Split-Path -Parent $PSScriptRoot)

$Py = if ($env:PYTHON) { $env:PYTHON } elseif (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { "py" }

New-Item -ItemType Directory -Force -Path logs | Out-Null
$Log = "logs/carga_inicial_{0}.log" -f (Get-Date -Format "yyyyMMddTHHmmss")
Start-Transcript -Path $Log -Append | Out-Null

$Fallos = New-Object System.Collections.Generic.List[string]

function Ejecutar([string]$Nombre, [string[]]$Argumentos) {
    Write-Host ""
    Write-Host ("==> [{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $Nombre)
    Write-Host ("    {0} -m src.ingest {1}" -f $Py, ($Argumentos -join " "))
    & $Py -m src.ingest @Argumentos
    if ($LASTEXITCODE -eq 0) {
        Write-Host "    OK: $Nombre"
    } else {
        Write-Host "    FALLO: $Nombre (código $LASTEXITCODE)" -ForegroundColor Red
        $script:Fallos.Add($Nombre)
    }
}

function HayFicheros([string]$Carpeta) {
    (Test-Path $Carpeta) -and @(Get-ChildItem $Carpeta -File -Recurse | Where-Object Name -ne ".gitkeep").Count -gt 0
}

$EnCurso = (-not $SinAnioEnCurso) -and ($HastaAnio -lt $AnioActual)

Write-Host "Piriscope · carga inicial"
Write-Host ("  Años completos : {0}..{1}{2}" -f $DesdeAnio, $HastaAnio, $(if ($EnCurso) { " + $AnioActual en curso" } else { "" }))
Write-Host ("  Fuentes        : {0}" -f ($Fuentes -join " "))
Write-Host ("  Hugging Face   : {0}" -f $(if ($SinSubir) { "no" } else { "sí" }))
Write-Host ("  Python         : {0}" -f (& $Py --version 2>&1))
Write-Host ("  Log            : {0}" -f $Log)

& $Py -c "import src.ingest" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "No se puede importar src.ingest. ¿Has activado .venv e instalado environment/requirements.txt?" -ForegroundColor Red
    Stop-Transcript | Out-Null
    exit 1
}

foreach ($Fuente in $Fuentes) {
    switch ($Fuente) {
        "ine_cnig" {
            if (-not (HayFicheros "data/landing/ine_cnig")) {
                Write-Host "AVISO: no hay límites del CNIG en data/landing/ine_cnig/; sólo se descargará el diccionario del INE." -ForegroundColor Yellow
            }
            Ejecutar "INE/CNIG" (@("ine_cnig") + $Subir)
        }
        "egif" {
            if (HayFicheros "data/landing/egif") {
                Ejecutar "EGIF" (@("egif") + $Subir)
            } else {
                Write-Host ""
                Write-Host "AVISO: se omite EGIF: no hay exportaciones en data/landing/egif/ (ver docs/plan_ingesta.md, paso A2)." -ForegroundColor Yellow
                $Fallos.Add("EGIF (sin ficheros)")
            }
        }
        "effis" {
            foreach ($y in $DesdeAnio..$HastaAnio) {
                Ejecutar "FWI $y" (@("effis", "--desde", "$y-01-01", "--hasta", "$y-12-31") + $Subir)
            }
            if ($EnCurso) { Ejecutar "FWI $AnioActual (en curso)" (@("effis", "--desde", "$AnioActual-01-01") + $Subir) }
        }
        "aemet" {
            foreach ($y in $DesdeAnio..$HastaAnio) {
                Ejecutar "AEMET $y" (@("aemet", "--desde", "$y-01-01", "--hasta", "$y-12-31") + $Subir)
            }
            if ($EnCurso) { Ejecutar "AEMET $AnioActual (en curso)" (@("aemet", "--desde", "$AnioActual-01-01") + $Subir) }
        }
    }
}

Write-Host ""
Write-Host "================================================================"
if ($Fallos.Count -eq 0) {
    Write-Host "Carga inicial terminada sin fallos. Revisa las incidencias de cada manifest.json." -ForegroundColor Green
    Stop-Transcript | Out-Null
    exit 0
}
Write-Host ("Terminado con {0} paso(s) fallido(s):" -f $Fallos.Count) -ForegroundColor Red
$Fallos | ForEach-Object { Write-Host "  - $_" }
Write-Host "Vuelve a lanzar el script: lo ya descargado se reutiliza."
Stop-Transcript | Out-Null
exit 1
