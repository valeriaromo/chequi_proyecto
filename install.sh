#!/usr/bin/env bash
set -euo pipefail
echo "======================================"
echo " Instalando dependencias del chequiproyecto"
echo "======================================"
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: Python 3 no está instalado."
    exit 1
fi

# crear entorno virtual si no existe
if [ ! -d ".venv" ]; then
    echo "Creando entorno virtual..."
    python3 -m venv .venv
fi

# activar entorno virtual
source .venv/bin/activate

# actualizar pip
echo "Actualizando pip..."
python -m pip install --upgrade pip

# verificar requirements.txt
if [ ! -f "requirements.txt" ]; then
    echo "Error: No se encontró requirements.txt"
    exit 1
fi

# instalar dependencias
echo "Instalando paquetes..."
python -m pip install -r requirements.txt

echo "======================================"
echo " Instalación completada correctamenteeeeee"
echo "======================================"
echo "Para activar el entorno virtual corre:"
echo "source .venv/bin/activate"