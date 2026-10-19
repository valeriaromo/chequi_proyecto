@echo off
setlocal
cd /d "%~dp0"

echo ======================================
echo Instalando dependencias de chequiproyecto
echo ======================================

where py >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON=py -3"
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo Error: Python 3 no esta instalado.
        pause
        exit /b 1
    )
    set "PYTHON=python"
)

if not exist ".venv\Scripts\python.exe" (
    echo Creando entorno virtual...
    %PYTHON% -m venv .venv
    if errorlevel 1 goto error
)

if not exist "requirements.txt" (
    echo Error: No se encontro requirements.txt
    pause
    exit /b 1
)

echo Actualizando pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto error

echo Instalando dependencias...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto error

echo.
echo Instalacion completada correctamente.
echo Activa el entorno con:
echo .venv\Scripts\activate.bat
pause
exit /b 0

:error
echo Error durante la instalacion.
pause
exit /b 1