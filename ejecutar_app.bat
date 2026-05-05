@echo off
echo ========================================
echo 🩸 Detector de Cancer de Sangre (ALL)
echo ========================================
echo.

REM Verificar si existe el entorno virtual
if not exist ".venv\Scripts\activate.bat" (
    echo ❌ Entorno virtual no encontrado.
    echo Ejecuta primero: python -m venv .venv
    pause
    exit /b 1
)

REM Activar entorno virtual
echo 🔧 Activando entorno virtual...
call .venv\Scripts\activate.bat

REM Verificar si streamlit está instalado
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo 📦 Instalando Streamlit...
    pip install streamlit==1.28.1
    if errorlevel 1 (
        echo ❌ Error al instalar Streamlit
        pause
        exit /b 1
    )
)

REM Verificar archivos necesarios
if not exist "evaluation_results\best_model.pth" (
    echo ❌ Modelo no encontrado en evaluation_results\best_model.pth
    echo Ejecuta primero el entrenamiento: python entrenamiento_y_metricas.py
    pause
    exit /b 1
)

echo ✅ Todo listo. Iniciando aplicación...
echo.
echo 🌐 La aplicación se abrirá en tu navegador en unos segundos
echo    URL: http://localhost:8501
echo.
echo Para detener: Ctrl+C en esta ventana
echo.

REM Ejecutar Streamlit
streamlit run app.py

REM Desactivar entorno virtual al salir
call .venv\Scripts\deactivate.bat

echo.
echo 👋 Aplicación cerrada.
pause