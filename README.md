# Detección binaria de cáncer por visión artificial

Este repositorio contiene los scripts y documentación para preparar un dataset de imágenes de sangre y realizar limpieza previa al entrenamiento.

> El dataset original no está incluido en el repositorio. Este proyecto usa el dataset `Blood cell Cancer [ALL]` localmente, y se debe respetar la autoría y licencia del creador original del dataset.

## Estructura del proyecto

- `Blood cell Cancer [ALL]/`  — dataset local, no se debe subir al repositorio.
- `cancer.py`          — preprocesamiento y división en `train`, `val` y `test`.
- `limpieza.py`        — limpieza y análisis de calidad del dataset.
- `splits/`            — CSV generados con las divisiones de datos.
- `cleanup_reports/`   — reportes generados sobre imágenes inválidas, duplicados y borrosas.
- `.venv/`             — entorno virtual de Python.

## Entorno virtual

1. Crear/activar el entorno:
   ```powershell
   .\.venv\Scripts\python.exe -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Instalar dependencias:
   ```powershell
   .\.venv\Scripts\python.exe -m pip install Pillow
   ```

## Descripción general

### `cancer.py`

- Identifica las clases del dataset.
- Convierte cada carpeta en una etiqueta binaria:
  - `Benign` -> `benign`
  - `[Malignant] early Pre-B`, `[Malignant] Pre-B`, `[Malignant] Pro-B` -> `malignant`
- Filtra archivos por extensión de imagen.
- Divide el dataset en `train`, `validation` y `test`.
- Guarda los resultados en CSV dentro de `splits/`.

### `limpieza.py`

- Verifica que cada archivo sea una imagen válida.
- Detecta archivos corruptos o con extensión inválida.
- Detecta duplicados exactos mediante hash MD5.
- Detecta duplicados perceptuales mediante un hash visual (`aHash`).
- Detecta imágenes borrosas usando la varianza del laplaciano.
- Genera reportes en `cleanup_reports/`.

## Qué no subir a Git

- El dataset `Blood cell Cancer [ALL]/` no debe subirse.
- `.venv/`
- `splits/`
- `cleanup_reports/`

## Aviso de autoría

El dataset pertenece al autor original del conjunto `Blood cell Cancer`. Este repositorio contiene únicamente el código de preprocesamiento y limpieza. Se debe dar crédito al creador del dataset original si se comparte o publica el proyecto.

## Pasos para un nuevo colaborador

1. Abrir el proyecto en VS Code.
2. Activar el entorno virtual:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
3. Instalar `Pillow` si es necesario:
   ```powershell
   .\.venv\Scripts\python.exe -m pip install Pillow
   ```
4. Ejecutar `python cancer.py`.
5. Ejecutar `python limpieza.py`.
6. Revisar los CSV en `splits/` y los reportes en `cleanup_reports/`.

## Qué falta

- Métricas de evaluación: `accuracy`, `precision`, `recall`, `F1 score`, `AUC`.
- Visualizaciones: matriz de confusión, curvas ROC, precisión/recall.
- Entrenamiento de un modelo de clasificación.
