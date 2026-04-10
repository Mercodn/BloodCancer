Proyecto de detección binaria de cáncer por visión artificial
=============================================================

Este proyecto usa un dataset de imágenes organizado en carpetas para detectar si una muestra es "benign" o "malignant".

Estructura del proyecto
-----------------------

- `Blood cell Cancer [ALL]/`
  - `Benign/`
  - `[Malignant] early Pre-B/`
  - `[Malignant] Pre-B/`
  - `[Malignant] Pro-B/`
- `cancer.py`          : script principal para preprocesamiento y división del dataset.
- `limpieza.py`        : script de limpieza y análisis de calidad del dataset.
- `splits/`            : carpeta generada con los CSV de división `train`, `val`, `test`.
- `cleanup_reports/`   : carpeta generada con los reportes de limpieza.
- `.venv/`             : entorno virtual de Python del proyecto.

Entorno virtual
----------------

El proyecto utiliza un entorno virtual Python en `.venv/`.
Esto asegura que las librerías necesarias se instalen localmente y no interfieran con otros proyectos.

Si alguien necesita recrear el entorno desde cero:

1. Abrir PowerShell en la carpeta del proyecto.
2. Crear el entorno virtual (si aún no existe):
   .\.venv\Scripts\python.exe -m venv .venv
3. Activar el entorno virtual:
   .\.venv\Scripts\Activate.ps1
4. Instalar dependencias necesarias:
   .\.venv\Scripts\python.exe -m pip install Pillow

Actualmente, el único paquete requerido por los scripts de limpieza y preprocesamiento es `Pillow`.

Cómo funciona el proyecto
------------------------

1. Preprocesamiento inicial (archivo `cancer.py`):

   - Detecta las clases en las carpetas del dataset.
   - Convierte los nombres de carpeta en etiquetas binarias:
     - `Benign` -> `benign`
     - `[Malignant] early Pre-B`, `[Malignant] Pre-B`, `[Malignant] Pro-B` -> `malignant`
   - Detecta solo archivos con extensiones válidas de imagen.
   - Divide el dataset en tres partes:
     - `train` (70%)
     - `validation` (15%)
     - `test` (15%)
   - Guarda tres archivos CSV en `splits/`:
     - `train.csv`
     - `val.csv`
     - `test.csv`

   Esta división es útil para entrenar modelos posteriormente y evaluar su desempeño.

2. Limpieza de datos (archivo `limpieza.py`):

   - Verifica que cada archivo sea una imagen válida y no esté corrupto.
   - Filtra extensiones inválidas para evitar datos no esperados.
   - Confirma que cada carpeta pertenezca a una etiqueta binaria conocida.
   - Detecta duplicados exactos usando un hash MD5 de cada archivo.
   - Detecta duplicados perceptuales usando un hash de imagen (`aHash`).
   - Detecta imágenes borrosas usando la varianza del laplaciano.
   - Genera reportes CSV en `cleanup_reports/` para revisar:
     - imágenes válidas
     - archivos inválidos
     - duplicados exactos
     - duplicados perceptuales
     - imágenes borrosas

   Este paso ayuda a garantizar que el modelo se entrene con datos limpios y de calidad.

Detalles del código
-------------------

`cancer.py`
- `DATA_DIR`: carpeta raíz del dataset.
- `OUTPUT_DIR`: carpeta donde se guardan los CSV de división.
- `VALID_EXTENSIONS`: extensiones de imagen permitidas.
- `collect_image_paths()`: recorre cada subcarpeta y devuelve rutas de imagen con etiquetas.
- `random_split()`: mezcla y divide el dataset en train/val/test.
- `save_split_csv()`: guarda cada división en un archivo CSV.
- `preprocess_image()`: función preparada para abrir y normalizar una imagen a 224x224.
  - Nota: hoy se usa solo para describir el proceso, pero se puede ampliar para generar tensores.

`limpieza.py`
- `DATA_DIR`: carpeta raíz del dataset.
- `OUTPUT_DIR`: carpeta donde se guardan reportes de limpieza.
- `CLASS_LABELS`: mapeo de carpeta a etiqueta binaria.
- `validate_image()`: abre cada imagen con Pillow y detecta corrupción.
- `compute_file_hash()`: genera un hash MD5 para detectar duplicados exactos.
- `average_hash()`: genera un hash perceptual para duplicados visuales.
- `variance_of_laplacian()`: calcula una métrica de nitidez para detectar imágenes borrosas.
- `detect_duplicates()`: encuentra grupos de imágenes idénticas o muy similares.
- `detect_blurry()`: encuentra imágenes con baja nitidez.
- `save_reports()`: guarda los resultados en varios CSV.

Qué hacer antes de subir a Git
------------------------------

- Asegurarse de que `.venv/` no se suba a Git.
- Incluir un archivo `.gitignore` con al menos:
  - `.venv/`
  - `__pycache__/`
  - `splits/`
  - `cleanup_reports/`
- Subir los scripts `cancer.py`, `limpieza.py` y `README.txt`.
- No es necesario subir los datos del dataset si son muy grandes; solo documentar dónde se almacenan.
- El dataset no es de propiedad del autor de este repositorio. En este proyecto solo se suben los scripts y la documentación, no el dataset completo.

Qué falta para completar el proyecto
-----------------------------------

En futuras etapas se debe agregar:

- métricas de evaluación como `accuracy`, `precision`, `recall`, `F1 score`, `AUC`.
- gráficos de análisis como matriz de confusión, curvas ROC y precisión/recall.
- modelado con una red neuronal convolucional o modelo de clasificación.
- entrenamiento real usando los CSV generados por `cancer.py`.
- validación de calidad después de limpieza.

Resumen para un nuevo compañero
--------------------------------

1. Abrir el proyecto en VS Code.
2. Activar el entorno virtual con `.\.venv\Scripts\Activate.ps1`.
3. Instalar dependencias si es necesario: `.\.venv\Scripts\python.exe -m pip install Pillow`.
4. Ejecutar `python cancer.py` para generar la división del dataset.
5. Ejecutar `python limpieza.py` para revisar y reportar problemas de datos.
6. Revisar los CSV generados en `splits/` y `cleanup_reports/`.

Con esto se deja el proyecto en un estado base en el que cualquier persona nueva puede comprender lo que hay, cómo se prepara el dataset y qué pasos deben hacerse luego para entrenar un modelo.
