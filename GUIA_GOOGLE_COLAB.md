# GUÍA PASO A PASO - Google Colab

## 🚀 Cómo Usar el Notebook en Google Colab

### OPCIÓN 1: Usando el Archivo Directamente

#### Paso 1: Abre Google Colab
1. Ve a: https://colab.research.google.com/
2. Inicia sesión con tu cuenta Google

#### Paso 2: Carga el Notebook
1. Click en "Archivo" → "Abrir notebook"
2. Click en la pestaña "Subir"
3. Selecciona: `Blood_Cancer_Detection_Google_Colab.ipynb`
4. Espera a que se cargue

#### Paso 3: Ejecuta las Celdas
```
Opción A - Una por una:
├─ Selecciona la primera celda
├─ Presiona Shift + Enter
└─ Repite para cada celda

Opción B - Todas automáticamente:
├─ Ve a Menú: "Entorno de ejecución"
├─ Click: "Ejecutar todas"
└─ Espera a que terminen
```

#### Paso 4: Descarga los Resultados
```
En el panel izquierdo:
├─ Click en icono de carpeta
├─ Navega a: evaluation_results/
├─ Click derecho → "Descargar"
└─ Repite para: metrics_report.txt, PNG files, etc.
```

---

### OPCIÓN 2: Crear Desde Cero en Colab

Si prefieres escribir manualmente en Colab:

#### Paso 1: Crea Nuevo Notebook
```
colab.research.google.com → Nuevo notebook
```

#### Paso 2: Copia cada celda del código

Copia de `Blood_Cancer_Detection_Google_Colab.ipynb`:

**Celda 1: Verificar GPU**
```python
import torch
print(f"GPU disponible: {torch.cuda.is_available()}")
print(f"Dispositivo: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
print(f"Versión PyTorch: {torch.__version__}")
```

**Celda 2: Instalar Dependencias**
```python
!pip install -q torch torchvision torchaudio
!pip install -q scikit-learn matplotlib seaborn pandas pillow
print("✅ Dependencias instaladas correctamente")
```

*(Continúa con el resto del código del notebook)*

---

## ⚡ CARACTERÍSTICAS DEL COLAB PARA TI

### GPU Gratuita
```
Activación automática si está disponible
Velocidad: 10x más rápido que CPU
```

### Almacenamiento
```
50 GB gratuito
Acceso a Google Drive
```

### Tiempo de Ejecución
```
Sin GPU: ~2 horas
Con GPU: ~12 minutos
```

---

## 📊 ESPERADO EN COLAB

### Salida Esperada después de ejecutar todo:

```
GPU disponible: True
Dispositivo: cuda
Versión PyTorch: 2.11.0

✅ Dependencias instaladas correctamente

📂 Recolectando imágenes...
✅ Total de imágenes encontradas: 3242
   benign: 512 imágenes (15.8%)
   malignant: 2730 imágenes (84.2%)

🔀 División del dataset:
   Train: 2269 (70.0%)
   Val:   486 (15.0%)
   Test:  487 (15.0%)

✅ CSV guardados
✅ Clase BloodCellDataset creada
✅ Arquitectura SimpleCNN definida

📊 Parámetros del modelo:
   Total: 2,816,514
   Entrenables: 2,816,514

✅ Transformaciones configuradas
📂 Cargando datasets...
✅ Train: 2269 imágenes
✅ Validation: 486 imágenes
✅ Test: 487 imágenes

✅ Funciones de entrenamiento definidas

Usando dispositivo: cuda

🚀 Comenzando entrenamiento...
----------------------------------------
Epoch 1/10
  Train Loss: 0.2244 | Train Acc: 90.88%
  Val Loss:   0.1534 | Val Acc:   93.00%
  ✓ Mejor modelo guardado (Acc: 93.00%)

...

Epoch 10/10
  Train Loss: 0.0562 | Train Acc: 98.24%
  Val Loss:   0.0851 | Val Acc:   98.77%
  ✓ Mejor modelo guardado (Acc: 98.77%)

🏁 Entrenamiento completado!

📊 Evaluando modelo en conjunto de test...

==================================================
RESUMEN FINAL - MÉTRICAS EN TEST
==================================================
🎯 Accuracy:  0.9897 (98.97%)
🎯 Precision: 0.9925 (99.25%)
🎯 Recall:    0.9950 (99.50%)
🎯 F1-Score:  0.9937 (99.37%)
🎯 AUC-ROC:   0.9958
==================================================
```

---

## 🛠️ TROUBLESHOOTING

### Problema: "RuntimeError: CUDA out of memory"
**Solución**:
```python
# En celda de configuración:
BATCH_SIZE = 16  # Reducir de 32 a 16
```

### Problema: "FileNotFoundError: Dataset no encontrado"
**Solución**:
```bash
# Ejecuta en celda:
!git clone https://github.com/Mercodn/BloodCancer.git
%cd BloodCancer
```

### Problema: "ModuleNotFoundError: scikit-learn"
**Solución**:
```bash
!pip install scikit-learn
```

### Problema: Las gráficas no se ven
**Solución**:
```python
# Agrega antes de plt.show():
%matplotlib inline
```

---

## 📝 DESCARGA DE ARCHIVOS

### Archivos que se generarán:

```
evaluation_results/
├── best_model.pth              (Modelo entrenado)
├── metrics_report.txt          (Métricas detalladas)
├── confusion_matrix.png        (Matriz confusión)
├── roc_curve.png              (Curva ROC)
└── training_history.png       (Gráficas de entrenamiento)

splits/
├── train.csv
├── val.csv
└── test.csv
```

### Cómo Descargar:

**Método 1: Desde Colab (Recomendado)**
```
1. Panel izquierdo → Carpeta
2. Navega a carpeta deseada
3. Click derecho en archivo
4. "Descargar"
```

**Método 2: Usando Código**
```python
# En una celda:
from google.colab import files
files.download("evaluation_results/metrics_report.txt")
files.download("evaluation_results/confusion_matrix.png")
files.download("evaluation_results/training_history.png")
```

---

## 🎓 VARIABLE DE ENTORNO RECOMENDADA

Si quieres ejecutar partes del código localmente después:

```bash
# Copia en tu terminal local:
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install torch torchvision torchaudio scikit-learn matplotlib seaborn pandas pillow
```

---

## 🔗 ENLACES ÚTILES

- **Google Colab**: https://colab.research.google.com/
- **PyTorch Docs**: https://pytorch.org/
- **Repositorio**: https://github.com/Mercodn/BloodCancer
- **Dataset**: Se descarga automáticamente del repositorio

---

## ✅ CHECKLIST ANTES DE ENTREGAR

- [ ] Ejecuté todas las celdas sin errores
- [ ] Las métricas se muestran correctamente
- [ ] Se generaron los 5 archivos en evaluation_results/
- [ ] Descargué el documento IEEE
- [ ] Descargué las gráficas PNG
- [ ] Guardé metrics_report.txt

---

## 💡 TIPS IMPORTANTES

1. **Guarda el Notebook**: File → Save (para futuros cambios)
2. **Reutiliza el Entorno**: Colab mantiene variables entre celdas
3. **Evita Reconectar**: Si se cae, inicia de nuevo desde arriba
4. **Monitorea Memoria**: Runtime → Manage sessions (Ver memoria usada)
5. **Exporte Modelos**: Descarga best_model.pth para usar localmente

---

**Última actualización**: Mayo 2026  
**Versión**: 1.0

