# 🩸 Detector de Cáncer de Sangre (ALL) - Aplicación Web

Una interfaz intuitiva y moderna para detectar células sanguíneas malignas usando Inteligencia Artificial.

## 🚀 Inicio Rápido

### 1. Instalar Dependencias

```bash
# Crear entorno virtual (opcional pero recomendado)
python -m venv venv
venv\Scripts\activate  # En Windows

# Instalar dependencias
pip install -r requirements_streamlit.txt
```

### 2. Ejecutar la Aplicación

```bash
streamlit run app.py
```

**O usando el script automático:**
```bash
ejecutar_app.bat
```

### 3. Abrir en el Navegador

La aplicación se abrirá automáticamente en: `http://127.0.0.1:8501`

**Nota:** La aplicación está configurada automáticamente para ejecutarse solo localmente en tu máquina.

---

## 📱 Características de la Aplicación

### 🏠 **Inicio**
- **Descripción general** del sistema y su propósito
- **Métricas principales** en tarjetas visuales
- **Gráfica del historial** de entrenamiento
- **Información clara** para usuarios no técnicos

### 🔍 **Hacer Predicción**
- **Subida de imágenes** (JPG, PNG, JPEG)
- **Análisis automático** con el modelo CNN
- **Resultados visuales** con colores intuitivos
- **Probabilidades detalladas** para cada clase
- **Advertencias médicas** importantes

### 📊 **Métricas del Modelo**
- **Rendimiento completo** en conjunto de test
- **Matriz de confusión** visual
- **Métricas por clase** (Benigno vs Maligno)
- **Reporte detallado** de evaluación

### 📈 **Estadísticas de Uso**
- **Contador de predicciones** realizadas
- **Historial completo** con timestamps
- **Distribución visual** de resultados
- **Información del sistema** y modelo

### ℹ️ **Información**
- **Detalles técnicos** del modelo
- **Limitaciones y advertencias**
- **Información del proyecto**
- **Recomendaciones de uso**

---

## 🎯 Para Quién Está Diseñada

### 👩‍⚕️ **Profesionales Médicos**
- Interfaz clara y directa
- Resultados con niveles de confianza
- Información médica contextualizada

### 👨‍💻 **Investigadores**
- Acceso a métricas detalladas
- Visualización de rendimiento
- Información técnica completa

### 👥 **Usuarios Generales**
- Lenguaje simple y accesible
- Explicaciones visuales
- Advertencias de limitaciones

---

## 📋 Requisitos del Sistema

### Hardware Mínimo
- **RAM**: 4GB
- **CPU**: Cualquier procesador moderno
- **Almacenamiento**: 500MB libres
- **GPU**: Opcional (acelera predicciones)

### Software
- **Python**: 3.8 o superior
- **Sistema Operativo**: Windows, macOS, Linux
- **Navegador**: Chrome, Firefox, Safari, Edge

### Archivos Requeridos
```
IA/
├── app.py                          # Aplicación principal
├── requirements_streamlit.txt      # Dependencias
├── evaluation_results/
│   ├── best_model.pth             # Modelo entrenado
│   ├── metrics_report.txt         # Reporte de métricas
│   └── *.png                      # Gráficas (opcional)
```

---

## 🎨 Interfaz de Usuario

### Diseño Moderno
- **Colores intuitivos**: Verde para benigno, rojo para maligno
- **Tarjetas métricas**: Información destacada
- **Navegación lateral**: Acceso rápido a secciones
- **Responsive**: Se adapta a diferentes tamaños de pantalla

### Experiencia de Usuario
- **Carga rápida**: Modelo optimizado para inferencia
- **Feedback visual**: Spinners y mensajes durante procesamiento
- **Resultados claros**: Interpretación fácil de entender
- **Advertencias**: Recordatorios de limitaciones médicas

---

## 🔧 Personalización

### Cambiar Puerto
```bash
streamlit run app.py --server.port 8502
```

### Modo Desarrollador
```bash
streamlit run app.py --logger.level debug
```

### Configuración Avanzada
Crear archivo `config.toml`:
```toml
[server]
headless = true
port = 8501

[browser]
gatherUsageStats = false
```

---

## 🚨 Consideraciones Médicas

### ⚠️ Limitaciones Importantes

**Esta aplicación es UNA HERRAMIENTA DE APOYO, NO:**
- Un reemplazo para diagnóstico médico profesional
- Un sustituto de análisis de laboratorio especializados
- Una herramienta para toma de decisiones clínicas finales

### 📞 Recomendaciones

1. **Siempre consulta** con especialistas médicos calificados
2. **Usa como complemento** a evaluaciones profesionales
3. **Proporciona contexto** al interpretar resultados
4. **Considera factores adicionales** no visibles en imágenes

---

## 🔄 Actualizaciones y Mejoras

### Próximas Versiones
- [ ] **API REST** para integración con sistemas médicos
- [ ] **Múltiples modelos** para comparación
- [ ] **Base de datos** para seguimiento de pacientes
- [ ] **Exportación de reportes** en PDF
- [ ] **Interfaz móvil** optimizada

### Mejoras Técnicas
- [ ] **Transfer Learning** con modelos pre-entrenados
- [ ] **Ensemble Methods** para mayor robustez
- [ ] **Validación cruzada** automática
- [ ] **Aumentación de datos** en tiempo real

---

## 📞 Soporte y Contacto

### Reportar Problemas
1. Verifica que todos los archivos requeridos existan
2. Revisa la consola para mensajes de error
3. Asegúrate de tener las dependencias instaladas

### Información del Proyecto
- **Nombre**: Detector de Cáncer de Sangre (ALL)
- **Versión**: 1.0
- **Fecha**: Mayo 2026
- **Framework**: Streamlit + PyTorch
- **Licencia**: MIT (Software Libre)

---

## 📊 Métricas de Rendimiento

### Modelo CNN Entrenado
- **Accuracy**: 98.97%
- **Precision**: 99.25%
- **Recall**: 99.50%
- **F1-Score**: 99.37%
- **AUC-ROC**: 0.9958

### Rendimiento de la Aplicación
- **Tiempo de carga**: < 2 segundos
- **Tiempo de predicción**: < 1 segundo
- **Memoria utilizada**: ~500MB
- **Compatibilidad**: Multiplataforma

---

## 🎯 Conclusión

Esta aplicación web proporciona una interfaz accesible y profesional para la detección de células sanguíneas malignas, combinando:

- **Tecnología avanzada** (Deep Learning con PyTorch)
- **Interfaz intuitiva** (Streamlit moderno)
- **Resultados confiables** (98.97% de precisión)
- **Consideraciones éticas** (advertencias médicas claras)

**¡Una herramienta poderosa para apoyar el diagnóstico médico con IA responsable!** 🩸🤖

---

*Desarrollado como parte del proyecto de Deep Learning para detección de cáncer de sangre (ALL) - 2026*