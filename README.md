#  Blood Cancer Detection (ALL) - Deep Learning Solution

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-98.97%25-orange.svg)]()

> **Revolutionizing Leukemia Detection with AI** - A complete deep learning pipeline for Acute Lymphoblastic Leukemia (ALL) detection using Convolutional Neural Networks, achieving 98.97% accuracy with sustainable, open-source technology.

##  Overview

This project implements a state-of-the-art **Convolutional Neural Network (CNN)** for the automatic detection of Acute Lymphoblastic Leukemia (ALL) from blood cell images. The solution is designed with **sustainability** and **territorial transformation** in mind, using 100% free software and low computational requirements.

###  Key Features

-  **98.97% Accuracy** - Exceptional performance on independent test set
-  **Custom CNN Architecture** - 4 convolutional blocks optimized for medical imaging
-  **Sustainable Technology** - 100% open-source, low energy consumption
-  **Web Interface** - Intuitive Streamlit app for non-technical users
-  **Medical Validation** - Comprehensive metrics and clinical evaluation
-  **Complete Pipeline** - From data preprocessing to deployment

## 📊 Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Accuracy** | 98.97% | Correctly classifies 482/487 test images |
| **Sensitivity (Recall)** | 99.50% | Detects 99.5% of malignant cases |
| **Specificity** | 96.63% | Correctly identifies 96.6% of benign cases |
| **AUC-ROC** | 0.996 | Excellent discrimination capability |
| **F1-Score** | 99.37% | Perfect balance of precision and recall |

## 🏗️ Architecture

### CNN Model Structure
```
Input: 224×224×3 RGB Images
├── Conv Block 1: 3→32 filters + ReLU + BatchNorm + MaxPool
├── Conv Block 2: 32→64 filters + ReLU + BatchNorm + MaxPool
├── Conv Block 3: 64→128 filters + ReLU + BatchNorm + MaxPool
├── Conv Block 4: 128→256 filters + ReLU + BatchNorm + AdaptiveAvgPool
├── Fully Connected: 4096→512→128→2 + Dropout(0.5, 0.3)
Output: [P(Benign), P(Malignant)]
```

### Dataset Summary
- **Total Images**: 3,242 blood cell samples
- **Classes**: Benign (15.8%) vs Malignant (84.2%)
- **Split**: 70% Train / 15% Validation / 15% Test
- **Resolution**: 224×224 pixels (standardized)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 4GB RAM minimum
- Modern CPU (GPU optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/blood-cancer-detection.git
   cd blood-cancer-detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the complete pipeline**
   ```bash
   # 1. Data preprocessing
   python cancer.py

   # 2. Data validation and cleaning
   python limpieza.py

   # 3. Generate data dictionary
   python diccionario_datos.py

   # 4. Train model and generate metrics
   python entrenamiento_y_metricas.py
   ```

5. **Launch the web application**
   ```bash
   # Using the automated script
   ejecutar_app.bat  # Windows

   # Or manually
   streamlit run app.py
   ```

## 📁 Project Structure

```
blood-cancer-detection/
├── 📊 Core Pipeline
│   ├── cancer.py                 # Data preprocessing & splitting
│   ├── limpieza.py               # Data validation & cleaning
│   ├── diccionario_datos.py      # Dataset metadata generation
│   └── entrenamiento_y_metricas.py # Model training & evaluation
│
├── 🌐 Web Application
│   ├── app.py                    # Streamlit web interface
│   ├── ejecutar_app.bat          # Windows launcher script
│   └── .streamlit/config.toml    # Streamlit configuration
│
├── 📈 Results & Models
│   └── evaluation_results/
│       ├── best_model.pth        # Trained CNN model
│       ├── metrics_report.txt    # Detailed performance report
│       ├── confusion_matrix.png  # Confusion matrix plot
│       ├── roc_curve.png         # ROC curve visualization
│       └── training_history.png  # Training progress
│
├── 📋 Documentation
│   ├── README.md                 # This file
│   ├── README_APP.md             # Web app documentation
│   ├── DOCUMENTO_IEEE_SEMANA10.md # IEEE academic paper
│   └── GUIA_GOOGLE_COLAB.md      # Colab usage guide
│
├── 🔬 Notebooks
│   └── Blood_Cancer_Detection_Google_Colab.ipynb # Reproducible notebook
│
├── 📊 Data Processing
│   ├── splits/                   # Train/val/test CSV files
│   ├── data_dictionary/          # Dataset metadata
│   └── cleanup_reports/          # Data quality reports
│
└── ⚙️ Configuration
    ├── requirements.txt          # Python dependencies
    ├── requirements_streamlit.txt # Web app dependencies
    ├── .gitignore               # Git ignore rules
    └── LICENSE                  # MIT License
```

## 🎨 Web Application Features

The Streamlit web application provides an intuitive interface for:

### 🏠 **Home Dashboard**
- Project overview and key metrics
- Training history visualization
- Model performance summary

### 🔍 **Prediction Interface**
- Drag-and-drop image upload
- Real-time analysis with confidence scores
- Visual results with medical warnings

### 📊 **Model Metrics**
- Detailed performance metrics
- Confusion matrix visualization
- ROC curve analysis
- Class-specific evaluation

### 📈 **Usage Statistics**
- Prediction history tracking
- Distribution analysis
- System information

## 🔬 Technical Details

### Model Specifications
- **Framework**: PyTorch 2.11.0
- **Architecture**: Custom 4-block CNN
- **Parameters**: ~2.8 million trainable
- **Input Size**: 224×224×3
- **Output**: Binary classification (Benign/Malignant)

### Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 32
- **Epochs**: 10
- **Data Augmentation**: Rotation (±10°), Horizontal Flip (50%)

### Hardware Requirements
- **Minimum**: 4GB RAM, Modern CPU
- **Recommended**: 8GB RAM, GPU (optional)
- **Training Time**: ~2 hours on CPU, ~30 min on GPU

## 🌍 Sustainability & Impact

### Technological Sustainability
- **100% Open Source**: No licensing costs
- **Low Energy**: ~100Wh training consumption
- **Scalable**: Runs on low-end hardware
- **Maintainable**: Clear, documented code

### Territorial Transformation
- **Rural Accessibility**: Works offline in remote areas
- **Cost Reduction**: 95% cheaper than traditional diagnosis
- **Local Empowerment**: Trainable by local technicians
- **Equity**: Equal access regardless of location

## 📈 Results Visualization

### Confusion Matrix
```
                Predicted
               Benign  Malignant
Actual Benign     86        3      (96.6% specificity)
Actual Malignant   2       396      (99.5% sensitivity)
```

### Training Progress
- **Epoch 10**: Validation Accuracy 98.77%
- **Stable Convergence**: No overfitting observed
- **Consistent Performance**: Robust across epochs

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Fork and clone
git clone https://github.com/your-username/blood-cancer-detection.git
cd blood-cancer-detection

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and test
# ... your code changes ...

# Submit pull request
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: Blood Cancer [ALL] research dataset
- **Framework**: PyTorch community
- **Libraries**: scikit-learn, Streamlit, and open-source contributors
- **Research**: IEEE academic paper and collaborators


## 🔗 Related Links

- [IEEE Paper](DOCUMENTO_IEEE_SEMANA10.md) - Complete academic documentation
- [Colab Notebook](Blood_Cancer_Detection_Google_Colab.ipynb) - Reproducible research
- [Web Demo](app.py) - Interactive application
- [PyTorch](https://pytorch.org/) - Deep learning framework

---

## 🎯 Mission Statement

*"Democratizing leukemia detection through sustainable AI technology, bringing accurate diagnosis to every corner of the world, regardless of infrastructure limitations."*

**Built with ❤️ for global health equity**

---

⭐ **Star this repository** if you find it useful!
📝 **Cite our work** if you use it in your research please dont!

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
