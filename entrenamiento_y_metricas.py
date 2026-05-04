# evaluate.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

# Configuración
DATA_DIR = Path("splits")  # Directorio con los CSV generados por cancer.py
OUTPUT_DIR = Path("evaluation_results")
BATCH_SIZE = 32
EPOCHS = 1
LEARNING_RATE = 0.001
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Usando dispositivo: {DEVICE}")


class BloodCellDataset(Dataset):
    """Dataset personalizado para imágenes de células sanguíneas"""
    
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.label_map = {"benign": 0, "malignant": 1}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = Path(self.data.iloc[idx]["image_path"])
        label_str = self.data.iloc[idx]["label"]
        label = self.label_map[label_str]
        
        # Cargar imagen
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            # Si no encuentra la ruta absoluta, intentar ruta relativa
            alt_path = Path("Blood cell Cancer [ALL]") / img_path.name
            if alt_path.exists():
                image = Image.open(alt_path).convert("RGB")
            else:
                raise FileNotFoundError(f"No se encontró la imagen: {img_path}")
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


class SimpleCNN(nn.Module):
    """CNN simple para clasificación binaria de células sanguíneas"""
    
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        
        # Capas convolucionales
        self.conv_layers = nn.Sequential(
            # Bloque 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            
            # Bloque 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            
            # Bloque 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            
            # Bloque 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Capas fully connected
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Entrena un epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validación del modelo"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def get_predictions(model, dataloader, device):
    """Obtiene todas las predicciones del modelo"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities[:, 1].cpu().numpy())  # Probabilidad de clase maligna
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Genera y guarda la matriz de confusión"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benigno', 'Maligno'],
                yticklabels=['Benigno', 'Maligno'])
    plt.title('Matriz de Confusión')
    plt.ylabel('Valor Real')
    plt.xlabel('Predicción')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(y_true, y_prob, save_path):
    """Genera y guarda la curva ROC"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'r--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return auc


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_dir):
    """Grafica el historial de entrenamiento"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Pérdida
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Pérdida durante entrenamiento')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc')
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy durante entrenamiento')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / "training_history.png")
    plt.close()


def save_metrics_report(y_true, y_pred, y_prob, output_path):
    """Guarda todas las métricas en un archivo de texto"""
    
    # Calcular métricas
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    auc = roc_auc_score(y_true, y_prob)
    
    # Calcular métricas por clase
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    # Reporte de clasificación
    class_report = classification_report(y_true, y_pred, 
                                        target_names=['Benigno', 'Maligno'])
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    
    # Guardar en archivo
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("REPORTE DE EVALUACIÓN DEL MODELO\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("📊 MÉTRICAS PRINCIPALES\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"Precision: {precision:.4f} ({precision*100:.2f}%)\n")
        f.write(f"Recall:    {recall:.4f} ({recall*100:.2f}%)\n")
        f.write(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)\n")
        f.write(f"AUC-ROC:   {auc:.4f}\n\n")
        
        f.write("📈 MÉTRICAS MACRO (promedio entre clases)\n")
        f.write("-" * 40 + "\n")
        f.write(f"Precision Macro: {precision_macro:.4f}\n")
        f.write(f"Recall Macro:    {recall_macro:.4f}\n")
        f.write(f"F1-Score Macro:  {f1_macro:.4f}\n\n")
        
        f.write("📋 REPORTE DE CLASIFICACIÓN DETALLADO\n")
        f.write("-" * 40 + "\n")
        f.write(class_report + "\n")
        
        f.write("🔢 MATRIZ DE CONFUSIÓN\n")
        f.write("-" * 40 + "\n")
        f.write("                Predicho\n")
        f.write("               Benigno  Maligno\n")
        f.write(f"Real Benigno     {cm[0,0]:5d}   {cm[0,1]:5d}\n")
        f.write(f"Real Maligno     {cm[1,0]:5d}   {cm[1,1]:5d}\n\n")
        
        f.write("💡 INTERPRETACIÓN\n")
        f.write("-" * 40 + "\n")
        f.write(f"• Verdaderos Negativos (VN): {cm[0,0]} - Benignos correctamente clasificados\n")
        f.write(f"• Falsos Positivos (FP):    {cm[0,1]} - Benignos clasificados como malignos\n")
        f.write(f"• Falsos Negativos (FN):    {cm[1,0]} - Malignos clasificados como benignos\n")
        f.write(f"• Verdaderos Positivos (VP): {cm[1,1]} - Malignos correctamente clasificados\n\n")
        
        if cm[0,1] > cm[1,0]:
            f.write("⚠️ El modelo tiende a clasificar imágenes como malignas (más FP)\n")
        elif cm[1,0] > cm[0,1]:
            f.write("⚠️ El modelo tiende a clasificar imágenes como benignas (más FN)\n")
        else:
            f.write("✓ El modelo tiene un balance equilibrado entre FP y FN\n")


def main():
    """Función principal"""
    
    # Crear directorio de resultados
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Verificar que existen los archivos CSV
    train_csv = DATA_DIR / "train.csv"
    val_csv = DATA_DIR / "val.csv"
    test_csv = DATA_DIR / "test.csv"
    
    for csv_path in [train_csv, val_csv, test_csv]:
        if not csv_path.exists():
            raise FileNotFoundError(f"No se encontró {csv_path}. Ejecuta primero cancer.py")
    
    print("📂 Cargando datasets...")
    
    # Transformaciones para las imágenes
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Crear datasets
    train_dataset = BloodCellDataset(train_csv, transform=train_transform)
    val_dataset = BloodCellDataset(val_csv, transform=val_transform)
    test_dataset = BloodCellDataset(test_csv, transform=val_transform)
    
    # Crear dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"✅ Train: {len(train_dataset)} imágenes")
    print(f"✅ Validation: {len(val_dataset)} imágenes")
    print(f"✅ Test: {len(test_dataset)} imágenes")
    
    # Inicializar modelo
    model = SimpleCNN(num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\n🚀 Comenzando entrenamiento...")
    print("-" * 40)
    
    # Historial de entrenamiento
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0
    
    # Entrenamiento
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Guardar mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pth")
            print(f"  ✓ Mejor modelo guardado (Acc: {val_acc:.2f}%)")
        print()
    
    print("🏁 Entrenamiento completado!")
    
    # Cargar mejor modelo para evaluación
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_model.pth"))
    
    # Evaluación en test
    print("\n📊 Evaluando modelo en conjunto de test...")
    y_true, y_pred, y_prob = get_predictions(model, test_loader, DEVICE)
    
    # Generar visualizaciones
    print("Generando visualizaciones...")
    plot_confusion_matrix(y_true, y_pred, OUTPUT_DIR / "confusion_matrix.png")
    auc = plot_roc_curve(y_true, y_prob, OUTPUT_DIR / "roc_curve.png")
    plot_training_history(train_losses, val_losses, train_accs, val_accs, OUTPUT_DIR)
    
    # Guardar reporte de métricas
    save_metrics_report(y_true, y_pred, y_prob, OUTPUT_DIR / "metrics_report.txt")
    
    # Mostrar resumen final
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    
    print("\n" + "=" * 50)
    print("RESUMEN FINAL - MÉTRICAS EN TEST")
    print("=" * 50)
    print(f"🎯 Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"🎯 Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"🎯 Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"🎯 F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"🎯 AUC-ROC:   {auc:.4f}")
    print("\n" + "=" * 50)
    print(f"✅ Resultados guardados en: {OUTPUT_DIR.resolve()}")
    print("  - metrics_report.txt  (reporte completo)")
    print("  - confusion_matrix.png (matriz de confusión)")
    print("  - roc_curve.png       (curva ROC)")
    print("  - training_history.png (historial de entrenamiento)")
    print("  - best_model.pth      (mejor modelo guardado)")
    print("=" * 50)


if __name__ == "__main__":
    main()