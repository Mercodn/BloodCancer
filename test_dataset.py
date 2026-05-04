import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path

# Configuración
DATA_DIR = Path("splits")
BATCH_SIZE = 32
IMAGE_SIZE = 224

# Verificar archivos
train_csv = DATA_DIR / "train.csv"
if not train_csv.exists():
    print(f"No se encontró {train_csv}")
    exit(1)

print("Cargando dataset...")

# Transformaciones
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# Dataset simple
class SimpleDataset:
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

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            alt_path = Path("Blood cell Cancer [ALL]") / img_path.name
            if alt_path.exists():
                image = Image.open(alt_path).convert("RGB")
            else:
                print(f"No se encontró: {img_path}")
                return None, label

        if self.transform:
            image = self.transform(image)

        return image, label

# Crear dataset
dataset = SimpleDataset(train_csv, transform=transform)
print(f"Dataset creado con {len(dataset)} imágenes")

# Probar cargar algunas imágenes
print("Probando cargar imágenes...")
for i in range(min(5, len(dataset))):
    img, label = dataset[i]
    if img is not None:
        print(f"Imagen {i}: shape {img.shape}, label {label}")
    else:
        print(f"Imagen {i}: no encontrada")

print("Prueba completada")