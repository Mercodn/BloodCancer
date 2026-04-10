from pathlib import Path
import csv
import random

DATA_DIR = Path("Blood cell Cancer [ALL]")
OUTPUT_DIR = Path("splits")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

MALIGNANT_CLASSES = {
    "[Malignant] early Pre-B",
    "[Malignant] Pre-B",
    "[Malignant] Pro-B",
}
BENIGN_CLASSES = {"Benign"}

try:
    from PIL import Image
except ImportError:
    Image = None


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in VALID_EXTENSIONS


def get_binary_label(class_name: str) -> str:
    if class_name in MALIGNANT_CLASSES:
        return "malignant"
    if class_name in BENIGN_CLASSES:
        return "benign"
    raise ValueError(f"Clase desconocida: {class_name}")


def collect_image_paths(data_dir: Path) -> list[tuple[Path, str]]:
    dataset = []
    for class_dir in sorted([p for p in data_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
        label = get_binary_label(class_dir.name)
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and is_image_file(image_path):
                dataset.append((image_path, label))
    return dataset


def random_split(items: list[tuple[Path, str]], train_frac=0.7, val_frac=0.15, seed=42):
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0")

    random.Random(seed).shuffle(items)
    total = len(items)
    train_end = int(total * train_frac)
    val_end = train_end + int(total * val_frac)

    train = items[:train_end]
    val = items[train_end:val_end]
    test = items[val_end:]
    return train, val, test


def save_split_csv(rows: list[tuple[Path, str]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        for image_path, label in rows:
            writer.writerow([str(image_path), label])


def print_summary(dataset: list[tuple[Path, str]]) -> None:
    counts = {"malignant": 0, "benign": 0}
    for _, label in dataset:
        counts[label] += 1
    print("Resumen del dataset:")
    print(f"  Total imágenes: {len(dataset)}")
    print(f"  Malignas: {counts['malignant']}")
    print(f"  Benignas: {counts['benign']}")


def preprocess_image(image_path: Path, image_size=(224, 224)):
    if Image is None:
        raise ImportError(
            "Para procesar imágenes necesita instalar Pillow: pip install Pillow"
        )
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize(image_size)
        pixels = [pixel / 255.0 for pixel in list(img.getdata())]
    return pixels


def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"No se encontró el directorio de datos: {DATA_DIR}")

    dataset = collect_image_paths(DATA_DIR)
    print_summary(dataset)

    train_rows, val_rows, test_rows = random_split(dataset, train_frac=0.7, val_frac=0.15, seed=42)

    print("División de datos:")
    print(f"  Train: {len(train_rows)}")
    print(f"  Validation: {len(val_rows)}")
    print(f"  Test: {len(test_rows)}")

    save_split_csv(train_rows, OUTPUT_DIR / "train.csv")
    save_split_csv(val_rows, OUTPUT_DIR / "val.csv")
    save_split_csv(test_rows, OUTPUT_DIR / "test.csv")

    print(f"Archivos CSV guardados en {OUTPUT_DIR}\n")



if __name__ == "__main__":
    main()
