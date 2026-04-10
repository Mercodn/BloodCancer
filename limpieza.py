from pathlib import Path
import csv
import hashlib
from collections import defaultdict

try:
    from PIL import Image, ImageFilter, ImageStat
except ImportError as exc:
    raise ImportError(
        "Pillow no está instalado. Instala con: .\\.venv\\Scripts\\python.exe -m pip install Pillow"
    ) from exc

DATA_DIR = Path("Blood cell Cancer [ALL]")
OUTPUT_DIR = Path("cleanup_reports")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

CLASS_LABELS = {
    "[Malignant] early Pre-B": "malignant",
    "[Malignant] Pre-B": "malignant",
    "[Malignant] Pro-B": "malignant",
    "Benign": "benign",
}

BLUR_THRESHOLD = 150.0
HASH_SIZE = 8


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in VALID_EXTENSIONS


def compute_file_hash(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_hash(image: Image.Image, hash_size: int = HASH_SIZE) -> str:
    gray = image.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    pixels = (
        list(gray.get_flattened_data())
        if hasattr(gray, "get_flattened_data")
        else list(gray.getdata())
    )
    avg = sum(pixels) / len(pixels)
    bits = ["1" if pixel > avg else "0" for pixel in pixels]
    return "".join(bits)


def variance_of_laplacian(image: Image.Image) -> float:
    gray = image.convert("L")
    kernel = ImageFilter.Kernel(
        size=(3, 3),
        kernel=(0, 1, 0, 1, -4, 1, 0, 1, 0),
        scale=1,
        offset=0,
    )
    lap = gray.filter(kernel)
    stats = ImageStat.Stat(lap)
    return stats.var[0]


def validate_image(path: Path) -> bool:
    if not is_image_file(path):
        return False

    try:
        with Image.open(path) as img:
            img.verify()
    except Exception:
        return False
    return True


def open_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def collect_dataset(data_dir: Path):
    rows = []
    invalid_files = []
    invalid_labels = []
    unknown_dirs = []

    for class_dir in sorted([p for p in data_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
        label = CLASS_LABELS.get(class_dir.name)
        if label is None:
            unknown_dirs.append(class_dir.name)
            continue

        for file_path in sorted(class_dir.iterdir()):
            if not file_path.is_file():
                continue

            if not is_image_file(file_path):
                invalid_files.append((file_path, "extensión inválida"))
                continue

            if not validate_image(file_path):
                invalid_files.append((file_path, "archivo corrupto o no soportado"))
                continue

            rows.append((file_path, class_dir.name, label))

    return rows, invalid_files, unknown_dirs


def detect_duplicates(rows):
    exact = defaultdict(list)
    perceptual = defaultdict(list)

    for path, class_name, label in rows:
        exact[compute_file_hash(path)].append((path, class_name, label))

    exact_duplicates = [group for group in exact.values() if len(group) > 1]

    for path, class_name, label in rows:
        with open_image(path) as img:
            perceptual[average_hash(img)].append((path, class_name, label))

    perceptual_duplicates = [group for group in perceptual.values() if len(group) > 1]
    return exact_duplicates, perceptual_duplicates


def detect_blurry(rows, threshold: float = BLUR_THRESHOLD):
    blurry = []
    for path, class_name, label in rows:
        with open_image(path) as img:
            score = variance_of_laplacian(img)
        if score < threshold:
            blurry.append((path, class_name, label, score))
    return blurry


def write_csv_report(rows, path: Path, headers):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows)


def save_reports(dataset_rows, invalid_files, unknown_dirs, exact_dups, perceptual_dups, blurry_rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    write_csv_report(
        [(str(p), label, class_name) for p, class_name, label in dataset_rows],
        OUTPUT_DIR / "dataset_valid.csv",
        ["image_path", "label", "class_dir"],
    )

    write_csv_report(
        [(str(p), reason) for p, reason in invalid_files],
        OUTPUT_DIR / "invalid_files.csv",
        ["image_path", "reason"],
    )

    if unknown_dirs:
        write_csv_report(
            [(name,) for name in sorted(set(unknown_dirs))],
            OUTPUT_DIR / "unknown_directories.csv",
            ["directory_name"],
        )

    for i, group in enumerate(exact_dups, start=1):
        write_csv_report(
            [(str(p), label, class_name) for p, class_name, label in group],
            OUTPUT_DIR / f"exact_duplicates_{i}.csv",
            ["image_path", "label", "class_dir"],
        )

    for i, group in enumerate(perceptual_dups, start=1):
        write_csv_report(
            [(str(p), label, class_name) for p, class_name, label in group],
            OUTPUT_DIR / f"perceptual_duplicates_{i}.csv",
            ["image_path", "label", "class_dir"],
        )

    write_csv_report(
        [(str(p), label, class_name, score) for p, class_name, label, score in blurry_rows],
        OUTPUT_DIR / "blurry_images.csv",
        ["image_path", "label", "class_dir", "laplacian_variance"],
    )


def print_summary(rows, invalid_files, unknown_dirs, exact_dups, perceptual_dups, blurry_rows):
    counts = {"malignant": 0, "benign": 0}
    for _, _, label in rows:
        counts[label] += 1

    print("Limpieza de datos completada")
    print("---------------------------")
    print(f"Imágenes válidas: {len(rows)}")
    print(f"  malignas: {counts['malignant']}")
    print(f"  benignas: {counts['benign']}")
    print(f"Invalid files: {len(invalid_files)}")
    print(f"Unknown directories: {len(unknown_dirs)}")
    print(f"Exact duplicate groups: {len(exact_dups)}")
    print(f"Perceptual duplicate groups: {len(perceptual_dups)}")
    print(f"Blurry images: {len(blurry_rows)}")
    print(f"Reportes guardados en: {OUTPUT_DIR.resolve()}\n")

    if invalid_files:
        print("Revisa invalid_files.csv para imágenes corruptas o con extensión inválida.")
    if exact_dups or perceptual_dups:
        print("Revisa los CSV de duplicados para decidir si eliminas imágenes repetidas.")
    if blurry_rows:
        print("Revisa blurry_images.csv para evaluar si quieres quitar imágenes borrosas.")


def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"No se encontró el directorio de datos: {DATA_DIR}")

    rows, invalid_files, unknown_dirs = collect_dataset(DATA_DIR)
    exact_dups, perceptual_dups = detect_duplicates(rows)
    blurry_rows = detect_blurry(rows)

    save_reports(rows, invalid_files, unknown_dirs, exact_dups, perceptual_dups, blurry_rows)
    print_summary(rows, invalid_files, unknown_dirs, exact_dups, perceptual_dups, blurry_rows)


if __name__ == "__main__":
    main()
