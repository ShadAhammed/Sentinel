"""
localize.py

Crop every labeled object from the two datasets and save the crops
for later CNN training.

Output layout (created when you run this file):

    Data/CNN/
        train/<label>/*.jpg
        test/<label>/*.jpg
        validation/<label>/*.jpg
        labels.csv

If two datasets use the same label name letter by letter (ignoring case),
the saved class name becomes:  <label>_<dataset>
Example: Soldier + soldier -> Soldier_KIIT-MiTA and soldier_military_object_dataset

Needs: Pillow  (pip install Pillow)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

# Pillow is the only extra library. It crops and saves images.
try:
    from PIL import Image
except ImportError:
    print("Pillow is missing. Install it with:  pip install Pillow")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Paths - resolved from this file, not from a hardcoded drive letter
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent
CNN_DIR = DATA_DIR / "CNN"
CSV_PATH = CNN_DIR / "labels.csv"

# Image types used by the three datasets
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

# Skip crops smaller than this (pixels). Tiny boxes are not useful for a CNN.
MIN_CROP_SIZE = 2


# ---------------------------------------------------------------------------
# Class names - copied from each dataset config so we do not need PyYAML
# ---------------------------------------------------------------------------
KIIT_NAMES = [
    "Artilary",
    "Missile",
    "Radar",
    "M. Rocket Launcher",
    "Soldier",
    "Tank",
    "Vehicle",
]

MILITARY_NAMES = [
    "camouflage_soldier",
    "weapon",
    "military_tank",
    "military_truck",
    "military_vehicle",
    "civilian",
    "soldier",
    "civilian_vehicle",
    "military_artillery",
    "trench",
    "military_aircraft",
    "military_warship",
]

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def safe_name(text: str) -> str:
    """Turn a label into a safe folder name (no path tricks, no illegal chars)."""
    cleaned = []
    for ch in text:
        # Keep letters, numbers, dash, underscore, and dot. Replace the rest.
        if ch.isalnum() or ch in "-_.":
            cleaned.append(ch)
        else:
            cleaned.append("_")
    name = "".join(cleaned).strip("._")
    if name == "":
        return "unknown"
    return name


def find_image(image_dir: Path, stem: str) -> Path | None:
    """Find an image file that has this name but any common extension."""
    for ext in IMAGE_EXTS:
        path = image_dir / f"{stem}{ext}"
        if path.is_file():
            return path
    return None


def clip_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int):
    """Keep the box inside the image. Return None if the box is too small."""
    x1_i = max(0, min(int(x1), width))
    y1_i = max(0, min(int(y1), height))
    x2_i = max(0, min(int(x2), width))
    y2_i = max(0, min(int(y2), height))

    # Guard: a reversed or tiny box is not a real object.
    if (x2_i - x1_i) < MIN_CROP_SIZE:
        return None
    if (y2_i - y1_i) < MIN_CROP_SIZE:
        return None
    return x1_i, y1_i, x2_i, y2_i


def yolo_to_pixels(parts: list[str], width: int, height: int):
    """
    YOLO line: class  x_center  y_center  w  h   (all 0 to 1)
    Convert to pixel x1, y1, x2, y2.
    """
    if len(parts) < 5:
        return None

    x_c = float(parts[1]) * width
    y_c = float(parts[2]) * height
    box_w = float(parts[3]) * width
    box_h = float(parts[4]) * height

    x1 = x_c - (box_w / 2.0)
    y1 = y_c - (box_h / 2.0)
    x2 = x_c + (box_w / 2.0)
    y2 = y_c + (box_h / 2.0)
    return clip_box(x1, y1, x2, y2, width, height)


def build_label_map() -> dict[tuple[str, str], str]:
    """
    Build (dataset, raw_label) -> folder label.

    If the same letters appear in more than one dataset (ignore case),
    join the label with the dataset name.
    """
    all_groups = {
        "KIIT-MiTA": KIIT_NAMES,
        "military_object_dataset": MILITARY_NAMES,
    }

    # Count how many datasets use each name, letter by letter.
    name_count: dict[str, int] = {}
    for labels in all_groups.values():
        for label in labels:
            key = label.casefold()
            name_count[key] = name_count.get(key, 0) + 1

    mapping: dict[tuple[str, str], str] = {}
    for dataset, labels in all_groups.items():
        for label in labels:
            if name_count[label.casefold()] > 1:
                # Same name in two datasets: keep both by adding the dataset.
                mapping[(dataset, label)] = f"{label}_{dataset}"
            else:
                mapping[(dataset, label)] = label
    return mapping


def output_path(cnn_split: str, label: str, file_name: str) -> Path:
    """
    Build CNN/split/label/file.jpg and refuse to write outside CNN.
    This blocks path-traversal if a label ever contains .. or slashes.
    """
    split_name = safe_name(cnn_split)
    label_name = safe_name(label)
    file_safe = safe_name(Path(file_name).stem) + ".jpg"

    path = (CNN_DIR / split_name / label_name / file_safe).resolve()
    cnn_root = CNN_DIR.resolve()

    # relative_to raises ValueError if path is not inside CNN.
    path.relative_to(cnn_root)
    return path


def save_crop(
    image: Image.Image,
    box: tuple[int, int, int, int],
    cnn_split: str,
    label: str,
    file_name: str,
) -> Path | None:
    """Crop one box and save it. Return the saved path, or None on failure."""
    try:
        dest = output_path(cnn_split, label, file_name)
    except ValueError:
        print(f"Skip unsafe path for label={label} file={file_name}")
        return None

    dest.parent.mkdir(parents=True, exist_ok=True)
    crop = image.crop(box)

    # JPEG needs RGB, not RGBA or P.
    if crop.mode != "RGB":
        crop = crop.convert("RGB")

    crop.save(dest, format="JPEG", quality=95)
    return dest


# ---------------------------------------------------------------------------
# Dataset processors
# ---------------------------------------------------------------------------
def process_yolo_split(
    dataset: str,
    class_names: list[str],
    image_dir: Path,
    label_dir: Path,
    cnn_split: str,
    label_map: dict[tuple[str, str], str],
    writer: csv.writer,
) -> int:
    """Crop every YOLO box in one split. Return how many crops were saved."""
    if not label_dir.is_dir() or not image_dir.is_dir():
        print(f"Skip missing folders: {image_dir} or {label_dir}")
        return 0

    saved = 0
    label_files = sorted(label_dir.glob("*.txt"))
    print(f"[{dataset}/{cnn_split}] {len(label_files)} label files")

    for label_file in label_files:
        image_path = find_image(image_dir, label_file.stem)
        if image_path is None:
            print(f"No image for {label_file.name}")
            continue

        try:
            with Image.open(image_path) as raw:
                image = raw.convert("RGB")
                width, height = image.size

                lines = label_file.read_text(encoding="utf-8", errors="ignore").splitlines()
                box_index = 0

                for line in lines:
                    parts = line.split()
                    if len(parts) < 5:
                        continue

                    # First number is the class id.
                    try:
                        class_id = int(float(parts[0]))
                    except ValueError:
                        continue

                    if class_id < 0 or class_id >= len(class_names):
                        print(f"Unknown class id {class_id} in {label_file.name}")
                        continue

                    box = yolo_to_pixels(parts, width, height)
                    if box is None:
                        continue

                    raw_label = class_names[class_id]
                    final_label = label_map[(dataset, raw_label)]
                    crop_name = f"{dataset}_{label_file.stem}_{box_index}.jpg"
                    dest = save_crop(image, box, cnn_split, final_label, crop_name)
                    if dest is None:
                        continue

                    writer.writerow(
                        [
                            cnn_split,
                            final_label,
                            dataset,
                            str(image_path.relative_to(DATA_DIR)),
                            str(dest.relative_to(CNN_DIR)),
                        ]
                    )
                    saved += 1
                    box_index += 1
        except Exception as err:
            # One bad image must not stop the whole run.
            print(f"Error on {image_path.name}: {err}")
            continue

    print(f"[{dataset}/{cnn_split}] saved {saved} crops")
    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Create CNN folders, crop both datasets, write labels.csv."""
    print("Building label names (join dataset name when labels match)...")
    label_map = build_label_map()

    # Show which names were joined so the user can check.
    for (dataset, raw), final in sorted(label_map.items()):
        if final != raw:
            print(f"  overlap: {raw} in {dataset} -> {final}")

    CNN_DIR.mkdir(parents=True, exist_ok=True)
    (CNN_DIR / "train").mkdir(exist_ok=True)
    (CNN_DIR / "test").mkdir(exist_ok=True)
    (CNN_DIR / "validation").mkdir(exist_ok=True)

    total = 0
    with CSV_PATH.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["split", "label", "dataset", "source_image", "crop_file"])

        # --- KIIT-MiTA (YOLO). Folder "valid" maps to CNN/validation. ---
        kiit = DATA_DIR / "KIIT-MiTA"
        for src_split, cnn_split in (("train", "train"), ("test", "test"), ("valid", "validation")):
            total += process_yolo_split(
                dataset="KIIT-MiTA",
                class_names=KIIT_NAMES,
                image_dir=kiit / src_split / "images",
                label_dir=kiit / src_split / "labels",
                cnn_split=cnn_split,
                label_map=label_map,
                writer=writer,
            )

        # --- military_object_dataset (YOLO). Folder "val" maps to CNN/validation. ---
        mil = DATA_DIR / "military_object_dataset"
        for src_split, cnn_split in (("train", "train"), ("test", "test"), ("val", "validation")):
            total += process_yolo_split(
                dataset="military_object_dataset",
                class_names=MILITARY_NAMES,
                image_dir=mil / src_split / "images",
                label_dir=mil / src_split / "labels",
                cnn_split=cnn_split,
                label_map=label_map,
                writer=writer,
            )

    print(f"Done. Saved {total} crops under {CNN_DIR}")
    print(f"Label list: {CSV_PATH}")


if __name__ == "__main__":
    main()
