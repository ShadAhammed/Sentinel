"""
add_mv_to_cnn.py

Add one copy of each MV source photo into Data/CNN-Data.

Carriers and tanks become military_vehicle. Soldiers become camouflage_soldier.
Air-fighter and bomber are already gone. Roboflow copies of the same photo are
not added: the train file is kept when it exists, otherwise validation, otherwise test.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image

import localize


DATA_DIR = Path(__file__).resolve().parent
MV_DIR = DATA_DIR / "YOLO" / "MV"
CNN_DATA_DIR = DATA_DIR / "CNN-Data"
LABELS_PATH = CNN_DATA_DIR / "labels.csv"

# Crops smaller than this are the ones EfficientNet refuses.
MIN_SHORT_SIDE = 32

# Current MV ids after air-fighter and bomber were removed.
MV_LABELS = {
    0: "military_vehicle",
    1: "camouflage_soldier",
    2: "military_vehicle",
}

SPLIT_ORDER = {"train": 0, "valid": 1, "test": 2}
CNN_SPLIT = {"train": "train", "valid": "validation", "test": "test"}


def mv_box_label(class_id: int) -> str | None:
    """Map one MV class id onto a CNN folder. Unknown ids are dropped."""
    return MV_LABELS.get(class_id)


def choose_sources() -> list[tuple[str, Path]]:
    """Pick one image per source photo. Train wins over validation and test."""
    groups: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for split in ("train", "valid", "test"):
        image_dir = MV_DIR / split / "images"
        for image in image_dir.iterdir():
            if not image.is_file():
                continue
            source = image.name.split(".rf.")[0]
            groups[source].append((split, image))

    chosen: list[tuple[str, Path]] = []
    for items in groups.values():
        items.sort(key=lambda item: (SPLIT_ORDER[item[0]], item[1].name))
        split, image = items[0]
        chosen.append((CNN_SPLIT[split], image))
    return chosen


def crop_one(split: str, image_path: Path) -> tuple[list[dict[str, str]], int]:
    """Crop the usable boxes in one MV photo. Returns new CSV rows and a refuse count."""
    label_path = MV_DIR / {"train": "train", "validation": "valid", "test": "test"}[split] / "labels" / f"{image_path.stem}.txt"
    if not label_path.is_file():
        return [], 0

    rows: list[dict[str, str]] = []
    refused = 0
    with Image.open(image_path) as raw:
        image = raw.convert("RGB")
        width, height = image.size
        lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        box_index = 0
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(float(parts[0]))
            except ValueError:
                continue
            box = localize.yolo_to_pixels(parts, width, height)
            box_index += 1
            if box is None:
                continue
            label = mv_box_label(class_id)
            if label is None:
                continue
            if min(box[2] - box[0], box[3] - box[1]) < MIN_SHORT_SIDE:
                refused += 1
                continue

            file_name = f"MV_{localize.safe_name(image_path.stem)}_{box_index}.jpg"
            dest = CNN_DATA_DIR / split / label / file_name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.is_file():
                crop = image.crop(box)
                if crop.mode != "RGB":
                    crop = crop.convert("RGB")
                crop.save(dest, format="JPEG", quality=95)
            rows.append(
                {
                    "split": split,
                    "label": label,
                    "dataset": "MV",
                    "source_image": str(image_path.relative_to(DATA_DIR)),
                    "crop_file": str(dest.relative_to(CNN_DATA_DIR)),
                }
            )
    return rows, refused


def write_labels(existing: list[dict[str, str]], added: list[dict[str, str]]) -> None:
    """Replace any older MV rows, then write the label file in a stable order."""
    kept = [row for row in existing if row.get("dataset") != "MV"]
    kept.extend(added)
    kept.sort(key=lambda row: (row["split"], row["label"], row["crop_file"]))
    fieldnames = ["split", "label", "dataset", "source_image", "crop_file"]
    with LABELS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)


def main() -> None:
    with LABELS_PATH.open(encoding="utf-8", newline="") as handle:
        existing = list(csv.DictReader(handle))

    added: list[dict[str, str]] = []
    refused = 0
    sources = choose_sources()
    for split, image_path in sources:
        rows, skipped = crop_one(split, image_path)
        added.extend(rows)
        refused += skipped

    write_labels(existing, added)
    print(f"MV photos considered: {len(sources)}")
    print(f"MV crops added: {len(added)}. Refused as too small: {refused}.")
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in added:
        counts[(row["split"], row["label"])] += 1
    for key in sorted(counts):
        print(f"  {key[0]} {key[1]}: {counts[key]}")


if __name__ == "__main__":
    main()
