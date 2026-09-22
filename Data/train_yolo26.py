"""
train_yolo26.py

Train one YOLO26m detector for each original dataset in Data/YOLO.

KIIT-MiTA and military_object_dataset follow the CNN label rules.
MV keeps the five class names stored in its own data.yaml.
"""

from __future__ import annotations

import gc
import os
import shutil
import sys
from pathlib import Path

import torch

import localize
import make_cnn_data


# Paths come from this file, not from a hardcoded drive letter.
DATA_DIR = Path(__file__).resolve().parent
YOLO_DIR = DATA_DIR / "YOLO"
TRAIN_DIR = DATA_DIR / "YOLO-Train"
RUNS_DIR = DATA_DIR / "YOLO-Runs"
MODEL_DIR = DATA_DIR / "YOLO-Models"
WEIGHTS = DATA_DIR / "yolo26m.pt"

# Same order as the CNN class list. Each dataset keeps only the names it still has.
CNN_LABEL_ORDER = [
    "Artillery",
    "M. Rocket Launcher",
    "Missile",
    "Radar",
    "Soldier",
    "camouflage_soldier",
    "military_aircraft",
    "military_vehicle",
    "military_warship",
    "trench",
]

# KIIT-MiTA is small, so 30 epochs is a full fine-tune.
EPOCHS = 30
PATIENCE = 10
BATCH = 8
# The military set is about 21k images. A larger batch and fewer epochs
# keep that run on the 12 GB GPU without a multi-hour wait.
MILITARY_EPOCHS = 8
MILITARY_BATCH = 16
IMAGE_SIZE = 640
RANDOM_SEED = 42

# MV is a separate Roboflow set. These names stay as written in data.yaml.
MV_NAMES = [
    "air-fighter",
    "armoured personnel carrier",
    "bomber",
    "soldier",
    "tank",
]


def cnn_names_for(raw_names: list[str]) -> list[str]:
    """Return the CNN labels this dataset can still produce, in CNN order."""
    present: set[str] = set()
    for raw in raw_names:
        mapped = make_cnn_data.mapped_label(raw)
        if mapped is None:
            continue
        if mapped not in CNN_LABEL_ORDER:
            raise RuntimeError(f"{raw} maps to {mapped}, which is not a CNN label.")
        present.add(mapped)
    return [name for name in CNN_LABEL_ORDER if name in present]


def rewrite_yolo_text(text: str, raw_names: list[str], class_to_idx: dict[str, int]) -> str:
    """Rewrite one YOLO label file. Dropped boxes are removed and class ids are remapped."""
    lines: list[str] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
        except ValueError:
            continue
        if class_id < 0 or class_id >= len(raw_names):
            continue
        mapped = make_cnn_data.mapped_label(raw_names[class_id])
        if mapped is None:
            continue
        coords = " ".join(parts[1:5])
        lines.append(f"{class_to_idx[mapped]} {coords}")
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def copy_native_text(text: str, class_count: int) -> str:
    """Keep boxes whose class id belongs to this dataset. Drop blank and unknown ids."""
    lines: list[str] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
        except ValueError:
            continue
        if class_id < 0 or class_id >= class_count:
            continue
        coords = " ".join(parts[1:5])
        lines.append(f"{class_id} {coords}")
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def link_image(source: Path, dest: Path) -> None:
    """Point dest at the original image. Copy only when a link cannot be made."""
    if dest.exists():
        return
    try:
        os.link(source, dest)
    except OSError:
        shutil.copy2(source, dest)


def write_data_yaml(path: Path, dataset_root: Path, class_names: list[str], include_test: bool = True) -> None:
    """Write the Ultralytics dataset file for one prepared dataset."""
    lines = [
        f"path: {dataset_root.as_posix()}",
        "train: images/train",
        "val: images/val",
    ]
    if include_test:
        lines.append("test: images/test")
    lines.append("names:")
    for index, name in enumerate(class_names):
        lines.append(f'  {index}: "{name}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_detection_dataset(
    name: str,
    source_root: Path,
    raw_names: list[str],
    split_dirs: dict[str, str],
) -> dict | None:
    """Hardlink images and write remapped labels. Returns None when nothing is left."""
    class_names = cnn_names_for(raw_names)
    if not class_names:
        print(f"{name}: no CNN labels remain. Skipping.")
        return None

    class_to_idx = {label: index for index, label in enumerate(class_names)}
    out_root = TRAIN_DIR / name
    if out_root.exists():
        shutil.rmtree(out_root)

    counts = {label: 0 for label in class_names}
    images_kept = {"train": 0, "val": 0, "test": 0}
    for split_name, folder_name in split_dirs.items():
        image_dir = source_root / folder_name / "images"
        label_dir = source_root / folder_name / "labels"
        dest_images = out_root / "images" / split_name
        dest_labels = out_root / "labels" / split_name
        dest_images.mkdir(parents=True, exist_ok=True)
        dest_labels.mkdir(parents=True, exist_ok=True)
        if not image_dir.is_dir():
            print(f"{name} {split_name}: missing {image_dir}")
            continue
        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in make_cnn_data.IMAGE_SUFFIXES:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            text = ""
            if label_path.is_file():
                text = label_path.read_text(encoding="utf-8", errors="ignore")
            rewritten = rewrite_yolo_text(text, raw_names, class_to_idx)
            # An image whose boxes were all removed is not used.
            if not rewritten:
                continue
            for line in rewritten.splitlines():
                class_id = int(line.split()[0])
                counts[class_names[class_id]] += 1
            link_image(image_path, dest_images / image_path.name)
            (dest_labels / f"{image_path.stem}.txt").write_text(rewritten, encoding="utf-8")
            images_kept[split_name] += 1
        print(f"{name} {split_name}: {images_kept[split_name]} images")

    yaml_path = out_root / "data.yaml"
    write_data_yaml(yaml_path, out_root, class_names)
    print(f"{name} classes:")
    for label in class_names:
        print(f"  {label}: {counts[label]} boxes")
    return {
        "name": name,
        "yaml": yaml_path,
        "names": class_names,
        "counts": counts,
        "images": images_kept,
    }


def prepare_native_dataset(
    name: str,
    source_root: Path,
    class_names: list[str],
    split_dirs: dict[str, str],
) -> dict:
    """Hardlink images and copy labels without renaming classes."""
    out_root = TRAIN_DIR / name
    if out_root.exists():
        shutil.rmtree(out_root)

    counts = {label: 0 for label in class_names}
    images_kept = {"train": 0, "val": 0, "test": 0}
    for split_name, folder_name in split_dirs.items():
        image_dir = source_root / folder_name / "images"
        label_dir = source_root / folder_name / "labels"
        dest_images = out_root / "images" / split_name
        dest_labels = out_root / "labels" / split_name
        dest_images.mkdir(parents=True, exist_ok=True)
        dest_labels.mkdir(parents=True, exist_ok=True)
        if not image_dir.is_dir():
            print(f"{name} {split_name}: missing {image_dir}")
            continue
        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in make_cnn_data.IMAGE_SUFFIXES:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            text = ""
            if label_path.is_file():
                text = label_path.read_text(encoding="utf-8", errors="ignore")
            copied = copy_native_text(text, len(class_names))
            # An image with no boxes is not used.
            if not copied:
                continue
            for line in copied.splitlines():
                class_id = int(line.split()[0])
                counts[class_names[class_id]] += 1
            link_image(image_path, dest_images / image_path.name)
            (dest_labels / f"{image_path.stem}.txt").write_text(copied, encoding="utf-8")
            images_kept[split_name] += 1
        print(f"{name} {split_name}: {images_kept[split_name]} images")

    yaml_path = out_root / "data.yaml"
    write_data_yaml(yaml_path, out_root, class_names)
    print(f"{name} classes:")
    for label in class_names:
        print(f"  {label}: {counts[label]} boxes")
    return {
        "name": name,
        "yaml": yaml_path,
        "names": class_names,
        "counts": counts,
        "images": images_kept,
    }


def format_metrics(title: str, metrics, names: list[str]) -> str:
    """Turn Ultralytics detection metrics into a short text report."""
    box = metrics.box
    lines = [
        title,
        f"precision {box.mp:.4f}",
        f"recall {box.mr:.4f}",
        f"mAP50 {box.map50:.4f}",
        f"mAP50-95 {box.map:.4f}",
    ]
    # ap_class_index lists only classes that had boxes in this split.
    scored = {int(class_id): row for row, class_id in enumerate(box.ap_class_index)}
    for class_id, name in enumerate(names):
        row = scored.get(class_id)
        if row is None:
            lines.append(f"  {name}: no scored instances")
            continue
        precision, recall, ap50, ap = box.class_result(row)
        lines.append(
            f"  {name}: precision {precision:.4f} recall {recall:.4f} "
            f"mAP50 {ap50:.4f} mAP50-95 {ap:.4f}"
        )
    return "\n".join(lines)


def train_one(prepared: dict, epochs: int = EPOCHS, batch: int = BATCH) -> str:
    """Train YOLO26m on one prepared dataset and score val plus test."""
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Refusing to train on CPU.")

    print(f"Training {prepared['name']} on GPU for {epochs} epochs, batch {batch}")
    model = YOLO(str(WEIGHTS))
    model.train(
        data=str(prepared["yaml"]),
        epochs=epochs,
        imgsz=IMAGE_SIZE,
        batch=batch,
        device=0,
        workers=2,
        patience=PATIENCE,
        project=str(RUNS_DIR),
        name=prepared["name"],
        exist_ok=True,
        seed=RANDOM_SEED,
        plots=False,
        verbose=True,
    )

    best_path = RUNS_DIR / prepared["name"] / "weights" / "best.pt"
    if not best_path.is_file():
        raise RuntimeError(f"Training did not save {best_path}")

    # Score the best checkpoint, not the last epoch.
    best_model = YOLO(str(best_path))
    val_metrics = best_model.val(
        data=str(prepared["yaml"]),
        split="val",
        imgsz=IMAGE_SIZE,
        batch=batch,
        device=0,
        workers=2,
        plots=False,
        verbose=False,
    )
    report = [format_metrics("Validation", val_metrics, prepared["names"])]
    if prepared["images"]["test"] > 0:
        test_metrics = best_model.val(
            data=str(prepared["yaml"]),
            split="test",
            imgsz=IMAGE_SIZE,
            batch=batch,
            device=0,
            workers=2,
            plots=False,
            verbose=False,
        )
        report.append(format_metrics("Test", test_metrics, prepared["names"]))

    dest_dir = MODEL_DIR / prepared["name"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, dest_dir / "yolo26m.pt")
    text = "\n\n".join(report) + "\n"
    (dest_dir / "results.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved model: {dest_dir / 'yolo26m.pt'}")

    # Free the GPU before the next dataset.
    del model
    del best_model
    gc.collect()
    torch.cuda.empty_cache()
    return text


def main() -> None:
    if not WEIGHTS.is_file():
        raise RuntimeError(f"Missing pretrained weights: {WEIGHTS}")

    kiit_model = MODEL_DIR / "KIIT-MiTA" / "yolo26m.pt"
    if kiit_model.is_file():
        print(f"KIIT-MiTA already trained: {kiit_model}")
    else:
        kiit = prepare_detection_dataset(
            "KIIT-MiTA",
            YOLO_DIR / "KIIT-MiTA",
            localize.KIIT_NAMES,
            {"train": "train", "val": "valid", "test": "test"},
        )
        if kiit is not None:
            train_one(kiit)

    # Drop the partial military run so the new epochs start from the COCO weights.
    partial_run = RUNS_DIR / "military_object_dataset"
    if partial_run.exists():
        shutil.rmtree(partial_run)

    military = prepare_detection_dataset(
        "military_object_dataset",
        YOLO_DIR / "military_object_dataset",
        localize.MILITARY_NAMES,
        {"train": "train", "val": "val", "test": "test"},
    )
    if military is not None:
        train_one(military, epochs=MILITARY_EPOCHS, batch=MILITARY_BATCH)


def run_mv() -> None:
    """Train YOLO26m on MV using the class names already in that dataset."""
    if not WEIGHTS.is_file():
        raise RuntimeError(f"Missing pretrained weights: {WEIGHTS}")
    prepared = prepare_native_dataset(
        "MV",
        YOLO_DIR / "MV",
        MV_NAMES,
        {"train": "train", "val": "valid", "test": "test"},
    )
    train_one(prepared)


if __name__ == "__main__":
    # Each dataset folder has its own short Ultralytics script.
    print("Run the script inside the dataset folder:")
    print("  Data/YOLO/KIIT-MiTA/run_yolo.py")
    print("  Data/YOLO/military_object_dataset/run_yolo.py")
    print("  Data/YOLO/MV/run_yolo.py")
