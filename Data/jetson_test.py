"""
jetson_test.py

Test EfficientNet and YOLO on their own pictures.

The Jetson folders next to this file are:
  YOLO/              the three finished detectors
  CNN/               EfficientNet-B0
  test_images_cnn/   five crops of each label, for EfficientNet
  test_images_yolo/  five full photos of each label, for YOLO

A crop is the cut-out object the CNN was trained on.
A full photo is the scene YOLO was trained on.
Each full photo is sent only to the YOLO file trained on that dataset.

YOLO names are turned into the 10 CNN labels before scoring.
A YOLO name with no CNN twin is ignored.

On this PC, build the upload tree with:
  python jetson_test.py prepare
"""

from __future__ import annotations

import csv
import random
import shutil
import sys
from pathlib import Path

from PIL import Image


# Paths come from this file, not from a hardcoded drive letter.
DATA_DIR = Path(__file__).resolve().parent
HERE = DATA_DIR
CNN_DATA_DIR = DATA_DIR / "CNN-Data"
YOLO_FINAL_DIR = DATA_DIR / "YOLO-Models-Final"
EFFNET_PATH = DATA_DIR / "EffNet_b0.pt"

# Shared label list. This is the EfficientNet folder order.
CNN_LABELS = (
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
)

# YOLO checkpoint name -> CNN label.
# MV "soldier" is camouflage. KIIT "Soldier" stays Soldier.
# "Vehicle" is not here: the CNN set dropped that class.
YOLO_TO_CNN = {
    "Artilary": "Artillery",
    "Artillery": "Artillery",
    "Missile": "Missile",
    "Radar": "Radar",
    "M. Rocket Launcher": "M. Rocket Launcher",
    "Soldier": "Soldier",
    "soldier": "camouflage_soldier",
    "camouflage_soldier": "camouflage_soldier",
    "Tank": "military_vehicle",
    "tank": "military_vehicle",
    "armoured personnel carrier": "military_vehicle",
    "military_vehicle": "military_vehicle",
    "military_aircraft": "military_aircraft",
    "military_warship": "military_warship",
    "trench": "trench",
}

# These three files are the finished detectors copied into YOLO/.
YOLO_FILES = (
    "KIIT-MiTA-yolo26s.pt",
    "MV-yolo26s.pt",
    "military-yolov8n.pt",
)

# Class id order in each dataset's label files. This is the original spelling.
DATASET_CLASS_NAMES = {
    "KIIT-MiTA": (
        "Artilary",
        "Missile",
        "Radar",
        "M. Rocket Launcher",
        "Soldier",
        "Tank",
        "Vehicle",
    ),
    "MV": (
        "armoured personnel carrier",
        "soldier",
        "tank",
    ),
    "military_object_dataset": (
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
    ),
}

# Which weight file was trained on that dataset.
DATASET_MODEL = {
    "KIIT-MiTA": "KIIT-MiTA-yolo26s.pt",
    "MV": "MV-yolo26s.pt",
    "military_object_dataset": "military-yolov8n.pt",
}

# Extra source spellings that do not appear as checkpoint names.
SOURCE_ONLY_TO_CNN = {
    "military_artillery": "Artillery",
    "military_tank": "military_vehicle",
    "military_truck": "military_vehicle",
}

IMAGES_PER_LABEL = 5
MIN_SHORT_SIDE = 32
RANDOM_SEED = 42
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def cnn_label(raw_name: str) -> str | None:
    """Return the shared CNN name for a YOLO class, or None when it was dropped."""
    mapped = YOLO_TO_CNN.get(raw_name)
    if mapped not in CNN_LABELS:
        return None
    return mapped


def source_label(dataset: str, raw_name: str) -> str | None:
    """Map a name written in a dataset label file onto the shared CNN list."""
    # The word soldier means camouflage only in the MV set.
    if dataset == "MV" and raw_name == "soldier":
        return "camouflage_soldier"
    if dataset == "military_object_dataset" and raw_name == "soldier":
        return "Soldier"
    if raw_name in SOURCE_ONLY_TO_CNN:
        return SOURCE_ONLY_TO_CNN[raw_name]
    return cnn_label(raw_name)


def read_test_rows(labels_csv: Path) -> dict[str, list[dict[str, str]]]:
    """Group CNN-Data test rows by label."""
    grouped: dict[str, list[dict[str, str]]] = {label: [] for label in CNN_LABELS}
    with labels_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "test":
                continue
            label = row["label"]
            if label in grouped:
                grouped[label].append(row)
    return grouped


def crop_is_big_enough(path: Path) -> bool:
    """Refuse a crop whose short side is under 32 pixels."""
    with Image.open(path) as image:
        width, height = image.size
    return min(width, height) >= MIN_SHORT_SIDE


def choose_crops(rows: list[dict[str, str]], cnn_data: Path, per_label: int, rng: random.Random) -> list[Path]:
    """Pick up to `per_label` crops, using a new source photo when one is left."""
    first_by_source: dict[str, Path] = {}
    extras: list[Path] = []
    for row in sorted(rows, key=lambda item: item["crop_file"]):
        crop = cnn_data / Path(row["crop_file"])
        if not crop.is_file() or not crop_is_big_enough(crop):
            continue
        # Keep the first crop of each photo. Further boxes from that photo are spares.
        if row["source_image"] not in first_by_source:
            first_by_source[row["source_image"]] = crop
        else:
            extras.append(crop)
    sources = sorted(first_by_source)
    rng.shuffle(sources)
    chosen = [first_by_source[source] for source in sources[:per_label]]
    # Trench has four test photos. Fill the last slot with another box from one of them.
    for crop in extras:
        if len(chosen) >= per_label:
            break
        chosen.append(crop)
    return chosen[:per_label]


def write_labels_file(folder: Path) -> None:
    """Write the same 10 names into a folder."""
    folder.mkdir(parents=True, exist_ok=True)
    text = "\n".join(CNN_LABELS) + "\n"
    (folder / "labels.txt").write_text(text, encoding="utf-8")


def find_photo(image_dir: Path, stem: str) -> Path | None:
    """Find the picture that belongs to one YOLO label file."""
    for suffix in IMAGE_SUFFIXES:
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def labels_in_file(dataset: str, label_path: Path) -> set[str]:
    """Read one YOLO label file and return the shared names inside it."""
    names = DATASET_CLASS_NAMES[dataset]
    found: set[str] = set()
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        class_id = int(float(parts[0]))
        if class_id < 0 or class_id >= len(names):
            continue
        mapped = source_label(dataset, names[class_id])
        if mapped is not None:
            found.add(mapped)
    return found


def collect_full_images(split: str) -> dict[str, list[tuple[Path, str]]]:
    """List photos in one split that contain each shared label."""
    grouped: dict[str, list[tuple[Path, str]]] = {label: [] for label in CNN_LABELS}
    for dataset, model_name in DATASET_MODEL.items():
        root = DATA_DIR / "YOLO" / dataset / split
        label_dir = root / "labels"
        image_dir = root / "images"
        # A dataset can omit a split. Skip it and keep looking in the others.
        if not label_dir.is_dir() or not image_dir.is_dir():
            continue
        for label_path in sorted(label_dir.glob("*.txt")):
            photo = find_photo(image_dir, label_path.stem)
            if photo is None:
                continue
            for label in labels_in_file(dataset, label_path):
                grouped[label].append((photo, model_name))
    return grouped


def unique_photos(pairs: list[tuple[Path, str]]) -> list[tuple[Path, str]]:
    """Drop repeated paths and keep the first model name."""
    unique: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for photo, model_name in pairs:
        if photo in seen:
            continue
        seen.add(photo)
        unique.append((photo, model_name))
    return unique


def write_yolo_images(image_dir: Path, rng: random.Random) -> None:
    """Copy five full photos for each shared label. Test photos come first."""
    from_test = collect_full_images("test")
    from_train = None
    manifest_rows: list[tuple[str, str, str]] = []
    for label in CNN_LABELS:
        pool = unique_photos(from_test[label])
        rng.shuffle(pool)
        chosen = pool[:IMAGES_PER_LABEL]
        # Ships are not in the YOLO test split. Fill the gap from train photos.
        if len(chosen) < IMAGES_PER_LABEL:
            if from_train is None:
                from_train = collect_full_images("train")
            extra = unique_photos(from_train[label])
            rng.shuffle(extra)
            need = IMAGES_PER_LABEL - len(chosen)
            chosen.extend(extra[:need])
            print(f"YOLO {label}: test split had {IMAGES_PER_LABEL - need}, added {min(need, len(extra))} train photos")
        if len(chosen) < IMAGES_PER_LABEL:
            raise RuntimeError(f"{label} has {len(chosen)} full photos, need {IMAGES_PER_LABEL}.")
        folder = image_dir / label
        folder.mkdir(parents=True)
        for photo, model_name in chosen:
            dataset = photo.parent.parent.parent.name
            dest_name = f"{dataset}__{photo.name}"
            shutil.copy2(photo, folder / dest_name)
            # Forward slashes so the same file works on the Jetson.
            manifest_rows.append((label, model_name, (Path(label) / dest_name).as_posix()))
        print(f"YOLO {label}: {len(chosen)} photos")
    with (image_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "model", "file"])
        writer.writerows(manifest_rows)


def prepare(dest: Path) -> None:
    """Build YOLO/, CNN/, test_images_cnn/, and test_images_yolo/ for the Jetson."""
    labels_csv = CNN_DATA_DIR / "labels.csv"
    if not labels_csv.is_file():
        raise RuntimeError(f"Missing {labels_csv}")
    if not EFFNET_PATH.is_file():
        raise RuntimeError(f"Missing {EFFNET_PATH}")

    if dest.exists():
        shutil.rmtree(dest)
    yolo_dir = dest / "YOLO"
    cnn_dir = dest / "CNN"
    cnn_image_dir = dest / "test_images_cnn"
    yolo_image_dir = dest / "test_images_yolo"
    yolo_dir.mkdir(parents=True)
    cnn_dir.mkdir(parents=True)

    for name in YOLO_FILES:
        source = YOLO_FINAL_DIR / name
        if not source.is_file():
            raise RuntimeError(f"Missing {source}")
        shutil.copy2(source, yolo_dir / name)
    shutil.copy2(EFFNET_PATH, cnn_dir / "EffNet_b0.pt")
    shutil.copy2(Path(__file__), dest / "jetson_test.py")
    write_labels_file(yolo_dir)
    write_labels_file(cnn_dir)

    grouped = read_test_rows(labels_csv)
    rng = random.Random(RANDOM_SEED)
    for label in CNN_LABELS:
        chosen = choose_crops(grouped[label], CNN_DATA_DIR, IMAGES_PER_LABEL, rng)
        if len(chosen) < IMAGES_PER_LABEL:
            raise RuntimeError(f"{label} has {len(chosen)} usable test crops, need {IMAGES_PER_LABEL}.")
        folder = cnn_image_dir / label
        folder.mkdir(parents=True)
        for crop in chosen:
            shutil.copy2(crop, folder / crop.name)
        print(f"CNN {label}: {len(chosen)} crops")
    write_yolo_images(yolo_image_dir, rng)
    print(f"Wrote {dest}")


def list_test_images(image_root: Path) -> list[tuple[Path, str]]:
    """Read test_images/<label>/ as the shared ground truth."""
    samples: list[tuple[Path, str]] = []
    for label in CNN_LABELS:
        folder = image_root / label
        if not folder.is_dir():
            raise RuntimeError(f"Missing test folder {folder}")
        files = sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            raise RuntimeError(f"No images in {folder}")
        for path in files:
            samples.append((path, label))
    return samples


def load_effnet(weight_path: Path, device):
    """Load EfficientNet-B0 from the checkpoint. No ImageNet download."""
    import torch
    from torch import nn
    from torchvision.models import efficientnet_b0

    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    classes = list(checkpoint["classes"])
    if tuple(classes) != CNN_LABELS:
        raise RuntimeError(f"Checkpoint classes do not match the shared list: {classes}")
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, classes


def effnet_predict(model, image_path: Path, device) -> str:
    """Return the CNN label for one crop."""
    import torch
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    batch = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        scores = model(batch)
        index = int(scores.argmax(dim=1)[0])
    return CNN_LABELS[index]


def yolo_boxes(model, image_path: Path, device_name: str) -> list[tuple[str, float]]:
    """Return mapped boxes from one YOLO file. Dropped names are left out."""
    result = model.predict(source=str(image_path), imgsz=640, conf=0.25, verbose=False, device=device_name)[0]
    found: list[tuple[str, float]] = []
    if result.boxes is None:
        return found
    names = model.names
    for box in result.boxes:
        mapped = cnn_label(str(names[int(box.cls[0])]))
        if mapped is None:
            continue
        found.append((mapped, float(box.conf[0])))
    return found


def load_yolo_models(yolo_dir: Path) -> list[tuple[str, object]]:
    """Load each finished weight file in YOLO/."""
    from ultralytics import YOLO

    models = []
    for name in YOLO_FILES:
        path = yolo_dir / name
        if not path.is_file():
            raise RuntimeError(f"Missing {path}")
        models.append((name, YOLO(str(path))))
    return models


def score_name(correct: int, total: int) -> str:
    """Format one accuracy line."""
    if total == 0:
        return "no images"
    return f"{correct}/{total}  {correct / total:.4f}"


def read_manifest(image_root: Path) -> list[tuple[str, str, Path]]:
    """Read which YOLO weight file should see each full photo."""
    manifest = image_root / "manifest.csv"
    if not manifest.is_file():
        raise RuntimeError(f"Missing {manifest}")
    rows: list[tuple[str, str, Path]] = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append((row["label"], row["model"], image_root / Path(row["file"])))
    return rows


def run_cnn(root: Path, device) -> None:
    """Score EfficientNet on the crop folder."""
    samples = list_test_images(root / "test_images_cnn")
    model, _classes = load_effnet(root / "CNN" / "EffNet_b0.pt", device)
    correct = 0
    by_label = {label: [0, 0] for label in CNN_LABELS}
    lines = ["true_label,cnn_pred,cnn_ok,file"]
    print("CNN crops")
    for path, true_label in samples:
        pred = effnet_predict(model, path, device)
        ok = pred == true_label
        correct += int(ok)
        by_label[true_label][0] += int(ok)
        by_label[true_label][1] += 1
        lines.append(f"{true_label},{pred},{int(ok)},{path.name}")
        print(f"  {true_label}  {pred}  {path.name}")
    print(f"CNN accuracy: {score_name(correct, len(samples))}")
    for label in CNN_LABELS:
        hit, total = by_label[label]
        print(f"  {label}: {hit}/{total}")
    report = root / "test_images_cnn" / "results.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {report}")


def run_yolo(root: Path, device_name: str) -> None:
    """Score each full photo with the one YOLO file trained on its dataset."""
    image_root = root / "test_images_yolo"
    rows = read_manifest(image_root)
    loaded = {name: model for name, model in load_yolo_models(root / "YOLO")}
    correct = 0
    by_label = {label: [0, 0] for label in CNN_LABELS}
    lines = ["true_label,found,best_label,best_conf,model,file"]
    print("YOLO full photos")
    for true_label, model_name, path in rows:
        boxes = yolo_boxes(loaded[model_name], path, device_name)
        found = any(label == true_label for label, _conf in boxes)
        best_label = ""
        best_conf = ""
        if boxes:
            label, conf = max(boxes, key=lambda item: item[1])
            best_label = label
            best_conf = f"{conf:.4f}"
        correct += int(found)
        by_label[true_label][0] += int(found)
        by_label[true_label][1] += 1
        lines.append(f"{true_label},{int(found)},{best_label},{best_conf},{model_name},{path.name}")
        print(f"  {true_label}  found {int(found)}  best {best_label or 'none'}  {path.name}")
    print(f"YOLO found the label in the photo: {score_name(correct, len(rows))}")
    for label in CNN_LABELS:
        hit, total = by_label[label]
        print(f"  {label}: {hit}/{total}")
    report = image_root / "results.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {report}")


def run_test(root: Path) -> None:
    """Score the CNN crops and the YOLO photos separately."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = "0" if device.type == "cuda" else "cpu"
    print(f"Device: {device}")
    run_cnn(root, device)
    run_yolo(root, device_name)


def main() -> None:
    """Prepare the upload tree, or run the test beside this file."""
    if len(sys.argv) > 1 and sys.argv[1] == "prepare":
        dest = Path(sys.argv[2]) if len(sys.argv) > 2 else DATA_DIR / "_jetson_upload"
        prepare(dest)
        return
    # On the Jetson this file sits beside YOLO/, CNN/, and test_images/.
    run_test(HERE)


if __name__ == "__main__":
    main()
