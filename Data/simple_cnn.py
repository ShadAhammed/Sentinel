"""
simple_cnn.py

Train a small CNN on the 10-class crop set plus the MV dataset.

MV folds into the existing labels:
    armoured personnel carrier, tank -> military_vehicle
    soldier -> camouflage_soldier

Air-fighter and bomber were removed from MV. They are not labels here.

Those three labels are redrawn from the full crop pool so MV can enter
the cap. The other seven labels stay on the current CNN-Data files, which
keeps the trench swap and the warship test split.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import localize
import make_cnn_data


# Paths come from this file, not from a hardcoded drive letter.
DATA_DIR = Path(__file__).resolve().parent
CNN_DIR = DATA_DIR / "CNN"
CNN_DATA_DIR = DATA_DIR / "CNN-Data"
CNN_MV_DIR = DATA_DIR / "CNN-MV"
COMBINED_DIR = DATA_DIR / "CNN-Combined"
MV_DIR = DATA_DIR / "YOLO" / "MV"
MODEL_PATH = DATA_DIR / "SimpleCNN.pt"

IMAGE_SIZE = 96
EPOCHS = 15
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
RANDOM_SEED = 42
# Same gate as the EfficientNet run. A crop smaller than this is not usable.
MIN_SHORT_SIDE = 32
# Validation uses a slice of the training budget, then train fills the rest.
VAL_CAP = 100

# MV's own names. soldier here is camouflage in the photos we checked,
# so it must not use the generic "soldier" -> Soldier rule.
MV_NAMES = [
    "armoured personnel carrier",
    "soldier",
    "tank",
]
MV_TO_CNN = {
    "armoured personnel carrier": "military_vehicle",
    "tank": "military_vehicle",
    "soldier": "camouflage_soldier",
}

# These three labels are rebuilt so the MV crops are inside the cap.
# military_vehicle stays at 3000 because that class was already three merged sources.
REBUILT_TRAIN_CAP = {
    "camouflage_soldier": 1000,
    "military_aircraft": 1000,
    "military_vehicle": 3000,
}
REBUILT_TEST_CAP = {
    "camouflage_soldier": 50,
    "military_aircraft": 50,
    "military_vehicle": 150,
}

CLASS_NAMES = [
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


class SmallCNN(nn.Module):
    """Four stride-2 convolutions and a linear layer. Input is 96x96."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        # Each block halves the image. 96 -> 48 -> 24 -> 12 -> 6.
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        flat = features.view(features.size(0), -1)
        return self.classifier(flat)


class CropSet(Dataset):
    """Image crops labeled by the combined class list."""

    def __init__(self, samples: list[tuple[Path, int]], transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        # Crops can be grayscale. The network expects three channels.
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def mv_final_label(raw_label: str) -> str | None:
    """Map one MV class name onto the 10 CNN labels. Unknown names are dropped."""
    return MV_TO_CNN.get(raw_label)


def keep_crop(width: int, height: int) -> bool:
    """Keep a crop when its short side is large enough to recognize."""
    return min(width, height) >= MIN_SHORT_SIDE


def set_seed(seed: int) -> None:
    """Make shuffling and weight init repeatable."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def take_rows(rows: list[dict[str, str]], cap: int, rng: random.Random, keep) -> list[dict[str, str]]:
    """Draw up to `cap` rows that pass `keep`. The draw is repeatable for one seed."""
    ordered = sorted(rows, key=lambda row: row["key"])
    if cap <= 0 or not ordered:
        return []

    unused = list(range(len(ordered)))
    chosen: list[dict[str, str]] = []
    while unused and len(chosen) < cap:
        need = cap - len(chosen)
        draw_count = min(len(unused), need)
        picks = rng.sample(unused, draw_count)
        picked = set(picks)
        unused = [index for index in unused if index not in picked]
        for index in picks:
            if keep(ordered[index]):
                chosen.append(ordered[index])
                if len(chosen) == cap:
                    break
    return chosen


def select_label_rows(
    rows: list[dict[str, str]],
    train_cap: int,
    test_cap: int,
    rng: random.Random,
    keep,
) -> list[dict[str, str]]:
    """Cap one label. Validation is taken first, then train fills the rest of the budget."""
    grouped: dict[str, list[dict[str, str]]] = {"train": [], "validation": [], "test": []}
    for row in rows:
        split = row["split"]
        if split not in grouped:
            raise ValueError(f"Unsupported split: {split}")
        grouped[split].append(row)

    validation = take_rows(grouped["validation"], min(VAL_CAP, train_cap), rng, keep)
    train = take_rows(grouped["train"], train_cap - len(validation), rng, keep)
    test = take_rows(grouped["test"], test_cap, rng, keep)
    return validation + train + test


def image_is_usable(path: Path) -> bool:
    """Read the header only and apply the short-side gate."""
    with Image.open(path) as image:
        width, height = image.size
    return keep_crop(width, height)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load a labels.csv file."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def crop_mv() -> list[dict[str, str]]:
    """Crop MV boxes into Data/CNN-MV and return one row per saved crop."""
    rows: list[dict[str, str]] = []
    refused = 0
    saved = 0
    for src_split, cnn_split in (("train", "train"), ("valid", "validation"), ("test", "test")):
        image_dir = MV_DIR / src_split / "images"
        label_dir = MV_DIR / src_split / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise RuntimeError(f"Missing MV split folders: {image_dir}")

        label_files = sorted(label_dir.glob("*.txt"))
        print(f"[MV/{cnn_split}] {len(label_files)} label files")
        for label_file in label_files:
            image_path = localize.find_image(image_dir, label_file.stem)
            if image_path is None:
                print(f"No image for {label_file.name}")
                continue

            with Image.open(image_path) as raw:
                image = raw.convert("RGB")
                width, height = image.size
                lines = label_file.read_text(encoding="utf-8", errors="ignore").splitlines()
                box_index = 0
                for line in lines:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    try:
                        class_id = int(float(parts[0]))
                    except ValueError:
                        continue
                    if class_id < 0 or class_id >= len(MV_NAMES):
                        continue

                    box = localize.yolo_to_pixels(parts, width, height)
                    box_index += 1
                    if box is None:
                        continue

                    label = mv_final_label(MV_NAMES[class_id])
                    if label is None:
                        continue
                    # The short side of the box is known before the file is written.
                    if not keep_crop(box[2] - box[0], box[3] - box[1]):
                        refused += 1
                        continue

                    file_name = f"MV_{localize.safe_name(label_file.stem)}_{box_index}.jpg"
                    dest = CNN_MV_DIR / cnn_split / label / file_name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if not dest.is_file():
                        crop = image.crop(box)
                        if crop.mode != "RGB":
                            crop = crop.convert("RGB")
                        crop.save(dest, format="JPEG", quality=95)
                    saved += 1
                    rows.append(
                        {
                            "split": cnn_split,
                            "label": label,
                            "dataset": "MV",
                            "key": str(dest),
                            "path": str(dest),
                        }
                    )

    print(f"MV crops kept: {saved}. Refused as too small: {refused}.")
    return rows


def rows_from_cnn_dump(label: str) -> list[dict[str, str]]:
    """Collect Data/CNN crops that already map to one rebuilt label."""
    rows: list[dict[str, str]] = []
    for source in read_csv_rows(CNN_DIR / "labels.csv"):
        mapped = make_cnn_data.mapped_label(source["label"])
        if mapped != label:
            continue
        path = CNN_DIR / source["crop_file"]
        if not path.is_file():
            continue
        rows.append(
            {
                "split": source["split"],
                "label": label,
                "dataset": source["dataset"],
                "key": str(path),
                "path": str(path),
            }
        )
    return rows


def rows_from_cnn_data() -> list[dict[str, str]]:
    """Load the seven labels that MV does not change."""
    rows: list[dict[str, str]] = []
    for source in read_csv_rows(CNN_DATA_DIR / "labels.csv"):
        if source["label"] in REBUILT_TRAIN_CAP:
            continue
        path = CNN_DATA_DIR / source["crop_file"]
        if not path.is_file():
            continue
        rows.append(
            {
                "split": source["split"],
                "label": source["label"],
                "dataset": source["dataset"],
                "key": str(path),
                "path": str(path),
            }
        )
    return rows


def build_combined_rows() -> list[dict[str, str]]:
    """Build the training list: current crops, with three labels redrawn to include MV."""
    rng = random.Random(RANDOM_SEED)
    mv_rows = crop_mv()
    chosen = rows_from_cnn_data()
    # Drop tiny crops from the seven unchanged labels. Their files are already on disk.
    kept_existing: list[dict[str, str]] = []
    refused_existing = 0
    for row in chosen:
        if image_is_usable(Path(row["path"])):
            kept_existing.append(row)
        else:
            refused_existing += 1
    print(f"Unchanged labels kept: {len(kept_existing)}. Refused as too small: {refused_existing}.")

    for label, train_cap in REBUILT_TRAIN_CAP.items():
        pool = rows_from_cnn_dump(label)
        pool.extend(row for row in mv_rows if row["label"] == label)
        print(f"[{label}] pool {len(pool)} before the cap")
        picked = select_label_rows(
            pool,
            train_cap,
            REBUILT_TEST_CAP[label],
            rng,
            lambda row: image_is_usable(Path(row["path"])),
        )
        print(f"[{label}] selected {len(picked)}")
        kept_existing.extend(picked)

    kept_existing.sort(key=lambda row: (row["split"], row["label"], row["key"]))
    return kept_existing


def write_manifest(rows: list[dict[str, str]]) -> None:
    """Save the combined list so the training set can be checked later."""
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    manifest = COMBINED_DIR / "labels.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "label", "dataset", "path"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "split": row["split"],
                    "label": row["label"],
                    "dataset": row["dataset"],
                    "path": row["path"],
                }
            )
    print(f"Combined list: {manifest}")


def build_transform(train: bool):
    """Resize every crop to 96x96. This network is trained from scratch, so use a plain scale."""
    steps = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if train:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=10),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transforms.Compose(steps)


def samples_for_split(rows: list[dict[str, str]], split: str) -> list[tuple[Path, int]]:
    """Turn manifest rows into (path, class index) pairs for one split."""
    class_to_idx = {name: index for index, name in enumerate(CLASS_NAMES)}
    samples: list[tuple[Path, int]] = []
    for row in rows:
        if row["split"] != split:
            continue
        samples.append((Path(row["path"]), class_to_idx[row["label"]]))
    return samples


def class_weights(samples: list[tuple[Path, int]], num_classes: int) -> torch.Tensor:
    """Give rare classes a higher loss weight so they are not ignored."""
    counts = [0] * num_classes
    for _, label in samples:
        counts[label] += 1
    weights = [0.0 if count == 0 else 1.0 / count for count in counts]
    scale = sum(weights) / num_classes
    return torch.tensor([weight / scale for weight in weights], dtype=torch.float32)


def make_loader(samples: list[tuple[Path, int]], train: bool, device: torch.device) -> DataLoader:
    """Build one data loader. Workers stay at 0 so Windows does not respawn the script."""
    dataset = CropSet(samples, build_transform(train))
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=train,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    """Train for one epoch, or evaluate when optimizer is None. Returns loss and accuracy."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    seen = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += labels.size(0)

    if seen == 0:
        return 0.0, 0.0
    return total_loss / seen, correct / seen


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> tuple[float, float, list[int], list[int]]:
    """Score one split. Returns loss, accuracy, correct counts, and support per class."""
    model.eval()
    total_loss = 0.0
    correct = 0
    seen = 0
    correct_by_class = [0] * num_classes
    support_by_class = [0] * num_classes

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            correct += (predictions == labels).sum().item()
            seen += labels.size(0)
            for label, prediction in zip(labels.tolist(), predictions.tolist()):
                support_by_class[label] += 1
                if label == prediction:
                    correct_by_class[label] += 1

    if seen == 0:
        return 0.0, 0.0, correct_by_class, support_by_class
    return total_loss / seen, correct / seen, correct_by_class, support_by_class


def count_parameters(model: nn.Module) -> int:
    """Count trainable weights."""
    return sum(parameter.numel() for parameter in model.parameters())


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. This training run does not fall back to CPU.")

    set_seed(RANDOM_SEED)
    device = torch.device("cuda")
    print(f"Device: {device} {torch.cuda.get_device_name(0)}")

    rows = build_combined_rows()
    write_manifest(rows)
    for split in ("train", "validation", "test"):
        counts: dict[str, int] = {name: 0 for name in CLASS_NAMES}
        for row in rows:
            if row["split"] == split:
                counts[row["label"]] += 1
        total = sum(counts.values())
        print(f"{split}: {total}")
        for name in CLASS_NAMES:
            print(f"  {name}: {counts[name]}")

    train_samples = samples_for_split(rows, "train")
    val_samples = samples_for_split(rows, "validation")
    test_samples = samples_for_split(rows, "test")
    # Train on train + validation together would hide the checkpoint choice.
    # Validation stays held out so the saved epoch is the best validation score.
    train_loader = make_loader(train_samples, train=True, device=device)
    val_loader = make_loader(val_samples, train=False, device=device)
    test_loader = make_loader(test_samples, train=False, device=device)

    model = SmallCNN(len(CLASS_NAMES)).to(device)
    parameter_count = count_parameters(model)
    print(f"Parameters: {parameter_count}")

    weights = class_weights(train_samples, len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_accuracy = -1.0
    best_state = None
    for epoch in range(1, EPOCHS + 1):
        learning_rate = optimizer.param_groups[0]["lr"]
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device, None)
        scheduler.step()
        print(
            f"Epoch {epoch:02d}/{EPOCHS}  "
            f"train loss {train_loss:.4f}  train acc {train_accuracy:.4f}  "
            f"val loss {val_loss:.4f}  val acc {val_accuracy:.4f}  "
            f"lr {learning_rate:.6f}"
        )
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.load_state_dict(best_state)
    test_loss, test_accuracy, correct_by_class, support_by_class = evaluate(
        model, test_loader, criterion, device, len(CLASS_NAMES)
    )
    torch.save(
        {
            "classes": CLASS_NAMES,
            "image_size": IMAGE_SIZE,
            "min_short_side": MIN_SHORT_SIDE,
            "state_dict": best_state,
        },
        MODEL_PATH,
    )
    file_bytes = MODEL_PATH.stat().st_size

    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test images scored: {len(test_samples)}")
    print("Per-class test recall:")
    for index, name in enumerate(CLASS_NAMES):
        support = support_by_class[index]
        if support == 0:
            print(f"  {name}: no test images")
            continue
        recall = correct_by_class[index] / support
        print(f"  {name}: {recall:.4f} ({correct_by_class[index]}/{support})")
    print(f"Parameters: {parameter_count}")
    print(f"Float32 weight bytes: {parameter_count * 4}")
    print(f"Saved file bytes: {file_bytes}")
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
