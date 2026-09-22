"""
EffNet.py

Train EfficientNet-B0 on the CNN crop set in Data/CNN-Data.

The network is ImageNet-pretrained, the input is 224x224, and the last layer
has one output per class folder under train/. Those folders are the 10 classes.
Crops with a short side under 32 pixels are refused and left out of training,
validation, and the test score.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


# Paths come from this file, not from a hardcoded drive letter.
DATA_DIR = Path(__file__).resolve().parent
CNN_DATA_DIR = DATA_DIR / "CNN-Data"
MODEL_PATH = DATA_DIR / "EffNet_b0.pt"

IMAGE_SIZE = 224
NUM_CLASSES = 10
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
RANDOM_SEED = 42
# Crops smaller than this are too small to classify. The short side is the limit.
MIN_SHORT_SIDE = 32
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ImageNet channel stats used with the pretrained EfficientNet-B0 weights.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CropSet(Dataset):
    """Image crops labeled by their class folder."""

    def __init__(self, samples: list[tuple[Path, int]], transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        # Crops can be grayscale or palette images. The model expects RGB.
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def set_seed(seed: int) -> None:
    """Make shuffling and weight init repeatable."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_names(train_dir: Path) -> list[str]:
    """Read the class folders in a stable order."""
    names = sorted(path.name for path in train_dir.iterdir() if path.is_dir())
    if len(names) != NUM_CLASSES:
        raise RuntimeError(f"Expected {NUM_CLASSES} class folders in {train_dir}, found {len(names)}.")
    return names


def keep_crop(width: int, height: int) -> bool:
    """Keep a crop when its short side is large enough to recognize."""
    return min(width, height) >= MIN_SHORT_SIDE


def list_samples(split_dir: Path, class_to_idx: dict[str, int]) -> tuple[list[tuple[Path, int]], int]:
    """Collect usable crop paths for one split. A missing class folder means zero images."""
    samples: list[tuple[Path, int]] = []
    refused = 0
    for name, index in class_to_idx.items():
        folder = split_dir / name
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            # Open the file only to read its size. Tiny boxes are refused.
            with Image.open(path) as image:
                width, height = image.size
            if not keep_crop(width, height):
                refused += 1
                continue
            samples.append((path, index))
    return samples, refused


def build_transform(train: bool):
    """Resize every crop to 224x224 and normalize with ImageNet stats."""
    steps = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if train:
        # Light augmentation. Rotation, color, and blur stay small so the object remains visible.
        steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
                    p=0.3,
                ),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(steps)


def build_model(num_classes: int) -> nn.Module:
    """Load ImageNet EfficientNet-B0 and replace the classifier for our classes."""
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


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


def main() -> None:
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    names = class_names(CNN_DATA_DIR / "train")
    class_to_idx = {name: index for index, name in enumerate(names)}
    print("Classes:")
    for name in names:
        print(f"  {class_to_idx[name]:2d}  {name}")

    train_samples, train_refused = list_samples(CNN_DATA_DIR / "train", class_to_idx)
    val_samples, val_refused = list_samples(CNN_DATA_DIR / "validation", class_to_idx)
    test_samples, test_refused = list_samples(CNN_DATA_DIR / "test", class_to_idx)
    print(
        f"Images: train {len(train_samples)} (refused {train_refused}), "
        f"validation {len(val_samples)} (refused {val_refused}), "
        f"test {len(test_samples)} (refused {test_refused})"
    )

    train_loader = make_loader(train_samples, train=True, device=device)
    val_loader = make_loader(val_samples, train=False, device=device)
    test_loader = make_loader(test_samples, train=False, device=device)

    model = build_model(len(names)).to(device)
    weights = class_weights(train_samples, len(names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # Decay the learning rate smoothly from LEARNING_RATE toward zero across the 10 epochs.
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

    # Score the test split with the epoch that did best on validation.
    # Refused crops are already absent, so this number is only recognizable boxes.
    model.load_state_dict(best_state)
    test_loss, test_accuracy, correct_by_class, support_by_class = evaluate(
        model, test_loader, criterion, device, len(names)
    )
    torch.save(
        {
            "classes": names,
            "image_size": IMAGE_SIZE,
            "min_short_side": MIN_SHORT_SIDE,
            "state_dict": best_state,
        },
        MODEL_PATH,
    )

    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test images scored: {len(test_samples)}. Refused as too small: {test_refused}.")
    print("Per-class test recall:")
    for index, name in enumerate(names):
        support = support_by_class[index]
        if support == 0:
            print(f"  {name}: no test images")
            continue
        recall = correct_by_class[index] / support
        print(f"  {name}: {recall:.4f} ({correct_by_class[index]}/{support})")
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
