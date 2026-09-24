"""
hog_svm.py

Train a linear SVM on HOG features from the same CNN-Data crops as EfficientNet.

The image is resized to 96x96 grayscale. Crops with a short side under 32 pixels
are refused, matching EffNet.py. The fit uses the train split only.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.svm import LinearSVC

import EffNet


DATA_DIR = Path(__file__).resolve().parent
CNN_DATA_DIR = DATA_DIR / "CNN-Data"
MODEL_PATH = DATA_DIR / "HogSvm.pkl"

HOG_SIZE = 96
# OpenCV HOG window, block, stride, cell, and bins. 96 is divisible by the cell size.
HOG = cv2.HOGDescriptor((HOG_SIZE, HOG_SIZE), (16, 16), (8, 8), (8, 8), 9)


def hog_features(path: Path) -> np.ndarray:
    """Resize one crop to grayscale and return its HOG vector."""
    with Image.open(path) as image:
        gray = np.array(image.convert("L").resize((HOG_SIZE, HOG_SIZE)))
    features = HOG.compute(gray)
    return features.reshape(-1)


def load_split(split: str, class_to_idx: dict[str, int]) -> tuple[np.ndarray, np.ndarray, int]:
    """Load one split. Returns features, labels, and how many crops were too small."""
    samples, refused = EffNet.list_samples(CNN_DATA_DIR / split, class_to_idx)
    features = []
    labels = []
    for index, (path, label) in enumerate(samples, start=1):
        features.append(hog_features(path))
        labels.append(label)
        if index % 1000 == 0:
            print(f"  {split} {index}/{len(samples)}")
    if not features:
        width = HOG.getDescriptorSize()
        return np.zeros((0, width), dtype=np.float32), np.zeros((0,), dtype=np.int64), refused
    return np.vstack(features).astype(np.float32), np.array(labels, dtype=np.int64), refused


def scores(model: LinearSVC, features: np.ndarray, labels: np.ndarray, names: list[str]) -> None:
    """Print accuracy and per-class recall for one split."""
    if len(labels) == 0:
        print("  no images")
        return
    predicted = model.predict(features)
    correct = int((predicted == labels).sum())
    print(f"  accuracy {correct / len(labels):.4f} ({correct}/{len(labels)})")
    print("  per-class recall:")
    for index, name in enumerate(names):
        support = int((labels == index).sum())
        if support == 0:
            print(f"    {name}: no images")
            continue
        hits = int(((labels == index) & (predicted == index)).sum())
        print(f"    {name}: {hits / support:.4f} ({hits}/{support})")


def main() -> None:
    names = EffNet.class_names(CNN_DATA_DIR / "train")
    class_to_idx = {name: index for index, name in enumerate(names)}
    print("Loading HOG features...")
    train_x, train_y, train_refused = load_split("train", class_to_idx)
    val_x, val_y, val_refused = load_split("validation", class_to_idx)
    test_x, test_y, test_refused = load_split("test", class_to_idx)
    print(
        f"Images: train {len(train_y)} (refused {train_refused}), "
        f"validation {len(val_y)} (refused {val_refused}), "
        f"test {len(test_y)} (refused {test_refused})"
    )
    print(f"HOG length: {train_x.shape[1]}")

    # dual=False is the primal solver, which fits this many samples more quietly.
    model = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, dual=False)
    model.fit(train_x, train_y)

    print("Validation:")
    scores(model, val_x, val_y, names)
    print("Test:")
    scores(model, test_x, test_y, names)

    with MODEL_PATH.open("wb") as handle:
        pickle.dump({"classes": names, "hog_size": HOG_SIZE, "model": model}, handle)
    print(f"Saved file bytes: {MODEL_PATH.stat().st_size}")
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
