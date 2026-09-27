"""
detection.py

YOLO box inference and EfficientNet crop classification for one video frame.

The window calls yolo_boxes() to get all boxes for a frame, then
cnn_prediction_for_crop() to name each one. Boxes whose short side
is under 32 pixels are skipped. Overlapping boxes from different YOLO
models are deduplicated by keep_best_boxes() before the window logs them.

efficientnet_hits() is the final deduplication step used by the window:
it keeps only boxes that EfficientNet named, and collapses any two boxes
that overlap by more than 50 percent into the one with the higher CNN score.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.labels import CNN_LABELS, cnn_label

# YOLO confidence gate. Boxes below this score are discarded.
CONFIDENCE = 0.25

# EfficientNet input size in pixels (height and width).
IMAGE_SIZE = 224

# ImageNet channel statistics required by the pretrained EfficientNet-B0 weights.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Crop whose short side is below this many pixels is not sent to EfficientNet.
MIN_SHORT_SIDE = 32

# BGR box color for each label. The same class always uses the same color.
LABEL_COLOR: dict[str, tuple[int, int, int]] = {
    "Artillery":         (40, 180, 240),
    "M. Rocket Launcher":(0, 140, 255),
    "Missile":           (60, 60, 220),
    "Radar":             (200, 180, 40),
    "Soldier":           (80, 200, 80),
    "camouflage_soldier":(40, 140, 40),
    "military_aircraft": (220, 160, 40),
    "military_vehicle":  (180, 80, 40),
    "military_warship":  (180, 80, 180),
    "trench":            (100, 100, 100),
}


def load_effnet(weight_path, device):
    """Load EfficientNet-B0 from a checkpoint file. No internet download needed.

    Returns (model, classes). The model is in eval mode on the given device.
    """
    from torch import nn
    from torchvision.models import efficientnet_b0

    # The checkpoint stores both state_dict and the class order.
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    classes = list(checkpoint["classes"])
    if tuple(classes) != CNN_LABELS:
        raise RuntimeError(f"Checkpoint classes do not match the shared list: {classes}")
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    # Replace the head with the right output count.
    model.classifier[1] = nn.Linear(in_features, len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, classes


def iou(box_a: list[float], box_b: list[float]) -> float:
    """Intersection-over-union of two xyxy boxes. Returns 0 when they do not overlap."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def keep_best_boxes(boxes: list[dict]) -> list[dict]:
    """Deduplicate by YOLO confidence: when two boxes overlap > 50%, keep the stronger one."""
    # Sort strongest first so the first box in each cluster is always kept.
    ordered = sorted(boxes, key=lambda b: b["confidence"], reverse=True)
    kept: list[dict] = []
    for box in ordered:
        if any(iou(box["xyxy"], old["xyxy"]) > 0.5 for old in kept):
            continue
        kept.append(box)
    return kept


def yolo_boxes(
    models: list[tuple[str, object]],
    frame: np.ndarray,
    device_name: str,
    half: bool = False,
) -> list[dict]:
    """Run all YOLO models on one frame. Return deduplicated boxes with CNN labels.

    half=True uses FP16 on the GPU. The first call is slow; later calls reuse the plan.
    """
    found = []
    for model_name, model in models:
        result = model.predict(
            frame,
            imgsz=640,
            conf=CONFIDENCE,
            verbose=False,
            device=device_name,
            half=half,
        )[0]
        if result.boxes is None:
            continue
        names = model.names
        for box in result.boxes:
            raw_name = str(names[int(box.cls[0])])
            label = cnn_label(raw_name)
            # Drop YOLO names that have no CNN counterpart.
            if label is None:
                continue
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            found.append({
                "xyxy": xyxy,
                "label": label,
                "confidence": float(box.conf[0]),
                "model": model_name,
            })
    return keep_best_boxes(found)


def cnn_prediction_for_crop(
    model, frame: np.ndarray, xyxy: list[float], device
) -> tuple[str, float] | None:
    """Name the object crop inside a YOLO box. Returns (label, confidence), or None if too small."""
    height, width = frame.shape[:2]
    x1 = max(0, int(xyxy[0]))
    y1 = max(0, int(xyxy[1]))
    x2 = min(width, int(xyxy[2]))
    y2 = min(height, int(xyxy[3]))
    # A crop too small to identify reliably is skipped rather than misclassified.
    if x2 - x1 < MIN_SHORT_SIDE or y2 - y1 < MIN_SHORT_SIDE:
        return None
    crop = frame[y1:y2, x1:x2]
    image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    batch = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(batch), dim=1)[0]
        index = int(probs.argmax())
    return CNN_LABELS[index], float(probs[index])


def confidence_percent(score: float) -> str:
    """Format a 0-1 confidence score as a whole-number percent string, e.g. '88%'."""
    return f"{int(round(score * 100))}%"


def final_label(yolo_label: str, cnn_label_name: str | None) -> str:
    """Return the EfficientNet name when the crop was classified, otherwise the YOLO name."""
    if cnn_label_name is None:
        return yolo_label
    return cnn_label_name


def efficientnet_hits(candidates: list[dict]) -> list[dict]:
    """Keep one EfficientNet observation per object.

    - Boxes with no CNN name are dropped entirely (not counted).
    - When two remaining boxes overlap > 50%, keep the one with the higher CNN confidence.
    """
    # Only boxes EfficientNet named count as observations.
    named = [item for item in candidates if item.get("cnn_name")]
    # Sort by CNN confidence so the best crop wins any overlap check.
    ordered = sorted(named, key=lambda item: item["cnn_confidence"], reverse=True)
    kept: list[dict] = []
    for item in ordered:
        if any(iou(item["xyxy"], old["xyxy"]) > 0.5 for old in kept):
            continue
        kept.append(item)
    return kept
