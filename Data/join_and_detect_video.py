"""
join_and_detect_video.py

Run the three YOLO weights on one frame, then name each crop with EfficientNet.

The demo window imports these helpers. YOLO places the box. EfficientNet
names the crop. A crop whose short side is under 32 pixels is skipped.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import jetson_test


CONFIDENCE = 0.25

# BGR colors, one per shared label, so the same class keeps the same box color.
LABEL_COLOR = {
    "Artillery": (40, 180, 240),
    "M. Rocket Launcher": (0, 140, 255),
    "Missile": (60, 60, 220),
    "Radar": (200, 180, 40),
    "Soldier": (80, 200, 80),
    "camouflage_soldier": (40, 140, 40),
    "military_aircraft": (220, 160, 40),
    "military_vehicle": (180, 80, 40),
    "military_warship": (180, 80, 180),
    "trench": (100, 100, 100),
}


def iou(box_a: list[float], box_b: list[float]) -> float:
    """Overlap of two xyxy boxes. 0 means they do not touch."""
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
    """Drop a weaker box when it sits on a stronger one."""
    ordered = sorted(boxes, key=lambda item: item["confidence"], reverse=True)
    kept = []
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
    """Run every YOLO file on one frame and map the names onto the CNN list."""
    found = []
    for model_name, model in models:
        # half is FP16 on the GPU. The first call is slow; later frames reuse it.
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
            label = jetson_test.cnn_label(raw_name)
            if label is None:
                continue
            xyxy = [float(value) for value in box.xyxy[0].tolist()]
            found.append(
                {
                    "xyxy": xyxy,
                    "label": label,
                    "confidence": float(box.conf[0]),
                    "model": model_name,
                }
            )
    return keep_best_boxes(found)


def cnn_prediction_for_crop(model, frame: np.ndarray, xyxy: list[float], device) -> tuple[str, float] | None:
    """Name the crop inside one box and return that class confidence. Tiny boxes are skipped."""
    height, width = frame.shape[:2]
    x1 = max(0, int(xyxy[0]))
    y1 = max(0, int(xyxy[1]))
    x2 = min(width, int(xyxy[2]))
    y2 = min(height, int(xyxy[3]))
    # A crop this small is not named. The box stays out of the count.
    if x2 - x1 < 32 or y2 - y1 < 32:
        return None
    crop = frame[y1:y2, x1:x2]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    transform = transforms.Compose(
        [
            transforms.Resize((jetson_test.IMAGE_SIZE, jetson_test.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(jetson_test.IMAGENET_MEAN, jetson_test.IMAGENET_STD),
        ]
    )
    batch = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        scores = model(batch)
        probs = torch.softmax(scores, dim=1)[0]
        index = int(probs.argmax())
    return jetson_test.CNN_LABELS[index], float(probs[index])
