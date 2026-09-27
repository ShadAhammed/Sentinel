"""
jetson_test.py

Shared labels and the EfficientNet loader for the demo window.

YOLO class names are turned into the 10 CNN labels before a box is kept.
A YOLO name with no CNN twin is ignored.
"""

from __future__ import annotations

from pathlib import Path


# Paths come from this file, not from a hardcoded drive letter.
DATA_DIR = Path(__file__).resolve().parent
YOLO_FINAL_DIR = DATA_DIR / "YOLO-Models-Final"
EFFNET_PATH = DATA_DIR / "EffNet_b0.pt"

# This is the EfficientNet class order stored in the checkpoint.
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
# "soldier" in lower case is the MV camouflage class.
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

# These three files are the detectors the window loads.
YOLO_FILES = (
    "KIIT-MiTA-yolo26s.pt",
    "MV-yolo26s.pt",
    "military-yolov8n.pt",
)

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def cnn_label(raw_name: str) -> str | None:
    """Return the shared CNN name for a YOLO class, or None when it was dropped."""
    mapped = YOLO_TO_CNN.get(raw_name)
    if mapped not in CNN_LABELS:
        return None
    return mapped


def load_effnet(weight_path: Path, device):
    """Load EfficientNet-B0 from the checkpoint. No ImageNet download."""
    import torch
    from torch import nn
    from torchvision.models import efficientnet_b0

    # The file stores the class list and the weights. Build the head to match.
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
