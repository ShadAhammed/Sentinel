"""
make_drone_video.py

Build a 30 second, 32 fps drone flight for detector testing.

Each shared label gets one real training photo. The camera drifts across
that photo, then fades into the next, so the clip feels like one flight
over the same kinds of objects the models were trained on.
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import cv2
import numpy as np

import jetson_test


# Paths come from this file, not from a hardcoded drive letter.
DATA_DIR = Path(__file__).resolve().parent
YOLO_DIR = DATA_DIR / "YOLO"
VIDEO_PATH = DATA_DIR / "drone_flight_30s.mp4"
OBJECTS_PATH = DATA_DIR / "drone_flight_30s_objects.csv"

LABELS = jetson_test.CNN_LABELS
FPS = 32
SECONDS = 30
FRAME_COUNT = FPS * SECONDS
VIEW_W = 1280
VIEW_H = 720
# Ten labels share the 960 frames. 96 frames is 3 seconds at 32 fps.
SCENE_FRAMES = FRAME_COUNT // len(LABELS)
FADE_FRAMES = 10
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# A box this large usually means a close-up, not a view from above.
MAX_BOX_AREA = 0.45
MIN_BOX_AREA = 0.004


def find_photo(image_dir: Path, stem: str) -> Path | None:
    """Find the picture that belongs to one YOLO label file."""
    for suffix in IMAGE_SUFFIXES:
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def collect_scenes() -> list[tuple[str, Path]]:
    """Pick one training photo for each shared label."""
    chosen: dict[str, Path] = {}
    # KIIT photos are already drone views. Fill the other labels from the military set.
    for dataset in ("KIIT-MiTA", "military_object_dataset", "MV"):
        if len(chosen) == len(LABELS):
            break
        names = jetson_test.DATASET_CLASS_NAMES[dataset]
        label_dir = YOLO_DIR / dataset / "train" / "labels"
        image_dir = YOLO_DIR / dataset / "train" / "images"
        if not label_dir.is_dir():
            continue
        for label_path in sorted(label_dir.glob("*.txt")):
            if len(chosen) == len(LABELS):
                break
            text = label_path.read_text(encoding="utf-8")
            for line in text.splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                class_id = int(float(parts[0]))
                if class_id < 0 or class_id >= len(names):
                    continue
                label = jetson_test.source_label(dataset, names[class_id])
                if label is None or label in chosen:
                    continue
                area = float(parts[3]) * float(parts[4])
                if area < MIN_BOX_AREA or area > MAX_BOX_AREA:
                    continue
                photo = find_photo(image_dir, label_path.stem)
                if photo is None or is_letterboxed(photo):
                    continue
                chosen[label] = photo
                break
    missing = [label for label in LABELS if label not in chosen]
    if missing:
        raise RuntimeError(f"No training photo for: {', '.join(missing)}")
    return [(label, chosen[label]) for label in LABELS]


def is_letterboxed(path: Path) -> bool:
    """Skip clips that are mostly black or green bars around a small picture."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return True
    top = image[:8].astype(np.float32)
    bottom = image[-8:].astype(np.float32)
    return float(top.std()) < 12 or float(bottom.std()) < 12


def ken_burns(image: np.ndarray, local_frame: int, scene_frames: int) -> np.ndarray:
    """Drift across one photo so the camera feels like it is flying forward."""
    height, width = image.shape[:2]
    # Extra scale leaves room to pan inside the photo.
    scale = max(VIEW_W / width, VIEW_H / height) * 1.2
    new_w = max(VIEW_W, int(width * scale))
    new_h = max(VIEW_H, int(height * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    progress = local_frame / max(scene_frames - 1, 1)
    max_x = resized.shape[1] - VIEW_W
    max_y = resized.shape[0] - VIEW_H
    x = int(max_x * (0.08 + 0.84 * progress))
    y = int(max_y * (0.05 + 0.9 * progress))
    return resized[y : y + VIEW_H, x : x + VIEW_W].copy()


def frame_scene(frame_index: int) -> tuple[int, int]:
    """Return which label scene this frame belongs to, and the frame inside it."""
    scene = min(frame_index // SCENE_FRAMES, len(LABELS) - 1)
    local = frame_index - scene * SCENE_FRAMES
    return scene, local


def render_frame(photos: list[np.ndarray], frame_index: int) -> np.ndarray:
    """Build one view, blending into the next photo at the start of a scene."""
    scene, local = frame_scene(frame_index)
    current = ken_burns(photos[scene], local, SCENE_FRAMES)
    if scene == 0 or local >= FADE_FRAMES:
        return current
    previous = ken_burns(photos[scene - 1], SCENE_FRAMES - 1, SCENE_FRAMES)
    # local 0 is still the previous place. By the end of the fade, the new photo is fully there.
    weight = local / FADE_FRAMES
    return cv2.addWeighted(previous, 1.0 - weight, current, weight, 0)


def write_objects(scenes: list[tuple[str, Path]]) -> None:
    """Save which training photo is on screen for each block of frames."""
    with OBJECTS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "start_frame", "end_frame", "file"])
        for index, (label, path) in enumerate(scenes):
            start = index * SCENE_FRAMES
            end = start + SCENE_FRAMES - 1
            writer.writerow([label, start, end, path.name])


def write_video(photos: list[np.ndarray]) -> None:
    """Stream every frame to ffmpeg. The file is 30 seconds at 32 fps."""
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{VIEW_W}x{VIEW_H}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(VIDEO_PATH),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("ffmpeg did not open a frame pipe.")
    try:
        for frame_index in range(FRAME_COUNT):
            frame = render_frame(photos, frame_index)
            process.stdin.write(frame.tobytes())
            if frame_index % 160 == 0:
                print(f"frame {frame_index}/{FRAME_COUNT}")
    finally:
        process.stdin.close()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited with code {return_code}")


def main() -> None:
    """Choose one training photo per label and fly across them."""
    print("Choosing training photos")
    scenes = collect_scenes()
    for label, path in scenes:
        print(f"  {label}: {path.name}")
    photos = []
    for _label, path in scenes:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read {path}")
        photos.append(image)
    write_objects(scenes)
    print("Writing video")
    write_video(photos)
    print(f"Wrote {VIDEO_PATH}")
    print(f"Wrote {OBJECTS_PATH}")


if __name__ == "__main__":
    main()
