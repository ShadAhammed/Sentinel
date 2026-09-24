"""
join_and_detect_video.py

Join the drone clips in Data/video with a short crossfade, then run the
three YOLO weights and EfficientNet-B0 on every frame.

YOLO draws a box. EfficientNet names the crop inside that box.
Both labels are written on the same combined video.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import jetson_test


DATA_DIR = Path(__file__).resolve().parent
VIDEO_DIR = DATA_DIR / "video"
COMBINED_PATH = VIDEO_DIR / "combined_drone.mp4"
DETECT_PATH = VIDEO_DIR / "combined_detections.mp4"

FADE_SECONDS = 0.8
CONFIDENCE = 0.25
VIEW_W = 1280
VIEW_H = 720
FPS = 24

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


def clip_paths() -> list[Path]:
    """Return the four drone clips in the order they were made."""
    clips = [path for path in VIDEO_DIR.glob("*.mp4") if path.name.startswith("Drone_")]
    # The time stamp is the digits at the end of the file name.
    clips.sort(key=lambda path: "".join(ch for ch in path.stem if ch.isdigit()))
    if len(clips) != 4:
        raise RuntimeError(f"Expected 4 drone clips in {VIDEO_DIR}, found {len(clips)}.")
    return clips


def clip_duration(path: Path) -> float:
    """Read the length of one clip in seconds."""
    raw = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        text=True,
    )
    return float(raw.strip())


def fade_offsets(durations: list[float], fade: float) -> list[float]:
    """Start time of each crossfade. Each fade overlaps the end of the clip so far."""
    offsets = []
    played = 0.0
    for index, duration in enumerate(durations[:-1]):
        played += duration
        offset = played - fade * (index + 1)
        if offset <= 0:
            raise RuntimeError("A clip is shorter than the crossfade.")
        offsets.append(offset)
    return offsets


def join_clips(clips: list[Path], durations: list[float]) -> None:
    """Write one mp4. The join is a fade, not a hard cut."""
    offsets = fade_offsets(durations, FADE_SECONDS)
    parts = []
    for index in range(len(clips)):
        parts.append(
            f"[{index}:v]fps={FPS},format=yuv420p,setpts=PTS-STARTPTS[v{index}]"
        )
    current = "v0"
    for index, offset in enumerate(offsets):
        out_name = "vout" if index == len(offsets) - 1 else f"x{index}"
        parts.append(
            f"[{current}][v{index + 1}]xfade=transition=fade:"
            f"duration={FADE_SECONDS}:offset={offset:.3f}[{out_name}]"
        )
        current = out_name
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for path in clips:
        command.extend(["-i", str(path)])
    command.extend(
        [
            "-filter_complex",
            ";".join(parts),
            "-map",
            "[vout]",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(FPS),
            "-movflags",
            "+faststart",
            str(COMBINED_PATH),
        ]
    )
    subprocess.run(command, check=True)


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


def cnn_label_for_crop(model, frame: np.ndarray, xyxy: list[float], device) -> str | None:
    """Classify the picture inside one box. Tiny boxes are skipped."""
    found = cnn_prediction_for_crop(model, frame, xyxy, device)
    if found is None:
        return None
    return found[0]


def draw_box(frame: np.ndarray, box: dict, cnn_name: str | None) -> None:
    """Paint the YOLO box and both names onto the frame."""
    x1, y1, x2, y2 = [int(value) for value in box["xyxy"]]
    color = LABEL_COLOR.get(box["label"], (255, 255, 255))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    yolo_line = f"YOLO {box['label']} {box['confidence']:.2f}"
    cnn_line = "CNN too small" if cnn_name is None else f"CNN {cnn_name}"
    text_y = max(36, y1 - 8)
    cv2.rectangle(frame, (x1, text_y - 34), (x1 + 280, text_y + 6), color, thickness=-1)
    cv2.putText(frame, yolo_line, (x1 + 4, text_y - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(frame, cnn_line, (x1 + 4, text_y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def detect_video() -> None:
    """Read the joined video and write a copy with both models drawn on it."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo_device = "0" if device.type == "cuda" else "cpu"
    print(f"Device: {device}")
    effnet, _classes = jetson_test.load_effnet(DATA_DIR / "EffNet_b0.pt", device)
    yolo_models = []
    from ultralytics import YOLO

    for name in jetson_test.YOLO_FILES:
        path = jetson_test.YOLO_FINAL_DIR / name
        yolo_models.append((name, YOLO(str(path))))

    capture = cv2.VideoCapture(str(COMBINED_PATH))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {COMBINED_PATH}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
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
        str(DETECT_PATH),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("ffmpeg did not open a frame pipe.")

    yolo_counts = {label: 0 for label in jetson_test.CNN_LABELS}
    cnn_counts = {label: 0 for label in jetson_test.CNN_LABELS}
    frames_with_box = 0
    try:
        for frame_index in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[1] != VIEW_W or frame.shape[0] != VIEW_H:
                frame = cv2.resize(frame, (VIEW_W, VIEW_H))
            boxes = yolo_boxes(yolo_models, frame, yolo_device)
            if boxes:
                frames_with_box += 1
            for box in boxes:
                cnn_name = cnn_label_for_crop(effnet, frame, box["xyxy"], device)
                yolo_counts[box["label"]] += 1
                if cnn_name is not None:
                    cnn_counts[cnn_name] += 1
                draw_box(frame, box, cnn_name)
            cv2.putText(
                frame,
                f"boxes {len(boxes)}",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            process.stdin.write(frame.tobytes())
            if frame_index % 40 == 0:
                print(f"frame {frame_index}/{frame_count}  boxes {len(boxes)}")
    finally:
        capture.release()
        process.stdin.close()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited with code {return_code}")

    print(f"Frames with a box: {frames_with_box}/{frame_count}")
    print("YOLO box counts:")
    for label in jetson_test.CNN_LABELS:
        print(f"  {label}: {yolo_counts[label]}")
    print("CNN crop counts:")
    for label in jetson_test.CNN_LABELS:
        print(f"  {label}: {cnn_counts[label]}")
    print(f"Wrote {DETECT_PATH}")


def main() -> None:
    """Join the four clips, then draw YOLO and EfficientNet on the result."""
    clips = clip_paths()
    durations = [clip_duration(path) for path in clips]
    for path, duration in zip(clips, durations):
        print(f"{duration:.2f}s  {path.name}")
    print("Joining with a crossfade")
    join_clips(clips, durations)
    print(f"Wrote {COMBINED_PATH}")
    print("Running YOLO and EfficientNet")
    detect_video()


if __name__ == "__main__":
    main()
