"""
sentinel_demo.py

SENTINEL-X demo window.

On this PC, run it from the video folder:

    python sentinel_demo.py

On the Jetson, the same file lives in the Sentinel folder. Run it there.
Detection and the chat model use that board's GPU.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
import urllib.error
import urllib.request
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk


def demo_dirs() -> tuple[Path, Path]:
    """Return the helper folder and the joined video.

    On this PC the script lives in Data/video and the helpers live in Data.
    On the Jetson the script lives in the Sentinel folder, beside the helpers
    and the video.
    """
    here = Path(__file__).resolve().parent
    if (here / "jetson_test.py").is_file():
        data_dir = here
    else:
        data_dir = here.parent
    return data_dir, here / "combined_drone.mp4"


# Helpers, weights, and the video are found from this file. No drive letter.
DATA_DIR, VIDEO_PATH = demo_dirs()
VIDEO_DIR = VIDEO_PATH.parent
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

# Field colors. Olive black, dull brass. Built for a booth, not a showcase reel.
BG = "#161915"
PANEL = "#22261f"
INK = "#e4e2d8"
MUTED = "#8e9686"
GOLD = "#a3986e"
GOLD_INK = "#1c1a14"
FIELD = "#121510"
CHAT_BG = "#000000"
MATRIX = "#00ff41"
MATRIX_DIM = "#1f8f32"

# Ollama tag. Same weights as qwen2.5:7b, stored as Q4_K_M (4-bit).
CHAT_MODEL = "qwen2.5:7b-instruct-q4_K_M"
CHAT_URL = "http://127.0.0.1:11434/api/chat"
GREETING = (
    "I am Sentinel, version 1.0. "
    "I am a reconnaissance assistant built on three object detectors "
    "and the local language model Qwen2.5 7B. "
    "I watch the aerial demonstration. "
    "YOLO localizes each object. "
    "EfficientNet names the crop. "
    "I report frames seen so far and assess the situation when asked. "
    "A person remains in the decision. "
    "I am here to detect potential dangers."
)
# Chat takes this share of the detection row. The video takes the rest.
CHAT_SHARE = 0.40
SYSTEM_PROMPT = (
    "You are Sentinel, the assistant at the SENTINEL-X stand. "
    "SENTINEL-X is a technology demonstrator for aerial reconnaissance. "
    "A person remains in the decision. You are not a weapon system. "
    "Answer only from the video context below. "
    "While playback is still running, talk only about frames up to the current time. "
    "After playback is complete, you may summarize everything detected in the clip. "
    "This is a demonstration video, not a mapped zone. Do not invent places or coordinates. "
    "If asked for an assessment, say what you think from the categories and counts. "
    "For example, soldiers, artillery, and military vehicles together can look like a war zone. "
    "If little military equipment has been seen, say so. "
    "Do not recommend the use of weapons. "
    "Say what is in the frames. State each category, how many were in view together, and the total. "
    "Do not explain how the counts are made, which model produced them, or how the report is organized. "
    "Answer in English. You already introduced yourself with: "
    + GREETING
)

# Spoken names that should open the crop window. Longer phrases come first.
SHOW_WORDS = ("show", "picture", "image", "crop", "display", "open the", "see the", "see a")
OBJECT_ALIASES = (
    ("military_warship", ("warship", "war ship", "naval")),
    ("military_aircraft", ("military aircraft", "aircraft", "airplane", "plane")),
    ("military_vehicle", ("military vehicle", "vehicle", "tank", "apc")),
    ("camouflage_soldier", ("camouflage soldier", "camouflaged soldier", "camouflage")),
    ("M. Rocket Launcher", ("rocket launcher", "rocket")),
    ("Artillery", ("artillery", "howitzer")),
    ("Missile", ("missile",)),
    ("Radar", ("radar",)),
    ("Soldier", ("soldier", "soldiers", "infantry", "troops")),
    ("trench", ("trench", "trenches")),
)

def split_view(total: int, gap: int = 12) -> tuple[int, int]:
    """Return chat width and video width. Chat is 40 percent of the row."""
    usable = max(total - gap, 2)
    chat = int(round(usable * CHAT_SHARE))
    video = usable - chat
    return chat, video


def dataset_credits() -> list[dict[str, str]]:
    """Return the three detection datasets and where they came from."""
    # Facts below are the published dataset pages plus this project's catalog.
    return [
        {
            "title": "KIIT-MiTA",
            "body": (
                "High-resolution drone images for military object detection.\n"
                "Contributor: Rajesh Chowdhury, KIIT University.\n"
                "Mendeley Data, 23 February 2026. DOI: 10.17632/drjmrf5kk5.1\n"
                "Please also cite: S. Chakrabarty, R. Chatterjee, S. Chakraborty, "
                "S. Roy Shuvo, and R. Chowdhury, \"Drones in Defense: Real-Time "
                "Vision-Based Military Target Surveillance and Tracking,\" "
                "ISACC 2025. DOI: 10.1109/ISACC65211.2025.10969335\n"
                "The dataset page limits use to education and research. "
                "Commercial use is prohibited. Attribution is required.\n"
                "Used here: 1,700 images, 7 classes. Detector: KIIT-MiTA-yolo26s.pt"
            ),
        },
        {
            "title": "military_object_dataset",
            "body": (
                "Military Assets Dataset, 12 classes, YOLOv8 format.\n"
                "Author: RAW (Ryan Madhuwala).\n"
                "Kaggle: rawsi18/military-assets-dataset-12-classes-yolo8-format\n"
                "License: CC BY 4.0\n"
                "26,315 labeled images (21,978 train, 2,941 validation, 1,396 test).\n"
                "Used here: detector military-yolov8n.pt"
            ),
        },
        {
            "title": "MV",
            "body": (
                "Military Vehicle Recognition.\n"
                "Recorded source: Roboflow Universe project "
                "military-vehicle-recognition, version 7.\n"
                "License: CC BY 4.0\n"
                "Original classes: armoured personnel carrier, soldier, tank, "
                "air-fighter, bomber.\n"
                "Air-fighter and bomber were removed in this demonstrator "
                "on 23 September 2026.\n"
                "Used here: 2,702 images, 3 classes. Detector: MV-yolo26s.pt"
            ),
        },
    ]


def fit_frame(frame: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    """Shrink a frame so it fits the video panel. Keep the aspect ratio."""
    height, width = frame.shape[:2]
    if width < 1 or height < 1:
        return frame
    scale = min(max_w / width, max_h / height)
    if scale >= 1:
        return frame
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def make_logo(size: int = 64) -> Image.Image:
    """Draw the SENTINEL-X mark. A gold diamond on a dark tile."""
    image = Image.new("RGB", (size, size), "#1c241c")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((1, 1, size - 2, size - 2), radius=12, outline=GOLD, width=2)
    center = size // 2
    # Outer diamond is the watch boundary. Inner diamond is the mark.
    draw.polygon(
        [(center, 12), (size - 12, center), (center, size - 12), (12, center)],
        outline=GOLD,
    )
    draw.polygon(
        [(center, 22), (size - 22, center), (center, size - 22), (22, center)],
        fill=GOLD,
    )
    return image


def confidence_percent(score: float) -> str:
    """Turn a 0-1 score into a whole percent."""
    return f"{int(round(score * 100))}%"


def final_label(yolo_label: str, cnn_label: str | None) -> str:
    """Keep the EfficientNet name when the crop could be classified."""
    # YOLO only localizes. If the two names differ, EfficientNet is the decision.
    if cnn_label is None:
        return yolo_label
    return cnn_label


def box_iou(box_a: list[float], box_b: list[float]) -> float:
    """Overlap of two boxes, from 0 to 1."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def efficientnet_hits(candidates: list[dict]) -> list[dict]:
    """One EfficientNet name per object. YOLO-only boxes are not observations."""
    named = []
    for item in candidates:
        # No EfficientNet name means the box stays out of the count.
        if not item.get("cnn_name"):
            continue
        named.append(item)
    ordered = sorted(named, key=lambda item: item["cnn_confidence"], reverse=True)
    kept = []
    for item in ordered:
        # Two YOLO models on the same object become one EfficientNet observation.
        if any(box_iou(item["xyxy"], old["xyxy"]) > 0.5 for old in kept):
            continue
        kept.append(item)
    return kept


def parse_gpu_line(line: str) -> tuple[int, int, int] | None:
    """Read utilization percent, used MiB, and free MiB from one nvidia-smi line."""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 3:
        return None
    try:
        return int(float(parts[0])), int(float(parts[1])), int(float(parts[2]))
    except ValueError:
        return None


def format_gpu_free(free_mib: int) -> str:
    """Show free GPU memory in gigabytes."""
    return f"{free_mib / 1024:.1f} GB"


def read_soc_gpu() -> tuple[int, int, int] | None:
    """Read the Jetson GPU load and unified free memory.

    nvidia-smi on the Orin leaves utilization and memory blank.
    The load file is 0 to 1000, which is 0 to 100 percent.
    Free memory is the SoC memory the GPU shares with the CPU.
    """
    load_file = Path("/sys/devices/platform/gpu.0/load")
    if not load_file.is_file():
        return None
    try:
        utilized = int(load_file.read_text().strip()) // 10
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    total = None
    available = None
    for line in meminfo.splitlines():
        if line.startswith("MemTotal:"):
            total = int(line.split()[1]) // 1024
        elif line.startswith("MemAvailable:"):
            available = int(line.split()[1]) // 1024
    if total is None or available is None:
        return None
    used = max(0, total - available)
    return utilized, used, available


def read_gpu() -> tuple[int, int, int] | None:
    """Ask nvidia-smi how busy the GPU is and how much memory is free."""
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return read_soc_gpu()
    lines = raw.strip().splitlines()
    if not lines:
        return read_soc_gpu()
    parsed = parse_gpu_line(lines[0])
    # The Orin report is blank. Use the SoC load file instead.
    if parsed is None:
        return read_soc_gpu()
    return parsed


def format_clock(seconds: float) -> str:
    """Show a player time as minutes and seconds."""
    whole = max(0, int(seconds))
    minutes, rest = divmod(whole, 60)
    return f"{minutes:02d}:{rest:02d}"


def parse_clock(clock: str) -> float:
    """Turn a player time like 01:15 back into seconds."""
    minutes, rest = clock.split(":")
    return int(minutes) * 60 + int(rest)


def probe_video(path: Path) -> tuple[float, float]:
    """Return frames per second and length in seconds. Zeros if the file is missing."""
    if not path.is_file():
        return 24.0, 0.0
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 24.0)
    frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    capture.release()
    if fps <= 0:
        fps = 24.0
    return fps, frames / fps


def cnn_caption(cnn_name: str | None, cnn_confidence: float | None) -> str | None:
    """The only words drawn on a detection box. Tiny crops get no label."""
    if cnn_name is None or cnn_confidence is None:
        return None
    return f"{cnn_name} {confidence_percent(cnn_confidence)}"


def draw_live_box(
    frame: np.ndarray,
    box: dict,
    cnn_name: str | None,
    cnn_confidence: float | None,
    decided_label: str,
) -> None:
    """Draw the box and the EfficientNet name only. Skip crops the CNN cannot read."""
    line = cnn_caption(cnn_name, cnn_confidence)
    if line is None:
        return
    import join_and_detect_video

    x1, y1, x2, y2 = [int(value) for value in box["xyxy"]]
    color = join_and_detect_video.LABEL_COLOR.get(decided_label, (255, 255, 255))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(line, font, 0.55, 1)
    block_h = text_h + 10
    if y1 > block_h + 4:
        top = y1 - block_h - 4
    else:
        top = min(y2 + 4, max(0, frame.shape[0] - block_h - 1))
    left = x1
    if left + text_w + 8 > frame.shape[1]:
        left = max(0, frame.shape[1] - text_w - 8)
    cv2.rectangle(frame, (left, top), (left + text_w + 8, top + block_h), color, thickness=-1)
    cv2.putText(frame, line, (left + 4, top + text_h + 2), font, 0.55, (0, 0, 0), 1)


def asked_object(text: str) -> str | None:
    """Return a class name when the visitor asks to see that object."""
    lowered = text.lower()
    if not any(word in lowered for word in SHOW_WORDS):
        return None
    for label, words in OBJECT_ALIASES:
        for word in words:
            if word in lowered:
                return label
    return None


def format_detection_context(report: dict) -> str:
    """Turn frames up to the on-screen time into the text the assistant may use."""
    frame_log = report.get("frame_log", [])
    if not frame_log:
        return "No frame has been read yet. There are no object counts."
    # The playhead is the frame on screen. Later logged frames stay out,
    # including after the reader has already reached the end of the file.
    limit = float(report.get("playhead", 0.0))
    used = [item for item in frame_log if item["seconds"] <= limit + 0.05]
    if report.get("finished"):
        status = "Playback is complete. You may talk about everything detected in the clip."
    else:
        status = (
            f"Playback is still running at {report.get('clock', '00:00')}. "
            "Use only the frames listed here. Do not guess about later frames."
        )
    if not used:
        return status + " No processed frame is at or before this time."
    totals: dict[str, int] = {}
    frames: dict[str, int] = {}
    maximums: dict[str, int] = {}
    for item in used:
        counts: dict[str, int] = {}
        for name, _confidence in item["hits"]:
            counts[name] = counts.get(name, 0) + 1
            totals[name] = totals.get(name, 0) + 1
        for name, count in counts.items():
            frames[name] = frames.get(name, 0) + 1
            maximums[name] = max(maximums.get(name, 0), count)
    latest = used[-1]
    lines = [
        status,
        f"Video time: {report.get('clock', latest['clock'])}",
        "This frame:",
    ]
    grouped: dict[str, list[float]] = {}
    for name, confidence in latest["hits"]:
        grouped.setdefault(name, []).append(confidence)
    if not grouped:
        lines.append("- no drawn objects")
    for name in sorted(grouped):
        scores = grouped[name]
        lines.append(
            f"- {name}: {len(scores)}, highest confidence {confidence_percent(max(scores))}"
        )
    lines.append("Categories in the frames so far:")
    names = sorted(maximums, key=lambda name: (-maximums[name], name))
    if not names:
        lines.append("- none yet")
    for name in names:
        lines.append(
            f"- {name}: at most {maximums[name]} at once, "
            f"in {frames[name]} frames, "
            f"{totals[name]} observations in total"
        )
    saved = report.get("crops", [])
    visible_crops = []
    for name, confidence, clock in saved:
        # A clearer crop from later in the clip must not leak into this answer.
        if parse_clock(clock) <= limit + 0.05:
            visible_crops.append((name, confidence, clock))
    if visible_crops:
        lines.append("Clearest crop saved for a picture window:")
        for name, confidence, clock in visible_crops:
            lines.append(f"- {name} at {clock}, {confidence_percent(confidence)}")
    return "\n".join(lines)


def stream_chat(history: list[dict[str, str]], context: str):
    """Ask Qwen on the local Ollama server and yield each text piece."""
    # The video tally is the only retrieved context for this answer.
    system = SYSTEM_PROMPT + "\n\nVideo context:\n" + context
    payload = {
        "model": CHAT_MODEL,
        "messages": [{"role": "system", "content": system}] + history,
        "stream": True,
        "keep_alive": "30m",
        "options": {"temperature": 0.3, "num_ctx": 2048, "num_predict": 360},
    }
    request = urllib.request.Request(
        CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        for raw in response:
            if not raw.strip():
                continue
            data = json.loads(raw.decode("utf-8"))
            if data.get("error"):
                raise RuntimeError(str(data["error"]))
            piece = data.get("message", {}).get("content", "")
            if piece:
                yield piece
            if data.get("done"):
                break


def warm_chat_model() -> None:
    """Load the 4-bit weights onto the GPU before the first question."""
    payload = {
        "model": CHAT_MODEL,
        "messages": [{"role": "user", "content": "Bereit?"}],
        "stream": False,
        "keep_alive": "30m",
        "options": {"num_predict": 1, "num_ctx": 512},
    }
    request = urllib.request.Request(
        CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        response.read()


class DemoWindow:
    """One window. The play button, the picture, and the detector share it."""

    def __init__(self, root: tk.Tk, preload: bool = False) -> None:
        self.root = root
        # The live demo loads the detectors while the window opens, so Play is not the first load.
        self.preload = preload
        self.models = None
        self.models_error: Exception | None = None
        self.models_ready = threading.Event()
        self.root.title("SENTINEL-X")
        self.root.configure(bg=BG)
        self.root.geometry("1180x760")
        self.root.minsize(980, 680)

        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.playing = threading.Event()
        self.restart = False
        self.ended = False
        self.worker: threading.Thread | None = None
        self.latest: dict | None = None
        self.gpu: tuple[int, int, int] | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.view_w = 640
        self.view_h = 460
        self.fps, self.duration = probe_video(VIDEO_PATH)
        self.scrubbing = False
        self.seek_seconds: float | None = None
        # Chat context uses the frame on screen, not the last frame the detector stored.
        self.shown_seconds = 0.0
        self.view_token = 0
        self.chat_busy = False
        self.chat_history = [{"role": "assistant", "content": GREETING}]
        self.chat_queue: queue.Queue = queue.Queue()
        self.report = self._blank_report()
        self.crops: dict[str, tuple[float, np.ndarray, str]] = {}
        self.picture_windows: list[tk.Toplevel] = []
        # The picture plays at video speed. The detector labels the freshest frame on its own thread.
        self.det_thread: threading.Thread | None = None
        self.det_input: tuple[np.ndarray, int, int] | None = None
        self.overlay_items: list[dict] = []
        self.overlay_token = -1

        self._build()
        self.gpu_thread = threading.Thread(target=self._gpu_loop, daemon=True)
        self.gpu_thread.start()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.intro_index = 0
        self.root.after(30, self._pump)
        # Detectors first. Qwen starts after they are warm so the two do not fight for the GPU.
        if self.preload and VIDEO_PATH.is_file():
            threading.Thread(target=self._preload_models, daemon=True).start()
        else:
            self.root.after(400, self._open_chat_model)
        self.root.after(840, self._begin_intro)

    def _build(self) -> None:
        """Lay out the header, the two tabs, and the play control."""
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=22, pady=(16, 8))

        logo = ImageTk.PhotoImage(make_logo(64))
        logo_label = tk.Label(header, image=logo, bg=BG)
        logo_label.image = logo
        logo_label.pack(side="left")

        titles = tk.Frame(header, bg=BG)
        titles.pack(side="left", padx=14)
        tk.Label(
            titles,
            text="SENTINEL-X",
            font=("Segoe UI", 22, "bold"),
            fg=INK,
            bg=BG,
        ).pack(anchor="w")
        tk.Label(
            titles,
            text="Aerial reconnaissance demonstrator",
            font=("Segoe UI", 11),
            fg=MUTED,
            bg=BG,
        ).pack(anchor="w")

        tk.Label(
            header,
            text="Human in the loop",
            font=("Segoe UI", 10),
            fg=GOLD,
            bg=BG,
        ).pack(side="right")

        tk.Frame(self.root, bg=GOLD, height=2).pack(fill="x", padx=22, pady=(0, 10))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Sentinel.TNotebook", background=BG, borderwidth=0)
        style.configure(
            "Player.Horizontal.TScale",
            background=BG,
            troughcolor="#3a4034",
            sliderlength=18,
        )
        style.configure(
            "Sentinel.TNotebook.Tab",
            background=PANEL,
            foreground=INK,
            padding=(18, 8),
            font=("Segoe UI", 11),
        )
        style.map(
            "Sentinel.TNotebook.Tab",
            background=[("selected", "#2c3328")],
            foreground=[("selected", GOLD)],
        )

        self.tabs = ttk.Notebook(self.root, style="Sentinel.TNotebook")
        self.tabs.pack(fill="both", expand=True, padx=22, pady=(0, 12))

        detection = tk.Frame(self.tabs, bg=BG)
        credit = tk.Frame(self.tabs, bg=BG)
        self.tabs.add(detection, text="Detection")
        self.tabs.add(credit, text="Credit")

        self._build_detection(detection)
        self._build_credit(credit)

    def _build_detection(self, parent: tk.Frame) -> None:
        """Video on the left. The names found on this frame sit on the right."""
        # Pack the player first so the picture cannot push the timeline off screen.
        self.status = tk.Label(
            parent,
            text=self._ready_status(),
            font=("Segoe UI", 10),
            fg=MUTED,
            bg=BG,
            anchor="w",
        )
        self.status.pack(side="bottom", fill="x", padx=8, pady=(0, 8))

        bar = tk.Frame(parent, bg=BG)
        bar.pack(side="bottom", fill="x", padx=8, pady=(0, 4))
        self.play_button = tk.Button(
            bar,
            text="Play",
            font=("Segoe UI", 11, "bold"),
            bg=GOLD,
            fg=GOLD_INK,
            activebackground="#b5aa80",
            activeforeground=GOLD_INK,
            relief="flat",
            padx=16,
            pady=6,
            cursor="hand2",
            command=self.on_play,
        )
        self.play_button.pack(side="left")
        self.time_now = tk.Label(
            bar,
            text="00:00",
            font=("Consolas", 11),
            fg=INK,
            bg=BG,
            width=6,
        )
        self.time_now.pack(side="left", padx=(12, 6))
        span = self.duration if self.duration > 0 else 1.0
        self.timeline = ttk.Scale(
            bar,
            from_=0,
            to=span,
            orient="horizontal",
            style="Player.Horizontal.TScale",
            command=self._on_timeline,
        )
        self.timeline.pack(side="left", fill="x", expand=True, padx=4)
        self.timeline.bind("<ButtonPress-1>", self._scrub_start)
        self.timeline.bind("<ButtonRelease-1>", self._scrub_end)
        self.time_end = tk.Label(
            bar,
            text=format_clock(self.duration),
            font=("Consolas", 11),
            fg=MUTED,
            bg=BG,
            width=6,
        )
        self.time_end.pack(side="left", padx=(6, 0))

        body = tk.Frame(parent, bg=BG)
        self.body = body
        body.pack(fill="both", expand=True, padx=8, pady=8)
        body.bind("<Configure>", self._on_body_resize)

        stage = tk.Frame(body, bg="#000000")
        stage.pack(side="left", fill="both", expand=True)
        stage.bind("<Configure>", self._on_stage_resize)
        self.stage = stage
        self.picture = tk.Label(
            stage,
            text="Press Play.\nDrag the timeline to move.",
            font=("Segoe UI", 14),
            fg=MUTED,
            bg="#000000",
            justify="center",
        )
        self.picture.pack(fill="both", expand=True)

        side = tk.Frame(body, bg=CHAT_BG, width=split_view(1000)[0])
        self.side = side
        side.pack(side="right", fill="y", padx=(12, 0))
        side.pack_propagate(False)
        tk.Label(
            side,
            text="Sentinel",
            font=("Consolas", 14, "bold"),
            fg=MATRIX,
            bg=CHAT_BG,
        ).pack(anchor="w", padx=14, pady=(14, 0))
        self.chat_status = tk.Label(
            side,
            text="Qwen2.5 7B Q4",
            font=("Consolas", 9),
            fg=MATRIX_DIM,
            bg=CHAT_BG,
            anchor="w",
        )
        self.chat_status.pack(anchor="w", padx=14, pady=(0, 8))

        # GPU and the question line stay visible under a long conversation.
        self.gpu_free = tk.Label(
            side,
            text="GPU free        --",
            font=("Consolas", 10),
            fg=MATRIX_DIM,
            bg=CHAT_BG,
            anchor="w",
        )
        self.gpu_free.pack(side="bottom", fill="x", padx=14, pady=(0, 12))
        self.gpu_util = tk.Label(
            side,
            text="GPU utilized    --",
            font=("Consolas", 10),
            fg=MATRIX_DIM,
            bg=CHAT_BG,
            anchor="w",
        )
        self.gpu_util.pack(side="bottom", fill="x", padx=14, pady=(8, 0))

        ask = tk.Frame(side, bg=CHAT_BG)
        ask.pack(side="bottom", fill="x", padx=12, pady=(0, 4))
        tk.Label(
            ask,
            text="Ask",
            font=("Consolas", 10),
            fg=MATRIX,
            bg=CHAT_BG,
        ).pack(side="left", padx=(0, 8))
        self.send_button = tk.Button(
            ask,
            text="Send",
            font=("Consolas", 10, "bold"),
            bg="#021a02",
            fg=MATRIX,
            activebackground="#063006",
            activeforeground=MATRIX,
            relief="flat",
            padx=12,
            pady=4,
            cursor="hand2",
            command=self.on_send,
        )
        self.send_button.pack(side="right")
        self.question = tk.Entry(
            ask,
            font=("Consolas", 11),
            bg=CHAT_BG,
            fg=MATRIX,
            insertbackground=MATRIX,
            relief="flat",
        )
        self.question.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 8))
        self.question.bind("<Return>", self._on_enter)
        self.question.configure(state="disabled")
        self.send_button.configure(state="disabled")

        self.chat = tk.Text(
            side,
            bg=CHAT_BG,
            fg=MATRIX,
            font=("Consolas", 11),
            height=8,
            relief="flat",
            wrap="word",
            padx=12,
            pady=8,
            highlightthickness=0,
        )
        self.chat.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.chat.tag_configure("who", font=("Consolas", 9, "bold"), foreground=MATRIX)
        self.chat.tag_configure("body", font=("Consolas", 11), foreground=MATRIX)
        self.chat.configure(state="disabled")

    def _on_body_resize(self, event: tk.Event) -> None:
        """Keep the chat at 40 percent of the row and the video at 60 percent."""
        if event.widget is not self.body:
            return
        chat, _video = split_view(event.width)
        self.side.configure(width=chat)

    def _on_stage_resize(self, event: tk.Event) -> None:
        """Remember the video panel size so each frame is scaled to fit."""
        if event.width > 20 and event.height > 20:
            self.view_w = event.width
            self.view_h = event.height

    def _build_credit(self, parent: tk.Frame) -> None:
        """Show one card for each of the three source datasets."""
        intro = tk.Label(
            parent,
            text="These three public datasets trained the detectors in this demonstrator.",
            font=("Segoe UI", 11),
            fg=MUTED,
            bg=BG,
            anchor="w",
        )
        intro.pack(fill="x", padx=12, pady=(12, 8))

        holder = tk.Frame(parent, bg=BG)
        holder.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        canvas = tk.Canvas(holder, bg=BG, highlightthickness=0)
        scroll = ttk.Scrollbar(holder, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        cards = tk.Frame(canvas, bg=BG)
        window_id = canvas.create_window((0, 0), window=cards, anchor="nw")

        for record in dataset_credits():
            card = tk.Frame(cards, bg=PANEL)
            card.pack(fill="x", pady=6)
            tk.Frame(card, bg=GOLD, width=4).pack(side="left", fill="y")
            inner = tk.Frame(card, bg=PANEL)
            inner.pack(fill="x", padx=14, pady=12)
            tk.Label(
                inner,
                text=record["title"],
                font=("Segoe UI", 14, "bold"),
                fg=GOLD,
                bg=PANEL,
                anchor="w",
            ).pack(fill="x")
            tk.Label(
                inner,
                text=record["body"],
                font=("Segoe UI", 10),
                fg=INK,
                bg=PANEL,
                justify="left",
                anchor="w",
                wraplength=860,
            ).pack(fill="x", pady=(6, 0))

        def _fit_cards(_event: object) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfigure(window_id, width=canvas.winfo_width())

        cards.bind("<Configure>", _fit_cards)
        canvas.bind("<Configure>", _fit_cards)

    def tab_names(self) -> list[str]:
        """Return the tab titles. Tests use this."""
        return [self.tabs.tab(tab_id, "text") for tab_id in self.tabs.tabs()]

    def _ready_status(self) -> str:
        """Tell the operator whether the joined video is in this folder."""
        if not VIDEO_PATH.is_file():
            return "combined_drone.mp4 is not in this folder."
        if self.preload and not self.models_ready.is_set():
            return "Loading YOLO and EfficientNet..."
        return "Ready. Play runs YOLO and EfficientNet on the joined video."

    def on_play(self) -> None:
        """Start live detection, or pause it if it is already running."""
        if not VIDEO_PATH.is_file():
            self.status.configure(text=self._ready_status())
            return
        if self.playing.is_set():
            self.playing.clear()
            self.play_button.configure(text="Play")
            self.status.configure(text="Paused")
            return

        self.playing.set()
        self.play_button.configure(text="Pause")
        if self.ended:
            self.restart = True
            self.ended = False
        if self.worker is None or not self.worker.is_alive():
            if not self._detectors_ready():
                self.status.configure(text="Loading YOLO and EfficientNet...")
            self.worker = threading.Thread(target=self._detect_loop, daemon=True)
            self.worker.start()

    def _detectors_ready(self) -> bool:
        """True once the background load has put the weights on the GPU."""
        return self.preload and self.models_ready.is_set() and self.models is not None

    def _detect_loop(self) -> None:
        """Play the clip at video speed. A separate thread labels frames and boxes appear over them."""
        capture = cv2.VideoCapture(str(VIDEO_PATH))
        if not capture.isOpened():
            with self.lock:
                self.latest = {"error": f"Could not open {VIDEO_PATH.name}"}
            self.playing.clear()
            return

        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        try:
            # Show the opening frame while the weights load, so Play is not a blank wait.
            if not self._detectors_ready():
                ok, first = capture.read()
                if ok:
                    with self.lock:
                        token = self.view_token
                    self._publish_frame(first.copy(), 1, total, 0, 0.0, "GPU", token, loading=True)
            try:
                models = self._models_for_playback()
            except Exception as exc:
                with self.lock:
                    self.latest = {"error": str(exc)}
                self.playing.clear()
                return
            if models is None or self.stop.is_set():
                return
            # Start over so the labels cover the clip from the beginning.
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._start_detector(models, total)
            self._play_loop(capture, models, total)
        finally:
            capture.release()

    def _start_detector(self, models: dict, total: int) -> None:
        """Run the detector on its own thread. Only one runs at a time."""
        if self.det_thread is not None and self.det_thread.is_alive():
            return
        self.det_thread = threading.Thread(
            target=self._detector_worker, args=(models, total), daemon=True
        )
        self.det_thread.start()

    def _play_loop(self, capture: "cv2.VideoCapture", models: dict, total: int) -> None:
        """Read and show frames on the video clock. Hand each frame to the detector."""
        period = (1.0 / self.fps) if self.fps > 0 else 0.0
        # The next frame is due at this wall-clock time. A jump or a resume resets it.
        target = time.perf_counter()
        last_shown = target
        while not self.stop.is_set():
            # A drag on the timeline jumps here even while playback is paused.
            seek = None
            with self.lock:
                if self.seek_seconds is not None:
                    seek = self.seek_seconds
                    self.seek_seconds = None
                restart = self.restart
                if restart:
                    self.restart = False
            if seek is not None:
                capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, seek) * 1000.0)
                self.ended = False
                target = time.perf_counter()
            elif restart:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._reset_report()
                target = time.perf_counter()
            elif not self.playing.is_set():
                time.sleep(0.05)
                target = time.perf_counter()
                continue
            ok, frame = capture.read()
            if not ok:
                self.playing.clear()
                self.ended = True
                with self.lock:
                    self.latest = {
                        "done": True,
                        "index": total,
                        "total": total,
                        "token": self.view_token,
                    }
                continue

            index = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
            with self.lock:
                token = self.view_token
            # Give the detector the freshest frame. An older one that it has not started is dropped.
            self._submit_detection(frame, index, token)
            # Draw the boxes the detector produced most recently onto this picture.
            shown = frame.copy()
            drawn = self._draw_overlay(shown, token)
            now = time.perf_counter()
            fps = 1.0 / max(now - last_shown, 1e-6)
            last_shown = now
            self._publish_frame(
                shown, index, total, drawn, fps, models["device_label"], token
            )
            # Hold each frame to the video clock so the picture plays at its real speed.
            target += period
            self._sleep_until(target)

    def _sleep_until(self, deadline: float) -> None:
        """Wait until this wall-clock time, but wake early on stop, pause, or a seek."""
        while time.perf_counter() < deadline:
            if self.stop.is_set() or not self.playing.is_set():
                return
            with self.lock:
                if self.seek_seconds is not None:
                    return
            time.sleep(0.005)

    def _submit_detection(self, frame: np.ndarray, index: int, token: int) -> None:
        """Hand one frame to the detector thread, replacing any frame it has not started."""
        with self.lock:
            self.det_input = (frame.copy(), index, token)

    def _detector_worker(self, models: dict, total: int) -> None:
        """Label the freshest frame, store its boxes, and log the hits."""
        while not self.stop.is_set():
            with self.lock:
                job = self.det_input
                self.det_input = None
            if job is None:
                time.sleep(0.005)
                continue
            frame, index, token = job
            with self.lock:
                stale = token != self.view_token or self.seek_seconds is not None
            if stale:
                continue
            items = self._detect_items(frame, models)
            hits = [(item["cnn_name"], item["cnn_confidence"], item["crop"]) for item in items]
            with self.lock:
                if token != self.view_token or self.seek_seconds is not None:
                    continue
                # These boxes are drawn on the picture until the next frame is labelled.
                self.overlay_items = items
                self.overlay_token = token
            self._add_hits(hits, index, token)

    def _draw_overlay(self, frame: np.ndarray, token: int) -> int:
        """Draw the most recent boxes on this frame. Old boxes from before a seek are skipped."""
        with self.lock:
            if self.overlay_token != token:
                return 0
            items = list(self.overlay_items)
        for item in items:
            draw_live_box(frame, item, item["cnn_name"], item["cnn_confidence"], item["decided"])
        return len(items)

    def _load_models(self) -> dict:
        """Open the three YOLO weights and EfficientNet once."""
        import torch
        from ultralytics import YOLO

        import jetson_test

        # The Orin GPU is cuda:0. These checkpoints do not run on the DLA, and CPU is refused.
        if not torch.cuda.is_available():
            raise RuntimeError("Sentinel runs on the board GPU. CUDA is not available.")
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        effnet, _classes = jetson_test.load_effnet(jetson_test.EFFNET_PATH, device)
        # Fixed 640 input. Let cuDNN pick a fast algorithm and reuse it.
        torch.backends.cudnn.benchmark = True
        yolo_models = []
        any_engine = False
        for name in jetson_test.YOLO_FILES:
            path = jetson_test.YOLO_FINAL_DIR / name
            engine = path.with_suffix(".engine")
            # Prefer the engine built on this board. A missing engine stays on the .pt file.
            weight = engine if engine.is_file() else path
            if weight.suffix == ".engine":
                any_engine = True
                # The engine file does not store the task. These three weights are detectors.
                model = YOLO(str(weight), task="detect")
            else:
                model = YOLO(str(weight))
            # An engine is already fused. Conv and batch-norm fuse is only for a .pt file.
            if weight.suffix != ".engine":
                try:
                    model.fuse()
                except Exception:
                    pass
            yolo_models.append((name, model))
        loaded = {
            "device": device,
            "yolo_device": "0",
            "device_label": gpu_name,
            "effnet": effnet,
            "yolo_models": yolo_models,
            # The engine is already FP16. Asking predict for half again is only for a .pt file.
            "half": not any_engine,
        }
        self._warm_detectors(loaded)
        return loaded

    def _preload_models(self) -> None:
        """Load and warm the detectors before Play. Qwen waits until that is done."""
        try:
            self.models = self._load_models()
        except Exception as exc:
            self.models_error = exc
        finally:
            self.models_ready.set()
        if self.stop.is_set():
            return
        self.chat_queue.put(("loading", None))
        self._warm_chat()

    def _models_for_playback(self) -> dict | None:
        """Use the weights loaded at startup, or load them now."""
        if not self.preload:
            return self._load_models()
        while not self.models_ready.wait(0.1):
            if self.stop.is_set():
                return None
        if self.stop.is_set():
            return None
        if self.models_error is not None:
            raise self.models_error
        return self.models

    def _warm_detectors(self, models: dict) -> None:
        """Run one blank frame so the first Play frame is not the cold start."""
        import torch

        import join_and_detect_video

        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        # A TensorRT engine is already FP16. A .pt file still tries FP16, then FP32.
        if models.get("half"):
            try:
                join_and_detect_video.yolo_boxes(
                    models["yolo_models"], blank, models["yolo_device"], half=True
                )
            except Exception:
                join_and_detect_video.yolo_boxes(
                    models["yolo_models"], blank, models["yolo_device"], half=False
                )
                models["half"] = False
        else:
            join_and_detect_video.yolo_boxes(
                models["yolo_models"], blank, models["yolo_device"], half=False
            )
        dummy = torch.zeros(1, 3, 224, 224, device=models["device"])
        with torch.inference_mode():
            models["effnet"](dummy)

    def _publish_frame(
        self,
        frame: np.ndarray,
        index: int,
        total: int,
        boxes: int,
        fps: float,
        device: str,
        token: int,
        loading: bool = False,
    ) -> None:
        """Hand one picture to the window thread."""
        with self.lock:
            if token != self.view_token or self.seek_seconds is not None:
                return
            self.latest = {
                "frame": frame,
                "index": index,
                "total": total,
                "boxes": boxes,
                "fps": fps,
                "device": device,
                "token": token,
                "loading": loading,
            }

    def _detect_items(self, frame: np.ndarray, models: dict) -> list[dict]:
        """YOLO places each box, EfficientNet names it. Return the boxes and crops, no drawing."""
        import join_and_detect_video

        boxes = join_and_detect_video.yolo_boxes(
            models["yolo_models"],
            frame,
            models["yolo_device"],
            half=bool(models.get("half")),
        )
        candidates = []
        for box in boxes:
            found = join_and_detect_video.cnn_prediction_for_crop(
                models["effnet"], frame, box["xyxy"], models["device"]
            )
            if found is None:
                # YOLO placed a box, but EfficientNet did not name it. Do not count it.
                continue
            cnn_name, cnn_confidence = found
            candidates.append(
                {
                    "xyxy": box["xyxy"],
                    "cnn_name": cnn_name,
                    "cnn_confidence": cnn_confidence,
                    "decided": final_label(box["label"], cnn_name),
                }
            )
        items = []
        for item in efficientnet_hits(candidates):
            x1, y1, x2, y2 = [int(value) for value in item["xyxy"]]
            height, width = frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            item["crop"] = frame[y1:y2, x1:x2].copy()
            items.append(item)
        return items

    def _blank_report(self) -> dict:
        """An empty detection log."""
        return {"clock": "00:00", "playhead": 0.0, "frame_log": []}

    def _reset_report(self) -> None:
        """Clear the tally when playback starts again from the beginning."""
        with self.lock:
            self.report = self._blank_report()
            self.crops = {}
            self.shown_seconds = 0.0
            self.latest = None
            self.view_token += 1

    def _add_hits(self, hits: list[tuple[str, float, np.ndarray]], index: int, token: int) -> None:
        """Fold one shown generation of frame into the log, and keep clearer crops."""
        if self.fps > 0:
            seconds = max(0.0, (index - 1) / self.fps)
        else:
            seconds = 0.0
        clock = format_clock(seconds)
        logged = [(name, confidence) for name, confidence, _crop in hits]
        with self.lock:
            # This frame belongs to an older seek. Leave it out of the chat log.
            if token != self.view_token or self.seek_seconds is not None:
                return
            self.report["clock"] = clock
            self.report["playhead"] = seconds
            self.report["frame_log"].append(
                {"seconds": seconds, "clock": clock, "hits": logged}
            )
            for name, confidence, crop in hits:
                history = self.crops.get(name)
                # Keep the older crop when a later frame is clearer, so a rewind can still use it.
                if history is None:
                    self.crops[name] = [(confidence, crop, clock, seconds)]
                elif confidence > history[-1][0]:
                    history.append((confidence, crop, clock, seconds))

    def _cut_log(self, seconds: float) -> None:
        """Drop every detection later than the frame the operator just moved to."""
        self.view_token += 1
        self.seek_seconds = seconds
        self.shown_seconds = seconds
        self.report["playhead"] = seconds
        self.report["clock"] = format_clock(seconds)
        self.report["frame_log"] = [
            item for item in self.report["frame_log"] if item["seconds"] <= seconds + 0.05
        ]
        trimmed = {}
        for name, history in self.crops.items():
            kept = [item for item in history if item[3] <= seconds + 0.05]
            if kept:
                trimmed[name] = kept
        self.crops = trimmed
        self.latest = None

    def _video_context(self) -> str:
        """Copy frames up to the picture on screen. Later detections stay out."""
        with self.lock:
            limit = self.shown_seconds
            at_end = self.duration > 0 and limit >= self.duration - 0.2
            crops = []
            for name, history in self.crops.items():
                eligible = [item for item in history if item[3] <= limit + 0.05]
                if not eligible:
                    continue
                confidence, _crop, clock, _seconds = max(eligible, key=lambda item: item[0])
                crops.append((name, confidence, clock))
            snapshot = {
                "finished": self.ended and at_end,
                "clock": format_clock(limit),
                "playhead": limit,
                "frame_log": [
                    {
                        "seconds": item["seconds"],
                        "clock": item["clock"],
                        "hits": list(item["hits"]),
                    }
                    for item in self.report["frame_log"]
                ],
                "crops": crops,
            }
        return format_detection_context(snapshot)

    def _pump(self) -> None:
        """Copy the newest detected frame onto the window."""
        if self.stop.is_set():
            return
        with self.lock:
            item = None if self.latest is None else dict(self.latest)
        self._refresh_gpu()
        self._drain_chat()
        self._note_detector_load()
        if item is not None:
            self._show(item)
        self.root.after(30, self._pump)

    def _note_detector_load(self) -> None:
        """Replace the loading line once the weights are on the GPU and playback is idle."""
        if not self.preload or self.playing.is_set() or self.latest is not None:
            return
        if not self.models_ready.is_set():
            return
        if self.models_error is not None:
            self.status.configure(text=str(self.models_error))
            return
        if str(self.status.cget("text")).startswith("Loading"):
            self.status.configure(text=self._ready_status())

    def _gpu_loop(self) -> None:
        """Keep a fresh GPU reading while the window is open."""
        while not self.stop.is_set():
            reading = read_gpu()
            with self.lock:
                self.gpu = reading
            self.stop.wait(0.5)

    def _refresh_gpu(self) -> None:
        """Write utilization and free memory into the right-hand panel."""
        with self.lock:
            reading = self.gpu
        if reading is None:
            self.gpu_util.configure(text="GPU utilized    --")
            self.gpu_free.configure(text="GPU free        --")
            return
        utilized, _used_mib, free_mib = reading
        self.gpu_util.configure(text=f"GPU utilized    {utilized}%")
        self.gpu_free.configure(text=f"GPU free        {format_gpu_free(free_mib)}")

    def _show(self, item: dict) -> None:
        """Paint one update. Errors and the finished state stay as text."""
        if "error" in item:
            self.play_button.configure(text="Play")
            self.status.configure(text=item["error"])
            return
        with self.lock:
            if item.get("token") != self.view_token:
                return
        if item.get("done"):
            self.play_button.configure(text="Play")
            self.status.configure(text="Detection finished.")
            return
        frame = item["frame"]
        fitted = fit_frame(frame, self.view_w, self.view_h)
        rgb = cv2.cvtColor(fitted, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.picture.configure(image=self.photo, text="")
        if item.get("loading"):
            self.status.configure(text="Loading YOLO and EfficientNet...")
            return
        if not self.scrubbing and self.fps > 0:
            seconds = max(0.0, (item["index"] - 1) / self.fps)
            self.timeline.set(min(seconds, self.duration or seconds))
            self.time_now.configure(text=format_clock(seconds))
            # The chat cutoff moves only when this frame is actually on screen.
            with self.lock:
                if item.get("token") == self.view_token:
                    self.shown_seconds = seconds

        paused = not self.playing.is_set() and not self.ended
        if paused:
            self.play_button.configure(text="Play")
        lead = "Paused    " if paused else ""
        self.status.configure(
            text=(
                f"{lead}Frame {item['index']} / {item['total']}"
                f"    {item['boxes']} boxes"
                f"    {item['fps']:.1f} fps"
                f"    {item['device']}"
            )
        )

    def _scrub_start(self, _event: object) -> None:
        """The operator is dragging the timeline. Do not fight that drag."""
        self.scrubbing = True

    def _on_timeline(self, value: str) -> None:
        """Show the time under the drag handle."""
        if not self.scrubbing:
            return
        seconds = float(value)
        self.time_now.configure(text=format_clock(seconds))
        # While the handle is moving, answers stop at that time.
        with self.lock:
            self.shown_seconds = seconds

    def _scrub_end(self, _event: object) -> None:
        """Jump the video to the released position."""
        self.scrubbing = False
        self._request_seek(float(self.timeline.get()))

    def _request_seek(self, seconds: float) -> None:
        """Ask the detector thread to move. Start it if Play has not been used yet."""
        if self.duration > 0:
            seconds = min(max(0.0, seconds), self.duration)
        with self.lock:
            self._cut_log(seconds)
        self.ended = False
        self.time_now.configure(text=format_clock(seconds))
        if not VIDEO_PATH.is_file():
            return
        if self.worker is None or not self.worker.is_alive():
            if not self._detectors_ready():
                self.status.configure(text="Loading YOLO and EfficientNet...")
            self.worker = threading.Thread(target=self._detect_loop, daemon=True)
            self.worker.start()

    def _open_crop(self, label: str) -> None:
        """Open the clearest YOLO crop of this class from the frames so far."""
        with self.lock:
            history = self.crops.get(label, [])
            limit = self.shown_seconds
            eligible = [item for item in history if item[3] <= limit + 0.05]
            saved = None if not eligible else max(eligible, key=lambda item: item[0])
            crop = None if saved is None else saved[1].copy()
        if saved is None or crop is None or crop.size == 0:
            self._write_turn("Sentinel", f"No {label} has been detected in the frames so far.")
            return
        confidence, _stored, clock, _seconds = saved
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        width, height = image.size
        long_side = max(width, height, 1)
        scale = 1.0
        if long_side < 280:
            scale = 280 / long_side
        if long_side * scale > 640:
            scale = 640 / long_side
        if scale != 1.0:
            image = image.resize(
                (max(1, int(width * scale)), max(1, int(height * scale))),
                Image.Resampling.BILINEAR,
            )
        window = tk.Toplevel(self.root)
        window.title(label)
        window.configure(bg=CHAT_BG)
        photo = ImageTk.PhotoImage(image)
        window.photo = photo
        tk.Label(window, image=photo, bg=CHAT_BG).pack(padx=16, pady=(16, 8))
        tk.Label(
            window,
            text=f"{label}   {confidence_percent(confidence)}   at {clock}",
            font=("Consolas", 12),
            fg=MATRIX,
            bg=CHAT_BG,
        ).pack(pady=(0, 16))
        self.picture_windows.append(window)

    def _write_turn(self, who: str, text: str) -> None:
        """Add one finished message to the chat."""
        self.chat.configure(state="normal")
        self.chat.insert("end", who + "\n", "who")
        self.chat.insert("end", text + "\n\n", "body")
        self.chat.see("end")
        self.chat.configure(state="disabled")

    def _on_enter(self, _event: object) -> None:
        """Enter sends the question."""
        self.on_send()

    def on_send(self) -> None:
        """Send the typed question to Qwen and stream the answer back."""
        text = self.question.get().strip()
        if not text or self.chat_busy:
            return
        self.question.delete(0, "end")
        self._write_turn("You", text)
        self.chat_history.append({"role": "user", "content": text})
        label = asked_object(text)
        if label is not None:
            self._open_crop(label)
        self.chat_busy = True
        self.chat.configure(state="normal")
        self.chat.insert("end", "Sentinel\n", "who")
        self.chat.see("end")
        history = list(self.chat_history)
        context = self._video_context()
        threading.Thread(target=self._reply, args=(history, context), daemon=True).start()

    def _reply(self, history: list[dict[str, str]], context: str) -> None:
        """Read the model stream on a background thread."""
        parts: list[str] = []
        try:
            for piece in stream_chat(history, context):
                parts.append(piece)
                self.chat_queue.put(("token", piece))
            answer = "".join(parts).strip()
            if answer:
                self.chat_history.append({"role": "assistant", "content": answer})
            self.chat_queue.put(("end", None))
        except (OSError, urllib.error.URLError, RuntimeError, json.JSONDecodeError) as exc:
            self.chat_queue.put(("error", str(exc)))

    def _drain_chat(self) -> None:
        """Move tokens from the model thread onto the chat widget."""
        while True:
            try:
                kind, payload = self.chat_queue.get_nowait()
            except queue.Empty:
                return
            if kind == "token":
                self.chat.configure(state="normal")
                self.chat.insert("end", payload, "body")
                self.chat.see("end")
            elif kind == "end":
                self.chat.configure(state="normal")
                self.chat.insert("end", "\n\n")
                self.chat.configure(state="disabled")
                self.chat_busy = False
            elif kind == "error":
                self.chat.configure(state="normal")
                self.chat.insert("end", "\nThe model is not answering right now.\n\n", "body")
                self.chat.configure(state="disabled")
                self.chat_status.configure(text="Qwen2.5 7B Q4 not ready")
                self.chat_busy = False
            elif kind == "loading":
                self.chat_status.configure(text="Qwen2.5 7B Q4 loading")
            elif kind == "ready":
                self.chat_status.configure(text="Qwen2.5 7B Q4 on the GPU")

    def _begin_intro(self) -> None:
        """Start the introduction after the window has been open for a moment."""
        if self.stop.is_set():
            return
        self.chat.configure(state="normal")
        self.chat.insert("end", "Sentinel\n", "who")
        self.chat.configure(state="disabled")
        self.intro_index = 0
        self.root.after(196, self._type_intro)

    def _type_intro(self) -> None:
        """Type one character so the introduction arrives slowly."""
        if self.stop.is_set():
            return
        if self.intro_index >= len(GREETING):
            self.chat.configure(state="normal")
            self.chat.insert("end", "\n\n")
            self.chat.configure(state="disabled")
            self.question.configure(state="normal")
            self.send_button.configure(state="normal")
            return
        character = GREETING[self.intro_index]
        self.intro_index += 1
        self.chat.configure(state="normal")
        self.chat.insert("end", character, "body")
        self.chat.see("end")
        self.chat.configure(state="disabled")
        # Pauses are 30 percent shorter than the first typewriter timing.
        if character in ".!?":
            delay = 294
        elif character == " ":
            delay = 25
        else:
            delay = 39
        self.root.after(delay, self._type_intro)

    def _open_chat_model(self) -> None:
        """Put the 4-bit model on the GPU once the window is open."""
        if self.stop.is_set():
            return
        self.chat_status.configure(text="Qwen2.5 7B Q4 loading")
        threading.Thread(target=self._warm_chat, daemon=True).start()

    def _warm_chat(self) -> None:
        """Load weights. The greeting is already on screen."""
        try:
            warm_chat_model()
            self.chat_queue.put(("ready", None))
        except (OSError, urllib.error.URLError, RuntimeError, json.JSONDecodeError) as exc:
            self.chat_queue.put(("error", str(exc)))

    def close(self) -> None:
        """Stop the detector thread, then close the window."""
        self.stop.set()
        self.playing.set()
        self.root.destroy()


def ssh_client_display(connection: str) -> str | None:
    """Return the X display of the PC that opened this SSH session."""
    parts = connection.split()
    if not parts:
        return None
    client_ip = parts[0]
    # A local forward is not the desktop monitor.
    if client_ip.startswith("127."):
        return None
    return client_ip + ":0.0"


def is_forwarded_display(display: str) -> bool:
    """True when SSH X11 forwarding set this DISPLAY (ssh -X or ssh -Y)."""
    # A forwarded display points at the SSH tunnel on this board, not a screen number.
    return display.startswith("localhost:") or display.startswith("127.")


def attach_desktop() -> None:
    """Send the window to the computer that ran SSH, when this board has no monitor."""
    in_ssh = bool(os.environ.get("SSH_CONNECTION"))
    display = os.environ.get("DISPLAY", "")
    # ssh -Y from a Mac (XQuartz) or Linux already set a tunnel display. Keep it and its cookie.
    if in_ssh and is_forwarded_display(display):
        return
    # No tunnel, but SSH knows the client. Draw straight on that computer's X server over TCP.
    remote = ssh_client_display(os.environ.get("SSH_CONNECTION", ""))
    if remote:
        os.environ["DISPLAY"] = remote
        os.environ.pop("XAUTHORITY", None)
        return
    if display:
        return
    if not Path("/tmp/.X11-unix/X0").exists():
        return
    os.environ["DISPLAY"] = ":0"
    authority = Path("/run/user") / str(os.getuid()) / "gdm" / "Xauthority"
    if authority.is_file():
        os.environ["XAUTHORITY"] = str(authority)


def main() -> None:
    """Open the demo window."""
    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    attach_desktop()
    root = tk.Tk()
    DemoWindow(root, preload=True)
    root.mainloop()


if __name__ == "__main__":
    main()
