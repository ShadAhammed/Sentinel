"""
window.py

Tkinter demo window for SENTINEL-X.

Run via run.py at the project root:

    python run.py

The window loads the three YOLO detectors and EfficientNet at startup
(before Play is pressed), then loads Qwen after the detectors are warm,
so the two models do not compete for the GPU during the cold start.

Layout:
  Header         - logo, title, "Human in the loop" badge
  Detection tab  - video (60%) + chat panel (40%)
  Credit tab     - dataset cards with license info
  Bottom bar     - Play / Pause, timeline, frame count, fps, device name
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
import tkinter as tk
import urllib.error
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

# The package modules supply all detection, chat, and hardware helpers.
from src.chat import (
    GREETING,
    asked_object,
    stream_chat,
    warm_chat_model,
)
from src.detection import (
    LABEL_COLOR,
    cnn_prediction_for_crop,
    confidence_percent,
    efficientnet_hits,
    final_label,
    load_effnet,
    yolo_boxes,
)
from src.gpu import format_gpu_free, read_gpu
from src.labels import EFFNET_PATH, VIDEO_PATH, YOLO_FILES, YOLO_FINAL_DIR

# ---- Module-level constants kept here because they only affect the window ----

# Color palette. Olive black background, dull brass highlights.
BG = "#161915"
PANEL = "#22261f"
INK = "#e4e2d8"
MUTED = "#8e9686"
GOLD = "#a3986e"
GOLD_INK = "#1c1a14"
CHAT_BG = "#000000"
MATRIX = "#00ff41"
MATRIX_DIM = "#1f8f32"

# The chat panel takes this fraction of the detection row. The video takes the rest.
CHAT_SHARE = 0.40


# ---- Small pure helpers used by the window --------------------------------


def split_view(total: int, gap: int = 12) -> tuple[int, int]:
    """Return (chat_width, video_width). Chat is 40 percent of the row."""
    usable = max(total - gap, 2)
    chat = int(round(usable * CHAT_SHARE))
    return chat, usable - chat


def dataset_credits() -> list[dict[str, str]]:
    """Return the three detection datasets with their license details for the Credit tab."""
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
    """Shrink a frame so it fits the panel without changing the aspect ratio."""
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
    """Draw the SENTINEL-X mark - a gold diamond on a dark rounded tile."""
    image = Image.new("RGB", (size, size), "#1c241c")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((1, 1, size - 2, size - 2), radius=12, outline=GOLD, width=2)
    center = size // 2
    # Outer diamond: the watch boundary. Inner diamond: the fill mark.
    draw.polygon(
        [(center, 12), (size - 12, center), (center, size - 12), (12, center)],
        outline=GOLD,
    )
    draw.polygon(
        [(center, 22), (size - 22, center), (center, size - 22), (22, center)],
        fill=GOLD,
    )
    return image


def format_clock(seconds: float) -> str:
    """Format a player position as 'MM:SS'."""
    whole = max(0, int(seconds))
    minutes, rest = divmod(whole, 60)
    return f"{minutes:02d}:{rest:02d}"


def parse_clock(clock: str) -> float:
    """Parse 'MM:SS' back to seconds as a float."""
    minutes, rest = clock.split(":")
    return int(minutes) * 60 + int(rest)


def probe_video(path: Path) -> tuple[float, float]:
    """Return (fps, duration_seconds) for a video file. Defaults to 24 fps if unreadable."""
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
    """Return the label string drawn on a detection box, or None when the crop was too small."""
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
    """Draw the bounding box and the EfficientNet label onto a frame in-place.

    Nothing is drawn when the crop was too small for EfficientNet to name.
    """
    line = cnn_caption(cnn_name, cnn_confidence)
    if line is None:
        return
    x1, y1, x2, y2 = [int(v) for v in box["xyxy"]]
    color = LABEL_COLOR.get(decided_label, (255, 255, 255))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(line, font, 0.55, 1)
    block_h = text_h + 10
    # Place the label above the box; fall below if there is no room.
    top = (y1 - block_h - 4) if y1 > block_h + 4 else min(y2 + 4, max(0, frame.shape[0] - block_h - 1))
    left = x1
    if left + text_w + 8 > frame.shape[1]:
        left = max(0, frame.shape[1] - text_w - 8)
    cv2.rectangle(frame, (left, top), (left + text_w + 8, top + block_h), color, thickness=-1)
    cv2.putText(frame, line, (left + 4, top + text_h + 2), font, 0.55, (0, 0, 0), 1)


def format_detection_context(report: dict) -> str:
    """Build the plain-text tally that is sent to the chat model.

    Only frames up to the playhead time are included so the model cannot
    see detections that are still in the future on the operator's screen.
    """
    frame_log = report.get("frame_log", [])
    if not frame_log:
        return "No frame has been read yet. There are no object counts."

    # Frames strictly after the on-screen position are excluded.
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

    # Accumulate per-category statistics across the used frames.
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
    lines = [status, f"Video time: {report.get('clock', latest['clock'])}", "This frame:"]

    # Describe what is in the most recent used frame.
    grouped: dict[str, list[float]] = {}
    for name, confidence in latest["hits"]:
        grouped.setdefault(name, []).append(confidence)
    if not grouped:
        lines.append("- no drawn objects")
    for name in sorted(grouped):
        scores = grouped[name]
        lines.append(f"- {name}: {len(scores)}, highest confidence {confidence_percent(max(scores))}")

    lines.append("Categories in the frames so far:")
    names = sorted(maximums, key=lambda n: (-maximums[n], n))
    if not names:
        lines.append("- none yet")
    for name in names:
        lines.append(
            f"- {name}: at most {maximums[name]} at once, "
            f"in {frames[name]} frames, "
            f"{totals[name]} observations in total"
        )

    # Add crop references up to the playhead.
    saved = report.get("crops", [])
    visible = [(n, c, clk) for n, c, clk in saved if parse_clock(clk) <= limit + 0.05]
    if visible:
        lines.append("Clearest crop saved for a picture window:")
        for name, confidence, clock in visible:
            lines.append(f"- {name} at {clock}, {confidence_percent(confidence)}")

    return "\n".join(lines)


# ---- DemoWindow -----------------------------------------------------------


class DemoWindow:
    """One window. The video, the detector thread, and the chat share it."""

    def __init__(self, root: tk.Tk, preload: bool = False) -> None:
        self.root = root
        # preload=True loads YOLO and EfficientNet while the window opens,
        # so Play does not trigger a cold-start delay.
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
        # shown_seconds is the timestamp of the frame currently on screen.
        # It controls the chat context cutoff.
        self.shown_seconds = 0.0
        self.view_token = 0
        self.chat_busy = False
        self.chat_history = [{"role": "assistant", "content": GREETING}]
        self.chat_queue: queue.Queue = queue.Queue()
        self.report = self._blank_report()
        self.crops: dict[str, list] = {}
        self.picture_windows: list[tk.Toplevel] = []
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

        # Start detector preload. Qwen loads after the detectors are warm
        # so they do not share the GPU during the cold start.
        if self.preload and VIDEO_PATH.is_file():
            threading.Thread(target=self._preload_models, daemon=True).start()
        else:
            self.root.after(400, self._open_chat_model)
        self.root.after(840, self._begin_intro)

    # ---- Layout -----------------------------------------------------------

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
        tk.Label(titles, text="SENTINEL-X", font=("Segoe UI", 22, "bold"), fg=INK, bg=BG).pack(anchor="w")
        tk.Label(titles, text="Aerial reconnaissance demonstrator", font=("Segoe UI", 11), fg=MUTED, bg=BG).pack(anchor="w")
        tk.Label(header, text="Human in the loop", font=("Segoe UI", 10), fg=GOLD, bg=BG).pack(side="right")

        tk.Frame(self.root, bg=GOLD, height=2).pack(fill="x", padx=22, pady=(0, 10))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Sentinel.TNotebook", background=BG, borderwidth=0)
        style.configure("Player.Horizontal.TScale", background=BG, troughcolor="#3a4034", sliderlength=18)
        style.configure("Sentinel.TNotebook.Tab", background=PANEL, foreground=INK, padding=(18, 8), font=("Segoe UI", 11))
        style.map("Sentinel.TNotebook.Tab", background=[("selected", "#2c3328")], foreground=[("selected", GOLD)])

        self.tabs = ttk.Notebook(self.root, style="Sentinel.TNotebook")
        self.tabs.pack(fill="both", expand=True, padx=22, pady=(0, 12))

        detection = tk.Frame(self.tabs, bg=BG)
        credit = tk.Frame(self.tabs, bg=BG)
        self.tabs.add(detection, text="Detection")
        self.tabs.add(credit, text="Credit")

        self._build_detection(detection)
        self._build_credit(credit)

    def _build_detection(self, parent: tk.Frame) -> None:
        """Video on the left, chat on the right, timeline and status at the bottom."""
        self.status = tk.Label(
            parent, text=self._ready_status(), font=("Segoe UI", 10), fg=MUTED, bg=BG, anchor="w"
        )
        self.status.pack(side="bottom", fill="x", padx=8, pady=(0, 8))

        bar = tk.Frame(parent, bg=BG)
        bar.pack(side="bottom", fill="x", padx=8, pady=(0, 4))

        self.play_button = tk.Button(
            bar, text="Play", font=("Segoe UI", 11, "bold"),
            bg=GOLD, fg=GOLD_INK, activebackground="#b5aa80", activeforeground=GOLD_INK,
            relief="flat", padx=16, pady=6, cursor="hand2", command=self.on_play,
        )
        self.play_button.pack(side="left")

        self.time_now = tk.Label(bar, text="00:00", font=("Consolas", 11), fg=INK, bg=BG, width=6)
        self.time_now.pack(side="left", padx=(12, 6))

        span = self.duration if self.duration > 0 else 1.0
        self.timeline = ttk.Scale(bar, from_=0, to=span, orient="horizontal", style="Player.Horizontal.TScale", command=self._on_timeline)
        self.timeline.pack(side="left", fill="x", expand=True, padx=4)
        self.timeline.bind("<ButtonPress-1>", self._scrub_start)
        self.timeline.bind("<ButtonRelease-1>", self._scrub_end)

        self.time_end = tk.Label(bar, text=format_clock(self.duration), font=("Consolas", 11), fg=MUTED, bg=BG, width=6)
        self.time_end.pack(side="left", padx=(6, 0))

        body = tk.Frame(parent, bg=BG)
        self.body = body
        body.pack(fill="both", expand=True, padx=8, pady=8)
        body.bind("<Configure>", self._on_body_resize)

        # Video area.
        stage = tk.Frame(body, bg="#000000")
        stage.pack(side="left", fill="both", expand=True)
        stage.bind("<Configure>", self._on_stage_resize)
        self.stage = stage
        self.picture = tk.Label(
            stage, text="Press Play.\nDrag the timeline to move.",
            font=("Segoe UI", 14), fg=MUTED, bg="#000000", justify="center",
        )
        self.picture.pack(fill="both", expand=True)

        # Chat panel.
        side = tk.Frame(body, bg=CHAT_BG, width=split_view(1000)[0])
        self.side = side
        side.pack(side="right", fill="y", padx=(12, 0))
        side.pack_propagate(False)

        tk.Label(side, text="Sentinel", font=("Consolas", 14, "bold"), fg=MATRIX, bg=CHAT_BG).pack(anchor="w", padx=14, pady=(14, 0))
        self.chat_status = tk.Label(side, text="Qwen2.5 7B Q4", font=("Consolas", 9), fg=MATRIX_DIM, bg=CHAT_BG, anchor="w")
        self.chat_status.pack(anchor="w", padx=14, pady=(0, 8))

        # GPU labels at the bottom of the chat column.
        self.gpu_free = tk.Label(side, text="GPU free        --", font=("Consolas", 10), fg=MATRIX_DIM, bg=CHAT_BG, anchor="w")
        self.gpu_free.pack(side="bottom", fill="x", padx=14, pady=(0, 12))
        self.gpu_util = tk.Label(side, text="GPU utilized    --", font=("Consolas", 10), fg=MATRIX_DIM, bg=CHAT_BG, anchor="w")
        self.gpu_util.pack(side="bottom", fill="x", padx=14, pady=(8, 0))

        # Question entry and Send button.
        ask = tk.Frame(side, bg=CHAT_BG)
        ask.pack(side="bottom", fill="x", padx=12, pady=(0, 4))
        tk.Label(ask, text="Ask", font=("Consolas", 10), fg=MATRIX, bg=CHAT_BG).pack(side="left", padx=(0, 8))
        self.send_button = tk.Button(
            ask, text="Send", font=("Consolas", 10, "bold"),
            bg="#021a02", fg=MATRIX, activebackground="#063006", activeforeground=MATRIX,
            relief="flat", padx=12, pady=4, cursor="hand2", command=self.on_send,
        )
        self.send_button.pack(side="right")
        self.question = tk.Entry(ask, font=("Consolas", 11), bg=CHAT_BG, fg=MATRIX, insertbackground=MATRIX, relief="flat")
        self.question.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 8))
        self.question.bind("<Return>", self._on_enter)
        self.question.configure(state="disabled")
        self.send_button.configure(state="disabled")

        # Chat text area.
        self.chat = tk.Text(side, bg=CHAT_BG, fg=MATRIX, font=("Consolas", 11), height=8, relief="flat", wrap="word", padx=12, pady=8, highlightthickness=0)
        self.chat.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.chat.tag_configure("who", font=("Consolas", 9, "bold"), foreground=MATRIX)
        self.chat.tag_configure("body", font=("Consolas", 11), foreground=MATRIX)
        self.chat.configure(state="disabled")

    def _on_body_resize(self, event: tk.Event) -> None:
        """Keep chat at 40 percent of the row when the window is resized."""
        if event.widget is not self.body:
            return
        chat, _video = split_view(event.width)
        self.side.configure(width=chat)

    def _on_stage_resize(self, event: tk.Event) -> None:
        """Remember the video panel size so frames are scaled to fit."""
        if event.width > 20 and event.height > 20:
            self.view_w = event.width
            self.view_h = event.height

    def _build_credit(self, parent: tk.Frame) -> None:
        """Scrollable cards showing the three source datasets and their licenses."""
        tk.Label(
            parent, text="These three public datasets trained the detectors in this demonstrator.",
            font=("Segoe UI", 11), fg=MUTED, bg=BG, anchor="w",
        ).pack(fill="x", padx=12, pady=(12, 8))

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
            tk.Label(inner, text=record["title"], font=("Segoe UI", 14, "bold"), fg=GOLD, bg=PANEL, anchor="w").pack(fill="x")
            tk.Label(inner, text=record["body"], font=("Segoe UI", 10), fg=INK, bg=PANEL, justify="left", anchor="w", wraplength=860).pack(fill="x", pady=(6, 0))

        def _fit_cards(_event: object) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfigure(window_id, width=canvas.winfo_width())

        cards.bind("<Configure>", _fit_cards)
        canvas.bind("<Configure>", _fit_cards)

    def tab_names(self) -> list[str]:
        """Return the tab titles. Used by tests."""
        return [self.tabs.tab(tab_id, "text") for tab_id in self.tabs.tabs()]

    # ---- Playback ---------------------------------------------------------

    def _ready_status(self) -> str:
        """Status bar text when the window is idle."""
        if not VIDEO_PATH.is_file():
            return "combined_drone.mp4 is not in this folder."
        if self.preload and not self.models_ready.is_set():
            return "Loading YOLO and EfficientNet..."
        return "Ready. Play runs YOLO and EfficientNet on the joined video."

    def on_play(self) -> None:
        """Start live detection, or pause it when it is already running."""
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
        """True once the background preload has put the weights on the GPU."""
        return self.preload and self.models_ready.is_set() and self.models is not None

    def _detect_loop(self) -> None:
        """Open the video, load models, and start the play and detector threads."""
        capture = cv2.VideoCapture(str(VIDEO_PATH))
        if not capture.isOpened():
            with self.lock:
                self.latest = {"error": f"Could not open {VIDEO_PATH.name}"}
            self.playing.clear()
            return
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        try:
            # Show the first frame while weights are loading so Play is not a blank wait.
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
            # Rewind so labels cover the clip from the beginning.
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._start_detector(models, total)
            self._play_loop(capture, models, total)
        finally:
            capture.release()

    def _start_detector(self, models: dict, total: int) -> None:
        """Start the detector thread. Only one runs at a time."""
        if self.det_thread is not None and self.det_thread.is_alive():
            return
        self.det_thread = threading.Thread(target=self._detector_worker, args=(models, total), daemon=True)
        self.det_thread.start()

    def _play_loop(self, capture, models: dict, total: int) -> None:
        """Read frames at the video clock. Hand each frame to the detector thread."""
        period = (1.0 / self.fps) if self.fps > 0 else 0.0
        target = time.perf_counter()
        last_shown = target
        while not self.stop.is_set():
            # Check for a seek or restart requested by the timeline or Play button.
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
                    self.latest = {"done": True, "index": total, "total": total, "token": self.view_token}
                continue
            index = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
            with self.lock:
                token = self.view_token
            # Give the detector the freshest frame; older pending frames are replaced.
            self._submit_detection(frame, index, token)
            # Draw the most recently produced overlay on this frame.
            shown = frame.copy()
            drawn = self._draw_overlay(shown, token)
            now = time.perf_counter()
            fps = 1.0 / max(now - last_shown, 1e-6)
            last_shown = now
            self._publish_frame(shown, index, total, drawn, fps, models["device_label"], token)
            target += period
            self._sleep_until(target)

    def _sleep_until(self, deadline: float) -> None:
        """Sleep until the deadline, but wake early on stop, pause, or a seek."""
        while time.perf_counter() < deadline:
            if self.stop.is_set() or not self.playing.is_set():
                return
            with self.lock:
                if self.seek_seconds is not None:
                    return
            time.sleep(0.005)

    def _submit_detection(self, frame: np.ndarray, index: int, token: int) -> None:
        """Give one frame to the detector thread, replacing any frame it has not started yet."""
        with self.lock:
            self.det_input = (frame.copy(), index, token)

    def _detector_worker(self, models: dict, total: int) -> None:
        """Label the freshest frame and store the boxes."""
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
                self.overlay_items = items
                self.overlay_token = token
            self._add_hits(hits, index, token)

    def _draw_overlay(self, frame: np.ndarray, token: int) -> int:
        """Draw the most recent boxes on a frame. Returns the count drawn."""
        with self.lock:
            if self.overlay_token != token:
                return 0
            items = list(self.overlay_items)
        for item in items:
            draw_live_box(frame, item, item["cnn_name"], item["cnn_confidence"], item["decided"])
        return len(items)

    # ---- Model loading ----------------------------------------------------

    def _load_models(self) -> dict:
        """Open the three YOLO weights and EfficientNet. Refuses to run on CPU."""
        import torch
        from ultralytics import YOLO

        # The Orin GPU is cuda:0. CPU inference is refused.
        if not torch.cuda.is_available():
            raise RuntimeError("Sentinel runs on the board GPU. CUDA is not available.")
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        effnet, _classes = load_effnet(EFFNET_PATH, device)
        # Let cuDNN pick a fast algorithm once and reuse it for all frames.
        torch.backends.cudnn.benchmark = True
        yolo_models = []
        any_engine = False
        for name in YOLO_FILES:
            path = YOLO_FINAL_DIR / name
            engine = path.with_suffix(".engine")
            # Use the TensorRT engine when it lives beside the .pt file.
            weight = engine if engine.is_file() else path
            if weight.suffix == ".engine":
                any_engine = True
                model = YOLO(str(weight), task="detect")
            else:
                model = YOLO(str(weight))
            # TensorRT engines are already fused. Fuse .pt files only.
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
            # A TensorRT engine is already FP16; do not ask predict() for half again.
            "half": not any_engine,
        }
        self._warm_detectors(loaded)
        return loaded

    def _preload_models(self) -> None:
        """Load and warm the detectors before Play. Qwen starts after."""
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
        """Return the preloaded models, or load them now if preload was not used."""
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
        """Run one blank frame through each model so the first real frame is not the cold start."""
        import torch

        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        if models.get("half"):
            try:
                yolo_boxes(models["yolo_models"], blank, models["yolo_device"], half=True)
            except Exception:
                yolo_boxes(models["yolo_models"], blank, models["yolo_device"], half=False)
                models["half"] = False
        else:
            yolo_boxes(models["yolo_models"], blank, models["yolo_device"], half=False)
        dummy = torch.zeros(1, 3, 224, 224, device=models["device"])
        with torch.inference_mode():
            models["effnet"](dummy)

    def _publish_frame(self, frame, index, total, boxes, fps, device, token, loading=False) -> None:
        """Hand one processed frame to the window thread."""
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
        """YOLO places boxes; EfficientNet names each crop. Return the kept boxes with crops."""
        boxes = yolo_boxes(
            models["yolo_models"],
            frame,
            models["yolo_device"],
            half=bool(models.get("half")),
        )
        candidates = []
        for box in boxes:
            found = cnn_prediction_for_crop(models["effnet"], frame, box["xyxy"], models["device"])
            if found is None:
                # EfficientNet could not name this crop - do not count it.
                continue
            cnn_name, cnn_confidence = found
            candidates.append({
                "xyxy": box["xyxy"],
                "cnn_name": cnn_name,
                "cnn_confidence": cnn_confidence,
                "decided": final_label(box["label"], cnn_name),
            })
        # Deduplicate overlapping boxes by EfficientNet confidence.
        items = []
        for item in efficientnet_hits(candidates):
            x1, y1, x2, y2 = [int(v) for v in item["xyxy"]]
            h, w = frame.shape[:2]
            item["crop"] = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)].copy()
            items.append(item)
        return items

    # ---- Detection log ----------------------------------------------------

    def _blank_report(self) -> dict:
        """Return an empty detection log."""
        return {"clock": "00:00", "playhead": 0.0, "frame_log": []}

    def _reset_report(self) -> None:
        """Clear the tally when playback restarts from the beginning."""
        with self.lock:
            self.report = self._blank_report()
            self.crops = {}
            self.shown_seconds = 0.0
            self.latest = None
            self.view_token += 1

    def _add_hits(self, hits: list, index: int, token: int) -> None:
        """Add one frame's detections to the log. Keep the clearest crop per label."""
        seconds = max(0.0, (index - 1) / self.fps) if self.fps > 0 else 0.0
        clock = format_clock(seconds)
        logged = [(name, confidence) for name, confidence, _crop in hits]
        with self.lock:
            # Drop frames that belong to an older seek position.
            if token != self.view_token or self.seek_seconds is not None:
                return
            self.report["clock"] = clock
            self.report["playhead"] = seconds
            self.report["frame_log"].append({"seconds": seconds, "clock": clock, "hits": logged})
            for name, confidence, crop in hits:
                history = self.crops.get(name)
                if history is None:
                    self.crops[name] = [(confidence, crop, clock, seconds)]
                elif confidence > history[-1][0]:
                    history.append((confidence, crop, clock, seconds))

    def _cut_log(self, seconds: float) -> None:
        """Drop all detections later than the frame the operator seeked to."""
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
        """Build the detection tally up to the frame currently on screen."""
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
                    {"seconds": i["seconds"], "clock": i["clock"], "hits": list(i["hits"])}
                    for i in self.report["frame_log"]
                ],
                "crops": crops,
            }
        return format_detection_context(snapshot)

    # ---- Window pump and display ------------------------------------------

    def _pump(self) -> None:
        """Copy the newest detection result onto the window. Called every 30 ms."""
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
        """Update the status bar once the weights are on the GPU and playback is idle."""
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
        """Keep a fresh GPU reading in the background."""
        while not self.stop.is_set():
            reading = read_gpu()
            with self.lock:
                self.gpu = reading
            self.stop.wait(0.5)

    def _refresh_gpu(self) -> None:
        """Write the latest GPU utilization and free memory into the chat panel."""
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
        """Paint one frame update. Errors and the finished state are shown as text."""
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
            with self.lock:
                if item.get("token") == self.view_token:
                    self.shown_seconds = seconds
        paused = not self.playing.is_set() and not self.ended
        if paused:
            self.play_button.configure(text="Play")
        lead = "Paused    " if paused else ""
        self.status.configure(text=(
            f"{lead}Frame {item['index']} / {item['total']}"
            f"    {item['boxes']} boxes"
            f"    {item['fps']:.1f} fps"
            f"    {item['device']}"
        ))

    # ---- Timeline ---------------------------------------------------------

    def _scrub_start(self, _event: object) -> None:
        """Record that the operator is dragging so the pump loop does not fight it."""
        self.scrubbing = True

    def _on_timeline(self, value: str) -> None:
        """Update the time label while the handle is being dragged."""
        if not self.scrubbing:
            return
        seconds = float(value)
        self.time_now.configure(text=format_clock(seconds))
        with self.lock:
            self.shown_seconds = seconds

    def _scrub_end(self, _event: object) -> None:
        """Jump the video to the released timeline position."""
        self.scrubbing = False
        self._request_seek(float(self.timeline.get()))

    def _request_seek(self, seconds: float) -> None:
        """Ask the detector thread to move to a new position."""
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

    # ---- Crop picture window ----------------------------------------------

    def _open_crop(self, label: str) -> None:
        """Open a small pop-up showing the clearest crop of the requested class."""
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
        image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
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
        window.photo = photo  # type: ignore[attr-defined]
        tk.Label(window, image=photo, bg=CHAT_BG).pack(padx=16, pady=(16, 8))
        tk.Label(
            window,
            text=f"{label}   {confidence_percent(confidence)}   at {clock}",
            font=("Consolas", 12), fg=MATRIX, bg=CHAT_BG,
        ).pack(pady=(0, 16))
        self.picture_windows.append(window)

    # ---- Chat interaction -------------------------------------------------

    def _write_turn(self, who: str, text: str) -> None:
        """Append one finished message to the chat widget."""
        self.chat.configure(state="normal")
        self.chat.insert("end", who + "\n", "who")
        self.chat.insert("end", text + "\n\n", "body")
        self.chat.see("end")
        self.chat.configure(state="disabled")

    def _on_enter(self, _event: object) -> None:
        """Enter key sends the question."""
        self.on_send()

    def on_send(self) -> None:
        """Send the typed question to Qwen and stream the answer back."""
        text = self.question.get().strip()
        if not text or self.chat_busy:
            return
        self.question.delete(0, "end")
        self._write_turn("You", text)
        self.chat_history.append({"role": "user", "content": text})
        # If the question asks to see a specific object, open its crop window.
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
        """Stream the model reply on a background thread. Puts tokens into the queue."""
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

    # ---- Intro typewriter -------------------------------------------------

    def _begin_intro(self) -> None:
        """Start typing the greeting after the window has settled."""
        if self.stop.is_set():
            return
        self.chat.configure(state="normal")
        self.chat.insert("end", "Sentinel\n", "who")
        self.chat.configure(state="disabled")
        self.intro_index = 0
        self.root.after(196, self._type_intro)

    def _type_intro(self) -> None:
        """Type one character of the greeting. Schedule the next character."""
        if self.stop.is_set():
            return
        if self.intro_index >= len(GREETING):
            # Greeting complete - unlock the question box.
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
        # Pauses after punctuation feel more natural.
        if character in ".!?":
            delay = 294
        elif character == " ":
            delay = 25
        else:
            delay = 39
        self.root.after(delay, self._type_intro)

    # ---- Chat model loading -----------------------------------------------

    def _open_chat_model(self) -> None:
        """Trigger Qwen warm-up once the window is open."""
        if self.stop.is_set():
            return
        self.chat_status.configure(text="Qwen2.5 7B Q4 loading")
        threading.Thread(target=self._warm_chat, daemon=True).start()

    def _warm_chat(self) -> None:
        """Warm-up Qwen. Called on a background thread."""
        try:
            warm_chat_model()
            self.chat_queue.put(("ready", None))
        except (OSError, urllib.error.URLError, RuntimeError, json.JSONDecodeError) as exc:
            self.chat_queue.put(("error", str(exc)))

    # ---- Close ------------------------------------------------------------

    def close(self) -> None:
        """Stop the detector thread and close the window."""
        self.stop.set()
        self.playing.set()
        self.root.destroy()


# ---- Display helpers for headless Jetson ----------------------------------


def ssh_client_display(connection: str) -> str | None:
    """Return the X display of the machine that opened this SSH session.

    Used to route the window to the connected laptop when the Jetson has no monitor.
    """
    parts = connection.split()
    if not parts:
        return None
    client_ip = parts[0]
    # A loopback client is a local forward, not the operator's desktop.
    if client_ip.startswith("127."):
        return None
    return client_ip + ":0.0"


def is_forwarded_display(display: str) -> bool:
    """True when SSH X11 forwarding (ssh -X or ssh -Y) already set DISPLAY.

    A forwarded display tunnels through localhost, so it starts with 'localhost:'
    or '127.'. This must be kept as-is and not overwritten.
    """
    return display.startswith("localhost:") or display.startswith("127.")


def attach_desktop() -> None:
    """Route the window to the correct display when the board has no monitor."""
    in_ssh = bool(os.environ.get("SSH_CONNECTION"))
    display = os.environ.get("DISPLAY", "")
    # An existing X11 tunnel from ssh -Y is already correct. Keep it.
    if in_ssh and is_forwarded_display(display):
        return
    # No tunnel but SSH knows the client IP. Draw on that machine's X server.
    remote = ssh_client_display(os.environ.get("SSH_CONNECTION", ""))
    if remote:
        os.environ["DISPLAY"] = remote
        os.environ.pop("XAUTHORITY", None)
        return
    if display:
        return
    if not Path("/tmp/.X11-unix/X0").exists():
        return
    # Fall back to the local display 0 when nothing else is set.
    os.environ["DISPLAY"] = ":0"
    authority = Path("/run/user") / str(os.getuid()) / "gdm" / "Xauthority"
    if authority.is_file():
        os.environ["XAUTHORITY"] = str(authority)


# ---- Entry point ----------------------------------------------------------


def main() -> None:
    """Open the demo window."""
    try:
        # Sharper rendering on Windows high-DPI displays.
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
