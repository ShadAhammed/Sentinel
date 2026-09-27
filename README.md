# SENTINEL-X

![SENTINEL-X banner](assets/banner.jpg)

**Real-time aerial reconnaissance demonstrator - air-gapped, edge-deployed, human in the loop.**

Three YOLOv8 detectors, an EfficientNet-B0 crop classifier, and a local Qwen 2.5 7B LLM running
together on a single NVIDIA Jetson Orin NX 16 GB. No cloud, no internet, no CPU fallback.
Demonstrated live at a defence and security expo in Essen, September 2026.

---

## What it does

A 30-second aerial video plays in the left panel.
Every frame is passed through three independent object detectors running on the GPU.
Each bounding box is cropped and fed to a fine-tuned EfficientNet-B0 that names the object.
A running tally - per class, per frame, peak concurrences - is kept throughout playback.

The right panel is a chat window backed by Qwen 2.5 7B running locally through Ollama.
The model is given only the tally up to the frame currently on screen, so it cannot
reveal detections the operator has not seen yet. The operator asks questions in natural
language. A person makes every decision.

---

## Architecture

```
Video frame (1280 x 720, 24 fps)
        |
        v
+---------------------------+
|  YOLO detection (GPU)     |   Three independent detectors:
|  - KIIT-MiTA-yolo26s      |   1,700 drone images, 7 classes
|  - MV-yolo26s             |   2,702 images, 3 vehicle classes
|  - military-yolov8n       |   26,315 images, 12 classes
+---------------------------+
        |
        |  IoU dedup: overlapping boxes -> keep highest YOLO confidence
        v
+---------------------------+
|  EfficientNet-B0 (GPU)    |   224 x 224 crops
|  10 output classes        |   Test accuracy: 89.01 % (494 / 555)
|  32 px short-side gate    |   Tiny crops skipped, not misclassified
+---------------------------+
        |
        |  IoU dedup again: overlapping named boxes -> keep highest CNN score
        v
  Unified 10-class label space
  Artillery | Missile | Radar | M. Rocket Launcher | Soldier
  camouflage_soldier | military_aircraft | military_vehicle
  military_warship | trench
        |
        v
+---------------------------+
|  Qwen 2.5 7B (Ollama)     |   4-bit quantized, fully local
|  Context cutoff at        |   Only sees frames <= playhead time
|  the playhead             |   No coordinates, no target advice
+---------------------------+
        |
        v
  Operator chat panel
  Human decides.
```

---

## Performance (Jetson Orin NX 16 GB)

| Stage | Time |
|---|---|
| TensorRT FP16 warm pass - KIIT-MiTA-yolo26s | ~17 ms |
| TensorRT FP16 warm pass - MV-yolo26s | ~27 ms |
| TensorRT FP16 warm pass - military-yolov8n | ~28 ms |
| Full pipeline throughput | 7-8 fps |
| Video playback rate | 24 fps |
| EfficientNet-B0 test accuracy | 89.01 % (494 / 555 crops) |

Video is decoded and displayed at 24 fps. Detection runs on a separate thread and draws
the most recently computed overlay on the current frame, so the picture never stutters
while the GPU is busy.

---

## Tech stack

| Concern | Tool |
|---|---|
| Object detection | YOLOv8 (Ultralytics) |
| Crop classification | EfficientNet-B0 (torchvision pretrained, custom head) |
| LLM | Qwen 2.5 7B Instruct Q4_K_M via Ollama |
| Edge inference | TensorRT 10.7 FP16 (Jetson only) |
| GPU | CUDA 12, cuDNN benchmark mode |
| Video | OpenCV |
| UI | Tkinter (no web server, no browser) |
| Language | Python 3.11 |

---

## Project structure

```
run.py               - entry point: python run.py
requirements.txt     - pip dependencies (see note for Jetson PyTorch)
src/
  labels.py          - unified CNN class list, YOLO-to-CNN name map, path constants
  detection.py       - YOLO inference, EfficientNet crop classifier, IoU dedup
  chat.py            - Qwen streaming via Ollama HTTP, context builder
  gpu.py             - GPU utilization reader (nvidia-smi + Jetson SoC sysfs fallback)
  window.py          - Tkinter demo window: video pane, chat pane, timeline
tests/
  test_labels.py     - label mapping and path constant tests
  test_detection.py  - IoU, dedup, crop gate, confidence formatting
  test_window.py     - clock, caption, split-view, GPU reader, DemoWindow widget tests
SENTINEL-X Project Description.md   - full technical write-up
```

---

## Hardware

The demo runs on one **NVIDIA Jetson Orin NX 16 GB** (hostname `yahboom`).
The board has no monitor. `attach_desktop()` in `src/window.py` routes the window to
the connected laptop over SSH X11 forwarding (`ssh -Y`).

On a development x86 machine the `.pt` weights are used directly.
On the Jetson, a `.engine` TensorRT file beside a `.pt` is loaded instead.

---

## Setup

### 1. Clone

```bash
git clone https://github.com/ShadAhammed/Sentinel.git
cd Sentinel
```

### 2. Install Python packages

```bash
# x86 with CUDA 12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Jetson Orin NX - torch and torchvision come from JetPack
pip install -r requirements.txt
```

### 3. Place the weights and video (not in the repo)

```
Data/
  EffNet_b0.pt
  video/
    combined_drone.mp4
  YOLO-Models-Final/
    KIIT-MiTA-yolo26s.pt
    MV-yolo26s.pt
    military-yolov8n.pt
```

On the Jetson, put a `.engine` file beside each `.pt` and it will be loaded instead.

### 4. Start Ollama and pull the model

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
```

---

## Run

```bash
python run.py
```

The window loads and warms YOLO and EfficientNet first, then loads Qwen.
The two model groups never compete for the GPU during the cold start.
Press **Play** to start detection. The chat is unlocked after the greeting finishes.

---

## Tests

No GPU, no weights, and no video file are needed to run the tests.

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

```
Ran 49 tests in ~1 s
OK
```

---

## Datasets

The three YOLO detectors were trained on public datasets. Full citation details are
in the **Credit** tab of the running application and in
[SENTINEL-X Project Description.md](SENTINEL-X%20Project%20Description.md).

| Detector | Dataset | License |
|---|---|---|
| KIIT-MiTA-yolo26s | KIIT-MiTA (Chowdhury, Mendeley Data, DOI 10.17632/drjmrf5kk5.1) | Education and research only |
| military-yolov8n | Military Assets Dataset (Ryan Madhuwala, Kaggle) | CC BY 4.0 |
| MV-yolo26s | Military Vehicle Recognition (Roboflow Universe, v7) | CC BY 4.0 |

---

## Notes

- Detection refuses to run on CPU. CUDA must be available.
- The Ollama server must be running locally on port 11434 before the window opens.
- Weights, video, and datasets are not committed to this repository.
- Older commits contain large binary blobs; a full clone may be several hundred MB.
