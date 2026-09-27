# SENTINEL-X - Technical Project Description

![SENTINEL-X banner](assets/banner.jpg)

**Category:** Embedded AI / Computer Vision / Edge Deployment
**Platform:** NVIDIA Jetson Orin NX 16 GB
**Language:** Python 3.11
**Status:** Completed demonstrator - exhibited at a defence and security expo, Essen, September 2026

---

## 1. Purpose

SENTINEL-X is a research demonstrator for real-time aerial reconnaissance on a single,
air-gapped edge computer. It processes a continuous video feed from an unmanned aerial
vehicle, identifies military objects frame by frame, and answers natural-language questions
from an operator about what has been detected so far.

The system is built around one principle: **a person remains in the decision loop at all
times.** The assistant reports. It does not act.

---

## 2. Design Goals

| Goal | Rationale |
|---|---|
| **Air-gapped operation** | No cloud dependency. The system runs entirely on the local board - inference, language model, and UI. A network outage or jamming has no effect. |
| **Single-board deployment** | All components - three detectors, a crop classifier, and a 7-billion-parameter language model - run on one Jetson Orin NX 16 GB. No external compute. |
| **Real-time throughput** | Detection keeps pace with live video at 7-8 fps on embedded hardware while the display plays the clip at full rate. |
| **Human oversight** | The chat interface is explicitly designed to report observations, not to recommend targets or actions. The system prompt prohibits coordinate invention and weapon advice. |
| **Reproducible research** | Three publicly licensed datasets, documented accuracy numbers, and a full unit-test suite make the work verifiable. |

---

## 3. System Overview

The pipeline has four sequential stages. Each stage feeds only its output to the next;
no raw frame data is stored or transmitted beyond the local machine.

```
Aerial video feed (1280 x 720, 24 fps)
        |
        v
  Stage 1 - Object detection
  Three YOLOv8 detectors running in parallel on the GPU.
  Each model was trained on a different public dataset.
  Boxes from all three are merged and deduplicated by spatial overlap.
        |
        v
  Stage 2 - Crop classification
  Every surviving bounding box is cropped and passed to EfficientNet-B0,
  a lightweight CNN fine-tuned on 10 military object classes.
  Crops smaller than 32 pixels on the short side are skipped rather than
  misclassified. A second deduplication pass keeps the highest-confidence
  observation when two boxes overlap.
        |
        v
  Stage 3 - Detection log
  Confirmed observations are accumulated into a per-frame tally:
  class name, confidence score, peak count, total appearances.
  Only frames up to the current playhead position are included,
  so the log reflects exactly what the operator has already seen.
        |
        v
  Stage 4 - Language model
  The operator types a question. The tally - not the raw pixels - is
  sent to a locally hosted Qwen 2.5 7B language model (4-bit quantized).
  The model answers from the evidence in the log. It cannot see the future
  of the clip. The operator reads the answer and decides.
```

---

## 4. Object Detection

### 4.1 Three-detector ensemble

A single detector trained on one dataset inherits that dataset's class vocabulary and
distribution biases. Running three independent models trained on three different datasets
and merging their outputs gives broader class coverage and more robust localisation.

All three detectors use a YOLOv8 backbone with a 640-pixel input. On the Jetson,
TensorRT FP16 engines are compiled from the trained weights at first deployment.
TensorRT fuses layers and quantizes weights to 16-bit floats, reducing inference
time by roughly half compared to the PyTorch original with no meaningful accuracy loss
on this task.

**Inference times (Jetson Orin NX, TensorRT FP16, warm pass):**

| Detector | Warm inference |
|---|---|
| KIIT-MiTA-yolo26s | ~17 ms |
| MV-yolo26s | ~27 ms |
| military-yolov8n | ~28 ms |

### 4.2 Unified label space

The three models were trained on different datasets with different class names. Before
any box is accepted, its class label is mapped onto a shared vocabulary of 10 categories.
Labels with no counterpart in the shared vocabulary - for example, generic civilian
vehicles - are discarded silently. This mapping is the only hand-crafted bridge between
the detector outputs and the rest of the pipeline.

**Unified label space (10 classes):**
Artillery, M. Rocket Launcher, Missile, Radar, Soldier, camouflage_soldier,
military_aircraft, military_vehicle, military_warship, trench.

### 4.3 Deduplication

When two bounding boxes from different models overlap by more than 50% (measured by
intersection-over-union), they describe the same physical object. The box with the lower
YOLO confidence score is discarded. This prevents double-counting and reduces noise in
the tally sent to the language model.

---

## 5. Crop Classification

EfficientNet-B0 was selected for its favourable accuracy-to-parameter ratio on
small-crop image classification tasks. The pretrained ImageNet backbone was kept frozen
and only the classification head was replaced and trained on a purpose-built crop dataset
derived from the three detection datasets.

**EfficientNet-B0 results (test split, 23 September 2026):**

| Metric | Value |
|---|---|
| Test accuracy | 89.01 % |
| Correct predictions | 494 / 555 |
| Input resolution | 224 x 224 px |
| Output classes | 10 |

The classifier provides a second, independent opinion on every YOLO box. If EfficientNet
cannot confidently assign one of the 10 labels to a crop, that box is excluded from the
tally entirely. This means every counted observation has been confirmed by two independent
models.

---

## 6. Language Model Integration

### 6.1 Model choice

Qwen 2.5 7B Instruct (4-bit GGUF quantization) was chosen because it fits entirely in
the 16 GB unified memory of the Jetson Orin NX alongside the two vision models, produces
coherent situational assessments from structured text input, and runs inference locally
through Ollama with no external dependency.

### 6.2 Context design

The model never receives raw video frames. The only context attached to each question is
the plain-text detection tally described in Section 3. This keeps the prompt short,
makes the model's reasoning auditable, and prevents any visual data from leaving the
inference pipeline.

A critical design constraint: **the tally is cut at the current playhead position.**
If the operator is at 00:15 in a 30-second clip, detections from frames after 00:15 are
excluded from the context. The model cannot reveal what has not yet appeared on screen.
Dragging the timeline backwards trims the log accordingly.

### 6.3 Guardrails

The system prompt instructs the model to:
- report only from the categories and counts in the tally
- never invent place names, coordinates, or map references
- never recommend use of force or identify targets
- state uncertainty when the evidence is thin

These constraints are applied at every request, not just at startup.

---

## 7. Software Architecture

### 7.1 Threading model

The application runs four concurrent threads to keep the UI responsive while the GPU
is occupied with inference:

| Thread | Responsibility |
|---|---|
| Main (Tkinter) | UI rendering, user input, pump loop at 30 ms intervals |
| Detection worker | Receives the latest frame, runs YOLO + EfficientNet, stores result |
| Chat worker | Streams tokens from the language model, feeds them to the UI queue |
| GPU monitor | Reads GPU utilization and free memory every 500 ms |

The play loop feeds frames to the detection worker but does not wait for the result.
The worker always processes the most recently submitted frame, discarding any that
arrived while it was busy. This decouples the display rate (24 fps) from the
detection rate (7-8 fps) cleanly.

### 7.2 Model loading sequence

YOLO and EfficientNet are loaded and warmed in the background as soon as the window
opens. Qwen is loaded only after the vision models have finished their warm-up pass.
This sequencing ensures the two model groups never compete for the 16 GB unified
memory during the cold start.

### 7.3 Module layout

The application is split into five focused modules:

| Module | Responsibility |
|---|---|
| `labels.py` | Shared label vocabulary, YOLO-to-CNN name map, path constants |
| `detection.py` | YOLO inference, EfficientNet crop classification, IoU deduplication |
| `chat.py` | Ollama HTTP streaming, context assembly, object-query parsing |
| `gpu.py` | GPU utilization reader with Jetson SoC and nvidia-smi support |
| `window.py` | Tkinter demo window: layout, playback, timeline, chat panel |

---

## 8. Hardware Deployment

The demonstrator runs on a single **NVIDIA Jetson Orin NX 16 GB** with no attached
monitor. The window is forwarded to the connected laptop over an SSH X11 tunnel.
The GPU in this module is reported by CUDA as `Orin`; utilization is read from the
SoC load interface because nvidia-smi does not expose those counters on this platform.

On a standard x86 workstation with a CUDA GPU the same codebase runs identically,
using the original PyTorch weights instead of TensorRT engines.

---

## 9. Testing

The test suite covers the core logic without requiring a GPU, trained weights, or a
video file. All external dependencies - YOLO models, EfficientNet, Ollama, OpenCV
video capture - are either stubbed or not reached by the test paths.

| Test file | Coverage area |
|---|---|
| `test_labels.py` | Label mapping, vocabulary integrity, path constant naming |
| `test_detection.py` | IoU calculation, box deduplication, crop gate, confidence formatting |
| `test_window.py` | Clock formatting, layout arithmetic, GPU reader, DemoWindow widget state |

**49 tests, all passing.** Run from the repository root with:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## 10. Datasets and Licensing

All training data is publicly licensed. The detectors are not redistributed with this
repository. Weights stay on the deployment machine.

| Dataset | Source | License | Use in this project |
|---|---|---|---|
| KIIT-MiTA | Chowdhury, Mendeley Data, DOI 10.17632/drjmrf5kk5.1 | Education and research only | 1,700 drone images, 7 classes - KIIT-MiTA-yolo26s detector |
| Military Assets Dataset | Ryan Madhuwala, Kaggle | CC BY 4.0 | 26,315 images, 12 classes - military-yolov8n detector |
| Military Vehicle Recognition | Roboflow Universe, v7 | CC BY 4.0 | 2,702 images, 3 classes - MV-yolo26s detector |

Full citations are displayed in the **Credit** tab of the running application.

---

## 11. Scope and Limitations

SENTINEL-X is a **research demonstrator**. It is not a fielded system, a certified
product, or a weapon. The following are explicitly out of scope:

- Autonomous target engagement or fire control
- Real-time geolocation or coordinate generation
- Classified or restricted data processing
- Operation outside a controlled demonstration environment

The project was developed to explore edge AI integration for situational awareness
assistance, with a deliberate emphasis on keeping a human operator in every decision
loop.
