# SENTINEL-X

Research demonstrator for aerial reconnaissance. The whole pipeline runs on one Jetson Orin NX: three YOLO detectors place boxes, EfficientNet-B0 names each crop, and a local Qwen2.5 7B answers from that tally. A person remains in the decision loop.

Architecture details are in [SENTINEL-X Project Description.md](SENTINEL-X%20Project%20Description.md).

## Repository layout

```
run.py               - entry point: python run.py
sentinel/
  __init__.py
  labels.py          - CNN class list, YOLO-to-CNN name map, weight/video paths
  detection.py       - YOLO loading, EfficientNet crop classifier, box dedup
  chat.py            - Qwen streaming chat, greeting, object-query helpers
  gpu.py             - GPU percent and free-memory reader (Orin + nvidia-smi)
  window.py          - Tkinter demo window (video pane + chat pane)
tests/
  test_labels.py
  test_detection.py
  test_window.py
```

## On your machine, not in Git

- `.env` (copy from `.env.example`)
- image datasets and crop folders under `Data/`
- video `Data/video/combined_drone.mp4`
- weights (not committed - download or train separately):
  - `Data/YOLO-Models-Final/KIIT-MiTA-yolo26s.pt`
  - `Data/YOLO-Models-Final/MV-yolo26s.pt`
  - `Data/YOLO-Models-Final/military-yolov8n.pt`
  - `Data/EffNet_b0.pt`

On the Jetson a `.engine` beside a `.pt` is loaded instead of that `.pt`.

## Run the window

Requires the weights, the clip, CUDA, and the local Ollama model:

```bash
python run.py
```

Detection refuses CPU. The chat model is the local Ollama tag `qwen2.5:7b-instruct-q4_K_M`.

## Run the tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

No GPU or weights needed - all tests mock or stub external dependencies.
