# SENTINEL-X

Research demonstrator for aerial reconnaissance on a Jetson Orin NX. Three YOLO detectors place boxes, EfficientNet-B0 names each crop, and a local Qwen2.5 7B answers from that tally. A person remains in the decision.

The architecture is in [SENTINEL-X Project Description.md](SENTINEL-X%20Project%20Description.md).

## In this repository

Python scripts, tests, YOLO `data.yaml` files, and the notes under `Data/`.

## On your machine, not in Git

- `.env` (copy from `.env.example`)
- image datasets and crop folders
- video (`*.mp4`)
- weights: `*.pt`, `*.engine`, `*.pkl`

The demo expects these files next to the scripts:

- `Data/YOLO-Models-Final/KIIT-MiTA-yolo26s.pt`
- `Data/YOLO-Models-Final/MV-yolo26s.pt`
- `Data/YOLO-Models-Final/military-yolov8n.pt`
- `Data/EffNet_b0.pt`
- `Data/video/combined_drone.mp4`

On the Jetson, a `.engine` beside a `.pt` is loaded instead of that `.pt`.

## Run the window

From `Data/video` on a machine with the weights, the clip, and CUDA:

```text
python sentinel_demo.py
```

Detection refuses CPU. The chat model is the local Ollama tag `qwen2.5:7b-instruct-q4_K_M`.
