# SENTINEL-X

SENTINEL-X is a research demonstrator for aerial reconnaissance on a disconnected edge computer. A person remains in the decision. The assistant reports what the detectors found. It does not take a military action.

The whole pipeline runs on one board, an NVIDIA Jetson Orin NX, in one window: `Data/video/sentinel_demo.py`. There is no second processor in this project.

## What runs

```text
combined_drone.mp4
        |
        v
Three YOLO detectors on the GPU (input 640)
  KIIT-MiTA-yolo26s
  MV-yolo26s
  military-yolov8n
  A TensorRT .engine beside a .pt is used when that file exists.
        |
        v
Each box is cropped
        |
        v
EfficientNet-B0 names the crop (10 classes, 224x224)
  A crop whose short side is under 32 pixels is skipped.
        |
        v
Boxes on the frame, and a tally of those names
        |
        v
Qwen2.5 7B on local Ollama (qwen2.5:7b-instruct-q4_K_M)
  The question is answered from the tally up to the frame on screen.
        |
        v
Operator, in the chat panel. The person decides.
```

The picture plays at the clip rate. Detection runs on its own thread and draws the newest boxes on the current frame. The window loads and warms YOLO and EfficientNet first, then loads Qwen, so the two do not share the GPU during the cold start. Detection uses `cuda:0`. CPU is refused. The Orin DLA is not used.

On the Orin, a warm TensorRT FP16 pass is about 17 ms, 27 ms, and 28 ms for the three detectors. The picture plays at 24 fps. Detection on that board is about 7 to 8 fps.

## Detection

The three weights were trained on three datasets. Their class names are mapped onto one list of 10 labels before a box is kept:

Artillery, M. Rocket Launcher, Missile, Radar, Soldier, camouflage_soldier, military_aircraft, military_vehicle, military_warship, trench.

YOLO places the box. EfficientNet names the crop. The name drawn on the box is the EfficientNet name. A YOLO box that EfficientNet does not name is left out of the count. When two boxes overlap by more than half, one observation is kept: the higher EfficientNet score.

The window plays `combined_drone.mp4`. That clip is 1280x720, 24 fps, 29.67 s, 712 frames.

## Chat

The operator types in the chat panel on the same window. The chat column is 40 percent of the detection row. The video is 60 percent. After a short pause the window types a short introduction. Ask and Send stay off until that line is finished.

Answers use the frame on screen (`shown_seconds`). Frames logged later than that time stay out of the prompt, including after the reader has reached the end of the file. A drag on the timeline moves that cutoff at once. After the picture reaches the end, an answer may cover the whole clip.

The only context sent with a question is that text tally. Qwen does not see the pixels. Asking to see a class, for example "show me the warship", opens the clearest crop of that class from the frames so far.

Answers are in English. The prompt tells the model to stay with the categories and counts, to avoid invented places, and to avoid weapon advice.

GPU use and free memory sit under the chat. On the Orin the percent comes from `/sys/devices/platform/gpu.0/load`.

If the board has no monitor, `attach_desktop()` keeps an SSH X11 tunnel when one is already set. Otherwise it draws on the SSH client's X server.

The Credit tab names KIIT-MiTA, military_object_dataset, and MV, with the licenses recorded in the source.

## Weights

The window does not train. It loads weights that already exist on the machine:

- `Data/YOLO-Models-Final/KIIT-MiTA-yolo26s.pt`
- `Data/YOLO-Models-Final/MV-yolo26s.pt`
- `Data/YOLO-Models-Final/military-yolov8n.pt`
- `Data/EffNet_b0.pt`

On the Jetson, a TensorRT `.engine` beside a `.pt` is loaded instead of that `.pt`.

EfficientNet-B0, measured on the crop test split on 23 September 2026: accuracy 0.8901 (494/555).

## What a clone contains

The demo window, the two helpers it imports, their tests, and these notes. Weights, image sets, and video stay on the machine that runs the demo.
