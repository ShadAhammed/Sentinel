# Edge Impulse CNN Workflow

This workflow trains the MCXN947 verifier model from `Data/CNN-Data`.

The model is an image classifier over object crops. The Jetson side should still
do object localization; the MCXN947 model verifies cropped candidate objects.

## Current Status

The uploader is ready, but upload needs either:

- an Edge Impulse key that can list projects and find `Sentinel-X` by name; or
- the numeric Edge Impulse project ID passed with `--project-id`.

The latest API key has the `ingestion_deployment` role. That role can be used
for upload only after the target project ID is known.

The script also detects an Edge Impulse HMAC key from `.env` and passes it to
the uploader for signed samples. The HMAC key does not replace the API key and
does not grant project lookup permissions.

## Dataset Mapping

- Local `train` and `validation` crops upload to Edge Impulse `training`.
- Local `test` crops upload to Edge Impulse `testing`.
- Labels come directly from `Data/CNN-Data/labels.csv`.
- Total samples prepared by the uploader: 43,521.

## Commands

Check counts without uploading:

```powershell
python Data\upload_to_edge_impulse.py --dry-run
```

Upload after `.env` contains the correct project API key:

```powershell
python Data\upload_to_edge_impulse.py --chunk-size 100
```

Upload when the key cannot list projects but you know the `Sentinel-X` project
ID:

```powershell
python Data\upload_to_edge_impulse.py --project-id 123456 --chunk-size 100
```

If the key uses a different variable name:

```powershell
python Data\upload_to_edge_impulse.py --key-name EDGE_IMPULSE_API_KEY --chunk-size 100
```

## Edge Impulse Model Setup

In Edge Impulse Studio, use an image classification impulse:

- Input: image crop.
- Processing block: Image.
- Learning block: Classification.
- Start with a small input size such as 96x96 or 160x160.
- Use RGB first if memory allows; switch to grayscale if MCXN947 estimates are
  too large.
- Train a compact transfer-learning model or other microcontroller-suitable CNN.
- Enable int8 quantization and compare quantized accuracy against float accuracy.

## MCXN947 Export

Use the Edge Impulse deployment option that best matches the NXP integration
path:

- C++ library for firmware integration.
- EON compiled output if supported for the selected target.
- Check Edge Impulse RAM, flash, latency, and ops estimates before calling the
  model MCXN947-ready.

## Validation Checklist

- Confirm upload counts match the dry-run summary.
- Inspect class balance and confusion matrix after training.
- Pay special attention to tiny classes such as `trench` and `civilian`.
- Record float accuracy, int8 accuracy, RAM, flash, and latency.
- Use the Edge Impulse testing set for final verifier reporting.
