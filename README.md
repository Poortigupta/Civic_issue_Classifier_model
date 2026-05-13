# Samadhan — Civic Issue Classifier

AI-powered image classification API that automatically identifies civic infrastructure problems from photos submitted by citizens.

---

## Overview

Samadhan uses a fine-tuned MobileNetV3-Small model to classify citizen-submitted images into three civic issue categories, while rejecting irrelevant or out-of-distribution images entirely. Built for integration into civic complaint portals and mobile apps.

**Model version:** v2  
**Framework:** PyTorch + FastAPI  
**Deployment target:** REST API (local or cloud)

---

## Classes

| Class | Description | Val F1 |
|-------|-------------|--------|
| `electricity_wire_issue` | Dangling wires, transformer faults, electrical hazards | 0.863 |
| `garbage_dump` | Illegal dumping, overflowing bins, waste accumulation | 0.884 |
| `road_pothole` | Potholes, road cracks, waterlogged road damage | 0.833 |
| `unknown` | Non-civic images — dogs, food, people, nature, etc. | 0.953 |

The `unknown` class acts as a rejection gate. Any image the model isn't confident is a civic issue gets returned as `not_a_civic_issue` rather than a wrong prediction.

---

## Model Architecture

```
Base model   : MobileNetV3-Small (timm — pretrained on ImageNet)
Input size   : 224 × 224 × 3 (RGB)
Output       : 4-class softmax head
Parameters   : ~2.5M (lightweight, mobile-friendly)
Export format: TorchScript (.pt) — no Python needed at inference
```

### Key design decisions

**Temperature scaling** — logits are divided by `T=2.0` before softmax. This prevents the model from outputting artificially high confidence (e.g. 0.99) on uncertain images. Calibrated on the validation set.

**Energy-score OOD detection** — a physics-inspired score computed from raw logits. Deeply out-of-distribution images (like animals or food) are caught at this stage before softmax is even computed.

**Per-class confidence thresholds** — each class has its own minimum confidence requirement. `road_pothole` uses a stricter threshold (0.70) because its raw precision was lower than other classes.

**Weighted random sampler** — training batches are balanced across classes regardless of how many images each class has, preventing the dominant class from monopolising learning.

---

## Project Structure

```
samadhan/
├── civic_classifier_v2.pt          # TorchScript model (inference-ready)
├── best_civic_classifier_v2.pth    # Raw weights (for further fine-tuning)
├── main.py                         # FastAPI application
├── samadhan_model_v2.py            # Full training script (Colab)
├── training_curves.png             # Accuracy + loss plot from training
└── README.md

civic_dataset/
├── train/
│   ├── electricity_wire_issue/
│   ├── garbage_dump/
│   ├── road_pothole/
│   └── unknown/                    # Diverse non-civic images for OOD training
└── val/
    ├── electricity_wire_issue/
    ├── garbage_dump/
    ├── road_pothole/
    └── unknown/
```

---

## Training

Trained on Google Colab (T4 GPU) using `samadhan_model_v2.py`.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 20 (with early stopping, patience=5) |
| Batch size | 32 |
| Optimizer | AdamW |
| Learning rate | 3e-4 (cosine annealed to 1e-6) |
| Weight decay | 1e-4 |
| Label smoothing | 0.05 |
| Dropout | 0.3 (before classifier head) |
| Mixed precision | Yes (float16 via GradScaler) |
| Gradient clipping | max_norm=1.0 |

### Augmentations (train only)

- Random horizontal + vertical flip
- Random rotation ±15°
- Color jitter (brightness, contrast, saturation, hue)
- Random grayscale (5%)
- MixUp (α=0.3)
- ImageNet normalisation

### Training results

```
Epoch   Train Acc   Val Acc   
──────────────────────────────
 1       65.6%      69.0%
 6       95.0%      86.2%
10       96.8%      89.2%
15       99.2%      91.8%  ← best checkpoint saved
20       99.3%      89.6%  (early stopping would fire here)
──────────────────────────────
Best val accuracy: 91.8%
Macro avg F1    : 0.883
```

### Class weights

Computed with `sklearn compute_class_weight('balanced')`, clipped to [0.5, 3.0] and re-normalised. This handles dataset imbalance without distorting the loss surface.

---

## API

### Run locally

```bash
pip install fastapi uvicorn torch torchvision pillow
python main.py
# Server starts at http://0.0.0.0:8000
```

### Endpoints

#### `GET /`
Health check.
```json
{ "message": "Samadhan Civic Classifier API is running", "version": "2.0" }
```

#### `GET /health`
Returns loaded class list and temperature setting.
```json
{
  "status": "ok",
  "classes": ["electricity_wire_issue", "garbage_dump", "road_pothole", "unknown"],
  "temperature": 2.0
}
```

#### `POST /predict_civic_issue`
Accepts a multipart image upload. Returns classification result.

**Request**
```bash
curl -X POST http://localhost:8000/predict_civic_issue \
  -F "file=@pothole.jpg"
```

**Response — classified**
```json
{
  "status": "classified",
  "predicted_class": "road_pothole",
  "confidence": 0.759,
  "reason": null,
  "probs": {
    "electricity_wire_issue": 0.104238,
    "garbage_dump": 0.076506,
    "road_pothole": 0.758953,
    "unknown": 0.060303
  }
}
```

**Response — rejected (OOD image, e.g. a dog photo)**
```json
{
  "status": "rejected",
  "predicted_class": "not_a_civic_issue",
  "confidence": null,
  "reason": "Image does not appear to be a civic issue (energy=3.21)",
  "probs": null
}
```

**Response — uncertain (low confidence)**
```json
{
  "status": "uncertain",
  "predicted_class": null,
  "confidence": 0.58,
  "reason": "Low confidence (58.0%) — please select issue type manually",
  "probs": { ... }
}
```

### Status codes

| Status | Meaning | Frontend action |
|--------|---------|-----------------|
| `classified` | Model is confident — prediction is reliable | Auto-fill issue type |
| `uncertain` | Model is unsure — show `probs` as hint | Show dropdown, pre-select top class |
| `rejected` | Not a civic issue image | Ask user to retake photo |

---

## Configuration

All tunable values are at the top of `main.py`:

```python
TEMPERATURE      = 2.0     # Softens overconfident softmax. Range: 1.5–3.0
ENERGY_THRESHOLD = 10.0    # OOD rejection gate. Set from calibration output.
                           # Run Section 15 of training script to get your value.

CLASS_THRESHOLDS = {
    'electricity_wire_issue': 0.65,
    'garbage_dump':           0.65,
    'road_pothole':           0.70,  # stricter — lower precision class
    'unknown':                0.65,
}
```

### Calibrating ENERGY_THRESHOLD

After training, run this in Colab on your val set to get the correct value:

```python
model.eval()
all_energies = []
with torch.no_grad():
    for imgs, _ in val_loader:
        logits = model(imgs.to(device))
        energy = -TEMPERATURE * torch.logsumexp(logits / TEMPERATURE, dim=1)
        all_energies.extend(energy.cpu().numpy())

import numpy as np
print(f"Set ENERGY_THRESHOLD to: {np.percentile(all_energies, 95):.2f}")
```

Paste that printed value into `ENERGY_THRESHOLD` in `main.py`.

---

## Known Limitations

**`electricity_wire_issue` val set is small (25 images)** — the reported F1 of 0.863 is statistically unreliable. A 2–3 image swing changes the score by ±4%. Collect at least 80 more val images for this class before trusting its metrics.

**`road_pothole` precision is 73.7%** — approximately 1 in 4 pothole predictions is a false positive. Mitigated in production by the stricter 0.70 confidence threshold. Will improve with more diverse training data.

**Watermarked stock images in training data** — if training images contain watermarks (e.g. Vecteezy, Shutterstock), the model may learn watermark features as a shortcut. Replace these with clean real-world photos.

**Night / extreme weather images** — the training set likely skews toward daytime, well-lit photos. Performance on night shots, heavy rain, or direct glare has not been evaluated.

---

## Improving the Model

Collect real-world images submitted by actual users in production — these improve the next training run far more than any synthetic augmentation. Target the following before retraining:

| Class | Current train size | Target |
|-------|--------------------|--------|
| `electricity_wire_issue` | ~200 | 500+ |
| `garbage_dump` | ~500 | maintain |
| `road_pothole` | ~300 | 500+ |
| `unknown` | ~500 | 600+ (diverse) |

Sources for `unknown` class images: [Intel Image Classification (Kaggle)](https://www.kaggle.com/datasets/puneet6060/intel-image-classification), [Animals-10 (Kaggle)](https://www.kaggle.com/datasets/alessiocorrado99/animals10), CIFAR-10 exports.

---

## Requirements

```
torch>=2.0
torchvision>=0.15
timm>=0.9
fastapi>=0.100
uvicorn>=0.23
pillow>=9.0
scikit-learn>=1.3
numpy>=1.24
matplotlib>=3.7
```

---

## License

Built for the Samadhan civic grievance redressal platform. Not licensed for commercial redistribution.