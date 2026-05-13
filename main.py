from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import torch.jit
import torch.nn.functional as F
import io
from PIL import Image
from torchvision import transforms
import uvicorn

app = FastAPI(title="Samadhan Civic Issue Classifier API")

# ─────────────────────────────────────────────────────────────────────
# CONFIG  — mirror these values from samadhan_model_v2.py Section 4
# ─────────────────────────────────────────────────────────────────────

# 4 classes now — must match the order ImageFolder assigned during training
# (alphabetical by folder name — double check with train_ds.class_to_idx in Colab)
CLASSES = [
    'electricity_wire_issue',
    'garbage_dump',
    'road_pothole',
    'unknown',          # ← added in v2
]

TEMPERATURE      = 2.0      # softens overconfident softmax — tune 1.5–3.0
ENERGY_THRESHOLD = 10.0    # above this = OOD — update from Section 15 calibration output

# Per-class confidence thresholds
# road_pothole is stricter because its precision was only 73.7%
CLASS_THRESHOLDS = {
    'electricity_wire_issue': 0.65,
    'garbage_dump':           0.65,
    'road_pothole':           0.70,
    'unknown':                0.65,
}

# ─────────────────────────────────────────────────────────────────────
# MODEL LOAD
# ─────────────────────────────────────────────────────────────────────

model = torch.jit.load('civic_classifier_v2.pt', map_location='cpu')
model.eval()
print(f"Model loaded — {len(CLASSES)} classes: {CLASSES}")

# ─────────────────────────────────────────────────────────────────────
# PREPROCESSING  — must exactly match val_transform in training script
# ─────────────────────────────────────────────────────────────────────

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

# ─────────────────────────────────────────────────────────────────────
# INFERENCE HELPER
# ─────────────────────────────────────────────────────────────────────

def run_predict(tensor: torch.Tensor) -> dict:
    """
    Two-stage OOD detection:
      Stage 1 — Energy score    : catches deeply out-of-distribution images (dogs, food, etc.)
      Stage 2 — Per-class conf  : catches uncertain predictions within known classes
    """
    with torch.no_grad():
        logits = model(tensor)                              # shape [1, 4]

        # ── Stage 1: Energy Score ─────────────────────────────────────
        energy = (-TEMPERATURE * torch.logsumexp(logits / TEMPERATURE, dim=1)).item()

        if energy > ENERGY_THRESHOLD:
            return {
                "status"          : "rejected",
                "predicted_class" : "not_a_civic_issue",
                "confidence"      : None,
                "reason"          : f"Image does not appear to be a civic issue (energy={energy:.2f})",
                "probs"           : None,
            }

        # ── Temperature-scaled softmax ────────────────────────────────
        probs           = F.softmax(logits / TEMPERATURE, dim=1).squeeze()   # shape [4]
        confidence, idx = probs.max(0)
        confidence      = confidence.item()
        class_name      = CLASSES[idx.item()]

        prob_dict = {CLASSES[i]: round(probs[i].item(), 6) for i in range(len(CLASSES))}

        # ── Stage 2: Per-class Confidence Threshold ───────────────────
        threshold = CLASS_THRESHOLDS.get(class_name, 0.65)

        if confidence < threshold:
            return {
                "status"          : "uncertain",
                "predicted_class" : None,
                "confidence"      : round(confidence, 4),
                "reason"          : f"Low confidence ({confidence:.1%}) — please select issue type manually",
                "probs"           : prob_dict,
            }

        # If predicted class is 'unknown', also reject — model is saying it's not civic
        if class_name == "unknown":
            return {
                "status"          : "rejected",
                "predicted_class" : "not_a_civic_issue",
                "confidence"      : round(confidence, 4),
                "reason"          : "Image classified as non-civic content",
                "probs"           : prob_dict,
            }

        return {
            "status"          : "classified",
            "predicted_class" : class_name,
            "confidence"      : round(confidence, 4),
            "reason"          : None,
            "probs"           : prob_dict,
        }

# ─────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Samadhan Civic Classifier API is running", "version": "2.0"}


@app.get("/health")
def health():
    return {"status": "ok", "classes": CLASSES, "temperature": TEMPERATURE}


@app.post("/predict_civic_issue")
async def predict_civic_issue(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are accepted")

    # Read and decode image
    try:
        contents = await file.read()
        image    = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=422, detail="Could not read image — file may be corrupted")

    # Preprocess and predict
    tensor = transform(image).unsqueeze(0)   # [1, 3, 224, 224]
    result = run_predict(tensor)

    # Debug log — remove in production if needed
    print(f"[{file.filename}] → {result['status']} | class={result['predicted_class']} | conf={result['confidence']}")

    return JSONResponse(content=result)


# ─────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)