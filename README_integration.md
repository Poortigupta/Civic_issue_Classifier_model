# Civic Issue Classifier API

A FastAPI-based image classification model that detects civic infrastructure issues from photos.

## What It Does

Classifies images into 4 categories:
- **electricity_wire_issue** - Damaged/loose electrical wires
- **garbage_dump** - Improper waste/litter
- **road_pothole** - Potholes and road damage
- **unknown** - Non-civic content

Uses two-stage validation:
1. **Energy score** - Rejects completely unrelated images (dogs, food, etc.)
2. **Confidence threshold** - Per-class confidence checks before classification

## Setup

```bash
pip install -r requirements.txt
python main.py
```
Server runs at: `http://localhost:8000`

## Endpoints

### 1. **GET** `/` 
Health check - verify API is running
```json
{
  "message": "Samadhan Civic Classifier API is running",
  "version": "2.0"
}
```

### 2. **GET** `/health`
Returns model config
```json
{
  "status": "ok",
  "classes": ["electricity_wire_issue", "garbage_dump", "road_pothole", "unknown"],
  "temperature": 2.0
}
```

### 3. **POST** `/predict_civic_issue`
**Input:** Multipart form with image file  
**Content-Type:** Only image files (jpg, png, etc.)

## Response Format

```json
{
  "status": "classified | uncertain | rejected",
  "predicted_class": "electricity_wire_issue | garbage_dump | road_pothole | not_a_civic_issue | null",
  "confidence": 0.95,
  "reason": "null or explanation string",
  "probs": {
    "electricity_wire_issue": 0.02,
    "garbage_dump": 0.01,
    "road_pothole": 0.95,
    "unknown": 0.02
  }
}
```

## Response Types

| Status | When | Action |
|--------|------|--------|
| **classified** | High confidence civic issue detected | Use `predicted_class` value |
| **uncertain** | Detected but below confidence threshold | Show all probabilities, ask user to confirm |
| **rejected** | Not a civic issue OR low quality | Reject image, ask user to resubmit |

## Example Integration

**cURL:**
```bash
curl -X POST "http://localhost:8000/predict_civic_issue" \
  -H "accept: application/json" \
  -F "file=@image.jpg"
```

**Python:**
```python
import requests

with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict_civic_issue",
        files={"file": f}
    )
    result = response.json()
    print(result["predicted_class"])  # e.g., "road_pothole"
    print(result["confidence"])       # e.g., 0.95
```

## Key Points

- ✅ Resamples all images to 224×224 automatically
- ✅ Returns per-class probabilities for transparency
- ✅ Handles corrupted/invalid images gracefully
- ✅ Confidence threshold prevents false positives

## Error Codes

- **400** - File is not an image
- **422** - Corrupted or unreadable image file
