import os
import io
from pathlib import Path

import torch
import numpy as np
import cv2
from PIL import Image
from tifffile import imread

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from pipeline import create_yolo_pipeline_from_saved_models

app = FastAPI(
    title="Building Damage Assessment API",
    description="Upload satellite images to detect and classify building damage",
    version="1.0"
)

# --- CORS (allow your Vercel site) ---
FRONTEND_ORIGIN = os.getenv(
    "FRONTEND_ORIGIN",
    "https://building-damage-assessment-app.vercel.app"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = None

@app.on_event("startup")
async def load_models():
    global pipeline
    print("Loading models...")

    device = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    pipeline = create_yolo_pipeline_from_saved_models(
        yolo_model_path=os.getenv("YOLO_MODEL_PATH", "models/best_yolo_50epochs.pt"),
        classification_model_path=os.getenv("CLS_MODEL_PATH", "models/efficientnet_stage3.pth"),
        device=device,
        confidence_threshold=float(os.getenv("CONF_THRESHOLD", "0.1"))
    )

    print(f"✓ Models loaded successfully! Using device: {device}")

@app.get("/")
async def root():
    return {
        "message": "Building Damage Assessment API",
        "version": "1.0",
        "status": "running",
        "endpoints": {
            "POST /predict": "Upload an image for damage assessment",
            "POST /predict-with-image": "Upload an image and get JSON + visualization",
            "GET /health": "Check API health status"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": pipeline is not None}

def _load_uploaded_image(file: UploadFile, contents: bytes) -> np.ndarray:
    if file.filename.lower().endswith((".tif", ".tiff")):
        image = imread(io.BytesIO(contents))
    else:
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image.dtype != np.uint8:
        image = (image / image.max() * 255).clip(0, 255).astype(np.uint8)

    return image

@app.post("/predict-with-image")
async def predict_with_visualization(file: UploadFile = File(...)):
    try:
        if pipeline is None:
            raise HTTPException(500, "Models not loaded. Please wait and try again.")

        if not file.filename.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
            raise HTTPException(400, "Unsupported file format")

        contents = await file.read()
        image = _load_uploaded_image(file, contents)

        results = pipeline.process_image(image)
        json_results = pipeline.results_to_json(results)

        vis_image = generate_visualization_image(results)

        import base64
        original_pil = Image.fromarray(image)
        original_buffer = io.BytesIO()
        original_pil.save(original_buffer, format="PNG")
        original_base64 = base64.b64encode(original_buffer.getvalue()).decode()

        vis_pil = Image.fromarray(vis_image)
        vis_buffer = io.BytesIO()
        vis_pil.save(vis_buffer, format="PNG")
        vis_base64 = base64.b64encode(vis_buffer.getvalue()).decode()

        json_results["images"] = {
            "original": f"data:image/png;base64,{original_base64}",
            "visualization": f"data:image/png;base64,{vis_base64}",
        }

        return JSONResponse(content=json_results)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")

@app.post("/predict")
async def predict_damage(file: UploadFile = File(...)):
    # optional: keep this simpler endpoint if you want JSON only
    try:
        if pipeline is None:
            raise HTTPException(500, "Models not loaded. Please wait and try again.")

        if not file.filename.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
            raise HTTPException(400, "Unsupported file format")

        contents = await file.read()
        image = _load_uploaded_image(file, contents)

        results = pipeline.process_image(image)
        json_results = pipeline.results_to_json(results)
        return JSONResponse(content=json_results)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")

def generate_visualization_image(results: dict) -> np.ndarray:
    image = results["original_tif_image"].copy()

    damage_colors = {
        "no-damage": (0, 255, 0),
        "minor-damage": (255, 255, 0),
        "major-damage": (255, 165, 0),
        "destroyed": (255, 0, 0),
    }

    for building in results["buildings"]:
        x1, y1, x2, y2 = building["bbox"]
        color = damage_colors.get(building["damage_class"], (255, 255, 255))

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

        label = f"{building['damage_class']}: {building['damage_confidence']:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        cv2.rectangle(image, (x1, y1 - th - 10), (x1 + tw + 5, y1), color, -1)
        cv2.putText(image, label, (x1 + 2, y1 - 5), font, font_scale, (0, 0, 0), thickness)

    return image

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
