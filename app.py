# app.py
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from tifffile import imread
import io
from typing import Optional
import cv2
from PIL import Image

# Import your pipeline
from pipeline import create_yolo_pipeline_from_saved_models

app = FastAPI(
    title="Building Damage Assessment API",
    description="Upload satellite images to detect and classify building damage",
    version="1.0"
)

# Enable CORS so web browsers can access your API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store the pipeline
pipeline = None

@app.on_event("startup")
async def load_models():
    """Load models when the API starts"""
    global pipeline
    print("Loading models...")
    
    # Update these paths to where your models are located
    pipeline = create_yolo_pipeline_from_saved_models(
        yolo_model_path='models/best_yolo_50epochs.pt',
        classification_model_path='models/efficientnet_stage3.pth',
        device='cpu',  # Using CPU for deployment
        confidence_threshold=0.1
    )
    
    print("✓ Models loaded successfully!")

@app.get("/")
async def root():
    """API information endpoint"""
    return {
        "message": "Building Damage Assessment API",
        "version": "1.0",
        "status": "running",
        "endpoints": {
            "POST /predict": "Upload an image for damage assessment",
            "GET /health": "Check API health status"
        }
    }

@app.get("/health")
async def health_check():
    """Check if the API and models are ready"""
    return {
        "status": "healthy",
        "models_loaded": pipeline is not None
    }

@app.post("/predict")
async def predict_damage(file: UploadFile = File(...)):
    """
    Upload a satellite image and get building damage assessment
    
    Accepts: .tif, .tiff, .png, .jpg, .jpeg files
    Returns: JSON with detected buildings and damage classifications
    """
    try:
        # Check if models are loaded
        if pipeline is None:
            raise HTTPException(500, "Models not loaded. Please wait and try again.")
        
        # Validate file format
        if not file.filename.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
            raise HTTPException(
                400, 
                "Unsupported file format. Please upload .tif, .png, or .jpg files"
            )
        
        print(f"Processing image: {file.filename}")
        
        # Read the uploaded file
        contents = await file.read()
        
        # Load image based on file type
        if file.filename.lower().endswith(('.tif', '.tiff')):
            image = imread(io.BytesIO(contents))
        else:
            # For PNG/JPG
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image / image.max() * 255).clip(0, 255).astype(np.uint8)
        
        print(f"Image loaded: {image.shape}")
        
        # Run the pipeline
        results = pipeline.process_image(image)
        
        # Convert to JSON format
        json_results = pipeline.results_to_json(results)
        
        print(f"✓ Detected {json_results['metadata']['total_buildings']} buildings")
        
        return JSONResponse(content=json_results)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(500, f"Prediction failed: {str(e)}")
    
@app.post("/predict-with-image")
async def predict_with_visualization(file: UploadFile = File(...)):
    """
    Upload a satellite image and get both JSON results AND visualization image
    """
    try:
        if pipeline is None:
            raise HTTPException(500, "Models not loaded. Please wait and try again.")
        
        if not file.filename.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
            raise HTTPException(400, "Unsupported file format")
        
        print(f"Processing image with visualization: {file.filename}")
        
        # Read the uploaded file
        contents = await file.read()
        
        # Load image
        if file.filename.lower().endswith(('.tif', '.tiff')):
            image = imread(io.BytesIO(contents))
        else:
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if image.dtype != np.uint8:
            image = (image / image.max() * 255).clip(0, 255).astype(np.uint8)
        
        # Run the pipeline
        results = pipeline.process_image(image)
        
        # Convert to JSON
        json_results = pipeline.results_to_json(results)
        
        # Generate visualization image
        vis_image = generate_visualization_image(results)
        
        # Convert original and visualization images to base64
        import base64
        
        # Original image
        original_pil = Image.fromarray(image)
        original_buffer = io.BytesIO()
        original_pil.save(original_buffer, format='PNG')
        original_base64 = base64.b64encode(original_buffer.getvalue()).decode()
        
        # Visualization image
        vis_pil = Image.fromarray(vis_image)
        vis_buffer = io.BytesIO()
        vis_pil.save(vis_buffer, format='PNG')
        vis_base64 = base64.b64encode(vis_buffer.getvalue()).decode()
        
        # Add images to results
        json_results['images'] = {
            'original': f'data:image/png;base64,{original_base64}',
            'visualization': f'data:image/png;base64,{vis_base64}'
        }
        
        print(f"✓ Detected {json_results['metadata']['total_buildings']} buildings with visualization")
        
        return JSONResponse(content=json_results)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(500, f"Prediction failed: {str(e)}")


def generate_visualization_image(results: dict) -> np.ndarray:
    """
    Generate visualization with bounding boxes and labels
    """
    image = results['original_tif_image'].copy()
    
    damage_colors = {
        'no-damage': (0, 255, 0),        # Green
        'minor-damage': (255, 255, 0),    # Yellow
        'major-damage': (255, 165, 0),    # Orange
        'destroyed': (255, 0, 0)          # Red
    }
    
    for building in results['buildings']:
        x1, y1, x2, y2 = building['bbox']
        color = damage_colors.get(building['damage_class'], (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        
        # Add label with background
        label = f"{building['damage_class']}: {building['damage_confidence']:.2f}"
        
        # Get text size for background rectangle
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw background rectangle for text
        cv2.rectangle(image, 
                     (x1, y1 - text_height - 10), 
                     (x1 + text_width + 5, y1), 
                     color, -1)
        
        # Draw text
        cv2.putText(image, label, (x1 + 2, y1 - 5), 
                   font, font_scale, (0, 0, 0), thickness)
    
    return image
    
# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/app")
async def serve_app():
    """Serve the web interface"""
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)