from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
import io
from PIL import Image
import base64
from typing import List, Dict, Any

# Import your custom model class
from model import PCBDefectModel

# Initialize FastAPI app
app = FastAPI(title="PCB Defect Detector", version="1.0.0")

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend) - go up one directory to find frontend folder
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

# Global model instance
pcb_model = None

@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global pcb_model
    try:
        pcb_model = PCBDefectModel("best.pt")  # Try custom model first
        if not pcb_model.load_model():
            # Fallback to pretrained
            pcb_model = PCBDefectModel()
            pcb_model.load_model()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        pcb_model = None

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert uploaded image bytes to OpenCV format"""
    try:
        # Convert bytes to PIL Image
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array (OpenCV format)
        image_array = np.array(pil_image)
        
        return image_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")

def draw_predictions(image: np.ndarray, detections: List[Dict[str, Any]]) -> str:
    """Draw bounding boxes on image and return as base64 string"""
    img_copy = image.copy()
    
    for detection in detections:
        bbox = detection["bbox"]
        label = detection["label"]
        confidence = detection["confidence"]
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw bounding box (green)
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label with confidence
        label_text = f"{label}: {confidence:.2f}"
        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        
        # Draw label background
        cv2.rectangle(img_copy, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(img_copy, label_text, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Convert to base64 for frontend display
    _, buffer = cv2.imencode('.jpg', img_copy)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return img_base64

@app.post("/predict")
async def predict_defects(file: UploadFile = File(...), confidence: float = 0.3):
    """Main prediction endpoint
    
    Args:
        file: Uploaded PCB image
        confidence: Confidence threshold for predictions (0.0-1.0)
    
    Returns:
        JSON with detections and annotated image
    """
    global pcb_model
    
    if pcb_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and preprocess image
        image_bytes = await file.read()
        image = preprocess_image(image_bytes)
        
        # Run inference using model class
        results = pcb_model.predict_image(image, conf_threshold=confidence)
        
        # Process predictions (model.py handles this)
        detections = pcb_model.process_results(results, confidence_threshold=confidence)
        
        # Generate annotated image
        annotated_image_b64 = draw_predictions(image, detections)
        
        return JSONResponse({
            "detections": detections,
            "total_defects": len(detections),
            "model_info": pcb_model.get_model_info(),
            "annotated_image": annotated_image_b64
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model"""
    global pcb_model
    
    if pcb_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return pcb_model.get_model_info()

@app.post("/model/retrain")
async def retrain_model(epochs: int = 10, data_path: str = "data/dataset.yaml"):
    """Retrain the model with new data (optional endpoint)"""
    global pcb_model
    
    if pcb_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        results = pcb_model.train_model(data_path, epochs=epochs)
        return {"message": "Model retrained successfully", "results": str(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": pcb_model is not None,
        "model_info": pcb_model.get_model_info() if pcb_model else None,
        "message": "PCB Defect Detector API is running"
    }

@app.get("/")
async def redirect_to_frontend():
    """Redirect root to frontend"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)