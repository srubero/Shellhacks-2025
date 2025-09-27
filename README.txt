# 🛠️ PCB Defect Detector

A FastAPI-based web application that uses YOLOv8 to detect defects in PCB images. Perfect for hackathons and rapid prototyping!

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone or create project directory
mkdir pcb-defect-detector
cd pcb-defect-detector

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application

```bash
# Start the FastAPI server
python main.py

# Or use uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Access the Application

- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📁 Project Structure

```
pcb-defect-detector/
│── main.py              # FastAPI application
│── model.py             # Model utilities and training
│── requirements.txt     # Python dependencies
│── README.md           # This file
│
│── models/             # Trained models (create this folder)
│   └── best.pt         # Your custom trained model (optional)
│
│── data/               # Dataset (create this folder)
│   ├── train/          # Training images
│   ├── val/            # Validation images
│   └── test/           # Test images
│
└── sample_images/      # Sample PCB images for testing
```

## 🎯 API Endpoints

### `POST /predict`
Upload PCB image for defect detection

**Request**: Multipart form with image file
**Response**:
```json
{
  "detections": [
    {
      "label": "solder_bridge",
      "confidence": 0.92,
      "bbox": [x1, y1, x2, y2]
    }
  ],
  "total_defects": 1,
  "annotated_image": "base64_encoded_image"
}
```

### `GET /health`
Health check endpoint

### `GET /`
Serves the web interface

## 🤖 Model Information

### Default Behavior
- Uses pretrained YOLOv8n model initially
- Automatically switches to custom model if `best.pt` exists
- Confidence threshold: 0.3 (adjustable)

### Custom Training
To train on your own PCB dataset:

```python
from model import PCBDefectModel, create_dataset_yaml

# 1. Organize your data
class_names = {0: "missing_hole", 1: "short", 2: "open_circuit"}
yaml_path = create_dataset_yaml("data/train", "data/val", class_names)

# 2. Train model
pcb_model = PCBDefectModel()
pcb_model.train_model(yaml_path, epochs=10)
```

### Supported Defect Types (Default)
- `missing_hole`: Missing drill holes
- `mouse_bite`: Rough edges from drilling
- `open_circuit`: Broken connections
- `short`: Unwanted connections
- `spur`: Extra copper fragments
- `spurious_copper`: Unwanted copper deposits

## 📊 Dataset Preparation

### YOLO Format Requirements
```
data/
├── train/
│   ├── images/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── labels/
│       ├── img1.txt
│       └── img2.txt
└── val/
    ├── images/
    └── labels/
```

### Label Format (YOLO)
Each `.txt` file contains:
```
class_id center_x center_y width height
0 0.5 0.3 0.2 0.1
1 0.8 0.7 0.15 0.08
```
Values are normalized (0-1).

## 🛠️ Development Tips

### Testing the API
```bash
# Test with curl
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/pcb_image.jpg"
```

### Custom Confidence Threshold
Modify in `main.py`:
```python
results = model(image, conf=0.5)  # Change confidence threshold
```

### Adding New Defect Types
Update `DEFECT_CLASSES` in `main.py`:
```python
DEFECT_CLASSES = {
    0: "your_defect_type",
    1: "another_defect",
    # ... add more
}
```

## 🏆 Hackathon Checklist

### Friday Night ✅
- [x] Environment setup
- [x] FastAPI skeleton
- [x] Basic YOLOv8 integration
- [x] Test inference on sample image

### Saturday Morning 📋
- [ ] Collect PCB images from EE team
- [ ] Test `/predict` endpoint
- [ ] Integrate frontend with backend
- [ ] Label sample images (if training)

### Saturday Afternoon 📋
- [ ] Fine-tune model (optional)
- [ ] Polish web interface
- [ ] Add confidence display
- [ ] Test with real PCB photos

### Sunday Morning 📋
- [ ] Final testing and demo prep
- [ ] Create presentation slides
- [ ] Record demo video
- [ ] Submit project

## 🎯 Demo Script

1. **Open application**: "Here's our PCB defect detector..."
2. **Upload image**: "We simply drag and drop a PCB photo..."
3. **Show results**: "The AI identifies defects with bounding boxes..."
4. **Explain impact**: "This helps manufacturers catch defects early..."

## 🔧 Troubleshooting

### Model Loading Issues
```bash
# Download YOLOv8 model manually
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Memory Issues
- Reduce batch size in training
- Use YOLOv8n (nano) instead of larger models
- Process smaller image sizes

### CORS Issues
Already configured in `main.py`, but if needed:
```python
app.add_middleware(CORSMiddleware, allow_origins=["*"])
```

## 📝 Customization Ideas

### For Better Demo
1. Add real-time camera capture
2. Batch processing multiple images
3. Export detection reports
4. Add defect severity scoring

### For Production
1. Add authentication
2. Database logging
3. Model versioning
4. Docker containerization

## 🚀 Deployment Options

### Local Demo
```bash
python main.py
```

### Docker (Optional)
```dockerfile
FROM python:3.10-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "main.py"]
```

### Cloud Deployment
- **Heroku**: Add `Procfile`
- **AWS/GCP**: Use container services
- **Gradio**: For quick web demos

## 🎉 Success Metrics

- ✅ Working web interface
- ✅ API returning predictions
- ✅ Bounding boxes displayed correctly
- ✅ Handles multiple defect types
- ✅ Fast inference (<5 seconds)
- ✅ Clear presentation ready

---

**Good luck with your hackathon! 🚀**

*Remember: Even with a pretrained model, you'll have a working demo. Focus on the user experience and presentation!*