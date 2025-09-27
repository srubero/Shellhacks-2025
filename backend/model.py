"""
Model utilities for PCB defect detection
This file handles model training, loading, and inference utilities
"""

from ultralytics import YOLO
import torch
import os
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

class PCBDefectModel:
    """Wrapper class for PCB defect detection model"""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the model
        
        Args:
            model_path: Path to custom trained model. If None, uses pretrained YOLOv8n
        """
        self.model = None
        self.model_path = model_path
        self.class_names = {
            0: "missing_hole",
            1: "mouse_bite", 
            2: "open_circuit",
            3: "short",
            4: "spur",
            5: "spurious_copper"
        }
        
    def load_model(self) -> bool:
        """Load the YOLO model"""
        try:
            if self.model_path and os.path.exists(self.model_path):
                print(f"Loading custom model from {self.model_path}")
                self.model = YOLO(self.model_path)
                
                # Try to load custom class names if available
                if hasattr(self.model, 'names'):
                    self.class_names = self.model.names
                    
            else:
                print("Loading pretrained YOLOv8n model")
                self.model = YOLO('yolov8n.pt')
                
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def train_model(self, 
                    data_yaml: str,
                    epochs: int = 10,
                    imgsz: int = 640,
                    batch: int = 16,
                    save_dir: str = "runs/detect/train"):
        """Train the model on custom dataset
        
        Args:
            data_yaml: Path to YAML file with dataset configuration
            epochs: Number of training epochs
            imgsz: Image size for training
            batch: Batch size
            save_dir: Directory to save training results
        """
        if not os.path.exists(data_yaml):
            raise FileNotFoundError(f"Dataset YAML file not found: {data_yaml}")
        
        # Start with pretrained YOLOv8n
        model = YOLO('yolov8n.pt')
        
        # Train the model
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            save_dir=save_dir,
            patience=5,  # Early stopping
            save=True,
            verbose=True
        )
        
        # Save the best model
        best_model_path = os.path.join(save_dir, "weights", "best.pt")
        if os.path.exists(best_model_path):
            self.model_path = best_model_path
            self.load_model()
            print(f"Model trained successfully! Best model saved to: {best_model_path}")
        
        return results
    
    def predict(self, image_path: str, conf_threshold: float = 0.5):
        """Run inference on an image"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        results = self.model(image_path, conf=conf_threshold)
        return results
    
    def predict_image(self, image_array, conf_threshold: float = 0.5):
        """Run inference on a numpy image array"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        results = self.model(image_array, conf=conf_threshold)
        return results
    
    def process_results(self, results, confidence_threshold: float = 0.5):
        """Convert YOLO predictions to JSON format"""
        detections = []
        
        if len(results) > 0:
            result = results[0]  # First (and only) image
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
                confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
                classes = result.boxes.cls.cpu().numpy()  # Class IDs
                
                for box, conf, cls in zip(boxes, confidences, classes):
                    if conf >= confidence_threshold:
                        x1, y1, x2, y2 = box
                        
                        # Get class name (use custom mapping or model's names)
                        class_id = int(cls)
                        if hasattr(self.model, 'names'):
                            class_name = self.model.names[class_id]
                        else:
                            class_name = self.class_names.get(class_id, f"defect_{class_id}")
                        
                        detection = {
                            "label": class_name,
                            "confidence": float(conf),
                            "bbox": [float(x1), float(y1), float(x2), float(y2)]
                        }
                        detections.append(detection)
        
        return detections
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model is None:
            return {
                "model_name": "Not loaded",
                "num_classes": 0,
                "is_custom": False
            }
        
        return {
            "model_name": "YOLOv8n" if not self.model_path else "Custom YOLOv8",
            "num_classes": len(self.class_names) if hasattr(self, 'class_names') else len(getattr(self.model, 'names', {})),
            "is_custom": bool(self.model_path and os.path.exists(self.model_path)),
            "classes": list(getattr(self.model, 'names', self.class_names).values()) if hasattr(self.model, 'names') else list(self.class_names.values())
        }
    
    def export_model(self, format: str = "onnx", save_path: Optional[str] = None):
        """Export model to different formats for deployment
        
        Args:
            format: Export format ('onnx', 'torchscript', 'tflite', etc.)
            save_path: Path to save exported model
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        exported_path = self.model.export(format=format, dynamic=True)
        
        if save_path:
            import shutil
            shutil.move(exported_path, save_path)
            return save_path
        
        return exported_path

def create_dataset_yaml(train_path: str, 
                       val_path: str, 
                       class_names: Dict[int, str],
                       save_path: str = "dataset.yaml") -> str:
    """Create YAML configuration file for YOLO training
    
    Args:
        train_path: Path to training images directory
        val_path: Path to validation images directory  
        class_names: Dictionary mapping class IDs to names
        save_path: Path to save the YAML file
        
    Returns:
        Path to created YAML file
    """
    
    dataset_config = {
        'path': os.path.dirname(train_path),  # Root directory
        'train': os.path.basename(train_path),  # Training images
        'val': os.path.basename(val_path),      # Validation images
        'nc': len(class_names),                 # Number of classes
        'names': list(class_names.values())     # Class names
    }
    
    with open(save_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"Dataset YAML created: {save_path}")
    return save_path

def download_sample_dataset():
    """Download a sample PCB defect dataset for quick testing"""
    try:
        from roboflow import Roboflow
        
        # This is a public PCB defect dataset - replace with your preferred one
        rf = Roboflow(api_key="your_api_key_here")  # Get free API key from roboflow.com
        project = rf.workspace("your-workspace").project("pcb-defects")
        dataset = project.version(1).download("yolov8")
        
        return dataset.location
        
    except ImportError:
        print("Roboflow not installed. Install with: pip install roboflow")
        return None
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Initialize model
    pcb_model = PCBDefectModel()
    
    # Load pretrained model
    if pcb_model.load_model():
        print("Model loaded successfully!")
        
        # Example prediction (replace with actual image path)
        # results = pcb_model.predict("sample_pcb.jpg")
        # print(f"Predictions: {results}")
        
        # Example training (uncomment when you have dataset)
        # class_names = {0: "defect", 1: "good"}
        # yaml_path = create_dataset_yaml("data/train", "data/val", class_names)
        # pcb_model.train_model(yaml_path, epochs=5)
        
    else:
        print("Failed to load model")