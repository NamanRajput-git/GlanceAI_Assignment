import torch
import json
import os
from transformers import pipeline
from PIL import Image

class GarmentDetector:
    def __init__(self, model_id="google/owlvit-base-patch32", lexicons_dir="data/lexicons"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading GarmentDetector on {self.device}...")
        self.detector = pipeline("zero-shot-object-detection", model=model_id, device=self.device if self.device == "cuda" else -1)
        
        garments_path = os.path.join(lexicons_dir, "garments.json")
        with open(garments_path, "r") as f:
            self.garment_classes = json.load(f)
            
    def detect_garments(self, image: Image.Image, threshold=0.1):
        """
        Runs object detection for fixed garment classes.
        Returns a list of dicts: {'box': [xmin, ymin, xmax, ymax], 'label': str, 'score': float, 'crop': PIL.Image}
        """
        results = self.detector(
            image,
            candidate_labels=self.garment_classes,
        )
        
        detected_items = []
        for res in results:
            if res["score"] > threshold:
                box = res["box"]
                crop_box = (box['xmin'], box['ymin'], box['xmax'], box['ymax'])
                crop = image.crop(crop_box)
                
                detected_items.append({
                    "box": crop_box,
                    "label": res["label"],
                    "score": res["score"],
                    "crop": crop
                })
                
        return detected_items
