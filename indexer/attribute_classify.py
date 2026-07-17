import torch
import json
import os
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class AttributeClassifier:
    def __init__(self, model_id="patrickjohncyh/fashion-clip", lexicons_dir="data/lexicons"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading AttributeClassifier ({model_id}) on {self.device}...")
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        
        with open(os.path.join(lexicons_dir, "colors.json"), "r") as f:
            self.colors = json.load(f)
        with open(os.path.join(lexicons_dir, "patterns.json"), "r") as f:
            self.patterns = json.load(f)
        with open(os.path.join(lexicons_dir, "garments.json"), "r") as f:
            self.garments = json.load(f)
            
        self.color_templates = [f"a photo of a {c} garment" for c in self.colors]
        self.pattern_templates = [f"a {p} patterned garment" for p in self.patterns]
        self.type_templates = [f"a photo of a {t}" for t in self.garments]

    def classify_crop(self, crop_image: Image.Image):
        """
        Runs zero-shot classification on a single garment crop.
        Returns: {type, color, pattern, confidence}
        """
        inputs = self.processor(
            text=self.color_templates + self.pattern_templates + self.type_templates,
            images=crop_image, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image[0]  # image-text similarity score
            
        color_logits = logits[:len(self.colors)]
        pattern_logits = logits[len(self.colors):len(self.colors)+len(self.patterns)]
        type_logits = logits[len(self.colors)+len(self.patterns):]
        
        color_idx = color_logits.argmax().item()
        pattern_idx = pattern_logits.argmax().item()
        type_idx = type_logits.argmax().item()
        
        c_prob = color_logits.softmax(dim=0).max().item()
        p_prob = pattern_logits.softmax(dim=0).max().item()
        t_prob = type_logits.softmax(dim=0).max().item()
        confidence = (c_prob + p_prob + t_prob) / 3.0
        
        predicted_color = self.colors[color_idx]
        verified_color = self.verify_color_hsv(crop_image, predicted_color)
        
        return {
            "type": self.garments[type_idx],
            "color": verified_color,
            "pattern": self.patterns[pattern_idx],
            "confidence": round(confidence, 4)
        }
        
    def verify_color_hsv(self, crop_image, predicted_color):
        try:
            hsv_array = np.array(crop_image.convert('HSV'))
            hues = hsv_array[:,:,0].flatten()
            sats = hsv_array[:,:,1].flatten()
            valid_hues = hues[sats > 50]
            if len(valid_hues) == 0:
                return predicted_color
            
            median_hue = np.median(valid_hues)
            hsv_color = None
            if median_hue < 15 or median_hue >= 240: hsv_color = "red"
            elif 15 <= median_hue < 30: hsv_color = "orange"
            elif 30 <= median_hue < 45: hsv_color = "yellow"
            elif 45 <= median_hue < 105: hsv_color = "green"
            elif 105 <= median_hue < 180: hsv_color = "blue"
            elif 180 <= median_hue < 240: hsv_color = "purple"
            
            if predicted_color == "purple" and hsv_color == "blue":
                return "blue"
            if predicted_color == "blue" and hsv_color == "purple":
                return "purple"
            if predicted_color == "red" and hsv_color == "orange":
                return "orange"
                
            return predicted_color
        except Exception:
            return predicted_color
