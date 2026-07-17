import torch
from torchvision import models, transforms
from PIL import Image
import urllib.request
import os
import json

class SceneClassifier:
    def __init__(self, lexicons_dir="data/lexicons"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SceneClassifier (ResNet50 Places365) on {self.device}...")
        
        
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(self.device)
        self.model.eval()
        
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        with open(os.path.join(lexicons_dir, "scenes.json"), "r") as f:
            self.scene_map = json.load(f)
            
    def classify_scene(self, image: Image.Image):
        """
        Classify scene and map to coarse categories.
        """
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            
        
        return "unknown"
