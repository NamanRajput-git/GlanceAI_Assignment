import json
import requests
import os
from PIL import Image
import base64
from io import BytesIO

class FallbackCaptioner:
    def __init__(self, api_key=None, lexicons_dir="data/lexicons"):
        self.api_url = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
        self.api_key = api_key or os.environ.get("HF_API_KEY")
        with open(os.path.join(lexicons_dir, "activities.json"), "r") as f:
            self.activities = json.load(f)
        with open(os.path.join(lexicons_dir, "materials.json"), "r") as f:
            self.materials = json.load(f)
        
    def get_structured_caption(self, image: Image.Image):
        """
        Uses HF Inference API to get a caption, then parses it for scene and style.
        Returns: {raw_caption, scene, style}
        """
        if not self.api_key:
            return {
                "raw_caption": "A person wearing clothes",
                "scene": "unknown",
                "style": "casual",
                "activity": "unknown",
                "materials": []
            }
            
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = buffered.getvalue()
        
        try:
            response = requests.post(self.api_url, headers=headers, data=img_str)
            result = response.json()
            if isinstance(result, list) and "generated_text" in result[0]:
                raw_caption = result[0]["generated_text"]
            else:
                raw_caption = "A person wearing clothes"
        except Exception as e:
            print(f"Caption API failed: {e}")
            raw_caption = "A person wearing clothes"
            
        style = "casual"
        if "suit" in raw_caption or "formal" in raw_caption or "tie" in raw_caption:
            style = "formal"
        elif "athletic" in raw_caption or "sport" in raw_caption:
            style = "athletic"
            
        scene = "unknown"
        if "office" in raw_caption or "desk" in raw_caption:
            scene = "office"
        elif "park" in raw_caption or "tree" in raw_caption:
            scene = "park"
        elif "street" in raw_caption or "road" in raw_caption:
            scene = "street"
            
        activity = "unknown"
        for a in self.activities:
            if a in raw_caption.lower():
                activity = a
                break
                
        materials_found = []
        for m in self.materials:
            if m in raw_caption.lower():
                materials_found.append(m)
                
        return {
            "raw_caption": raw_caption,
            "scene": scene,
            "style": style,
            "activity": activity,
            "materials": materials_found
        }
