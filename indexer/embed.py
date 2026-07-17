import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class Embedder:
    def __init__(self, text_model_id="sentence-transformers/all-mpnet-base-v2", image_model_id="patrickjohncyh/fashion-clip"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Embedders on {self.device}...")
        
        self.text_model = SentenceTransformer(text_model_id, device=self.device)
        
        self.image_model = CLIPModel.from_pretrained(image_model_id).to(self.device)
        self.image_processor = CLIPProcessor.from_pretrained(image_model_id)

    def embed_text(self, text: str):
        embedding = self.text_model.encode(text)
        return embedding.tolist()

    def embed_image(self, image: Image.Image):
        inputs = self.image_processor(images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            image_features = self.image_model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features[0].cpu().tolist()

    def embed_text_clip(self, text: str):
        inputs = self.image_processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_features = self.image_model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features[0].cpu().tolist()
