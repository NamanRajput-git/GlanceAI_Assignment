import os
import json
from datasets import load_dataset
from PIL import Image

def get_supplementary_images(data_dir: str):
    """
    Looks for supplementary images in the specified directory.
    Returns a list of dictionaries with 'image' and 'id'.
    """
    images = []
    if os.path.exists(data_dir):
         for f in os.listdir(data_dir):
             if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
                 continue
             if f.startswith('fashionpedia_'):
                 continue
             img_path = os.path.join(data_dir, f)
             try:
                 images.append({
                     "id": f"supp_{f.split('.')[0]}",
                     "image": Image.open(img_path).convert("RGB"),
                     "path": img_path
                 })
             except Exception as e:
                 print(f"Failed to load {img_path}: {e}")
    return images

def load_fashion_dataset(num_samples=50, save_dir="data/images"):
    """
    Loads samples from Fashionpedia and saves them locally for the UI to display.
    """
    os.makedirs(save_dir, exist_ok=True)
    dataset_records = []
    try:
        print("Loading Fashionpedia subset...")
        ds = load_dataset("detection-datasets/fashionpedia", split=f"train[:{num_samples}]")
        for i, row in enumerate(ds):
            img = row["image"].convert("RGB")
            img_path = os.path.join(save_dir, f"fashionpedia_{i}.jpg")
            if not os.path.exists(img_path):
                img.save(img_path)
                
            dataset_records.append({
                "id": f"fashionpedia_{i}",
                "image": img,
                "path": img_path
            })
    except Exception as e:
        print(f"Could not load Fashionpedia directly. Error: {e}")
        
    return dataset_records
