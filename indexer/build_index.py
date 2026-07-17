import os
from indexer.dataset import load_fashion_dataset, get_supplementary_images
from indexer.garment_detection import GarmentDetector
from indexer.attribute_classify import AttributeClassifier
from indexer.scene_classify import SceneClassifier
from indexer.structured_caption import FallbackCaptioner
from indexer.merge_schema import merge_to_schema
from indexer.embed import Embedder
from indexer.vector_store import VectorStore

def build_index():
    print("Loading datasets...")
    records = load_fashion_dataset(num_samples=1000) # Increased to 1000 images
    records.extend(get_supplementary_images("data/images"))
    
    if not records:
        print("No images to index.")
        return

    print("Initializing models (this may take a moment to download weights)...")
    detector = GarmentDetector()
    attr_clf = AttributeClassifier()
    scene_clf = SceneClassifier()
    captioner = FallbackCaptioner()
    embedder = Embedder()
    vs = VectorStore()
    
    print(f"Indexing {len(records)} images...")
    for i, rec in enumerate(records):
        img_id = rec["id"]
        image = rec["image"]
        path = rec["path"]
        
        print(f"[{i+1}/{len(records)}] Processing {img_id}...")
        
        detections = detector.detect_garments(image, threshold=0.1)
        
        garments = []
        for det in detections:
            crop = det["crop"]
            attrs = attr_clf.classify_crop(crop)
            garments.append({
                "type": attrs["type"], 
                "color": attrs["color"],
                "pattern": attrs["pattern"],
                "confidence": attrs["confidence"]
            })
            
        scene_label = scene_clf.classify_scene(image)
        caption_data = captioner.get_structured_caption(image)
        structured_data = merge_to_schema(garments, scene_label, caption_data)
        
        schema_text = vs._schema_to_sentence(structured_data)
        text_emb = embedder.embed_text(schema_text)
        image_emb = embedder.embed_image(image)
        
        vs.add_item(img_id, path, structured_data, text_emb, image_emb)
        
    print("Indexing complete.")

if __name__ == "__main__":
    build_index()
