import chromadb
import json

class VectorStore:
    def __init__(self, persist_directory="data/chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name="fashion_index")
        
    def add_item(self, image_id, img_path, structured_data, text_embedding, image_embedding):
        metadata = {
            "path": img_path if img_path else "",
            "structured_json": json.dumps(structured_data),
            "image_embedding": json.dumps(image_embedding) 
        }
        
        doc_text = self._schema_to_sentence(structured_data)
        
        self.collection.upsert(
            ids=[image_id],
            embeddings=[text_embedding],
            documents=[doc_text],
            metadatas=[metadata]
        )
        
    def _schema_to_sentence(self, data):
        garments_str = ", ".join([f"{g.get('color', '')} {g.get('pattern', '')} {g.get('material', '')} {g.get('type', '')}".strip() for g in data.get("garments", [])])
        activity_str = f"{data.get('activity')} activity" if data.get('activity') and data.get('activity') != "unknown" else ""
        materials_str = f"made of {' '.join(data.get('materials', []))}" if data.get('materials') else ""
        return f"{garments_str}, {data.get('style', 'casual')} style, {data.get('scene', 'unknown')} setting, {activity_str} {materials_str}".strip().replace("  ", " ")

    def search_text(self, query_embedding, n_results=50):
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['embeddings', 'metadatas', 'documents', 'distances']
        )
        return results
