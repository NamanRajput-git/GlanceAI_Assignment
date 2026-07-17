import os
import json

from retriever.query_parser import QueryParser
from indexer.embed import Embedder
from indexer.vector_store import VectorStore
from retriever.search import compute_blended_score

def run_eval():
    print("Loading models for evaluation...")
    qp = QueryParser(lexicons_dir="../data/lexicons") if os.path.exists("../data/lexicons") else QueryParser(lexicons_dir="data/lexicons")
    embedder = Embedder()
    vs = VectorStore(persist_directory="../data/chroma_db") if os.path.exists("../data/chroma_db") else VectorStore(persist_directory="data/chroma_db")
    
    queries = [
        "red shirt with blue pants",
        "a person in a modern office",
        "striped shirt and solid tie",
        "casual weekend walk in the park",
        "blue jacket red pants"
    ]
    
    for q in queries:
        print(f"\n{'='*50}\nEVALUATING QUERY: '{q}'\n{'='*50}")
        parsed_query = qp.parse_query(q)
        print(f"Parsed Slots: {json.dumps(parsed_query, indent=2)}")
        
        query_text_emb = embedder.embed_text(q)
        query_clip_emb = embedder.embed_text_clip(q)
        
        results = vs.search_text(query_text_emb, n_results=10)
        
        if not results['ids'] or not results['ids'][0]:
            print("No results found in index.")
            continue
            
        candidates = []
        for i in range(len(results['ids'][0])):
            img_id = results['ids'][0][i]
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            
            structured_data = json.loads(metadata['structured_json'])
            image_emb = json.loads(metadata['image_embedding'])
            
            score_breakdown = compute_blended_score(
                parsed_query, 
                query_clip_emb, 
                structured_data, 
                query_text_emb, 
                image_emb, 
                distance
            )
            
            candidates.append({
                "id": img_id,
                "structured_data": structured_data,
                "scores": score_breakdown
            })
            
        candidates.sort(key=lambda x: x['scores']['final_score'], reverse=True)
        
        print("Top 5 Results:")
        for idx, cand in enumerate(candidates[:5]):
            print(f"\nRank {idx+1} | Image ID: {cand['id']}")
            print(f"Final Score: {cand['scores']['final_score']:.4f}")
            print(f"Breakdown: Emb={cand['scores']['embedding_score']:.3f}, Attr={cand['scores']['attribute_score']:.3f}, Img={cand['scores']['image_score']:.3f}, Scene={cand['scores']['scene_score']:.3f}, Style={cand['scores']['style_score']:.3f}")
            print(f"Garments: {cand['structured_data'].get('garments', [])}")
            print(f"Scene: {cand['structured_data'].get('scene')} | Style: {cand['structured_data'].get('style')}")

if __name__ == "__main__":
    run_eval()
