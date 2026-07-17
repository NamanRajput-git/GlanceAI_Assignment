import json
import numpy as np
from .config import RetrieverConfig

def attribute_match_score(query_slots, candidate_garments):
    if not query_slots:
        return 0.0
        
    score = 0.0
    for qg in query_slots:
        q_type = qg.get("type")
        q_color = qg.get("color")
        q_pattern = qg.get("pattern")
        q_material = qg.get("material")
        
        best_match_for_qg = 0.0
        for cg in candidate_garments:
            if cg.get("type") == q_type:
                match_val = 0.5
                if q_color and cg.get("color") == q_color:
                    match_val += 0.3
                if q_pattern and cg.get("pattern") == q_pattern:
                    match_val += 0.2
                if q_material and cg.get("material") == q_material:
                    match_val += 0.2
                
                if q_color and cg.get("color") != q_color:
                    match_val -= 0.5
                if q_pattern and cg.get("pattern") != q_pattern:
                    match_val -= 0.3
                    
                conf = cg.get("confidence", 1.0)
                final_cg_score = match_val * conf
                    
                best_match_for_qg = max(best_match_for_qg, final_cg_score)
                
        score += max(0, best_match_for_qg)
        
    return score / len(query_slots)

def compute_blended_score(query_parsed, query_image_emb, candidate_data, candidate_text_emb, candidate_image_emb, text_distance):
    embedding_score = 1.0 - (text_distance / 2.0)
    
    attribute_score = attribute_match_score(query_parsed.get("garments", []), candidate_data.get("garments", []))
    
    query_image_emb = np.array(query_image_emb)
    candidate_image_emb = np.array(candidate_image_emb)
    image_score = np.dot(query_image_emb, candidate_image_emb) / (np.linalg.norm(query_image_emb) * np.linalg.norm(candidate_image_emb) + 1e-9)
    
    q_scene = query_parsed.get("scene")
    c_scene = candidate_data.get("scene")
    scene_score = 0.0
    if q_scene:
        if q_scene == c_scene:
            scene_score = 1.0
        elif c_scene and c_scene != "unknown":
            scene_score = -0.3
            
    q_activity = query_parsed.get("activity")
    c_activity = candidate_data.get("activity")
    activity_score = 0.0
    if q_activity:
        if q_activity == c_activity:
            activity_score = 1.0
        elif c_activity and c_activity != "unknown":
            activity_score = -0.2
            
    style_score = 1.0 if query_parsed.get("style") == candidate_data.get("style") else 0.0
    
    env_score = 0.0
    env_weight = RetrieverConfig.DELTA
    env_comps = []
    if q_scene: env_comps.append(scene_score)
    if query_parsed.get("style"): env_comps.append(style_score)
    if q_activity: env_comps.append(activity_score)
    
    if env_comps:
        env_score = sum(env_comps) / len(env_comps)
    else:
        env_weight = 0.0
        
    final_score = (RetrieverConfig.ALPHA * embedding_score + 
                   RetrieverConfig.BETA * attribute_score + 
                   RetrieverConfig.GAMMA * float(image_score) + 
                   env_weight * env_score)
                   
    total_weight = RetrieverConfig.ALPHA + RetrieverConfig.BETA + RetrieverConfig.GAMMA + env_weight
    
    return {
        "final_score": final_score / total_weight,
        "embedding_score": embedding_score,
        "attribute_score": attribute_score,
        "image_score": float(image_score),
        "scene_score": scene_score,
        "style_score": style_score,
        "activity_score": activity_score
    }
