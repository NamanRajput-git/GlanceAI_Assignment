import json

def merge_to_schema(garments, scene_label, caption_data):
    """
    Merges detection/classification results with scene and caption fallbacks into a single JSON schema.
    """
    degraded = len(garments) == 0
    
    final_scene = scene_label
    scene_source = "places365"
    if final_scene == "unknown" and caption_data["scene"] != "unknown":
        final_scene = caption_data["scene"]
        scene_source = "vlm_caption"
        
    return {
        "garments": garments,
        "scene": final_scene,
        "scene_source": scene_source,
        "style": caption_data["style"],
        "style_source": "vlm_caption",
        "activity": caption_data.get("activity", "unknown"),
        "materials": caption_data.get("materials", []),
        "raw_caption": caption_data["raw_caption"],
        "degraded": degraded
    }
