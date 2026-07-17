import json
import os
import re

class QueryParser:
    def __init__(self, lexicons_dir="data/lexicons"):
        with open(os.path.join(lexicons_dir, "colors.json"), "r") as f:
            self.colors = json.load(f)
        with open(os.path.join(lexicons_dir, "patterns.json"), "r") as f:
            self.patterns = json.load(f)
        with open(os.path.join(lexicons_dir, "garments.json"), "r") as f:
            self.garments = json.load(f)
        with open(os.path.join(lexicons_dir, "scenes.json"), "r") as f:
            self.scenes = json.load(f)
        with open(os.path.join(lexicons_dir, "styles.json"), "r") as f:
            self.styles = json.load(f)
        with open(os.path.join(lexicons_dir, "activities.json"), "r") as f:
            self.activities = json.load(f)
        with open(os.path.join(lexicons_dir, "materials.json"), "r") as f:
            self.materials = json.load(f)
            
    def parse_query(self, query: str):
        query_lower = query.lower()
        
        parsed_garments = []
        for garment in self.garments:
            if re.search(r'\b' + re.escape(garment) + r'\b', query_lower):
                match = re.search(r'(.*?)\b' + re.escape(garment) + r'\b', query_lower)
                prefix = match.group(1) if match else ""
                
                g_color = None
                for c in self.colors:
                    if re.search(r'\b' + re.escape(c) + r'\b', prefix):
                        g_color = c
                        break
                        
                g_pattern = None
                for p in self.patterns:
                    if re.search(r'\b' + re.escape(p) + r'\b', prefix):
                        g_pattern = p
                        break
                        
                g_material = None
                for m in self.materials:
                    if re.search(r'\b' + re.escape(m) + r'\b', prefix):
                        g_material = m
                        break
                        
                parsed_garments.append({
                    "type": garment,
                    "color": g_color,
                    "pattern": g_pattern,
                    "material": g_material
                })
                
        parsed_scene = None
        for scene_key, keywords in self.scenes.items():
            for kw in keywords:
                if re.search(r'\b' + re.escape(kw) + r'\b', query_lower):
                    parsed_scene = scene_key
                    break
            if parsed_scene:
                break
                
        parsed_style = None
        for style in self.styles:
            if re.search(r'\b' + re.escape(style) + r'\b', query_lower):
                parsed_style = style
                break
                
        parsed_activity = None
        for activity in self.activities:
            if re.search(r'\b' + re.escape(activity) + r'\b', query_lower):
                parsed_activity = activity
                break
                
        return {
            "garments": parsed_garments,
            "scene": parsed_scene,
            "style": parsed_style,
            "activity": parsed_activity
        }
