import pytest
from retriever.search import attribute_match_score
from indexer.merge_schema import merge_to_schema

def test_color_garment_swap():
    query_slots = [
        {"type": "shirt", "color": "red", "pattern": None},
        {"type": "pants", "color": "blue", "pattern": None}
    ]
    
    candidate_correct = [
        {"type": "shirt", "color": "red", "pattern": "solid"},
        {"type": "pants", "color": "blue", "pattern": "solid"}
    ]
    
    candidate_swapped = [
        {"type": "shirt", "color": "blue", "pattern": "solid"},
        {"type": "pants", "color": "red", "pattern": "solid"}
    ]
    
    score_correct = attribute_match_score(query_slots, candidate_correct)
    score_swapped = attribute_match_score(query_slots, candidate_swapped)
    
    assert score_correct > score_swapped, f"Expected {score_correct} > {score_swapped}"

def test_pattern_garment_swap():
    query_slots = [
        {"type": "shirt", "color": None, "pattern": "striped"}
    ]
    
    candidate_correct = [
        {"type": "shirt", "color": "white", "pattern": "striped"}
    ]
    
    candidate_swapped = [
        {"type": "shirt", "color": "white", "pattern": "solid"},
        {"type": "tie", "color": "red", "pattern": "striped"}
    ]
    
    score_correct = attribute_match_score(query_slots, candidate_correct)
    score_swapped = attribute_match_score(query_slots, candidate_swapped)
    
    assert score_correct > score_swapped, "Correct pattern binding must score higher than a swapped binding"

def test_degraded_mode_flag():
    garments = []
    caption_data = {"raw_caption": "A nice shirt", "scene": "unknown", "style": "casual"}
    
    schema = merge_to_schema(garments, "unknown", caption_data)
    assert schema["degraded"] is True
    
    garments_valid = [{"type": "shirt", "color": "red", "pattern": "solid", "confidence": 0.9}]
    schema_valid = merge_to_schema(garments_valid, "office", caption_data)
    assert schema_valid["degraded"] is False
