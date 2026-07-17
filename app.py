import streamlit as st
import json
import numpy as np
from PIL import Image
import os

from retriever.query_parser import QueryParser
from indexer.embed import Embedder
from indexer.vector_store import VectorStore
from retriever.search import compute_blended_score

st.set_page_config(page_title="GlanceAI Retrieval", page_icon="🔍", layout="wide")

# ==========================================
# CUSTOM CSS FOR PREMIUM AESTHETICS
# ==========================================
st.markdown("""
<style>
    /* Import modern typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    /* Premium Title Styling */
    h1 {
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding-bottom: 20px;
    }
    
    /* Beautiful Image Cards */
    img {
        border-radius: 12px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    img:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.3);
    }
    
    /* Metric Card Styling */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 10px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    qp = QueryParser(lexicons_dir="data/lexicons")
    embedder = Embedder()
    vs = VectorStore(persist_directory="data/chroma_db")
    return qp, embedder, vs

st.title("Multimodal Fashion Engine")
st.markdown("<p style='text-align: center; color: #a0a0a0; font-size: 18px;'>Find exactly what you're looking for with composable natural language queries.</p>", unsafe_allow_html=True)
st.write("---")

qp, embedder, vs = load_models()

def reset_page():
    st.session_state.page = 1
def next_page():
    st.session_state.page += 1
def prev_page():
    st.session_state.page -= 1

if "page" not in st.session_state:
    st.session_state.page = 1

col_search, _ = st.columns([2, 1])
with col_search:
    query = st.text_input("Describe the perfect outfit or scene...", placeholder="e.g., 'red leather jacket walking in the city'", on_change=reset_page)

if query:
    parsed_query = qp.parse_query(query)
    with st.expander("See How the AI Parsed Your Request"):
        st.json(parsed_query)
    
    with st.spinner("Analyzing neural embeddings and scoring attributes..."):
        query_text_emb = embedder.embed_text(query)
        query_clip_emb = embedder.embed_text_clip(query)
        
        results = vs.search_text(query_text_emb, n_results=50)
        
        if not results['ids'] or not results['ids'][0]:
            st.error("No matching garments found in the visual database.")
        else:
            candidates = []
            for i in range(len(results['ids'][0])):
                img_id = results['ids'][0][i]
                metadata = results['metadatas'][0][i]
                if metadata is None:
                    continue
                    
                distance = results['distances'][0][i]
                
                structured_data = json.loads(metadata['structured_json'])
                image_emb = json.loads(metadata['image_embedding'])
                path = metadata['path']
                
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
                    "path": path,
                    "structured_data": structured_data,
                    "scores": score_breakdown
                })
                
            candidates.sort(key=lambda x: x['scores']['final_score'], reverse=True)
            
            RESULTS_PER_PAGE = 3
            total_pages = max(1, (len(candidates) + RESULTS_PER_PAGE - 1) // RESULTS_PER_PAGE)
            
            start_idx = (st.session_state.page - 1) * RESULTS_PER_PAGE
            end_idx = start_idx + RESULTS_PER_PAGE
            page_candidates = candidates[start_idx:end_idx]
            
            st.markdown(f"### Top Matches (Page {st.session_state.page} of {total_pages})")
            st.write("")
            
            cols = st.columns(3)
            for idx, cand in enumerate(page_candidates):
                col = cols[idx % 3]
                with col:
                    if os.path.exists(cand['path']):
                        image = Image.open(cand['path'])
                        st.image(image, use_column_width=True)
                    else:
                        st.warning(f"Image missing: {cand['path']}")
                        
                    score = cand['scores']['final_score']
                    st.markdown(f"**Match Score: {score:.3f}**")
                    st.progress(min(1.0, max(0.0, score)))
                    
                    with st.expander("View Mathematical Breakdown"):
                        s = cand['scores']
                        c1, c2 = st.columns(2)
                        c1.metric("Attribute Match", f"{s['attribute_score']:.3f}")
                        c2.metric("Environment", f"{s['scene_score']:.3f}")
                        c1.metric("Text Embed", f"{s['embedding_score']:.3f}")
                        c2.metric("Visual Embed", f"{s['image_score']:.3f}")
                        
                        st.markdown("**Detected Composition**")
                        for g in cand['structured_data'].get('garments', []):
                            conf = g.get('confidence', 0.0)
                            c = g.get('color', '') or ''
                            p = g.get('pattern', '') or ''
                            t = g.get('type', '') or ''
                            desc = f"{c} {p} {t}".strip()
                            st.caption(f"- **{desc}** *(Conf: {conf*100:.1f}%)*")
                        
                        st.markdown("**Context & Environment**")
                        scene = cand['structured_data'].get('scene', 'unknown').capitalize()
                        style = cand['structured_data'].get('style', 'unknown').capitalize()
                        activity = cand['structured_data'].get('activity', 'unknown').capitalize()
                        materials = cand['structured_data'].get('materials', [])
                        mat_str = ", ".join(materials).capitalize() if materials else "None explicitly detected"
                        
                        st.caption(f"**Scene:** {scene} | **Style:** {style} | **Activity:** {activity}")
                        st.caption(f"**Global Materials:** {mat_str}")
                        
                        st.markdown("**Raw VLM Caption**")
                        raw_cap = cand['structured_data'].get('raw_caption', 'N/A')
                        st.caption(f'*"{raw_cap}"*')
                        
            st.write("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                st.button("Previous Page", on_click=prev_page, disabled=(st.session_state.page <= 1), use_container_width=True)
            with col3:
                st.button("Next Page", on_click=next_page, disabled=(st.session_state.page >= total_pages), use_container_width=True)
