import streamlit as st
import json
import numpy as np
from PIL import Image
import os

from retriever.query_parser import QueryParser
from indexer.embed import Embedder
from indexer.vector_store import VectorStore
from retriever.search import compute_blended_score

st.set_page_config(layout="wide")

@st.cache_resource
def load_models():
    qp = QueryParser(lexicons_dir="data/lexicons")
    embedder = Embedder()
    vs = VectorStore(persist_directory="data/chroma_db")
    return qp, embedder, vs

st.title("Multimodal Fashion & Context Retrieval")
qp, embedder, vs = load_models()

def reset_page():
    st.session_state.page = 1
def next_page():
    st.session_state.page += 1
def prev_page():
    st.session_state.page -= 1

if "page" not in st.session_state:
    st.session_state.page = 1

query = st.text_input("Enter your search query (e.g., 'red shirt in a modern office'):", on_change=reset_page)

if query:
    parsed_query = qp.parse_query(query)
    with st.expander("Parsed Query Slots"):
        st.json(parsed_query)
    
    with st.spinner("Searching..."):
        query_text_emb = embedder.embed_text(query)
        query_clip_emb = embedder.embed_text_clip(query)
        
        results = vs.search_text(query_text_emb, n_results=50)
        
        if not results['ids'] or not results['ids'][0]:
            st.warning("No results found in the index.")
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
            
            RESULTS_PER_PAGE = 5
            total_pages = max(1, (len(candidates) + RESULTS_PER_PAGE - 1) // RESULTS_PER_PAGE)
            
            start_idx = (st.session_state.page - 1) * RESULTS_PER_PAGE
            end_idx = start_idx + RESULTS_PER_PAGE
            page_candidates = candidates[start_idx:end_idx]
            
            st.write(f"### Top Results (Page {st.session_state.page} of {total_pages})")
            cols = st.columns(3)
            for idx, cand in enumerate(page_candidates):
                col = cols[idx % 3]
                with col:
                    if os.path.exists(cand['path']):
                        image = Image.open(cand['path'])
                        st.image(image, use_column_width=True)
                    else:
                        st.warning(f"Image missing: {cand['path']}")
                        
                    st.write(f"**Score:** {cand['scores']['final_score']:.3f}")
                    with st.expander("Score Breakdown & Metadata"):
                        st.json(cand['scores'])
                        st.json(cand['structured_data'])
                        
            st.write("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                st.button("Previous Page", on_click=prev_page, disabled=(st.session_state.page <= 1))
            with col3:
                st.button("Next Page", on_click=next_page, disabled=(st.session_state.page >= total_pages))
