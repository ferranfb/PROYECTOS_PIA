import streamlit as st
import os
import faiss
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# --- CONFIGURACIÓN Y RUTAS (Asegúrate de que coincidan con indexarembeddings.py) ---
MODEL_NAME = "openai/clip-vit-base-patch32"
INDEX_FILE = "faiss_index.bin"
MAPPING_FILE = "image_paths.npy"
ASSETS_DIR = "assets"

# --- RUTA ABSOLUTA (Para asegurar que Streamlit encuentra los archivos) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, INDEX_FILE)
MAPPING_PATH = os.path.join(BASE_DIR, MAPPING_FILE)

# --- 2. Funciones de Carga y Preproceso ---

@st.cache_resource
def load_resources():
    """Carga el modelo, el índice FAISS y el mapeo solo una vez."""
    try:
        # Cargar Modelo CLIP
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
        processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        
        # Cargar Índice FAISS
        index = faiss.read_index(INDEX_PATH)
        
        # Cargar Mapeo de Rutas
        image_paths = np.load(MAPPING_PATH, allow_pickle=True)
        
        st.success(f"Recursos cargados: Modelo en {device}, {index.ntotal} vectores indexados.")
        return model, processor, index, image_paths, device
    except Exception as e:
        st.error(f"Error al cargar recursos. ¿Ejecutaste indexarembeddings.py? Error: {e}")
        return None, None, None, None, None

def get_query_embedding(query, model, processor, device):
    """Genera el embedding de la consulta (imagen o texto)."""
    with torch.no_grad():
        if isinstance(query, Image.Image):
            # Consulta por Imagen
            inputs = processor(images=query.convert("RGB"), return_tensors="pt").to(device)
            features = model.get_image_features(**inputs)
        elif isinstance(query, str):
            # Consulta por Texto (Ampliación)
            inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
            features = model.get_text_features(**inputs)
        else:
            return None

        # Normalizar el vector de consulta
        normalized_features = features / features.norm(p=2, dim=-1, keepdim=True)
        return normalized_features.cpu().numpy().astype('float32')

def search_index(query_embedding, index, image_paths, top_k):
    """Busca en el índice FAISS."""
    # D: Distancias (scores); I: IDs de los vecinos
    distances, indices = index.search(query_embedding, top_k)
    
    # Las distancias en IndexFlatIP (producto interior) son la similitud del coseno.
    results = []
    for i, (idx, score) in enumerate(zip(indices[0], distances[0])):
        # idx es el ID, image_paths[idx] es la ruta real del archivo
        results.append({
            'path': image_paths[idx],
            'score': score
        })
    return results

# --- 3. INTERFAZ DE STREAMLIT ---

def main():
    st.set_page_config(page_title="🖼️ Buscador Visual por Similitud", layout="wide")
    st.title("🖼️ Buscador Visual por Similitud (CLIP + FAISS)")
    st.markdown("---")

    # Cargar recursos (se cachean)
    model, processor, index, image_paths, device = load_resources()

    if index is None:
        st.stop() # Detener si no se pudieron cargar los recursos

    # Opciones de búsqueda
    st.sidebar.header("Opciones de Búsqueda")
    search_type = st.sidebar.radio("Tipo de Consulta:", ("Por Imagen", "Por Texto (Ampliación)"))
    top_k = st.sidebar.slider("Número de Resultados (Top K):", 1, 20, 5)

    query = None
    query_text = ""

    if search_type == "Por Imagen":
        uploaded_file = st.file_uploader("Sube una imagen de consulta", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            query = Image.open(uploaded_file)
            st.sidebar.image(query, caption="Imagen de Consulta", use_container_width=True)
            
    elif search_type == "Por Texto (Ampliación)":
        query_text = st.sidebar.text_input("Introduce la descripción de búsqueda:", "un paisaje tranquilo")
        if query_text:
            query = query_text # La consulta es el texto

    st.markdown("## Resultados de la Búsqueda")
    
    # --- EJECUTAR BÚSQUEDA ---
    if query is not None:
        if st.button("Buscar Similares"):
            with st.spinner("Buscando los vecinos más cercanos..."):
                # 1. Generar embedding de la consulta
                query_embedding = get_query_embedding(query, model, processor, device)
                
                if query_embedding is not None:
                    # 2. Buscar en el índice
                    results = search_index(query_embedding, index, image_paths, top_k)
                    
                    if results:
                        # 3. Mostrar Resultados
                        cols = st.columns(top_k)
                        
                        for i, result in enumerate(results):
                            # Usamos os.path.join para asegurarnos de que la ruta de la imagen funcione
                            full_path = result['path']
                            
                            try:
                                img = Image.open(full_path)
                                with cols[i]:
                                    st.image(img, caption=f"Score: {result['score']:.4f}", use_container_width=True)
                            except FileNotFoundError:
                                st.error(f"No se encontró el archivo: {full_path}")
                                
                    else:
                        st.warning("No se encontraron resultados.")
                else:
                    st.error("Error al generar el embedding de la consulta.")

if __name__ == "__main__":
    main()