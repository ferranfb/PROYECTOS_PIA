import os
import glob
from PIL import Image
import torch
import numpy as np
import faiss
from transformers import CLIPProcessor, CLIPModel

# --- CONFIGURACIÓN ---
ASSETS_DIR = "assets"
MODEL_NAME = "openai/clip-vit-base-patch32"
INDEX_FILE = "faiss_index.bin"
MAPPING_FILE = "image_paths.npy"

# --- 1. CARGA DEL MODELO ---
def load_clip_model():
    """Carga el modelo CLIP y el procesador."""
    print(f"Cargando modelo: {MODEL_NAME}...")
    # Intenta usar GPU si está disponible, si no, usa CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    
    print(f"Modelo cargado y ejecutándose en: {device}")
    return model, processor, device

# --- 2. GENERACIÓN DE EMBEDDINGS ---
def get_image_embeddings(image_path, model, processor, device):
    """Procesa una imagen y devuelve su embedding."""
    try:
        # 1. Cargar y convertir a RGB (CLIP lo requiere)
        image = Image.open(image_path).convert("RGB")
        
        # 2. Preprocesar la imagen (resizing, normalización, etc.)
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        # 3. Obtener el embedding (vector)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        
        # 4. Normalizar el vector (clave para la similitud del coseno/FAISS)
        # Esto asegura que la longitud del vector sea 1.
        normalized_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        
        # 5. Devolver como array de NumPy
        return normalized_features.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error procesando {image_path}: {e}")
        return None

# --- 3. PROCESO PRINCIPAL DE INDEXACIÓN ---
def index_dataset(model, processor, device):
    """Genera embeddings para todo el dataset y construye el índice FAISS."""
    # Buscar todas las imágenes (soporta jpg, jpeg, png)
    image_paths = glob.glob(os.path.join(ASSETS_DIR, "*.[jJpP][pPeE][gGgG]")) 
    
    if not image_paths:
        print(f"🚨 ¡Advertencia! No se encontraron imágenes en la carpeta '{ASSETS_DIR}'.")
        return

    all_embeddings = []
    indexed_paths = []

    print(f"\nGenerando embeddings para {len(image_paths)} imágenes...")
    
    # Procesar cada imagen
    for i, path in enumerate(image_paths):
        print(f"  Procesando {i+1}/{len(image_paths)}: {os.path.basename(path)}")
        embedding = get_image_embeddings(path, model, processor, device)
        
        if embedding is not None:
            all_embeddings.append(embedding)
            indexed_paths.append(path)

    # Si no se pudo generar ningún embedding
    if not all_embeddings:
        print("🚨 Error: No se pudo generar ningún embedding. Terminando.")
        return

    # Convertir a array de NumPy
    embeddings_matrix = np.array(all_embeddings).astype('float32')
    D = embeddings_matrix.shape[1] # Dimensión del embedding (debería ser 512)

    # 4. CONSTRUCCIÓN DEL ÍNDICE FAISS
    # IndexFlatIP: Índice de Producto Interior. Perfecto para vectores normalizados (Similitud del Coseno).
    print(f"\nConstruyendo índice FAISS con {len(indexed_paths)} vectores de dimensión {D}...")
    index = faiss.IndexFlatIP(D)
    index.add(embeddings_matrix) 
    
    # 5. PERSISTENCIA
    # Guardar el índice y el mapping (ruta de archivo -> ID de FAISS)
    faiss.write_index(index, INDEX_FILE)
    np.save(MAPPING_FILE, indexed_paths)

    print("\n✅ ¡INDEXACIÓN COMPLETADA Y GUARDADA!")
    print(f"   - Índice FAISS: {INDEX_FILE}")
    print(f"   - Mapeo de rutas: {MAPPING_FILE}")
    print("   -> Ya puedes pasar a la 'Página Buscar'.")

# --- EJECUTAR ---
if __name__ == "__main__":
    model, processor, device = load_clip_model()
    index_dataset(model, processor, device)

    print(f"\nBuscando imágenes en la ruta absoluta: {os.path.abspath(ASSETS_DIR)}")
    all_files_in_dir = os.listdir(ASSETS_DIR)
    print(f"  Debug: Archivos encontrados en 'assets': {all_files_in_dir}")