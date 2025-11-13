# 🖼️ Buscador Visual por Similitud (Proyecto Final)

Este proyecto es una aplicación web que permite realizar búsquedas visuales por similitud dentro de un repositorio de imágenes local. La búsqueda se puede realizar de dos maneras:
1.  **Por Imagen:** Subiendo una imagen de consulta.
2.  **Por Texto:** Escribiendo una descripción (ej: "un perro corriendo").

## 🎯 Objetivo

El objetivo es implementar un sistema que utiliza **embeddings** (vectores de IA) para encontrar las imágenes más similares (búsqueda *Top-K*) y mostrar los resultados en una interfaz gráfica con su puntuación (*score*) de similitud.

## 🛠️ Pila Tecnológica (Tecnologías Usadas)

* **Modelo de IA (Embeddings):** `openai/clip-vit-base-patch32` (de Hugging Face). Elegida a dedo per Rafa
* **¿Qué es?** FAISS (Facebook AI Similarity Search) no es una base de datos tradicional (como SQL). Es una librería especializada y optimizada únicamente para almacenar y buscar **vectores** de alta dimensión (como nuestros embeddings de 512 dimensiones).
    * **¿Por qué la elegimos?** Se eligió por dos razones clave:
        1.  **Velocidad Extrema:** Está diseñada para comparar millones de vectores en milisegundos. Es mucho más rápida que una búsqueda manual.
        2.  **Integración:** Funciona perfectamente en Python y se integra de forma nativa con `Numpy`, facilitando el flujo de trabajo con los embeddings generados por PyTorch/CLIP.
* **Base de Datos Vectorial:** `FAISS` (de Facebook AI). Se utiliza para almacenar los miles de vectores de imagen y realizar búsquedas de similitud (`K-Nearest Neighbors`) de forma ultrarrápida.
* **Interfaz de Usuario (UI):** `Streamlit`. Permite crear una aplicación web interactiva usando únicamente código Python, ideal para prototipar rápido.
* **Librerías de Python:** `PyTorch`, `Transformers` (para cargar CLIP), `Numpy` (para manejo de vectores) y `Pillow` (para imágenes).

## 🚀 Cómo Ejecutar Localmente (Para la Demo)

Para ejecutar el proyecto en local, solo se necesitan estos pasos:

```bash
# 1. Crear un entorno virtual (ej: venv_proyecto) y activarlo
python -m venv venv_proyecto
.\venv_proyecto\Scripts\activate

# 2. Instalar las dependencias necesarias
pip install streamlit numpy Pillow faiss-cpu torch transformers

# 3. (Paso OBLIGATORIO) Indexar las imágenes de la carpeta /assets
# Esto crea los archivos faiss_index.bin y image_paths.npy
python indexarembeddings.py

# 4. Ejecutar la aplicación web
streamlit run buscadorvisual_app.py

```
## Como Cargamos los modelos CLIP/FAISS y los embeddings para nuestro proyecto
```
@st.cache_resource
def load_resources():
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
```
## PER A EJECUTAR AMB STREAMLIT, LA INTERFAZ DE USUARI TRIADA
```
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

```
## Busqueda por Imagen (Score Alto)Qué ocurre: Cuando subes una foto (ej: un perro específico), estás comparando una instancia específica contra otras instancias específicas (las fotos de la base de datos).
Resultado: El modelo encuentra imágenes muy similares (ej: otros perros de la misma raza o en la misma postura). Como el parecido es tan concreto y detallado, el score de similitud es alto (ej: 0.70 - 0.85).

## Búsqueda por Texto (Score Bajo)Qué ocurre: Cuando escribes "perro", estás comparando un concepto general y abstracto contra instancias específicas (las fotos).
Resultado: El modelo CLIP entiende que la palabra "perro" es un "promedio" de todos los perros posibles (diferentes razas, colores, posturas). Por lo tanto, la similitud entre el concepto de "perro" 
y una foto específica de un perro siempre será matemáticamente menor que la similitud entre dos fotos específicas.

## Conclusión: Un score bajo (ej: 0.23 - 0.27) no significa que la búsqueda haya fallado. Significa que es la mejor coincidencia para un concepto general, y es la forma correcta en que el modelo funciona.

