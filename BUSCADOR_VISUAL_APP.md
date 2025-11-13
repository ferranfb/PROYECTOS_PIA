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

## Busqueda por Imagen (Score Alto)Qué ocurre: Cuando subes una foto (ej: un perro específico), estás comparando una instancia específica contra otras instancias específicas (las fotos de la base de datos).
Resultado: El modelo encuentra imágenes muy similares (ej: otros perros de la misma raza o en la misma postura). Como el parecido es tan concreto y detallado, el score de similitud es alto (ej: 0.70 - 0.85).

## Búsqueda por Texto (Score Bajo)Qué ocurre: Cuando escribes "perro", estás comparando un concepto general y abstracto contra instancias específicas (las fotos).
Resultado: El modelo CLIP entiende que la palabra "perro" es un "promedio" de todos los perros posibles (diferentes razas, colores, posturas). Por lo tanto, la similitud entre el concepto de "perro" 
y una foto específica de un perro siempre será matemáticamente menor que la similitud entre dos fotos específicas.

## Conclusión: Un score bajo (ej: 0.23 - 0.27) no significa que la búsqueda haya fallado. Significa que es la mejor coincidencia para un concepto general, y es la forma correcta en que el modelo funciona.

