import streamlit as st
from PIL import Image, ImageDraw
import io
import os
import uuid
import base64
import json
import shutil
import pickle
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="Visor de Imágenes",
    page_icon="🖼️",
    layout="wide"
)

# Estilo CSS personalizado
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .gallery-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 20px;
    }
    .image-container {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        font-size: 16px;
        font-weight: 500;
        color: rgb(49, 51, 63);
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
    }
    /* Estilos para la composición */
    .composition-canvas {
        position: relative;
        border: 2px dashed #ccc;
        background-color: #f9f9f9;
        margin: 10px 0;
        padding: 5px;
    }
    .layer-controls {
        border: 1px solid #eee;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Definir constantes para almacenamiento persistente
STORAGE_DIR = ".streamlit/image_gallery_storage"
METADATA_FILE = os.path.join(STORAGE_DIR, "gallery_metadata.json")
COMPOSITIONS_FILE = os.path.join(STORAGE_DIR, "compositions.json")

# Crear directorio de almacenamiento persistente si no existe
os.makedirs(STORAGE_DIR, exist_ok=True)

# Título de la aplicación
st.title("🖼️ Visualizador de Imágenes")
st.write("Una aplicación para visualizar imágenes PNG y crear composiciones con ellas.")

# Cargar datos de sesión persistentes si existen
def load_persisted_data():
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, 'r') as f:
                data = json.load(f)
                
                # Filtrar para asegurarnos de que solo tengamos entradas válidas sin duplicados
                unique_paths = set()
                filtered_data = []
                
                for item in data:
                    if item['path'] not in unique_paths and os.path.exists(item['path']):
                        unique_paths.add(item['path'])
                        filtered_data.append(item)
                
                return filtered_data
        except Exception as e:
            st.error(f"Error al cargar datos persistentes: {e}")
            return []
    return []

# Cargar composiciones si existen
def load_compositions():
    if os.path.exists(COMPOSITIONS_FILE):
        try:
            with open(COMPOSITIONS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error al cargar composiciones: {e}")
            return []
    return []

# Guardar datos de sesión de forma persistente
def save_persisted_data(gallery_data):
    try:
        with open(METADATA_FILE, 'w') as f:
            json.dump(gallery_data, f)
    except Exception as e:
        st.error(f"Error al guardar datos persistentes: {e}")

# Guardar composiciones de forma persistente
def save_compositions(compositions_data):
    try:
        with open(COMPOSITIONS_FILE, 'w') as f:
            json.dump(compositions_data, f)
    except Exception as e:
        st.error(f"Error al guardar composiciones: {e}")

# Inicializar y limpiar variables de estado
if 'stored_gallery_images' not in st.session_state:
    st.session_state.stored_gallery_images = load_persisted_data()
else:
    # Verificar si hay duplicados y limpiarlos
    unique_paths = set()
    unique_images = []
    
    for img in st.session_state.stored_gallery_images:
        if img['path'] not in unique_paths:
            unique_paths.add(img['path'])
            unique_images.append(img)
    
    # Actualizar el estado con imágenes únicas
    if len(unique_images) != len(st.session_state.stored_gallery_images):
        st.session_state.stored_gallery_images = unique_images
        save_persisted_data(unique_images)

# Inicializar composiciones
if 'compositions' not in st.session_state:
    st.session_state.compositions = load_compositions()

# Inicializar la composición actual si no existe
if 'current_composition' not in st.session_state:
    st.session_state.current_composition = {
        'id': str(uuid.uuid4()),
        'name': 'Nueva Composición',
        'background_image': None,
        'layers': []
    }

# Crear un directorio temporal para guardar las imágenes si no existe
if 'temp_gallery_dir' not in st.session_state:
    st.session_state.temp_gallery_dir = STORAGE_DIR

# Función para guardar imagen subida de forma persistente - SOLO PARA LA GALERÍA
def save_uploaded_image(uploaded_file):
    if uploaded_file is not None:
        # Crear un ID único para la imagen
        image_id = str(uuid.uuid4())
        image_path = os.path.join(STORAGE_DIR, f"{image_id}.png")
        
        # Guardar la imagen
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Crear registro de metadatos
        image_data = {
            'id': image_id,
            'path': image_path,
            'name': uploaded_file.name
        }
        
        # Verificar duplicados por nombre
        exists = False
        for img in st.session_state.stored_gallery_images:
            if img['name'] == uploaded_file.name:
                exists = True
                break
                
        if not exists:
            st.session_state.stored_gallery_images.append(image_data)
            # Actualizar datos persistentes
            save_persisted_data(st.session_state.stored_gallery_images)
        
        return image_path
    return None

# Función para componer imágenes
def compose_images(background_path, layers):
    try:
        if not background_path or not os.path.exists(background_path):
            return None
            
        # Cargar imagen de fondo
        background = Image.open(background_path).convert("RGBA")
        canvas = background.copy()
        
        # Agregar cada capa
        for layer in layers:
            if not os.path.exists(layer['image_path']):
                continue
                
            # Cargar imagen de la capa
            layer_img = Image.open(layer['image_path']).convert("RGBA")
            
            # Redimensionar si es necesario
            if layer.get('scale', 100) != 100:
                new_width = int(layer_img.width * layer['scale'] / 100)
                new_height = int(layer_img.height * layer['scale'] / 100)
                layer_img = layer_img.resize((new_width, new_height), Image.LANCZOS)
            
            # Calcular posición
            pos_x = layer.get('pos_x', 0)
            pos_y = layer.get('pos_y', 0)
            
            # Pegar la imagen en la posición
            canvas.paste(layer_img, (pos_x, pos_y), layer_img)
        
        return canvas
    except Exception as e:
        st.error(f"Error al componer imágenes: {e}")
        return None

# Crear pestañas
tab1, tab2, tab3 = st.tabs(["📸 Visor Individual", "🖼️ Galería de Imágenes", "🎨 Editor de Composiciones"])

# Pestaña 1: Visor de imagen individual
with tab1:
    st.header("Visor de imagen individual")
    st.write("Sube una imagen PNG para visualizarla en tiempo real.")
    
    # Mantener la última imagen entre sesiones
    if 'last_image' not in st.session_state:
        st.session_state.last_image = None
    
    # Cargar imagen desde caché si existe
    if 'last_image_path' not in st.session_state:
        last_image_cache_path = os.path.join(STORAGE_DIR, "last_image.png")
        if os.path.exists(last_image_cache_path):
            st.session_state.last_image_path = last_image_cache_path
            st.session_state.last_image_info = {
                'path': last_image_cache_path,
                'name': "Imagen anterior"
            }
            try:
                st.session_state.last_image = Image.open(last_image_cache_path)
            except:
                st.session_state.last_image = None
        else:
            st.session_state.last_image_path = None
            st.session_state.last_image_info = None
    
    uploaded_file = st.file_uploader("Elige una imagen PNG", type=["png"], key="upload_single_image")
    
    if uploaded_file is not None:
        try:
            # Abrir y mostrar la imagen
            image = Image.open(uploaded_file)
            
            # Guardar la imagen en caché únicamente, NO añadirla a la galería
            last_image_cache_path = os.path.join(STORAGE_DIR, "last_image.png")
            with open(last_image_cache_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Actualizar la información del estado
            st.session_state.last_image = image
            st.session_state.last_image_path = last_image_cache_path
            st.session_state.last_image_info = {
                'path': last_image_cache_path,
                'name': uploaded_file.name,
                'size': uploaded_file.size,
                'width': image.width,
                'height': image.height
            }
            
            # Guardar la información en un archivo adicional para metadata
            last_image_meta_path = os.path.join(STORAGE_DIR, "last_image_meta.json")
            with open(last_image_meta_path, "w") as f:
                json.dump({
                    'name': uploaded_file.name,
                    'size': uploaded_file.size,
                    'width': image.width,
                    'height': image.height
                }, f)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption=f"Nombre: {uploaded_file.name}", use_container_width=True)
            
            with col2:
                # Mostrar información de la imagen
                st.subheader("Información de la imagen")
                st.write(f"**Nombre:** {uploaded_file.name}")
                st.write(f"**Tamaño:** {uploaded_file.size} bytes")
                st.write(f"**Dimensiones:** {image.width} x {image.height} píxeles")
                
                # Botón para descargar la imagen
                img_bytes = io.BytesIO()
                image.save(img_bytes, format='PNG')
                img_b64 = base64.b64encode(img_bytes.getvalue()).decode()
                href = f'<a href="data:image/png;base64,{img_b64}" download="{uploaded_file.name}" class="download-btn">Descargar imagen</a>'
                st.markdown(href, unsafe_allow_html=True)
                
                # Botón para usar como fondo en el editor
                if st.button("Usar como fondo en el editor"):
                    # También guardar en la galería para uso futuro
                    image_path = save_uploaded_image(uploaded_file)
                    
                    # Establecer como fondo en la composición actual
                    st.session_state.current_composition['background_image'] = last_image_cache_path
                    st.success("Imagen establecida como fondo en el editor de composiciones.")
                
        except Exception as e:
            st.error(f"Error al procesar la imagen: {e}")
    
    # Mostrar la última imagen si no hay nueva carga pero existe una en caché
    elif st.session_state.last_image is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(st.session_state.last_image, caption="Última imagen subida (recuperada del caché)", use_container_width=True)
        
        with col2:
            # Intentar cargar metadata si existe
            last_image_meta_path = os.path.join(STORAGE_DIR, "last_image_meta.json")
            if os.path.exists(last_image_meta_path):
                try:
                    with open(last_image_meta_path, "r") as f:
                        meta = json.load(f)
                    
                    st.subheader("Información de la imagen (caché)")
                    st.write(f"**Nombre:** {meta.get('name', 'Desconocido')}")
                    st.write(f"**Tamaño:** {meta.get('size', 'Desconocido')} bytes")
                    st.write(f"**Dimensiones:** {meta.get('width', 'N/A')} x {meta.get('height', 'N/A')} píxeles")
                    
                    # Botón para usar como fondo en el editor
                    if st.button("Usar como fondo en el editor"):
                        st.session_state.current_composition['background_image'] = last_image_cache_path
                        st.success("Imagen establecida como fondo en el editor de composiciones.")
                except:
                    st.write("**Imagen recuperada del caché local**")
            else:
                st.write("**Imagen recuperada del caché local**")

# Pestaña 2: Galería de imágenes
with tab2:
    st.header("Galería de imágenes")
    st.write("Sube múltiples imágenes PNG para crear una galería persistente.")
    
    # Uploader para múltiples imágenes
    uploaded_files = st.file_uploader("Elige varias imágenes PNG", type=["png"], accept_multiple_files=True, key="upload_gallery_images")
    
    if uploaded_files:
        # Procesar las nuevas imágenes subidas
        for uploaded_file in uploaded_files:
            save_uploaded_image(uploaded_file)
        
        # Mostrar mensaje de éxito
        st.success(f"{len(uploaded_files)} imagen(es) añadida(s) a la galería")
        
    # Mostrar botón para limpiar la galería
    if st.session_state.stored_gallery_images:
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Eliminar todas las imágenes"):
                # Eliminar archivos (excepto last_image)
                for img in st.session_state.stored_gallery_images:
                    if os.path.exists(img['path']) and not img['path'].endswith("last_image.png"):
                        os.remove(img['path'])
                
                # Limpiar lista y actualizar datos persistentes
                st.session_state.stored_gallery_images = []
                save_persisted_data([])
                st.rerun()
        
        with col2:
            # Incluir un botón para eliminar imágenes seleccionadas
            if st.button("Actualizar galería", help="Verifica las imágenes y elimina las que ya no existen"):
                # Filtrar imágenes que aún existen
                valid_images = []
                for img in st.session_state.stored_gallery_images:
                    if os.path.exists(img['path']):
                        valid_images.append(img)
                
                st.session_state.stored_gallery_images = valid_images
                save_persisted_data(valid_images)
                st.rerun()
    
    # Mostrar la galería de imágenes
    if st.session_state.stored_gallery_images:
        st.subheader(f"Imágenes en la galería ({len(st.session_state.stored_gallery_images)})")
        
        # Crear filas de 3 imágenes
        cols = st.columns(3)
        
        # Lista para almacenar imágenes a eliminar
        if 'images_to_delete' not in st.session_state:
            st.session_state.images_to_delete = []
        
        # Mostrar cada imagen en la galería
        for i, img_data in enumerate(st.session_state.stored_gallery_images):
            # No mostrar last_image en la galería
            if "last_image.png" in img_data['path']:
                continue
                
            col_idx = i % 3
            
            with cols[col_idx]:
                if os.path.exists(img_data['path']):
                    try:
                        img = Image.open(img_data['path'])
                        st.image(img, caption=f"Imagen {i+1}: {img_data['name']}", use_container_width=True)
                        
                        # Crear un identificador único para cada botón
                        button_col1, button_col2 = st.columns(2)
                        
                        with button_col1:
                            # Botón para eliminar la imagen específica
                            delete_button_key = f"delete_{img_data['id']}"
                            if st.button("Eliminar", key=delete_button_key):
                                # Marcar la imagen para eliminar después de la iteración
                                st.session_state.images_to_delete.append(img_data)
                                st.info(f"Imagen marcada para eliminar. Actualizando...")
                                st.rerun()
                        
                        with button_col2:
                            # Botón para agregar a la composición actual
                            add_button_key = f"add_to_comp_{img_data['id']}"
                            if st.button("Añadir a composición", key=add_button_key):
                                # Agregar como capa a la composición actual
                                new_layer = {
                                    'id': str(uuid.uuid4()),
                                    'image_path': img_data['path'],
                                    'name': img_data['name'],
                                    'pos_x': 0,
                                    'pos_y': 0,
                                    'scale': 100
                                }
                                st.session_state.current_composition['layers'].append(new_layer)
                                st.success(f"Imagen añadida a la composición actual.")
                                
                    except Exception as e:
                        st.error(f"Error al cargar imagen {i+1}: {e}")
                else:
                    st.warning(f"Imagen {i+1} no encontrada")
        
        # Procesar imágenes para eliminar (fuera del bucle de renderizado)
        if st.session_state.images_to_delete:
            for img_to_delete in st.session_state.images_to_delete:
                try:
                    # Eliminar el archivo
                    if os.path.exists(img_to_delete['path']):
                        os.remove(img_to_delete['path'])
                    
                    # Eliminar de la lista de imágenes
                    if img_to_delete in st.session_state.stored_gallery_images:
                        st.session_state.stored_gallery_images.remove(img_to_delete)
                except Exception as e:
                    st.error(f"Error al eliminar imagen: {e}")
            
            # Limpiar la lista de imágenes a eliminar
            st.session_state.images_to_delete = []
            
            # Guardar los cambios
            save_persisted_data(st.session_state.stored_gallery_images)
            
    else:
        st.info("No hay imágenes en la galería. Sube algunas imágenes para verlas aquí.")

# Pestaña 3: Editor de Composiciones
with tab3:
    st.header("Editor de Composiciones")
    st.write("Crea composiciones con imágenes de tu galería. Selecciona una imagen de fondo y añade capas.")
    
    # Sección para administrar composiciones guardadas
    st.subheader("Mis Composiciones")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        # Campo para nombrar la composición actual
        composition_name = st.text_input(
            "Nombre de la composición actual:", 
            value=st.session_state.current_composition.get('name', 'Nueva Composición')
        )
        st.session_state.current_composition['name'] = composition_name
    
    with col2:
        # Selector de composiciones guardadas
        composition_options = [comp['name'] for comp in st.session_state.compositions]
        if composition_options:
            selected_composition = st.selectbox(
                "Cargar composición guardada:", 
                options=["Nueva Composición"] + composition_options
            )
            
            if selected_composition != "Nueva Composición":
                for comp in st.session_state.compositions:
                    if comp['name'] == selected_composition:
                        if st.button("Cargar esta composición"):
                            st.session_state.current_composition = comp.copy()
                            st.success(f"Composición '{selected_composition}' cargada.")
                            st.rerun()
    
    with col3:
        # Botones para guardar/eliminar composición
        if st.button("Guardar Composición"):
            # Verificar que tenga un nombre y una imagen de fondo
            if not st.session_state.current_composition.get('name'):
                st.error("Por favor, asigna un nombre a la composición.")
            elif not st.session_state.current_composition.get('background_image'):
                st.error("Por favor, selecciona una imagen de fondo.")
            else:
                # Comprobar si ya existe una composición con este nombre
                exists = False
                for i, comp in enumerate(st.session_state.compositions):
                    if comp['name'] == st.session_state.current_composition['name']:
                        # Actualizar composición existente
                        st.session_state.compositions[i] = st.session_state.current_composition.copy()
                        exists = True
                        break
                
                if not exists:
                    # Agregar nueva composición
                    st.session_state.compositions.append(st.session_state.current_composition.copy())
                
                # Guardar composiciones
                save_compositions(st.session_state.compositions)
                st.success(f"Composición '{st.session_state.current_composition['name']}' guardada.")
    
    # Sección para seleccionar imagen de fondo
    st.subheader("Imagen de Fondo")
    
    # Mostrar imagen de fondo actual si existe
    background_image_path = st.session_state.current_composition.get('background_image')
    
    if background_image_path and os.path.exists(background_image_path):
        try:
            bg_image = Image.open(background_image_path)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(bg_image, caption="Imagen de fondo actual", use_container_width=True)
            with col2:
                st.write(f"**Dimensiones:** {bg_image.width} x {bg_image.height} píxeles")
                
                # Selector para elegir otra imagen de fondo de la galería
                bg_options = []
                bg_paths = {}
                
                # Agregar opción para la última imagen
                if st.session_state.last_image_path and os.path.exists(st.session_state.last_image_path):
                    last_img_name = "Última imagen (cargada en visor)"
                    bg_options.append(last_img_name)
                    bg_paths[last_img_name] = st.session_state.last_image_path
                
                # Agregar opciones de la galería
                for img in st.session_state.stored_gallery_images:
                    if os.path.exists(img['path']):
                        bg_options.append(img['name'])
                        bg_paths[img['name']] = img['path']
                
                if bg_options:
                    selected_bg = st.selectbox("Cambiar fondo por otra imagen:", options=["Mantener actual"] + bg_options)
                    
                    if selected_bg != "Mantener actual" and st.button("Aplicar cambio de fondo"):
                        if selected_bg in bg_paths:
                            st.session_state.current_composition['background_image'] = bg_paths[selected_bg]
                            st.success(f"Fondo cambiado a: {selected_bg}")
                            st.rerun()
        except Exception as e:
            st.error(f"Error al cargar imagen de fondo: {e}")
    else:
        st.info("No has seleccionado una imagen de fondo. Por favor, selecciona una imagen de la galería o sube una en el Visor Individual.")
        
        # Selector para elegir una imagen de fondo de la galería
        bg_options = []
        bg_paths = {}
        
        # Agregar opción para la última imagen
        if st.session_state.last_image_path and os.path.exists(st.session_state.last_image_path):
            last_img_name = "Última imagen (cargada en visor)"
            bg_options.append(last_img_name)
            bg_paths[last_img_name] = st.session_state.last_image_path
        
        # Agregar opciones de la galería
        for img in st.session_state.stored_gallery_images:
            if os.path.exists(img['path']):
                bg_options.append(img['name'])
                bg_paths[img['name']] = img['path']
        
        if bg_options:
            selected_bg = st.selectbox("Seleccionar imagen de fondo:", options=bg_options)
            
            if st.button("Establecer como fondo"):
                if selected_bg in bg_paths:
                    st.session_state.current_composition['background_image'] = bg_paths[selected_bg]
                    st.success(f"Fondo establecido: {selected_bg}")
                    st.rerun()
        else:
            st.warning("No hay imágenes disponibles. Por favor, sube algunas en el Visor Individual o en la Galería.")
    
    # Sección para gestionar las capas
    st.subheader("Capas")
    
    layers = st.session_state.current_composition.get('layers', [])
    
    if not layers:
        st.info("No hay capas en esta composición. Añade imágenes desde la galería para empezar a componer.")
    else:
        # Mostrar la previsualización de la composición
        preview_image = compose_images(
            st.session_state.current_composition.get('background_image'),
            layers
        )
        
        if preview_image:
            st.subheader("Vista previa de la composición")
            st.image(preview_image, caption="Composición actual", use_container_width=True)
            
            # Botón para descargar la composición
            img_bytes = io.BytesIO()
            preview_image.save(img_bytes, format='PNG')
            img_b64 = base64.b64encode(img_bytes.getvalue()).decode()
            href = f'<a href="data:image/png;base64,{img_b64}" download="{st.session_state.current_composition["name"]}.png" class="download-btn">Descargar composición</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        # Mostrar controles para cada capa
        st.write("Ajustar posición y escala de las capas:")
        
        # Crear una lista para almacenar capas a eliminar
        layers_to_remove = []
        
        for i, layer in enumerate(layers):
            with st.expander(f"Capa {i+1}: {layer.get('name', 'Sin nombre')}"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Mostrar miniatura de la imagen de la capa
                    if os.path.exists(layer['image_path']):
                        try:
                            layer_img = Image.open(layer['image_path'])
                            st.image(layer_img, caption=layer.get('name', 'Sin nombre'), width=150)
                        except Exception as e:
                            st.error(f"Error al cargar imagen de la capa: {e}")
                
                with col2:
                    # Controles para posición X e Y
                    pos_x = st.slider(
                        "Posición X:", 
                        min_value=-500, 
                        max_value=1000, 
                        value=layer.get('pos_x', 0),
                        key=f"pos_x_{layer['id']}"
                    )
                    layer['pos_x'] = pos_x
                    
                    pos_y = st.slider(
                        "Posición Y:", 
                        min_value=-500, 
                        max_value=1000, 
                        value=layer.get('pos_y', 0),
                        key=f"pos_y_{layer['id']}"
                    )
                    layer['pos_y'] = pos_y
                    
                    scale = st.slider(
                        "Escala (%):", 
                        min_value=10, 
                        max_value=200, 
                        value=layer.get('scale', 100),
                        key=f"scale_{layer['id']}"
                    )
                    layer['scale'] = scale
                    
                    # Botones para gestionar la capa
                    col_btn1, col_btn2, col_btn3 = st.columns(3)
                    
                    with col_btn1:
                        if st.button("Eliminar capa", key=f"remove_{layer['id']}"):
                            layers_to_remove.append(i)
                            st.info("Capa marcada para eliminar.")
                    
                    with col_btn2:
                        # Mover capa arriba (al frente)
                        if i > 0 and st.button("Mover arriba", key=f"up_{layer['id']}"):
                            layers[i], layers[i-1] = layers[i-1], layers[i]
                            st.rerun()
                    
                    with col_btn3:
                        # Mover capa abajo (al fondo)
                        if i < len(layers)-1 and st.button("Mover abajo", key=f"down_{layer['id']}"):
                            layers[i], layers[i+1] = layers[i+1], layers[i]
                            st.rerun()
        
        # Procesar eliminaciones de capas (fuera del bucle)
        if layers_to_remove:
            # Eliminar en orden inverso para no alterar los índices
            for idx in sorted(layers_to_remove, reverse=True):
                if 0 <= idx < len(layers):
                    del layers[idx]
            st.success("Capas eliminadas correctamente.")
            st.rerun()
        
        # Botón para crear nueva composición
        if st.button("Nueva composición en blanco"):
            # Crear una nueva composición en blanco
            st.session_state.current_composition = {
                'id': str(uuid.uuid4()),
                'name': 'Nueva Composición',
                'background_image': None,
                'layers': []
            }
            st.success("Nueva composición creada.")
            st.rerun()

# Manejo de errores y limpieza
st.sidebar.title("Información")
st.sidebar.info("""
Esta aplicación guarda las imágenes y composiciones de forma local usando el sistema de archivos.
Las imágenes y composiciones permanecerán entre sesiones mientras uses el mismo dispositivo y navegador.
""")

# Opción para exportar/importar la galería (opcional)
with st.sidebar.expander("Opciones avanzadas"):
    if st.button("Verificar almacenamiento"):
        # Verificar que el directorio existe
        if not os.path.exists(STORAGE_DIR):
            os.makedirs(STORAGE_DIR, exist_ok=True)
            st.sidebar.success("Directorio de almacenamiento creado correctamente.")
        else:
            # Contar archivos
            file_count = len([f for f in os.listdir(STORAGE_DIR) if f.endswith('.png')])
            st.sidebar.success(f"Almacenamiento OK: {file_count} imágenes en disco.")
            
    # Opción para forzar la recarga del estado desde el disco
    if st.button("Recargar desde disco"):
        if os.path.exists(STORAGE_DIR):
            # Recargar metadatos
            st.session_state.stored_gallery_images = load_persisted_data()
            
            # Recargar composiciones
            st.session_state.compositions = load_compositions()
            
            # Verificar la imagen individual
            last_image_path = os.path.join(STORAGE_DIR, "last_image.png")
            if os.path.exists(last_image_path):
                try:
                    st.session_state.last_image = Image.open(last_image_path)
                    st.session_state.last_image_path = last_image_path
                except:
                    st.session_state.last_image = None
            
            st.sidebar.success("Datos recargados desde el disco.")
            st.rerun()
        else:
            st.sidebar.error("No se encontró almacenamiento en disco.")
    
    # Botón para eliminar composiciones
    if st.session_state.compositions:
        if st.button("Eliminar todas las composiciones"):
            st.session_state.compositions = []
            
            # Guardar (o en este caso, sobrescribir con lista vacía)
            save_compositions([])
            
            st.sidebar.success("Todas las composiciones han sido eliminadas.")
            st.rerun()

# Pie de página
st.markdown("---")
st.markdown("Aplicación de visualización y composición de imágenes con almacenamiento local persistente.")

# Limpieza al cerrar la aplicación (aunque esto no siempre funciona en Streamlit)
def cleanup():
    # Ya no eliminamos los archivos al cerrar, para mantenerlos entre sesiones
    pass

# Intentar registrar la función de limpieza
import atexit
atexit.register(cleanup)