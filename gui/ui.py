import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import types
import torch

# Patch torch.classes so Streamlit doesn't crash
if not hasattr(torch, 'classes'):
    torch.classes = types.SimpleNamespace()
torch.classes.__path__ = []  # Prevent Streamlit from crashing on __path__ lookup


PATH = os.path.abspath(os.path.dirname(__file__))
WARDROBE_DIR = os.path.join(PATH, "wardrobe_items")
os.makedirs(WARDROBE_DIR, exist_ok=True)

@st.cache_resource
def load_model():
    model_path = os.path.join(PATH, "..", "weights", "best.pt")
    return YOLO(model_path)

def extract_clothes(image):
    model = load_model()
    results = model(image)
    orig = np.array(image.convert("RGB"))
    clothes = []

    result = results[0]  # YOLO result object
    masks = result.masks.data.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    names = result.names  # class ID to label name mapping

    for i, mask in enumerate(masks):
        label = names[class_ids[i]]

        mask = cv2.resize(mask, (orig.shape[1], orig.shape[0]))
        masked_img = np.zeros((orig.shape[0], orig.shape[1], 4), dtype=np.uint8)
        for c in range(3):
            masked_img[:, :, c] = orig[:, :, c] * mask.astype(np.uint8)
        masked_img[:, :, 3] = (mask * 255).astype(np.uint8)

        clothes.append((Image.fromarray(masked_img), label))  # return label with image

    return clothes

def extract_dominant_colors(pil_img, k=5):
    np_image = np.array(pil_img.convert("RGBA"))
    mask = np_image[:, :, 3] > 0
    pixels = np_image[:, :, :3][mask]

    if len(pixels) > 10000:
        pixels = pixels[np.random.choice(len(pixels), 10000, replace=False)]

    kmeans = KMeans(n_clusters=k, random_state=42).fit(pixels)
    return kmeans.cluster_centers_.astype(int)

def plot_colors(colors):
    fig, ax = plt.subplots(figsize=(6, 1))
    for i, color in enumerate(colors):
        hex_color = "#{:02x}{:02x}{:02x}".format(*color)
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=hex_color))
        ax.text(i + 0.5, -0.3, hex_color, ha="center", fontsize=8)
    plt.xlim(0, len(colors))
    plt.axis('off')
    st.pyplot(fig)


def center_and_scale(image, output_size=(256, 256)):
    """
    Takes a PIL RGBA image and returns a centered, scaled version on a square canvas.
    """
    # Ensure RGBA mode
    image = image.convert("RGBA")

    # Get alpha channel and compute bounding box
    alpha = image.split()[-1]
    bbox = alpha.getbbox()

    if bbox:
        # Crop image to the bounding box
        cropped = image.crop(bbox)
    else:
        cropped = image

    # Resize while keeping aspect ratio
    cropped.thumbnail(output_size, Image.Resampling.LANCZOS)

    # Create a new transparent canvas and paste the resized item centered
    canvas = Image.new("RGBA", output_size, (0, 0, 0, 0))
    paste_x = (output_size[0] - cropped.width) // 2
    paste_y = (output_size[1] - cropped.height) // 2
    canvas.paste(cropped, (paste_x, paste_y), mask=cropped)

    return canvas

def save_item(image, label, idx):
    label_folder = os.path.join(WARDROBE_DIR, label)
    os.makedirs(label_folder, exist_ok=True)
    
    filename = f"item_{idx}_{np.random.randint(1e6)}.png"
    path = os.path.join(label_folder, filename)
    image.save(path)
    return path

def list_wardrobe_items():
    items = []
    for label_dir in os.listdir(WARDROBE_DIR):
        label_path = os.path.join(WARDROBE_DIR, label_dir)
        if os.path.isdir(label_path):
            for f in os.listdir(label_path):
                if f.endswith(".png"):
                    items.append((label_dir, os.path.join(label_path, f)))
    return items

# --- UI Setup ---
st.set_page_config(page_title="AI Wardrobe", layout="centered")

st.sidebar.title("🧭 Navigation")
page = st.sidebar.selectbox("Go to", ["Upload Clothes", "View Wardrobe"])

# --- Pages ---
if page == "Upload Clothes":
    st.title("👕 Upload Clothes")
    uploaded_file = st.file_uploader("Upload a photo of yourself", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Segmenting clothes..."):
            clothes = extract_clothes(image)

        st.success(f"Found {len(clothes)} item(s).")

        for idx, (item, label) in enumerate(clothes):
            centered_item = center_and_scale(item)
            st.image(centered_item, caption=f"Clothing Item {idx+1}", use_container_width=True)

            with st.expander(f"🎨 Dominant Colors for Item {idx+1}"):
                colors = extract_dominant_colors(centered_item)
                plot_colors(colors)

            if st.button(f"Add {label} to Wardrobe ({idx+1}/{len(clothes)})", key=f"add_{idx}"):
                save_item(center_and_scale(item), label, idx)
                st.success(f"Item saved to wardrobe under '{label}'!")


elif page == "View Wardrobe":
    st.title("👗 Your Wardrobe")

    wardrobe = list_wardrobe_items()
    if not wardrobe:
        st.info("No items in wardrobe yet. Try uploading from the Upload Clothes tab.")
    else:
        cols = st.columns(3)
        for i, (label, item_path) in enumerate(wardrobe):
            with cols[i % 3]:
                st.image(item_path, use_container_width=True)
                st.caption(f"Item {i}: {label.capitalize()}")
                if st.button("Remove", key=f"remove_{i}"):
                    os.remove(item_path)
                    st.rerun()  # Refresh to update the list and UI