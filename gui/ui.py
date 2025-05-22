import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
import types
import torch
import json
from sklearn.metrics.pairwise import cosine_similarity

# Patch torch.classes so Streamlit doesn't crash
if not hasattr(torch, 'classes'):
    torch.classes = types.SimpleNamespace()
torch.classes.__path__ = []  # Prevent Streamlit from crashing on __path__ lookup


PATH = os.path.abspath(os.path.dirname(__file__))
WARDROBE_DIR = os.path.join(PATH, "wardrobe_items")
os.makedirs(WARDROBE_DIR, exist_ok=True)

METADATA_FILE = os.path.join(WARDROBE_DIR, "wardrobe_metadata.json")

# --- Metadata Loading and Saving ---
def load_metadata():
    """
    Loads wardrobe metadata from a JSON file if it exists.
    Returns an empty dictionary if the file does not exist.
    """
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    return {}

def save_metadata(metadata):
    """
    Saves the given metadata dictionary to a JSON file.
    """
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

# --- Streamlit Dialogs ---
@st.dialog("Suggested Outfit")
def show_outfit_dialog(target_id, metadata):
    """
    Shows a dialog with suggested outfit items based on the target item.
    """
    outfit_ids = suggest_outfit(target_id, metadata)
    
    if not outfit_ids:
        st.warning("No suitable outfit suggestions found.")
    else:
        st.markdown("### Matching Items:")
        for oid in outfit_ids:
            item = metadata[oid]
            olabel = item["label"]
            opath = os.path.join(WARDROBE_DIR, oid)
            st.image(opath, caption=f"{olabel.capitalize()} (Suggested)", use_container_width=True)

@st.dialog("Clothing Item Details")
def show_item_details(item_path, meta):
    """
    Shows a dialog with details for a clothing item, including style and colors.
    """
    st.image(item_path, use_container_width=True)
    
    if meta.get("style"):
        st.markdown(f"**Predicted Style:** `{meta['style']}`")

    if meta.get("colors"):
        colors = np.array(meta["colors"])
        st.markdown("**Dominant Colors:**")
        plot_colors(colors)

# --- Model Loading ---
@st.cache_resource
def load_clip():
    """
    Loads the CLIP processor and model from Hugging Face.
    """
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    return processor, model

@st.cache_resource
def load_yolo():
    """
    Loads the YOLO model for clothing segmentation.
    """
    model_path = os.path.join(PATH, "..", "weights", "best.pt")
    return YOLO(model_path)

STYLE_LABELS = ["casual", "formal", "sporty", "streetwear", "bohemian", "business", "vintage", "chic"]

@st.cache_resource
def predict_style(pil_image):
    """
    Predicts the clothing style of a PIL image using CLIP.
    Returns the most probable style and the full probability distribution.
    """
    processor, model = load_clip()
    inputs = processor(text=STYLE_LABELS, images=pil_image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

    # Return the most probable label and full distribution
    style = STYLE_LABELS[np.argmax(probs)]
    return style, dict(zip(STYLE_LABELS, map(float, probs)))

def outfit_similarity(item_a, item_b, weight_style=0.6, weight_color=0.4):
    """
    Computes a similarity score between two wardrobe items based on style and color.
    """
    # Style similarity
    style_sim = cosine_similarity(
        [item_a["style_probs"]], [item_b["style_probs"]]
    )[0][0]

    # Color similarity (mean color per item)
    color_a = np.mean(item_a["colors"], axis=0)
    color_b = np.mean(item_b["colors"], axis=0)
    color_sim = cosine_similarity([color_a], [color_b])[0][0]

    return weight_style * style_sim + weight_color * color_sim

def suggest_outfit(target_id, metadata, top_n=2):
    """
    Suggests outfit items that best match the target item, based on similarity.
    Only items from different categories are considered.
    Returns the IDs of the top N matching items.
    """
    target = metadata[target_id]
    label = target["label"]

    # Only compare with items in different categories
    candidates = {
        k: v for k, v in metadata.items()
        if k != target_id and v["label"] != label
    }

    if not candidates:
        return []

    scores = []
    for cid, data in candidates.items():
        score = outfit_similarity(target, data)
        scores.append((cid, score))

    top_matches = sorted(scores, key=lambda x: -x[1])[:top_n]
    return [match[0] for match in top_matches]

def extract_clothes(image):
    """
    Uses YOLO to segment clothing items from an image.
    Returns:
    - List of (PIL image, label) tuples for each detected item.
    - Rendered segmentation image (RGB format).
    """
    model = load_yolo()
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

        clothes.append((Image.fromarray(masked_img), label))

    # Get rendered YOLO image and convert BGR to RGB
    rendered_image_bgr = result.plot()  # BGR format
    rendered_image_rgb = cv2.cvtColor(rendered_image_bgr, cv2.COLOR_BGR2RGB)

    return clothes, rendered_image_rgb

# --- Color Extraction ---
def extract_dominant_colors(pil_img, k=5):
    """
    Extracts the k dominant colors from a PIL RGBA image using KMeans clustering.
    Returns an array of RGB color values.
    """
    np_image = np.array(pil_img.convert("RGBA"))
    mask = np_image[:, :, 3] > 0
    pixels = np_image[:, :, :3][mask]

    if len(pixels) > 10000:
        pixels = pixels[np.random.choice(len(pixels), 10000, replace=False)]

    kmeans = KMeans(n_clusters=k, random_state=42).fit(pixels)
    return kmeans.cluster_centers_.astype(int)

def plot_colors(colors):
    """
    Plots a horizontal bar of the given colors using matplotlib and displays it in Streamlit.
    """
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
    The image is cropped to its alpha bounding box and resized to fit the output size.
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

def save_item(image, label, idx, style=None, style_probs=None, colors=None):
    """
    Saves a clothing item image and its metadata to the wardrobe directory.
    Updates the metadata JSON file with the new item.
    Returns the file path of the saved image.
    """
    label_folder = os.path.join(WARDROBE_DIR, label)
    os.makedirs(label_folder, exist_ok=True)

    filename = f"item_{idx}_{np.random.randint(1e6)}.png"
    path = os.path.join(label_folder, filename)
    image.save(path)

    metadata = load_metadata()
    id = "/".join(path.split("/")[-2:])

    color_list = [color.tolist() for color in colors] if colors is not None else None

    metadata[id] = {
        "label": label,
        "style": style,
        "style_probs": list(style_probs.values()) if style_probs else None,
        "colors": color_list,
    }
    save_metadata(metadata)

    return path

def list_wardrobe_items():
    """
    Lists all wardrobe items by scanning the wardrobe directory.
    Returns a list of (label, file path) tuples for each item.
    """
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
        uploaded_image_slot = st.empty()  # reserve a space for the image
        uploaded_image_slot.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Segmenting clothes..."):
            clothes, segmented_image = extract_clothes(image)

        uploaded_image_slot.image(segmented_image, caption="Segmented Image", use_container_width=True)
        st.success(f"Found {len(clothes)} item(s).")

        for idx, (item, label) in enumerate(clothes):
            centered_item = center_and_scale(item)
            st.image(centered_item, caption=f"Clothing Item {idx+1}", use_container_width=True)

            colors = extract_dominant_colors(centered_item)
            with st.expander(f"🎨 Dominant Colors for Item {idx+1}"):
                plot_colors(colors)

            # Predict style
            style, probs = predict_style(centered_item)
            
            st.markdown(f"🧷 **Style Prediction for Item {idx+1}**")
            st.markdown(f"**Predicted Style:** `{style}`")
            st.progress(probs[style])

            # Show detailed probabilities in a separate expander (not nested)
            with st.expander(f"🔍 All Style Probabilities for Item {idx+1}"):
                for s, p in sorted(probs.items(), key=lambda x: -x[1]):
                    st.write(f"{s.capitalize()}: {p:.2%}")

            # Add to wardrobe
            if st.button(f"Add {label} to Wardrobe ({idx+1}/{len(clothes)})", key=f"add_{idx}"):
                save_item(centered_item, label, idx, style=style, style_probs=probs, colors=colors)
                st.success(f"Item saved to wardrobe under '{label}' with style '{style}'!")

elif page == "View Wardrobe":
    st.title("👗 Your Wardrobe")

    wardrobe = list_wardrobe_items()
    metadata = load_metadata()

    if not wardrobe:
        st.info("No items in wardrobe yet. Try uploading from the Upload Clothes tab.")
    else:
        cols = st.columns(3)
        for i, (label, item_path) in enumerate(wardrobe):
            with cols[i % 3]:
                st.image(item_path, use_container_width=True)
                st.caption(f"Item {i}: {label.capitalize()}")

                item_id = "/".join(item_path.split("/")[-2:])
                meta = metadata.get(item_id, {})

                if meta.get("style"):
                    st.markdown(f"**Style:** `{meta['style']}`")

                btn_col1, btn_col2 = st.columns([1, 1])
                with btn_col1:
                    if st.button("Details", key=f"details_{i}"):
                        show_item_details(item_path, meta)
                with btn_col2:
                    if st.button("Remove", key=f"remove_{i}"):
                        os.remove(item_path)
                        if item_id in metadata:
                            del metadata[item_id]
                            save_metadata(metadata)
                        st.rerun()
                if st.button("Get Outfit", key=f"get_outfit_{i}"):
                    show_outfit_dialog(item_id, metadata)