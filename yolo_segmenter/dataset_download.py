import json
from pathlib import Path
import cv2, numpy as np
from PIL import Image
from datasets import load_dataset
from torch.utils.data import random_split        # for simple 95 / 5 split

# ---------------- configuration ----------------------------------
DATASET_NAME = "sergiomadrid/imaterialist"
OUT_ROOT     = Path("dataset")
VAL_RATIO    = 0.05
NUM_CLASSES  = 46                # IDs 0-45 in the mask
SEED         = 42

CLASS_NAMES = [
    "shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "cardigan", "jacket",
    "vest", "pants", "shorts", "skirt", "coat", "dress", "jumpsuit", "cape",
    "glasses", "hat", "headband, head covering, hair accessory", "tie", "glove",
    "watch", "belt", "leg warmer", "tights, stockings", "sock", "shoe",
    "bag, wallet", "scarf", "umbrella", "hood", "collar", "lapel", "epaulette",
    "sleeve", "pocket", "neckline", "buckle", "zipper", "applique", "bead",
    "bow", "flower", "fringe", "ribbon", "rivet", "ruffle", "sequin", "tassel"
]

assert len(CLASS_NAMES) == NUM_CLASSES

# ---------------- create directory tree --------------------------
for sub in (
    "images/train", "images/val",
    "masks/train",  "masks/val",
    "labels/train", "labels/val",
):
    (OUT_ROOT / sub).mkdir(parents=True, exist_ok=True)

# ---------------- helper to write one split ---------------------
def save_split(split_ds, split_name: str):
    for ex in split_ds:
        img_id = ex["ImageId"]

        # --- image -------------------------------------------------
        img: Image.Image = ex["image"]
        img.save(OUT_ROOT / f"images/{split_name}/{img_id}.jpg", quality=95)

        # --- mask --------------------------------------------------
        mask = np.array(ex["mask"], dtype=np.int32)
        h, w  = mask.shape
        Image.fromarray(mask.astype(np.uint8), mode="L")\
             .save(OUT_ROOT / f"masks/{split_name}/{img_id}.png")

        # --- labels -----------------------------------------------
        with open(OUT_ROOT / f"labels/{split_name}/{img_id}.txt", "w") as f:
            # Finds the polygons for each class in the mask
            for ds_cls in range(0, NUM_CLASSES):         
                cls_mask = (mask == ds_cls).astype(np.uint8)
                if not cls_mask.any():
                    continue
                contours, _ = cv2.findContours(
                    cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for cnt in contours:
                    eps  = 0.01 * cv2.arcLength(cnt, True)
                    poly = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
                    if len(poly) < 3:
                        continue
                    coords = []
                    # Writes the label into a YOLO-style .txt label file 
                    # (polygon segmentation format), normalized coordinates.
                    for x, y in poly:
                        coords += [x / w, y / h]
                    yolo_cls = ds_cls
                    f.write(" ".join([str(yolo_cls)]
                                     + [f"{v:.6f}" for v in coords]) + "\n")

# ---------------- main -------------------------------------------
print("🔄  Loading dataset …")
hf_ds = load_dataset(DATASET_NAME)

# train / val split
val_size   = int(VAL_RATIO * len(hf_ds["train"]))
train_size = len(hf_ds["train"]) - val_size
train_ds, val_ds = random_split(hf_ds["train"], [train_size, val_size],
                                generator=None)

print(f"✍️  Saving {train_size} train images …")
save_split(train_ds, "train")

print(f"✍️  Saving {val_size} val images …")
save_split(val_ds, "val")

# data.yaml
yaml_path = OUT_ROOT / "data.yaml"
if not yaml_path.exists():
    yaml = {
        "path": str(OUT_ROOT),
        "train": "images/train",
        "val":   "images/val",
        "nc":    NUM_CLASSES,
        "names": CLASS_NAMES,
    }
    yaml_path.write_text(json.dumps(yaml, indent=2))
    print(f"📄  Wrote {yaml_path}")

print("✅  Done!")

