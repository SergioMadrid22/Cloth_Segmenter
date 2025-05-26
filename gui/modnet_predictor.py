# models.py
import os
import sys
import torch

from collections import OrderedDict
import torch, re

modnet_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gui','models', 'MODNet', 'src'))
if modnet_path not in sys.path:
    sys.path.append(modnet_path)

from models.modnet import MODNet

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_modnet_weights(model, ckpt_path, device='cpu', strict=True):
    ckpt = torch.load(ckpt_path, map_location=device)

    # 1. unwrap Lightning-style checkpoints
    sd = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt

    # 2. strip 'module.' prefixes from DataParallel saves
    clean_sd = OrderedDict()
    for k, v in sd.items():
        clean_sd[re.sub(r'^module\.', '', k)] = v

    missing, unexpected = model.load_state_dict(clean_sd, strict=strict)
    print(f"✓ loaded with {len(missing)} missing / {len(unexpected)} unexpected keys")
    return model

_modnet = MODNet(backbone_pretrained=False).to(_device)
load_modnet_weights(
    _modnet,
    "models/MODNet/pretrained/modnet_photographic_portrait_matting.ckpt",
    device=_device
)
_modnet.eval()

def modnet_predict(image_bgr):
    """
    Parameters
    ----------
    image_bgr : np.ndarray  (H, W, 3) uint8, OpenCV BGR

    Returns
    -------
    matte : np.ndarray  (H, W) float32, 0‒1
    """
    import cv2, torch, torchvision.transforms as T, torch.nn.functional as F

    h, w = image_bgr.shape[:2]
    im_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    im_tensor = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])(im_rgb).unsqueeze(0).to(_device)

    # MODNet recommends half-resolution inference for speed
    ref_size = 512
    rh, rw = (h // 32) * 32, (w // 32) * 32
    im_tensor = torch.nn.functional.interpolate(im_tensor, size=(rh, rw),
                                                mode='area')

    with torch.no_grad():
        _, _, matte = _modnet(im_tensor, True)

    matte = torch.nn.functional.interpolate(matte, size=(h, w),
                                            mode='bilinear')
    return matte[0, 0].cpu().numpy()
