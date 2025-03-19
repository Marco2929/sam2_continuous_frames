import os
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_camera_predictor


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_bbox(bbox, ax, marker_size=200):
    tl, br = bbox[0], bbox[1]
    w, h = (br - tl)[0], (br - tl)[1]
    x, y = tl[0], tl[1]
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=None, edgecolor="blue", linewidth=2))


def init_first_frame(predictor, image_path):
    image = cv2.imread(image_path)  # Load image
    if image is None:
        raise ValueError(f"Error: Unable to read image at {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    predictor.load_first_frame(image)

    ann_frame_idx = 0
    ann_obj_id = 1

    bbox = np.array([[450, 200], [700, 400]], dtype=np.float32)

    plt.figure(figsize=(12, 8))
    plt.title(f"Frame {ann_frame_idx}")
    plt.imshow(image)

    _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
        frame_idx=ann_frame_idx, obj_id=ann_obj_id, bbox=bbox
    )
    show_bbox(bbox, plt.gca())
    plt.show()


def find_segmentation(predictor, image_path):
    image = cv2.imread(image_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    out_obj_ids, out_mask_logits = predictor.track(image)

    # Ensure logits is a NumPy array
    mask_logits = out_mask_logits[0].cpu().numpy()

    # Check if all values in mask_logits are -1024
    if np.all(mask_logits == -1024):
        object_visible = False
        print("No object detected in the image.")
    else:
        object_visible = True
        print("Object detected in the image.")

    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    show_mask(
        (out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0]
    )
    plt.show()

    return object_visible


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent

    sam2_checkpoint = os.path.join(base_dir, r"checkpoints\sam2.1_hiera_large.pt")
    model_cfg = os.path.join(base_dir, r"checkpoints\sam2.1_hiera_l.yaml")

    predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

    init_first_frame(predictor, os.path.join(base_dir, "data/init_image.jpg"))
    find_segmentation(predictor, os.path.join(base_dir, "data/test_image.jpg"))
    find_segmentation(predictor, os.path.join(base_dir, "data/test_image2.jpg"))
