import numpy as np
from segment_anything import (
    SamAutomaticMaskGenerator,
    sam_model_registry,
)

def load_sam_model(checkpoint, model_type="vit_h", device="cuda"):
    """Load a SAM model from a local checkpoint."""
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device=device)
    sam.eval()
    return sam

def build_mask_generator(sam_model, points_per_side, box_nms_thresh, crop_n_layers, crop_nms_thresh):
    mask_generator = SamAutomaticMaskGenerator(
        model=sam_model,
        points_per_side=points_per_side,
        box_nms_thresh=box_nms_thresh,
        crop_n_layers=crop_n_layers,
        crop_nms_thresh=crop_nms_thresh,
        output_mode="binary_mask",
    )
    return mask_generator

def sam_masks_to_label_image(masks, image_shape):
    h, w = image_shape[:2]
    label_img = np.zeros((h, w), dtype=np.int32)

    masks_sorted = sorted(masks, key=lambda x: x["area"], reverse=True)

    current_label = 1
    occupied = np.zeros((h, w), dtype=bool)

    for m in masks_sorted:
        seg = m["segmentation"].astype(bool)
        seg_new = seg & (~occupied)

        if seg_new.sum() == 0:
            continue

        label_img[seg_new] = current_label
        occupied[seg_new] = True
        current_label += 1

    return label_img


def sam_masks_to_label_image_by_iou(masks, image_shape):
    h, w = image_shape[:2]
    label_img = np.zeros((h, w), dtype=np.int32)
    masks_sorted = sorted(masks, key=lambda x: x["predicted_iou"], reverse=True)

    current_label = 1
    occupied = np.zeros((h, w), dtype=bool)

    for m in masks_sorted:
        seg = m["segmentation"].astype(bool)
        seg_new = seg & (~occupied)

        if seg_new.sum() == 0:
            continue

        label_img[seg_new] = current_label
        occupied[seg_new] = True
        current_label += 1

    return label_img