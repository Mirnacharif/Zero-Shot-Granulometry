import os
import matplotlib.pyplot as plt
import numpy as np
import csv
import cv2
import tifffile
from skimage.color import label2rgb
from skimage.segmentation import mark_boundaries
from skimage.measure import label

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)




def save_overlap_debug_csv(debug_info, debug_kept_pairs, out_dir, prefix="debug"):
    os.makedirs(out_dir, exist_ok=True)

    rows = [] 

    for j, d in enumerate(debug_info):
        rows.append({
            "debug_id": j,
            "kept_sorted_idx": d["kept_sorted_idx"],
            "removed_sorted_idx": d["removed_sorted_idx"],
            "overlap_ratio": d["overlap_ratio"],
            "intersection_pixels": d["intersection_pixels"],
            "kept_area": d["kept_area"],
            "removed_area": d["removed_area"],
            "kept_predicted_iou": d["kept_predicted_iou"],
            "removed_predicted_iou": d["removed_predicted_iou"]
        })

    csv_path = os.path.join(out_dir, f"{prefix}_removed_debug.csv")
    if len(rows) > 0:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    # ==========================
    kept_rows = []

    for j, d in enumerate(debug_kept_pairs):
        kept_rows.append({
            "debug_id": j,
            "mask1_sorted_idx": d["current_sorted_idx"],
            "mask2_sorted_idx": d["kept_sorted_idx"],
            "overlap_ratio": d["overlap_ratio"],
            "intersection_pixels": d["intersection_pixels"],
            "mask1_predicted_iou": d["current_predicted_iou"],
            "mask2_predicted_iou": d["kept_predicted_iou"],
        })

    kept_csv_path = os.path.join(out_dir, f"{prefix}_kept_pairs_debug.csv")

    if len(kept_rows) > 0:
        with open(kept_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=kept_rows[0].keys())
            writer.writeheader()
            writer.writerows(kept_rows)

    return rows, kept_rows

def show_removed_example(image_rgb, debug_info, j):
    d = debug_info[j]

    removed = d["removed_mask"]["segmentation"]
    kept = d["kept_mask"]["segmentation"]
    inter = removed & kept

    ratio = d["overlap_ratio"]

    plt.figure(figsize=(8, 8))
    plt.imshow(image_rgb)

    overlay = np.zeros((*removed.shape, 4), dtype=float)

    overlay[kept] = [0, 1, 0, 0.35]     
    overlay[removed] = [1, 0, 0, 0.35]  
    overlay[inter] = [1, 1, 0, 0.65]    

    plt.imshow(overlay)
    plt.title(
        f"Exemple suppression j={j}\n"
        f"kept idx={d['kept_sorted_idx']} | removed idx={d['removed_sorted_idx']}\n"
        f"overlap ratio={ratio:.3f}"
    )
    plt.axis("off")
    plt.show()

def show_removed_from_removed_info(image_rgb, masks_after_filter1, cleaned_masks, removed_info, j):
    idx = removed_info[j]["idx"]

    original = masks_after_filter1[idx]["segmentation"].astype(bool)
    cleaned = cleaned_masks[idx]["segmentation"].astype(bool)

    removed = original & (~cleaned)

    plt.figure(figsize=(8, 8))
    plt.imshow(image_rgb)

    overlay = np.zeros((*original.shape, 4), dtype=float)
    overlay[cleaned] = [0, 1, 0, 0.40]   
    overlay[removed] = [1, 0, 0, 0.60]   

    plt.imshow(overlay)
    plt.title(
        f"Mask idx={idx}\n"
        f"components={removed_info[j]['nb_components']} | "
        f"removed area={removed_info[j]['removed_area']}"
    )
    plt.axis("off")
    plt.show()


def mask_cmap(masks, cmap='BrBG'):
    values = np.unique(masks)
    colors = plt.cm.get_cmap(cmap, len(values))
    colors = [colors(i) for i in range(len(values))]
    np.random.shuffle(colors)
    return colors

def show_masks(
    image_path,
    label_path,
    save_path=None,
    alpha=0.5,
    bg_color="black",
    contour_color=(0, 0, 0), 
    contour_mode="thick",
    dpi=300,
    figsize=(2, 3)
):


    image_path = str(image_path)
    label_path = str(label_path)

    if save_path is not None:
        save_path = str(save_path)
        
    image_bgr = cv2.imread(image_path)
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    
    if label_path.endswith(".tif") or label_path.endswith(".tiff"):
        pred = tifffile.imread(label_path)

    elif label_path.endswith(".pkl"):
        with open(label_path, "rb") as f:
            masks = pickle.load(f)
        masks = sorted(masks, key=lambda m: m["area"], reverse=True)
        H, W = masks[0]["segmentation"].shape
        pred = np.zeros((H, W), dtype=np.int32)

        for i, m in enumerate(masks, start=1):
            pred[m["segmentation"]] = i

    else:
        raise ValueError("Unsupported file format")

    colors = mask_cmap(pred)

    msks = label2rgb(
        pred,
        image=img,
        colors=colors,
        bg_label=0,
        alpha=alpha,
        bg_color=bg_color
    )

    result = mark_boundaries(
        msks,
        label(pred),
        color=contour_color,
        mode=contour_mode
    )

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(result)
    ax.axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)

    plt.show()

