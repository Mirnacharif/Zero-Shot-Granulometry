import torch
from skimage.measure import label, regionprops
import numpy as np

def remove_overlapping_masks(masks, device="cuda", min_overlap_ratio=0.9):
    """
    Removes masks that overlap with an already retained mask,
    only if:

        intersection(A, B) / min(area(A), area(B)) > min_overlap_ratio

    Masks with the highest predicted_iou are given priority.

    Returns:
        kept_masks
        removed_masks
        debug_info
        debug_kept_pairs: a list of dictionaries containing, for each
        retained mask, the pairs with an overlap ratio below the threshold.
    """
    if len(masks) == 0:
        return [], [], [], []

    
    sorted_indices = sorted(
        range(len(masks)),
        key=lambda i: masks[i].get("predicted_iou", 0.0),
        reverse=True
    )
    sorted_masks = [masks[i] for i in sorted_indices]

    
    mask_tensor = torch.stack(
        [
            torch.from_numpy(m["segmentation"]).to(device=device)
            for m in sorted_masks
        ],
        dim=0
    )

    n, h, w = mask_tensor.shape
    print(f"Processing {n} masks with shape ({h}, {w})")

    kept_indices = []
    removed_indices = []
    debug_info = []
    debug_kept_pairs = [] 

    for i in range(n):
        current_mask = mask_tensor[i]
        current_area = sorted_masks[i]["area"] 

        overlapping_kept_idx = None

        
        for k in kept_indices: 
            kept_mask = mask_tensor[k]
            kept_area = sorted_masks[k]["area"]
            inter = current_mask & kept_mask
            inter_sum = int(inter.sum().item())
            min_area = min(current_area, kept_area)

            overlap_ratio = inter_sum / min_area

            if overlap_ratio > min_overlap_ratio:
                overlapping_kept_idx = k
                intersection_pixels = inter_sum
                break
            if overlap_ratio <= min_overlap_ratio and inter_sum > 0:
                debug_kept_pairs.append({
                    "current_sorted_idx": i, 
                    "kept_sorted_idx": k,
                    "current_mask": sorted_masks[i],
                    "kept_mask": sorted_masks[k],
                    "intersection_pixels": inter_sum,
                    "overlap_ratio": overlap_ratio,
                    "current_predicted_iou": sorted_masks[i].get("predicted_iou", None),
                    "kept_predicted_iou": sorted_masks[k].get("predicted_iou", None)
                })



       
        if overlapping_kept_idx is None:
            kept_indices.append(i)
        else:
            removed_indices.append(i)

            debug_info.append({
                "removed_sorted_idx": i,
                "kept_sorted_idx": overlapping_kept_idx,
                "removed_mask": sorted_masks[i],
                "kept_mask": sorted_masks[overlapping_kept_idx],
                "intersection_pixels": intersection_pixels, 
                "removed_area": sorted_masks[i]["area"], 
                "kept_area": sorted_masks[overlapping_kept_idx]["area"],
                "removed_predicted_iou": sorted_masks[i].get("predicted_iou", None),
                "kept_predicted_iou": sorted_masks[overlapping_kept_idx].get("predicted_iou", None),
                "overlap_ratio": overlap_ratio,
            })

    kept_masks = [sorted_masks[i] for i in kept_indices]
    removed_masks = [sorted_masks[i] for i in removed_indices]

    return kept_masks, removed_masks, debug_info, debug_kept_pairs



def keep_largest_component(masks_list):
    cleaned_masks = []
    removed_info = []

    for i, m in enumerate(masks_list):
        mask = m["segmentation"].astype(bool)

        lab = label(mask, connectivity=2) 
        props = regionprops(lab)

        if len(props) <= 1:
            cleaned_masks.append(m)
            continue

        
        largest = max(props, key=lambda r: r.area)
        largest_mask = lab == largest.label

        removed_area = int(mask.sum() - largest_mask.sum())

        removed_info.append({
            "idx": i,
            "nb_components": len(props),
            "removed_area": removed_area
        })

        m_clean = m.copy()
        m_clean["segmentation"] = largest_mask
        m_clean["area"] = int(largest_mask.sum())

        cleaned_masks.append(m_clean)

    return cleaned_masks, removed_info








def filter_sam_list_by_shape_components(masks_list,
                                        image_shape=None,
                                        min_area=0,
                                        max_area_frac=0.40,
                                        min_solidity=0.70,
                                        max_eccentricity=0.996,
                                        min_circularity=0.10):

    kept_masks = []
    removed_masks = []

    if image_shape is None:
        first = masks_list[0]
        image_shape = first["segmentation"].shape if isinstance(first, dict) else first.shape

    image_area = image_shape[0] * image_shape[1]

    for i, m in enumerate(masks_list):

        
        mask_bin = m["segmentation"].astype(bool)
        mask_lab = label(mask_bin)
        props = regionprops(mask_lab)

        if len(props) == 0:
            removed_masks.append((i, m, "empty"))
            continue

       
        props_sorted = sorted(props, key=lambda x: x.area, reverse=True)

       
        main_region = props_sorted[0]
        main_mask = mask_lab == main_region.label

        
        artifact_mask = np.zeros_like(mask_bin, dtype=bool)

        for r_art in props_sorted[1:]:
            artifact_mask[mask_lab == r_art.label] = True

        if artifact_mask.sum() > 0:
            removed_masks.append((i, artifact_mask, "artifact component"))

        area = main_region.area
        perimeter = main_region.perimeter

        if perimeter == 0:
            removed_masks.append((i, main_mask, "zero perimeter"))
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)

        reason = None

        if area < min_area:
            reason = "too small"
        elif area > max_area_frac * image_area:
            reason = "too large/background"
        elif main_region.solidity < min_solidity:
            reason = "low solidity"
        elif main_region.eccentricity > max_eccentricity:
            reason = "too elongated"
            print(f"Mask {i} removed: eccentricity {main_region.eccentricity} > max_eccentricity {max_eccentricity}")
        elif circularity < min_circularity:
            reason = "low circularity"

        if reason is None:
            if isinstance(m, dict):
                m_clean = m.copy()
                m_clean["segmentation"] = main_mask
                m_clean["area"] = int(main_mask.sum())
                kept_masks.append(m_clean)
            else:
                kept_masks.append(main_mask)
        else:
            removed_masks.append(
                (i, main_mask, reason, area,
                 main_region.solidity,
                 main_region.eccentricity,
                 circularity)
            )

    return kept_masks, removed_masks

def filter_by_shape_labelmask(mask, 
                    min_solidity=0.70,
                    max_eccentricity=0.996,
                    min_circularity=0.10):
    
    mask_lab = mask
    h, w = mask_lab.shape
    image_area = h * w

    out = np.zeros_like(mask_lab, dtype=np.int32)

    removed = []
    kept = []

    new_id = 1

    for r in regionprops(mask_lab):
        area = r.area
        perimeter = r.perimeter

        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)

        reason = None

        if r.solidity < min_solidity:
            reason = "low solidity"

        elif r.eccentricity > max_eccentricity:
            reason = "too elongated"

        elif circularity < min_circularity:
            reason = "low circularity"

        if reason is None:
            out[mask_lab == r.label] = new_id
            kept.append(r.label)
            new_id += 1
        else:
            removed.append((r.label, area, r.solidity, r.eccentricity, circularity, reason))

    return out, kept, removed
