import torch

def coverage_union(masks, H, W, device):

    union_mask = torch.zeros((H, W), dtype=torch.bool, device=device)

    for m in masks:
        mask = torch.from_numpy(m["segmentation"]).to(device=device, dtype=torch.bool)
        union_mask |= mask  
    covered_pixels = union_mask.sum().item()
    image_area = H * W

    return (covered_pixels / image_area) * 100

def recouvrement(masks, H, W, device):

    count_mask = torch.zeros((H, W), dtype=torch.int32, device=device)

    for m in masks:
        mask = torch.from_numpy(m["segmentation"]).to(device=device, dtype=torch.bool)
        count_mask += mask  

    pixels_union = (count_mask >= 1).sum().item()
    pixels_overlap = (count_mask >= 2).sum().item()

    if pixels_union == 0:
        return 0.0
    return (pixels_overlap / pixels_union) * 100

def coverage_labelmask(label_mask, device):
    label_t = torch.as_tensor(label_mask, device=device)

    image_area = label_t.numel() 
    covered_pixels = torch.sum(label_t > 0).item()

    return (covered_pixels / image_area) * 100