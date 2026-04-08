#
# Copyright (C) 2026, GS4City
# All rights reserved.

import os
import sys
import cv2
import torch
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
import json
from typing import Dict, Optional

import clip
from PIL import Image


# ===================== Config helpers =====================

def _load_full_config(script_dir: str) -> Dict:
    config_path = os.path.join(script_dir, "mask", "config.json")
    assert os.path.exists(config_path), f"config.json not found: {config_path}"
    with open(config_path, "r") as f:
        return json.load(f)


def _get_clip_cfg(full_config: Dict) -> Dict:
    cfg = full_config.get("clip", {})
    assert isinstance(cfg, dict), "`clip` in config.json must be a dict"
    return cfg


# ===================== CLIP helpers =====================

def load_clip_model(device, model_name: str):
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    return model, preprocess


def encode_texts(model, device, texts):
    with torch.no_grad():
        tokens = clip.tokenize(texts).to(device)
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats  # [N, D]


def classify_sam_instance_with_clip(
    model,
    preprocess,
    device,
    image_rgb,       # H x W x 3, uint8, RGB
    mask_bool_np,    # H x W, bool (numpy)
    text_feats_city,
    text_feats_fore,
    margin: float,
) -> str:
    """
    Use CLIP to classify a SAM instance as either building-like ("city") or foreground-like ("fore").
    If margin > 0, only classify as "fore" when foreground similarity is sufficiently higher than city similarity.
    """
    ys, xs = np.where(mask_bool_np)
    if ys.size == 0:
        return "city"

    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    # Add a small padding to avoid overly tight crops
    H, W = image_rgb.shape[:2]
    pad = 4
    y1 = max(0, y1 - pad)
    x1 = max(0, x1 - pad)
    y2 = min(H - 1, y2 + pad)
    x2 = min(W - 1, x2 + pad)

    crop = image_rgb[y1:y2 + 1, x1:x2 + 1, :]
    mask_crop = mask_bool_np[y1:y2 + 1, x1:x2 + 1]
    crop = crop.copy()
    crop[~mask_crop] = 255  # paint non-instance area to white

    pil_img = Image.fromarray(crop)

    with torch.no_grad():
        clip_in = preprocess(pil_img).unsqueeze(0).to(device)
        img_feat = model.encode_image(clip_in)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        sim_city = (img_feat @ text_feats_city.T).max().item()
        sim_fore = (img_feat @ text_feats_fore.T).max().item()

    if sim_fore > sim_city + margin:
        return "fore"
    else:
        return "city"


# ===================== Visualization helpers =====================

def get_n_different_colors(n: int, seed: int) -> np.ndarray:
    """Generate n random colors for instance mask visualization."""
    np.random.seed(int(seed))
    return np.random.randint(1, 256, (n, 3), dtype=np.uint8)


def visualize_mask(mask: np.ndarray, color_seed: int) -> np.ndarray:
    """
    Convert an instance label image (0=background, 1..N=instances) into a colorful RGB image for visualization.
    """
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    num_masks = int(mask.max())
    if num_masks == 0:
        return color_mask

    random_colors = get_n_different_colors(num_masks, seed=color_seed)
    for i in range(num_masks):
        color_mask[mask == (i + 1)] = random_colors[i]
    return color_mask


# ===================== SAM model construction =====================

def get_seg_model(
    config: Dict,
    seg_method: str,
    device: str,
):
    """
    Build the segmentation model based on config.json.
    All SAM-related parameters (encoder, checkpoint, points_per_side, etc.) are read from config.json only.
    """
    if seg_method == "sam":
        from segment_anything import sam_model_registry

        # Explicitly add the mask directory into sys.path for local imports under mask/
        script_dir = os.path.dirname(os.path.abspath(__file__))
        mask_dir = os.path.join(script_dir, "mask")
        if mask_dir not in sys.path:
            sys.path.append(mask_dir)
        from mask.automatic_mask_generator import SamAutomaticMaskGenerator

        sam = sam_model_registry[config["sam_encoder_version"]](
            checkpoint=config["sam_checkpoint_path"]
        ).to(device=device)

        pps = config.get("sam_num_points_per_side", 32)
        ppb = config.get("sam_num_points_per_batch", 64)
        piou = config.get("sam_pred_iou_threshold", 0.86)
        cnl = config.get("sam_crop_n_layers", 0)

        auto_sam = SamAutomaticMaskGenerator(
            sam,
            points_per_side=pps,
            points_per_batch=ppb,
            pred_iou_thresh=piou,
            crop_n_layers=cnl,
        )
        return auto_sam

    else:
        raise NotImplementedError("only sam seg_method is supported for now")


# ===================== SAM + CityGML + CLIP foreground filtering =====================

def get_sam_mask(
    auto_sam,
    image: np.ndarray,                  # RGB, numpy
    confidence_threshold: float,
    city_mask: Optional[np.ndarray] = None,
    overlap_clip_thresh: float = 0.8,
    clip_model=None,
    clip_preprocess=None,
    clip_device: str = "cuda",
    text_feats_city=None,
    text_feats_fore=None,
    clip_margin: float = 0.05,
    min_sam_pixels: int = 50,
    enable_clip: bool = True,
) -> np.ndarray:
    """
    Generate a foreground instance segmentation map as numpy.uint16:
    0 = background/building, 1..N = foreground instances.
    """
    H, W = image.shape[:2]

    with torch.no_grad():
        mask_data = auto_sam.generate(image)

    pred_masks = mask_data["masks"].float()   # [N, H, W]
    pred_scores = mask_data["iou_preds"]      # [N]

    pred_masks = pred_masks.to("cuda" if torch.cuda.is_available() else "cpu")
    pred_scores = pred_scores.to(pred_masks.device)
    device = pred_masks.device

    keep_idx = (pred_scores >= confidence_threshold)
    if keep_idx.sum().item() == 0:
        return np.zeros((H, W), dtype=np.uint16)

    pred_masks = pred_masks[keep_idx]
    pred_scores = pred_scores[keep_idx]

    masks_bool = (pred_masks > 0.5)

    city_bin_np: Optional[np.ndarray] = None
    if city_mask is not None:
        if city_mask.ndim == 3:
            city_bin_np = (city_mask[..., :3].sum(axis=2) > 0)
        else:
            city_bin_np = (city_mask > 0)
        city_bin_np = city_bin_np.astype(np.bool_)

    areas = masks_bool.sum(dim=(1, 2))
    area_keep = (areas >= min_sam_pixels)
    if area_keep.sum().item() == 0:
        return np.zeros((H, W), dtype=np.uint16)

    masks_bool = masks_bool[area_keep]
    pred_scores = pred_scores[area_keep]
    areas = areas[area_keep]
    K2 = masks_bool.shape[0]

    if city_bin_np is not None:
        city_bin_torch = torch.from_numpy(city_bin_np).to(device)
        overlaps = (masks_bool & city_bin_torch).sum(dim=(1, 2))
        overlap_ratio = overlaps.float() / areas.float().clamp_min(1)
    else:
        city_bin_torch = None
        overlaps = torch.zeros_like(areas, dtype=torch.long, device=device)
        overlap_ratio = torch.zeros_like(areas, dtype=torch.float32, device=device)

    use_clip = (
        enable_clip and
        (clip_model is not None) and
        (clip_preprocess is not None) and
        (text_feats_city is not None) and
        (text_feats_fore is not None)
    )

    if city_bin_torch is None:
        keep_direct_mask = torch.ones(K2, dtype=torch.bool, device=device)
        needs_clip_mask = torch.zeros(K2, dtype=torch.bool, device=device)
    else:
        keep_direct_mask = (overlap_ratio < overlap_clip_thresh)
        needs_clip_mask = (overlap_ratio >= overlap_clip_thresh)

    if not use_clip:
        needs_clip_mask = torch.zeros_like(needs_clip_mask)

    idx_keep_direct = torch.nonzero(keep_direct_mask, as_tuple=False).flatten()
    idx_needs_clip = torch.nonzero(needs_clip_mask, as_tuple=False).flatten()

    kept_masks = []
    kept_scores = []

    if idx_keep_direct.numel() > 0:
        for idx in idx_keep_direct.tolist():
            kept_masks.append(masks_bool[idx])
            kept_scores.append(pred_scores[idx])

    if use_clip and city_bin_torch is not None and idx_needs_clip.numel() > 0:
        image_rgb = image
        for idx in idx_needs_clip.tolist():
            m_tensor = masks_bool[idx]
            m_np_bool = m_tensor.detach().to("cpu").numpy().astype(bool)
            sem = classify_sam_instance_with_clip(
                clip_model,
                clip_preprocess,
                clip_device,
                image_rgb,
                m_np_bool,
                text_feats_city,
                text_feats_fore,
                margin=clip_margin,
            )
            if sem == "fore":
                kept_masks.append(m_tensor)
                kept_scores.append(pred_scores[idx])

    if len(kept_masks) == 0:
        return np.zeros((H, W), dtype=np.uint16)

    kept_masks = torch.stack(kept_masks, dim=0)
    kept_scores = torch.stack(kept_scores, dim=0)

    mask_id = torch.zeros((H, W), dtype=torch.int32, device=device)

    _, order = torch.sort(kept_scores, descending=False)

    for new_idx, idx in enumerate(order.tolist()):
        m_bool = kept_masks[idx]
        mask_id[m_bool] = int(new_idx + 1)

    output_mask_t = torch.zeros((H, W), dtype=torch.int32, device=device)
    unique_ids = mask_id.unique()

    cur_id = 1
    for tmp_id in unique_ids.tolist():
        if tmp_id == 0:
            continue
        cur_mask = (mask_id == tmp_id)
        kept_area = int(cur_mask.sum().item())
        if kept_area < min_sam_pixels:
            continue
        output_mask_t[cur_mask] = cur_id
        cur_id += 1

    output_mask = output_mask_t.to("cpu").numpy().astype(np.uint16)
    return output_mask


# ===================== Main pipeline =====================

def run_segmentation(
    scene: str,
    seg_method: str = "sam",
    visualize: bool = False,
    use_gml: bool = False,
    use_clip: bool = False,
    scale: Optional[float] = None,
) -> None:
    """
    Main pipeline:
    - Read data from <script_dir>/dataset/<scene>
    - Read images from images/
    - If use_gml=True, read building masks from gml_mask/ to help filter foreground
    - If use_clip=True, load CLIP and classify highly-overlapping instances
    - Build the segmentation model from mask/config.json and run segmentation
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    full_config = _load_full_config(script_dir)
    clip_cfg = _get_clip_cfg(full_config)

    assert seg_method in full_config, f"seg_method '{seg_method}' not found in config.json"
    config = full_config[seg_method]

    clip_model_name = clip_cfg["clip_model_name"]
    city_text_prompts = clip_cfg["city_text_prompts"]
    fore_text_prompts = clip_cfg["fore_text_prompts"]

    default_scale = float(clip_cfg["default_scale"])
    overlap_clip_thresh = float(clip_cfg["default_overlap_clip_thresh"])
    clip_margin = float(clip_cfg["default_clip_margin"])
    min_sam_pixels = int(clip_cfg["default_min_sam_pixels"])
    default_conf_threshold = float(clip_cfg["default_confidence_threshold"])
    color_seed = int(clip_cfg["color_seed"])

    if scale is None:
        scale = default_scale

    dataset_root = os.path.normpath(os.path.join(script_dir, "dataset"))

    scene_folder = os.path.join(dataset_root, scene)
    assert os.path.exists(scene_folder), f"scene folder not found: {scene_folder}"

    image_folder = os.path.join(scene_folder, "images")
    assert os.path.exists(image_folder), f"image folder not found: {image_folder}"

    output_folder = os.path.join(scene_folder, f"raw_{seg_method}_mask")
    os.makedirs(output_folder, exist_ok=True)

    vis_output_folder = None
    if visualize:
        vis_output_folder = os.path.join(scene_folder, f"raw_{seg_method}_mask_vis")
        os.makedirs(vis_output_folder, exist_ok=True)

    confidence_threshold = float(config.get("confidence_threshold", default_conf_threshold))

    print(f"[INFO] loading {seg_method} model ...")
    seg_model = get_seg_model(
        config=config,
        seg_method=seg_method,
        device=device,
    )

    if use_clip:
        clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] loading CLIP model '{clip_model_name}' on {clip_device} ...")
        clip_model, clip_preprocess = load_clip_model(clip_device, model_name=clip_model_name)

        text_feats_city = encode_texts(clip_model, clip_device, city_text_prompts)
        text_feats_fore = encode_texts(clip_model, clip_device, fore_text_prompts)
    else:
        print("[INFO] CLIP filtering disabled (use_clip = False).")
        clip_device = "cpu"
        clip_model = None
        clip_preprocess = None
        text_feats_city = None
        text_feats_fore = None

    citygml_dir = None
    if use_gml:
        citygml_dir = os.path.join(scene_folder, "gml_mask")
        assert os.path.exists(citygml_dir), f"gml_mask dir not found: {citygml_dir}"
        print(f"[INFO] use citygml dir: {citygml_dir}")
    else:
        print("[INFO] GML filtering disabled (use_gml = False).")

    image_names = sorted(os.listdir(image_folder))
    print(f"[INFO] found {len(image_names)} files in {image_folder}")

    for image_name in tqdm(image_names):
        if not image_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            continue

        img_path = os.path.join(image_folder, image_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] cannot read image: {img_path}, skip.")
            continue

        h, w = img.shape[:2]
        if scale != 1.0:
            img_small = cv2.resize(
                img,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            img_small = img

        img_small_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

        city_small = None
        if citygml_dir is not None:
            base = os.path.splitext(image_name)[0]

            city_path_npy = os.path.join(citygml_dir, base + ".npy")
            city_path_png = os.path.join(citygml_dir, base + ".png")

            city_arr = None
            if os.path.exists(city_path_npy):
                city_arr = np.load(city_path_npy)
            elif os.path.exists(city_path_png):
                city_arr = cv2.imread(city_path_png, cv2.IMREAD_UNCHANGED)

            if city_arr is not None:
                if scale != 1.0:
                    city_small = cv2.resize(
                        city_arr,
                        (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_NEAREST,
                    )
                else:
                    city_small = city_arr

        mask_small = get_sam_mask(
            auto_sam=seg_model,
            image=img_small_rgb,
            confidence_threshold=confidence_threshold,
            city_mask=city_small,
            overlap_clip_thresh=overlap_clip_thresh,
            clip_model=clip_model,
            clip_preprocess=clip_preprocess,
            clip_device=clip_device,
            text_feats_city=text_feats_city,
            text_feats_fore=text_feats_fore,
            clip_margin=clip_margin,
            min_sam_pixels=min_sam_pixels,
            enable_clip=use_clip,
        )

        if scale != 1.0:
            mask = cv2.resize(
                mask_small,
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.uint16)
        else:
            mask = mask_small.astype(np.uint16)

        base_name = os.path.splitext(image_name)[0]
        save_path_npy = os.path.join(output_folder, base_name + ".npy")
        np.save(save_path_npy, mask)

        if visualize and vis_output_folder is not None:
            vis_mask = visualize_mask(mask, color_seed=color_seed)
            vis_path = os.path.join(
                vis_output_folder,
                base_name + ".png",
            )
            cv2.imwrite(vis_path, vis_mask[:, :, ::-1])

    print("[INFO] segmentation finished.")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--scene",
        "-s",
        default="subset_building1_29_copy",
        type=str,
        help="scene name under dataset/, e.g. 'mipnerf360/room'",
    )

    parser.add_argument(
        "--seg_method",
        "-m",
        default="sam",
        type=str,
        help="segmentation method, currently only 'sam' is supported",
    )

    parser.add_argument(
        "--visualize",
        "-v",
        action="store_true",
        help="if set, also save colorful visualization of instance masks",
    )

    parser.add_argument(
        "--gml",
        "-g",
        action="store_true",
        help="if set, use dataset/<scene>/gml_mask masks to filter SAM results",
    )

    parser.add_argument(
        "--clip",
        "-c",
        action="store_true",
        help="if set, use CLIP to classify highly-overlapping SAM instances as foreground/building",
    )

    args = parser.parse_args()

    run_segmentation(
        scene=args.scene,
        seg_method=args.seg_method,
        visualize=args.visualize,
        use_gml=args.gml,
        use_clip=args.clip,
    )
