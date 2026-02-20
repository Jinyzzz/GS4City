#
# Copyright (C) 2026, CityGMLGaussian
# All rights reserved.
#

import os
from argparse import ArgumentParser
from typing import Optional, Tuple, Dict
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import clip


def id_to_color(idx: int) -> Tuple[int, int, int]:
    """
    Given an instance id, generate a deterministic RGB color.
    This ensures the same id has the same color across different views.
    """
    h = (idx * 2654435761) & 0xFFFFFFFF
    r = (h & 0xFF)
    g = ((h >> 8) & 0xFF)
    b = ((h >> 16) & 0xFF)
    if r == 0 and g == 0 and b == 0:
        r = 50
    return int(r), int(g), int(b)


def load_mask_auto(base_path_no_ext: str) -> Optional[np.ndarray]:
    """
    Automatically load a mask from base_path_no_ext.* supporting:
      - .npy
      - .png / .jpg / .jpeg / .bmp / .tif / .tiff

    Returns an HxW integer mask (0=background, >0=instance id), or None if not found.
    """
    npy_path = base_path_no_ext + ".npy"
    if os.path.exists(npy_path):
        m = np.load(npy_path)
        if m is not None:
            if m.ndim == 3:
                m = m[..., 0]
            if not np.issubdtype(m.dtype, np.integer):
                m = m.astype(np.int64)
            return m

    exts = [
        ".png", ".PNG",
        ".jpg", ".JPG",
        ".jpeg", ".JPEG",
        ".bmp", ".BMP",
        ".tif", ".tiff", ".TIF", ".TIFF",
    ]
    for ext in exts:
        p = base_path_no_ext + ext
        if os.path.exists(p):
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if img.ndim == 3:
                img = img[..., 0]
            if not np.issubdtype(img.dtype, np.integer):
                img = img.astype(np.int64)
            return img

    return None


def load_rgb_image(scene_folder: str, stem: str) -> Optional[np.ndarray]:
    """
    Load the corresponding RGB image from <scene>/images (returned as BGR in OpenCV format).
    """
    img_dir = os.path.join(scene_folder, "images")
    exts = [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG", ".bmp", ".BMP"]
    for ext in exts:
        p = os.path.join(img_dir, stem + ext)
        if os.path.exists(p):
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is not None:
                return img
    return None


def get_instance_bbox_from_mask(mask_np: np.ndarray, instance_id: int) -> Optional[Tuple[int, int, int, int]]:
    """
    Compute bbox for a given instance id from a 2D integer mask.
    Returns (y_min, y_max, x_min, x_max) or None if not present.
    """
    ys, xs = np.where(mask_np == instance_id)
    if ys.size == 0 or xs.size == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def compute_clip_feature_for_instance(
    clip_model,
    clip_preprocess,
    device: torch.device,
    rgb_bgr: np.ndarray,
    mask_int: np.ndarray,
    instance_id: int,
    bg_value: int = 127,
    pad: int = 4,
) -> Optional[torch.Tensor]:
    """
    Extract CLIP image embedding for a specific instance id from an RGB image and instance mask.
    Uses bbox crop and fills non-instance pixels with a constant background value.
    Returns a L2-normalized torch tensor on CPU with shape [D], or None.
    """
    if rgb_bgr is None or mask_int is None:
        return None
    if mask_int.ndim != 2:
        raise ValueError(f"mask_int must be 2D, got shape {mask_int.shape}")

    bbox = get_instance_bbox_from_mask(mask_int, instance_id)
    if bbox is None:
        return None

    y1, y2, x1, x2 = bbox
    H, W = mask_int.shape

    y1 = max(0, y1 - pad)
    x1 = max(0, x1 - pad)
    y2 = min(H - 1, y2 + pad)
    x2 = min(W - 1, x2 + pad)

    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    crop = rgb[y1:y2 + 1, x1:x2 + 1, :].copy()
    m_crop = (mask_int[y1:y2 + 1, x1:x2 + 1] == instance_id)

    if crop.size == 0 or m_crop.size == 0 or (not m_crop.any()):
        return None

    bg = np.array([bg_value, bg_value, bg_value], dtype=crop.dtype)
    crop[~m_crop] = bg

    pil_img = Image.fromarray(crop)

    with torch.no_grad():
        inp = clip_preprocess(pil_img).unsqueeze(0).to(device)
        feat = clip_model.encode_image(inp).squeeze(0)  # [D]
        feat = feat / feat.norm(dim=-1, keepdim=True)

    return feat.detach().cpu()


def fuse_masks_for_scene(
    scene_folder: str,
    sam_id_offset: int = 10000,
    write_vis: bool = True,
) -> Tuple[str, str]:
    """
    Fuse <scene>/gml_mask and <scene>/sam_mask into <scene>/fused_mask.
    Returns (fused_mask_dir, fused_vis_dir).
    """
    gml_dir = os.path.join(scene_folder, "gml_mask")
    sam_dir = os.path.join(scene_folder, "sam_mask")
    out_gray_dir = os.path.join(scene_folder, "fused_mask")
    out_vis_dir = os.path.join(scene_folder, "fused_mask_vis")

    if not os.path.exists(gml_dir):
        raise FileNotFoundError(f"gml_mask folder not found: {gml_dir}")
    if not os.path.exists(sam_dir):
        raise FileNotFoundError(f"sam_mask folder not found: {sam_dir}")

    os.makedirs(out_gray_dir, exist_ok=True)
    if write_vis:
        os.makedirs(out_vis_dir, exist_ok=True)

    sam_files = sorted(f for f in os.listdir(sam_dir) if f.lower().endswith(".npy"))
    print(f"[INFO] Found {len(sam_files)} SAM mask (.npy) files in {sam_dir}")

    for name in tqdm(sam_files, desc="Fusing masks"):
        stem, _ = os.path.splitext(name)

        sam_path = os.path.join(sam_dir, name)
        sam = np.load(sam_path)
        if sam is None:
            print(f"[WARN] cannot read SAM mask: {sam_path}, skip.")
            continue

        if sam.ndim == 3:
            sam = sam[..., 0]
        if not np.issubdtype(sam.dtype, np.integer):
            sam = sam.astype(np.int64)
        else:
            sam = sam.astype(np.int64)

        gml_base = os.path.join(gml_dir, stem)
        gml = load_mask_auto(gml_base)
        if gml is None:
            H_sam, W_sam = sam.shape[:2]
            gml = np.zeros((H_sam, W_sam), dtype=np.int64)

        if gml.ndim == 3:
            gml = gml[..., 0]
        if not np.issubdtype(gml.dtype, np.integer):
            gml = gml.astype(np.int64)
        else:
            gml = gml.astype(np.int64)

        h_gml, w_gml = gml.shape[:2]
        h_sam, w_sam = sam.shape[:2]
        target_h = min(h_gml, h_sam)
        target_w = min(w_gml, w_sam)
        target_size = (target_w, target_h)

        if (h_gml, w_gml) != (target_h, target_w):
            gml = cv2.resize(gml.astype(np.int32), target_size, interpolation=cv2.INTER_NEAREST).astype(np.int64)
        if (h_sam, w_sam) != (target_h, target_w):
            sam = cv2.resize(sam.astype(np.int32), target_size, interpolation=cv2.INTER_NEAREST).astype(np.int64)

        H, W = target_h, target_w

        sam_nonzero = (sam != 0)
        if sam_nonzero.any():
            sam[sam_nonzero] = sam[sam_nonzero] + int(sam_id_offset)

        fused = np.zeros((H, W), dtype=np.uint16)
        fused[sam != 0] = sam[sam != 0].astype(np.uint16)
        fused[(sam == 0) & (gml != 0)] = gml[(sam == 0) & (gml != 0)].astype(np.uint16)

        out_gray_path = os.path.join(out_gray_dir, stem + ".npy")
        np.save(out_gray_path, fused)

        if write_vis:
            vis = np.zeros((H, W, 3), dtype=np.uint8)
            unique_ids = np.unique(fused)
            unique_ids = unique_ids[unique_ids != 0]
            for uid in unique_ids:
                r, g, b = id_to_color(int(uid))
                vis[fused == uid] = (b, g, r)  # BGR
            out_vis_path = os.path.join(out_vis_dir, stem + ".png")
            cv2.imwrite(out_vis_path, vis)

    return out_gray_dir, out_vis_dir


def compute_clip_index_from_fused_masks(
    scene_folder: str,
    fused_mask_dir: str,
    output_npz_path: str,
    model_name: str = "ViT-B/16",
    device_str: Optional[str] = None,
    min_pixels: int = 10,
    skip_ids: Tuple[int, ...] = (0,),
    bg_value: int = 127,
) -> None:
    """
    Compute per-object CLIP features using fused masks across views, then save as .npz.
    """
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading CLIP model: {model_name}")
    clip_model, clip_preprocess = clip.load(model_name, device=device)
    clip_model.eval()

    fused_files = sorted(f for f in os.listdir(fused_mask_dir) if f.lower().endswith(".npy"))
    if not fused_files:
        raise RuntimeError(f"No fused mask .npy files found in {fused_mask_dir}")

    id_to_feat_sum: Dict[int, torch.Tensor] = {}
    id_to_count = defaultdict(int)

    for fname in tqdm(fused_files, desc="Computing CLIP features"):
        stem, _ = os.path.splitext(fname)
        mask_path = os.path.join(fused_mask_dir, fname)

        fused = np.load(mask_path)
        if fused is None:
            continue
        if fused.ndim == 3:
            fused = fused[..., 0]
        if not np.issubdtype(fused.dtype, np.integer):
            fused = fused.astype(np.int64)
        else:
            fused = fused.astype(np.int64)

        rgb_bgr = load_rgb_image(scene_folder, stem)
        if rgb_bgr is None:
            print(f"[WARN] Missing RGB image for {stem}, skipping feature extraction for this view.")
            continue

        rgb_h, rgb_w = rgb_bgr.shape[:2]
        if fused.shape[0] != rgb_h or fused.shape[1] != rgb_w:
            fused = cv2.resize(fused.astype(np.int32), (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST).astype(np.int64)

        unique_ids = np.unique(fused)
        unique_ids = unique_ids[unique_ids != 0]

        for inst_id in unique_ids.tolist():
            inst_id_int = int(inst_id)
            if inst_id_int in skip_ids:
                continue

            inst_mask = (fused == inst_id_int)
            if int(inst_mask.sum()) < int(min_pixels):
                continue

            feat = compute_clip_feature_for_instance(
                clip_model=clip_model,
                clip_preprocess=clip_preprocess,
                device=device,
                rgb_bgr=rgb_bgr,
                mask_int=fused,
                instance_id=inst_id_int,
                bg_value=bg_value,
                pad=4,
            )
            if feat is None:
                continue

            if inst_id_int not in id_to_feat_sum:
                id_to_feat_sum[inst_id_int] = feat.clone()
            else:
                id_to_feat_sum[inst_id_int] += feat
            id_to_count[inst_id_int] += 1

    all_ids = sorted(id_to_feat_sum.keys())
    if not all_ids:
        raise RuntimeError(
            "No instance features were computed. Check images, masks, min_pixels, and skip_ids."
        )

    features = []
    counts = []
    for inst_id in all_ids:
        sum_feat = id_to_feat_sum[inst_id]
        cnt = int(id_to_count[inst_id])
        avg_feat = sum_feat / float(max(cnt, 1))
        avg_feat = avg_feat / (avg_feat.norm(dim=-1, keepdim=True) + 1e-12)
        features.append(avg_feat.numpy().astype(np.float32))
        counts.append(cnt)

    ids_np = np.array(all_ids, dtype=np.int64)
    feats_np = np.stack(features, axis=0).astype(np.float32)
    counts_np = np.array(counts, dtype=np.int32)

    out_dir = os.path.dirname(os.path.abspath(output_npz_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez(
        output_npz_path,
        ids=ids_np,
        features=feats_np,
        counts=counts_np,
        model_name=model_name,
    )

    print(f"[INFO] Saved {len(ids_np)} object features to {output_npz_path}")
    print("[INFO] You can query by object id using this .npz file.")


def main():
    parser = ArgumentParser(
        description="Fuse (SAM + CityGML) masks and compute per-object CLIP features from fused masks."
    )
    parser.add_argument(
        "--scene",
        "-s",
        type=str,
        required=True,
        help="Scene name under dataset/, e.g. 'subset_building1_16'",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Dataset root. If not set, uses <script_dir>/dataset.",
    )
    parser.add_argument(
        "--sam-id-offset",
        type=int,
        default=10000,
        help="Fixed offset added to SAM ids to avoid conflicts with CityGML ids.",
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Disable visualization output (fused_mask_vis).",
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default="ViT-B/16",
        help="CLIP model name (e.g., 'ViT-B/16', 'ViT-B/32').",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device string for PyTorch/CLIP (e.g., 'cuda' or 'cpu'). If None, auto-detect.",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=10,
        help="Minimum number of pixels per instance to be considered for CLIP feature extraction.",
    )
    parser.add_argument(
        "--skip-id",
        type=int,
        nargs="*",
        default=[0],
        help="Instance ids to skip (e.g., background 0).",
    )
    parser.add_argument(
        "--bg-value",
        type=int,
        default=127,
        help="Background value (0-255) used to fill non-instance pixels in masked crops.",
    )
    parser.add_argument(
        "--output-npz",
        type=str,
        default=None,
        help="Output .npz path for CLIP index. If not set, saves to <scene>/object_clip_index.npz.",
    )

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_root = args.dataset_root if args.dataset_root is not None else os.path.join(script_dir, "dataset")
    scene_folder = os.path.join(dataset_root, args.scene)

    if not os.path.exists(scene_folder):
        raise FileNotFoundError(f"scene folder not found: {scene_folder}")

    write_vis = (not args.no_vis)

    print(f"[INFO] scene folder: {scene_folder}")
    print(f"[INFO] SAM id offset: {args.sam_id_offset}")
    print(f"[INFO] write visualization: {write_vis}")

    fused_mask_dir, _ = fuse_masks_for_scene(
        scene_folder=scene_folder,
        sam_id_offset=int(args.sam_id_offset),
        write_vis=write_vis,
    )

    if args.output_npz is not None:
        output_npz_path = args.output_npz
    else:
        output_npz_path = os.path.join(scene_folder, "object_clip_index.npz")

    compute_clip_index_from_fused_masks(
        scene_folder=scene_folder,
        fused_mask_dir=fused_mask_dir,
        output_npz_path=output_npz_path,
        model_name=args.clip_model,
        device_str=args.device,
        min_pixels=int(args.min_pixels),
        skip_ids=tuple(int(x) for x in args.skip_id),
        bg_value=int(args.bg_value),
    )


if __name__ == "__main__":
    main()
