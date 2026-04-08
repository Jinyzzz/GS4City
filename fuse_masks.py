#
# Copyright (C) 2026, GS4City
# All rights reserved.
#

import os
from argparse import ArgumentParser
from typing import Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from extract_object_clip_features import compute_clip_index_from_fused_masks


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
            print(f"[WARN] Cannot read SAM mask: {sam_path}, skip.")
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
            h_sam, w_sam = sam.shape[:2]
            gml = np.zeros((h_sam, w_sam), dtype=np.int64)

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
            gml = cv2.resize(
                gml.astype(np.int32),
                target_size,
                interpolation=cv2.INTER_NEAREST
            ).astype(np.int64)

        if (h_sam, w_sam) != (target_h, target_w):
            sam = cv2.resize(
                sam.astype(np.int32),
                target_size,
                interpolation=cv2.INTER_NEAREST
            ).astype(np.int64)

        h, w = target_h, target_w

        sam_nonzero = (sam != 0)
        if sam_nonzero.any():
            sam[sam_nonzero] = sam[sam_nonzero] + int(sam_id_offset)

        fused = np.zeros((h, w), dtype=np.uint16)
        fused[sam != 0] = sam[sam != 0].astype(np.uint16)
        fused[(sam == 0) & (gml != 0)] = gml[(sam == 0) & (gml != 0)].astype(np.uint16)

        out_gray_path = os.path.join(out_gray_dir, stem + ".npy")
        np.save(out_gray_path, fused)

        if write_vis:
            vis = np.zeros((h, w, 3), dtype=np.uint8)
            unique_ids = np.unique(fused)
            unique_ids = unique_ids[unique_ids != 0]
            for uid in unique_ids:
                r, g, b = id_to_color(int(uid))
                vis[fused == uid] = (b, g, r)  # BGR for OpenCV
            out_vis_path = os.path.join(out_vis_dir, stem + ".png")
            cv2.imwrite(out_vis_path, vis)

    return out_gray_dir, out_vis_dir


def main():
    parser = ArgumentParser(
        description="Fuse (SAM + CityGML) masks and compute per-object OpenCLIP features from fused masks."
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
        default="ViT-B-16",
        help="OpenCLIP model name, e.g. 'ViT-B-16'.",
    )
    parser.add_argument(
        "--clip-pretrained",
        type=str,
        default="laion2b_s34b_b88k",
        help="OpenCLIP pretrained checkpoint name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device string for PyTorch/OpenCLIP (e.g. 'cuda' or 'cpu'). If None, auto-detect.",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=10,
        help="Minimum number of pixels per instance to be considered for feature extraction.",
    )
    parser.add_argument(
        "--skip-id",
        type=int,
        nargs="*",
        default=[0],
        help="Instance ids to skip (e.g. background 0).",
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
        raise FileNotFoundError(f"Scene folder not found: {scene_folder}")

    write_vis = (not args.no_vis)

    print(f"[INFO] Scene folder: {scene_folder}")
    print(f"[INFO] SAM id offset: {args.sam_id_offset}")
    print(f"[INFO] Write visualization: {write_vis}")

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
        pretrained=args.clip_pretrained,
        device_str=args.device,
        min_pixels=int(args.min_pixels),
        skip_ids=tuple(int(x) for x in args.skip_id),
        bg_value=int(args.bg_value),
    )


if __name__ == "__main__":
    main()