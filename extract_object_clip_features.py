#
# Copyright (C) 2026, GS4City
# All rights reserved.
#

import os
from argparse import ArgumentParser
from collections import defaultdict
from typing import Optional, Tuple, Dict

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import open_clip


CITY_OPENCLIP_MODEL_NAME = "ViT-B-16"
CITY_OPENCLIP_PRETRAINED = "laion2b_s34b_b88k"


def load_rgb_image(scene_folder: str, stem: str) -> Optional[np.ndarray]:
    """
    Load the corresponding RGB image from <scene>/images.
    Returns a BGR image in OpenCV format.
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
    Compute the bounding box for a given instance id from a 2D integer mask.
    Returns (y_min, y_max, x_min, x_max), or None if not present.
    """
    ys, xs = np.where(mask_np == instance_id)
    if ys.size == 0 or xs.size == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def create_openclip_model_and_preprocess(
    model_name: str,
    pretrained: str,
    device: torch.device,
):
    """
    Create OpenCLIP model and preprocessing transform.
    """
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
        device=device,
    )
    model.eval()
    return model, preprocess


def compute_openclip_feature_for_instance(
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
    Extract an OpenCLIP image embedding for a specific instance id from an RGB image and instance mask.
    Uses bounding-box crop and fills non-instance pixels with a constant background value.
    Returns an L2-normalized torch tensor on CPU with shape [D], or None.
    """
    if rgb_bgr is None or mask_int is None:
        return None

    if mask_int.ndim != 2:
        raise ValueError(f"mask_int must be 2D, got shape {mask_int.shape}")

    bbox = get_instance_bbox_from_mask(mask_int, instance_id)
    if bbox is None:
        return None

    y1, y2, x1, x2 = bbox
    h, w = mask_int.shape

    y1 = max(0, y1 - pad)
    x1 = max(0, x1 - pad)
    y2 = min(h - 1, y2 + pad)
    x2 = min(w - 1, x2 + pad)

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
        feat = clip_model.encode_image(inp).squeeze(0)
        feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-12)

    return feat.detach().cpu()


def compute_clip_index_from_fused_masks(
    scene_folder: str,
    fused_mask_dir: str,
    output_npz_path: str,
    model_name: str = CITY_OPENCLIP_MODEL_NAME,
    pretrained: str = CITY_OPENCLIP_PRETRAINED,
    device_str: Optional[str] = None,
    min_pixels: int = 10,
    skip_ids: Tuple[int, ...] = (0,),
    bg_value: int = 127,
) -> None:
    """
    Compute per-object OpenCLIP features using fused masks across views, then save as .npz.
    """
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading OpenCLIP model: {model_name}")
    print(f"[INFO] Using pretrained weights: {pretrained}")

    clip_model, clip_preprocess = create_openclip_model_and_preprocess(
        model_name=model_name,
        pretrained=pretrained,
        device=device,
    )

    fused_files = sorted(f for f in os.listdir(fused_mask_dir) if f.lower().endswith(".npy"))
    if not fused_files:
        raise RuntimeError(f"No fused mask .npy files found in {fused_mask_dir}")

    id_to_feat_sum: Dict[int, torch.Tensor] = {}
    id_to_count = defaultdict(int)

    for fname in tqdm(fused_files, desc="Computing OpenCLIP features"):
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
            print(f"[WARN] Missing RGB image for {stem}, skipping this view.")
            continue

        rgb_h, rgb_w = rgb_bgr.shape[:2]
        if fused.shape[0] != rgb_h or fused.shape[1] != rgb_w:
            fused = cv2.resize(
                fused.astype(np.int32),
                (rgb_w, rgb_h),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.int64)

        unique_ids = np.unique(fused)
        unique_ids = unique_ids[unique_ids != 0]

        for inst_id in unique_ids.tolist():
            inst_id_int = int(inst_id)
            if inst_id_int in skip_ids:
                continue

            inst_mask = (fused == inst_id_int)
            if int(inst_mask.sum()) < int(min_pixels):
                continue

            feat = compute_openclip_feature_for_instance(
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
            "No instance features were computed. Check images, fused masks, min_pixels, and skip_ids."
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
        pretrained=pretrained,
    )

    print(f"[INFO] Saved {len(ids_np)} object features to {output_npz_path}")
    print("[INFO] You can query by object id using this .npz file.")


def resolve_scene_folder(scene: Optional[str], dataset_root: Optional[str], scene_folder: Optional[str]) -> str:
    """
    Resolve scene folder from either:
      1) explicit --scene-folder
      2) --dataset-root + --scene
    """
    if scene_folder is not None:
        return os.path.abspath(scene_folder)

    if dataset_root is None or scene is None:
        raise ValueError("Either --scene-folder or both --dataset-root and --scene must be provided.")

    return os.path.abspath(os.path.join(dataset_root, scene))


def main():
    parser = ArgumentParser(
        description="Compute per-object OpenCLIP features from <scene>/images and <scene>/fused_mask."
    )

    parser.add_argument(
        "--scene",
        "-s",
        type=str,
        default=None,
        help="Scene name under dataset root, e.g. 'subset_building1_16'.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Dataset root path. Used together with --scene.",
    )
    parser.add_argument(
        "--scene-folder",
        type=str,
        default=None,
        help="Direct path to the scene folder containing images/ and fused_mask/.",
    )
    parser.add_argument(
        "--fused-mask-dir",
        type=str,
        default=None,
        help="Optional explicit fused mask directory. If not set, uses <scene>/fused_mask.",
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default=CITY_OPENCLIP_MODEL_NAME,
        help="OpenCLIP model name.",
    )
    parser.add_argument(
        "--clip-pretrained",
        type=str,
        default=CITY_OPENCLIP_PRETRAINED,
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
        help="Output .npz path. If not set, saves to <scene>/object_clip_index.npz.",
    )

    args = parser.parse_args()

    scene_folder = resolve_scene_folder(
        scene=args.scene,
        dataset_root=args.dataset_root,
        scene_folder=args.scene_folder,
    )

    if not os.path.exists(scene_folder):
        raise FileNotFoundError(f"Scene folder not found: {scene_folder}")

    if args.fused_mask_dir is not None:
        fused_mask_dir = os.path.abspath(args.fused_mask_dir)
    else:
        fused_mask_dir = os.path.join(scene_folder, "fused_mask")

    if not os.path.exists(fused_mask_dir):
        raise FileNotFoundError(f"Fused mask folder not found: {fused_mask_dir}")

    images_dir = os.path.join(scene_folder, "images")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images folder not found: {images_dir}")

    if args.output_npz is not None:
        output_npz_path = os.path.abspath(args.output_npz)
    else:
        output_npz_path = os.path.join(scene_folder, "object_clip_index.npz")

    print(f"[INFO] Scene folder: {scene_folder}")
    print(f"[INFO] Images folder: {images_dir}")
    print(f"[INFO] Fused mask folder: {fused_mask_dir}")
    print(f"[INFO] Output npz: {output_npz_path}")

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