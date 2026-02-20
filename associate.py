#
# Copyright (C) 2026, CityGMLGaussian
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaga
# Gaga research group, https://github.com/weijielyu/Gaga
#

import os
import json
import cv2
import numpy as np
import torch
from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams, get_combined_args
from mask.projector import GaussianProjector

# Only keep such non-hyperparameter constants here
VIS_IMAGE_EXT = ".png"


def finalize_masks_and_features(projector: GaussianProjector):
    """
    Post-process the per-view matched masks written by GaussianProjector.build_mask_association
    into projector.associated_mask_folder (.npy), and finalize outputs:

    1. Filter global mask ids by number of views (>= projector.min_views);
    2. Optional: filter by 3D group center distance to cameras (> max_group_distance treated as background);
    3. Remap labels so that 1..N are contiguous;
    4. Overwrite the .npy masks with remapped labels;
    5. Generate colorful visualization PNGs using projector.visualize_mask_association(new_mask);
    6. If projector.global_clip_features exists, save remapped clip_features.npy:
       - shape [num_mask+1, D], where row i corresponds to mask id = i (row 0 is background zeros).
    """

    associated_mask_folder = projector.associated_mask_folder
    visualize = getattr(projector, "visualize", False)
    visualize_folder = getattr(projector, "visualize_folder", None) if visualize else None

    # min_views comes entirely from config (projector.min_views), not hard-coded in this file
    min_views = int(getattr(projector, "min_views", 1))

    # Only .npy masks are supported (skip clip_features)
    mask_files = sorted(
        f for f in os.listdir(associated_mask_folder)
        if f.lower().endswith(".npy") and f != "clip_features.npy"
    )
    if len(mask_files) == 0:
        print(f"[finalize] No .npy masks found in {associated_mask_folder}.")
        return

    # ====== First pass: count view occurrences and infer max_label (vectorized) ======
    view_count_dict = {}  # label -> number of views
    max_label = 0
    base_dtype = None

    for fname in mask_files:
        path = os.path.join(associated_mask_folder, fname)
        mask = np.load(path)  # H x W

        # Only process 2D HxW masks
        if mask.ndim != 2:
            print(f"[finalize] Skip non-2D npy file in first pass: {fname}, shape={mask.shape}")
            continue

        # Force integer type for bincount/indexing
        if not np.issubdtype(mask.dtype, np.integer):
            mask = mask.astype(np.int64)

        if base_dtype is None:
            base_dtype = mask.dtype

        if mask.size == 0:
            continue

        max_lab_local = int(mask.max())
        if max_lab_local == 0:
            continue
        max_label = max(max_label, max_lab_local)

        # bincount length = max_lab_local+1, index corresponds to label
        counts = np.bincount(mask.ravel(), minlength=max_lab_local + 1)
        labels = np.nonzero(counts)[0]
        labels = labels[labels > 0]  # remove background 0
        if labels.size == 0:
            continue

        # View-counting: if a label appears in this view, count +1 for that label
        for lab in labels.tolist():
            view_count_dict[lab] = view_count_dict.get(lab, 0) + 1

    if base_dtype is None or max_label == 0 or len(view_count_dict) == 0:
        print("[finalize] All masks are background or empty. Nothing to finalize.")
        return

    num_mask = int(max_label)

    # dict -> array, index = label
    view_counts = np.zeros(num_mask + 1, dtype=np.int32)
    for lab, cnt in view_count_dict.items():
        if lab <= num_mask:
            view_counts[lab] = cnt

    # ====== Additional: filter by distance to cameras (remove far background groups) ======
    # Threshold comes from projector.max_group_distance (<=0 means disabled)
    max_group_dist = float(getattr(projector, "max_group_distance", 0.0))

    # Default: keep all labels w.r.t. distance filtering
    dist_keep_mask = np.ones(num_mask + 1, dtype=bool)
    dist_keep_mask[0] = False  # background is excluded by definition

    if max_group_dist > 0.0 and getattr(projector, "gaussian_idx_bank", None):
        # projector.gaussian_idx_bank: list of tensors (one tensor per group) of gaussian indices
        # projector.gaussians_xyz: [N, 3] gaussian 3D positions
        xyz = projector.gaussians_xyz.detach().cpu()  # [N, 3] on CPU

        # Collect all train camera centers in world coordinates
        cam_centers = []
        for cam in projector.viewpoint_camera:
            # GraphDECO Camera provides camera_center (world coordinates)
            c = cam.camera_center
            if isinstance(c, torch.Tensor):
                c = c.detach().cpu().numpy()
            else:
                c = np.asarray(c, dtype=np.float32)
            cam_centers.append(c)
        if len(cam_centers) > 0:
            cam_centers = np.stack(cam_centers, axis=0)  # [M, 3]
        else:
            cam_centers = None

        group_dists = []

        for g in projector.gaussian_idx_bank:
            if (
                isinstance(g, torch.Tensor)
                and g.numel() > 0
                and cam_centers is not None
            ):
                # Group center = mean of its gaussian points
                pts = xyz[g.long().cpu()]           # [K, 3]
                center = pts.mean(dim=0).numpy()    # [3]

                # Distance to all camera centers, take the minimum
                dists_to_cams = np.linalg.norm(cam_centers - center[None, :], axis=1)
                dist = float(dists_to_cams.min())
            else:
                # No gaussians or no cameras -> treat as extremely far
                dist = float("inf")
            group_dists.append(dist)

        group_dists = np.asarray(group_dists, dtype=np.float32)  # length ~= num_groups

        # Align lengths safely: gaussian_idx_bank length may differ from num_mask
        if group_dists.shape[0] < num_mask:
            pad = np.full((num_mask - group_dists.shape[0],), np.inf, dtype=np.float32)
            group_dists = np.concatenate([group_dists, pad], axis=0)
        elif group_dists.shape[0] > num_mask:
            print(
                f"[finalize] Warning: len(gaussian_idx_bank)={group_dists.shape[0]} > num_mask={num_mask}, "
                f"truncating group_dists."
            )
            group_dists = group_dists[:num_mask]

        print(
            "[debug] group distance (to nearest camera) stats: "
            f"min={group_dists.min():.3f}, "
            f"max={group_dists.max():.3f}, "
            f"p50={np.percentile(group_dists, 50):.3f}, "
            f"p90={np.percentile(group_dists, 90):.3f}, "
            f"p99={np.percentile(group_dists, 99):.3f}"
        )

        # Distance filtering for labels 1..num_mask
        dist_keep_mask[1:] = (group_dists <= max_group_dist)

    # ====== Filter by view count: keep labels with view_counts >= min_views (ignore 0) ======
    keep_mask = (view_counts >= min_views)
    keep_mask[0] = False

    # Combine both filters: must satisfy view-count and distance constraints
    keep_mask = keep_mask & dist_keep_mask

    keep_labels = np.where(keep_mask[1:])[0] + 1  # 1-based
    original_num = num_mask
    new_num = int(keep_labels.size)

    if new_num == 0:
        print(f"[finalize] Warning: no masks meet min_views={min_views}.")
        return

    print(f"[finalize] Filtering by views: {original_num} -> {new_num} (min_views={min_views})")

    # remap: old label -> new label; 0 stays 0
    remap = np.zeros(num_mask + 1, dtype=np.int32)
    for new_label, old_label in enumerate(keep_labels.tolist(), start=1):
        remap[old_label] = new_label

    # Update projector.num_mask
    projector.num_mask = new_num

    # ====== If projector has global CLIP features, remap and save them ======
    if hasattr(projector, "global_clip_features") and projector.global_clip_features is not None:
        old_feats = projector.global_clip_features  # [old_num+1, D]
        if isinstance(old_feats, np.ndarray) and old_feats.ndim == 2:
            D = old_feats.shape[1]
            clip_feats_new = np.zeros((new_num + 1, D), dtype=old_feats.dtype)
            clip_feats_new[0] = 0.0
            for old_label in keep_labels.tolist():
                new_label = int(remap[old_label])
                clip_feats_new[new_label] = old_feats[old_label]
            out_path = os.path.join(associated_mask_folder, "clip_features.npy")
            np.save(out_path, clip_feats_new)
            print(f"[finalize] Saved remapped CLIP features to {out_path}")
        else:
            print("[finalize] projector.global_clip_features has unexpected shape, skip saving CLIP features.")
    else:
        print("[finalize] No global_clip_features found in projector, skip saving CLIP features.")

    # ====== Second pass: overwrite .npy masks and save visualizations ======
    for fname in mask_files:
        path = os.path.join(associated_mask_folder, fname)
        mask = np.load(path)

        if mask.ndim != 2:
            print(f"[finalize] Skip non-2D npy file in second pass: {fname}, shape={mask.shape}")
            continue

        if not np.issubdtype(mask.dtype, np.integer):
            mask = mask.astype(np.int64)

        new_mask = remap[mask]
        np.save(path, new_mask)

        if visualize and visualize_folder is not None:
            vis = projector.visualize_mask_association(new_mask)
            vis_path = os.path.join(
                visualize_folder,
                os.path.splitext(fname)[0] + VIS_IMAGE_EXT,
            )
            cv2.imwrite(vis_path, vis)

    # info.json: dump the effective projector configuration
    info = {
        "num_mask": int(projector.num_mask),
        "raw_mask_folder": projector.raw_mask_folder,
        "associated_mask_folder": projector.associated_mask_folder,
        "front_percentage": float(projector.front_percentage),
        "iou_threshold": float(projector.iou_threshold),
        "num_patch": int(projector.num_patches),
        "min_views": int(min_views),
        "use_clip_filter": bool(getattr(projector, "use_clip", False)),
        "clip_sim_threshold": float(getattr(projector, "clip_sim_threshold", 0.0)),
        "max_group_distance": float(getattr(projector, "max_group_distance", 0.0)),
    }
    info_path = os.path.join(associated_mask_folder, "info.json")
    json.dump(info, open(info_path, "w"))
    print(f"[finalize] Done. Final num_mask = {projector.num_mask}")


# ===================== main =====================

if __name__ == "__main__":
    parser = ArgumentParser()
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)

    # At runtime we only care about:
    #   --visualize : whether to save colorful visualizations
    #   --clip      : whether to enable CLIP-assisted matching (must be explicitly enabled)
    parser.add_argument("--visualize", "-v", action="store_true")
    parser.add_argument(
        "--clip",
        "-c",
        action="store_true",
        help="if set, use CLIP features inside GaussianProjector for matching",
    )

    # iteration is no longer needed; internally we always use -1
    args = get_combined_args(parser)

    # Extract dataset / pipeline parameters via Gaga's argument system
    dataset_params = model.extract(args)
    pipeline_params = pipeline.extract(args)

    # Only override non-core keys (visualize/use_clip); all other settings come from mask/config.json["projector"]
    params_override = {
        "visualize": args.visualize,
        "use_clip": args.clip,
    }

    with torch.no_grad():
        projector = GaussianProjector(
            dataset_params,
            pipeline_params,
            iteration=-1,
            params=params_override,
        )

        # Always rebuild cross-view association (overwrites .npy masks under the SAM mask folder)
        print(f"[main] Building association into {projector.associated_mask_folder} ...")
        projector.build_mask_association()

        # Then apply view-count filtering + visualization + save averaged CLIP features
        finalize_masks_and_features(projector)
