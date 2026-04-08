#
# Copyright (C) 2026, GS4City
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

VIS_IMAGE_EXT = ".png"


def finalize_masks_and_features(projector: GaussianProjector, skip_min_views: bool = False):
    associated_mask_folder = projector.associated_mask_folder
    visualize = getattr(projector, "visualize", False)
    visualize_folder = getattr(projector, "visualize_folder", None) if visualize else None

    min_views = int(getattr(projector, "min_views", 1))

    mask_files = sorted(
        f for f in os.listdir(associated_mask_folder)
        if f.lower().endswith(".npy") and f != "clip_features.npy"
    )
    if len(mask_files) == 0:
        print(f"[finalize] No .npy masks found in {associated_mask_folder}.")
        return

    view_count_dict = {}
    max_label = 0
    base_dtype = None

    for fname in mask_files:
        path = os.path.join(associated_mask_folder, fname)
        mask = np.load(path)

        if mask.ndim != 2:
            print(f"[finalize] Skip non-2D npy file in first pass: {fname}, shape={mask.shape}")
            continue

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

        counts = np.bincount(mask.ravel(), minlength=max_lab_local + 1)
        labels = np.nonzero(counts)[0]
        labels = labels[labels > 0]
        if labels.size == 0:
            continue

        for lab in labels.tolist():
            view_count_dict[lab] = view_count_dict.get(lab, 0) + 1

    if base_dtype is None or max_label == 0 or len(view_count_dict) == 0:
        print("[finalize] All masks are background or empty. Nothing to finalize.")
        return

    num_mask = int(max_label)

    view_counts = np.zeros(num_mask + 1, dtype=np.int32)
    for lab, cnt in view_count_dict.items():
        if lab <= num_mask:
            view_counts[lab] = cnt

    max_group_dist = float(getattr(projector, "max_group_distance", 0.0))

    dist_keep_mask = np.ones(num_mask + 1, dtype=bool)
    dist_keep_mask[0] = False

    if max_group_dist > 0.0 and getattr(projector, "gaussian_idx_bank", None):
        xyz = projector.gaussians_xyz.detach().cpu()

        cam_centers = []
        for cam in projector.viewpoint_camera:
            c = cam.camera_center
            if isinstance(c, torch.Tensor):
                c = c.detach().cpu().numpy()
            else:
                c = np.asarray(c, dtype=np.float32)
            cam_centers.append(c)
        if len(cam_centers) > 0:
            cam_centers = np.stack(cam_centers, axis=0)
        else:
            cam_centers = None

        group_dists = []

        for g in projector.gaussian_idx_bank:
            if (
                isinstance(g, torch.Tensor)
                and g.numel() > 0
                and cam_centers is not None
            ):
                pts = xyz[g.long().cpu()]
                center = pts.mean(dim=0).numpy()
                dists_to_cams = np.linalg.norm(cam_centers - center[None, :], axis=1)
                dist = float(dists_to_cams.min())
            else:
                dist = float("inf")
            group_dists.append(dist)

        group_dists = np.asarray(group_dists, dtype=np.float32)

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

        dist_keep_mask[1:] = (group_dists <= max_group_dist)

    if skip_min_views:
        keep_mask = np.ones(num_mask + 1, dtype=bool)
        keep_mask[0] = False
    else:
        keep_mask = (view_counts >= min_views)
        keep_mask[0] = False

    keep_mask = keep_mask & dist_keep_mask

    keep_labels = np.where(keep_mask[1:])[0] + 1
    original_num = num_mask
    new_num = int(keep_labels.size)

    if new_num == 0:
        if skip_min_views:
            print("[finalize] Warning: no masks remain after distance filtering.")
        else:
            print(f"[finalize] Warning: no masks meet min_views={min_views}.")
        return

    if skip_min_views:
        print(f"[finalize] Filtering by distance only: {original_num} -> {new_num}")
    else:
        print(f"[finalize] Filtering by views: {original_num} -> {new_num} (min_views={min_views})")

    remap = np.zeros(num_mask + 1, dtype=np.int32)
    for new_label, old_label in enumerate(keep_labels.tolist(), start=1):
        remap[old_label] = new_label

    projector.num_mask = new_num

    if hasattr(projector, "global_clip_features") and projector.global_clip_features is not None:
        old_feats = projector.global_clip_features
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


if __name__ == "__main__":
    parser = ArgumentParser()
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)

    parser.add_argument("--visualize", "-v", action="store_true")
    parser.add_argument(
        "--clip",
        "-c",
        action="store_true",
        help="if set, use CLIP features inside GaussianProjector for matching",
    )
    parser.add_argument(
        "--skip_finalize",
        action="store_true",
        help="if set, only skip min_views filtering inside finalize_masks_and_features",
    )

    args = get_combined_args(parser)

    dataset_params = model.extract(args)
    pipeline_params = pipeline.extract(args)

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

        print(f"[main] Building association into {projector.associated_mask_folder} ...")
        projector.build_mask_association()

        finalize_masks_and_features(projector, skip_min_views=args.skip_finalize)