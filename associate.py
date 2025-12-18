# Copyright (C) 2024, Gaga
# Gaga research group, https://github.com/weijielyu/Gaga
# All rights reserved.
#

import os
import json
import cv2
import numpy as np
import torch
from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams, get_combined_args
from mask.projector import GaussianProjector

# 只有这种“非超参”的常量保留
VIS_IMAGE_EXT = ".png"


def finalize_masks_and_features(projector: GaussianProjector):
    """
    针对已经在 GaussianProjector.build_mask_association 中完成跨视角匹配、
    并将结果写到 projector.associated_mask_folder 的 .npy mask，做最后的整理：

    1. 按视角数（>= projector.min_views）过滤全局 mask id；
    2. 可选：按 3D 高斯组团中心与相机的距离过滤（> max_group_distance 视为后景）；
    3. 重映射 label，使得 1..N 连续；
    4. 覆盖写回 .npy；
    5. 用 projector.visualize_mask_association(new_mask) 生成彩色可视化 PNG；
    6. 若 projector.global_clip_features 存在，则按照重映射保存 clip_features.npy：
       - 形状 [num_mask+1, D]，第 i 行对应 mask id = i（0 行为背景，全 0）。
    """

    associated_mask_folder = projector.associated_mask_folder
    visualize = getattr(projector, "visualize", False)
    visualize_folder = getattr(projector, "visualize_folder", None) if visualize else None

    # min_views 完全来自 config（projector.min_views），不在这个文件写死
    min_views = int(getattr(projector, "min_views", 1))

    # 只支持 .npy（跳过 clip 特征）
    mask_files = sorted(
        f for f in os.listdir(associated_mask_folder)
        if f.lower().endswith(".npy") and f != "clip_features.npy"
    )
    if len(mask_files) == 0:
        print(f"[finalize] No .npy masks found in {associated_mask_folder}.")
        return

    # ====== 第一遍：统计 view_count + 推断 max_label（向量化） ======
    view_count_dict = {}  # label -> 视角数
    max_label = 0
    base_dtype = None

    for fname in mask_files:
        path = os.path.join(associated_mask_folder, fname)
        mask = np.load(path)  # H x W

        # 只处理 2D 的 HxW 图像
        if mask.ndim != 2:
            print(f"[finalize] Skip non-2D npy file in first pass: {fname}, shape={mask.shape}")
            continue

        # 强制转为整型，后面要用作 np.bincount / 索引
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

        # bincount 长度 = max_lab_local+1，index 对应 label
        counts = np.bincount(mask.ravel(), minlength=max_lab_local + 1)
        labels = np.nonzero(counts)[0]
        labels = labels[labels > 0]  # 去掉背景 0
        if labels.size == 0:
            continue

        # 按视角计数：某个 label 在这一视角中出现→+1
        for lab in labels.tolist():
            view_count_dict[lab] = view_count_dict.get(lab, 0) + 1

    if base_dtype is None or max_label == 0 or len(view_count_dict) == 0:
        print("[finalize] All masks are background or empty. Nothing to finalize.")
        return

    num_mask = int(max_label)

    # dict -> array，index = label
    view_counts = np.zeros(num_mask + 1, dtype=np.int32)
    for lab, cnt in view_count_dict.items():
        if lab <= num_mask:
            view_counts[lab] = cnt

    # ====== 新增：按“与相机的距离”过滤（后景剔除） ======
    # 距离阈值来自 projector.max_group_distance（<=0 表示关闭）
    max_group_dist = float(getattr(projector, "max_group_distance", 0.0))

    # 默认所有 label 在“距离维度”上都是可用的
    dist_keep_mask = np.ones(num_mask + 1, dtype=bool)
    dist_keep_mask[0] = False  # 背景本来就不算

    if max_group_dist > 0.0 and getattr(projector, "gaussian_idx_bank", None):
        # projector.gaussian_idx_bank: list[len≈实际全局组数]，每个元素是一个高斯 index 的 Tensor
        # projector.gaussians_xyz: [N, 3]，全局高斯的 3D 坐标
        xyz = projector.gaussians_xyz.detach().cpu()  # [N, 3] on CPU

        # ===== 先收集所有训练相机的世界坐标中心 =====
        cam_centers = []
        for cam in projector.viewpoint_camera:
            # GraphDECO 的 Camera 有 camera_center 属性（世界坐标下）
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
                # 高斯点坐标 -> 组团中心
                pts = xyz[g.long().cpu()]           # [K, 3]
                center = pts.mean(dim=0).numpy()    # [3]

                # 到所有相机中心的距离，取最小
                dists_to_cams = np.linalg.norm(cam_centers - center[None, :], axis=1)
                dist = float(dists_to_cams.min())
            else:
                # 没有任何高斯的组 或 没有相机 → 视为“极远”
                dist = float("inf")
            group_dists.append(dist)

        group_dists = np.asarray(group_dists, dtype=np.float32)  # 长度 ≈ num_groups

        # 对齐长度：group_dists 的长度可能和 num_mask 不完全一致，这里做个安全处理
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

        # 1..num_mask 的距离过滤：距离最近相机 > max_group_dist 的组直接丢掉
        dist_keep_mask[1:] = (group_dists <= max_group_dist)

    # ====== 基于视角数过滤：保留视角数 >= min_views 的 label（忽略 0） ======
    keep_mask = (view_counts >= min_views)
    keep_mask[0] = False

    # ★ 把“视角数过滤”和“距离过滤”合并：两者都要满足
    keep_mask = keep_mask & dist_keep_mask

    keep_labels = np.where(keep_mask[1:])[0] + 1  # 1-based
    original_num = num_mask
    new_num = int(keep_labels.size)

    if new_num == 0:
        print(f"[finalize] Warning: no masks meet min_views={min_views}.")
        return

    print(f"[finalize] Filtering by views: {original_num} -> {new_num} (min_views={min_views})")

    # remap: 旧 label -> 新 label；0 始终保持 0
    remap = np.zeros(num_mask + 1, dtype=np.int32)
    for new_label, old_label in enumerate(keep_labels.tolist(), start=1):
        remap[old_label] = new_label

    # 更新 projector.num_mask
    projector.num_mask = new_num

    # ====== 若 projector 有全局 CLIP 特征，则重排并保存 ======
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

    # ====== 第二遍：重写 .npy & 保存可视化结果 ======
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

    # info.json：这里只是把当前 projector 的实际配置 dump 出来
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

    # 运行时只关心：
    #   --visualize : 是否保存彩色可视化
    #   --clip      : 是否启用 CLIP 筛选（用户必须显式指定）
    parser.add_argument("--visualize", "-v", action="store_true")
    parser.add_argument("--clip", "-c", action="store_true",
                        help="if set, use CLIP features inside GaussianProjector for matching")

    # 不再需要 iteration，内部直接用 -1
    args = get_combined_args(parser)

    # 从 Gaga 的 Argument 系统提取 dataset / pipeline
    dataset_params = model.extract(args)
    pipeline_params = pipeline.extract(args)

    # 只传「覆盖配置」的键（visualize/use_clip），其余全从 mask/config.json["projector"] 来
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

        # 总是重新做跨视角关联（会覆盖 sam_mask 里的 .npy）
        print(f"[main] Building association into {projector.associated_mask_folder} ...")
        projector.build_mask_association()

        # 然后再做视角数过滤 + 可视化 + 保存 CLIP 平均特征
        finalize_masks_and_features(projector)
