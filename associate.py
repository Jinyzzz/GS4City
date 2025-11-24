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


def filter_masks_by_view_count(projector: GaussianProjector, min_views: int = 3):
    """
    从已经生成好的关联灰度图（projector.associated_mask_folder 下的 PNG）
    统计每个全局 mask id 在多少个视角中出现，并过滤只在 < min_views 个视角中出现的 id。

    灰度图约定：
      - 0 为背景
      - 1..N 为全局 mask id
    """
    associated_mask_folder = projector.associated_mask_folder
    visualize = getattr(projector, "visualize", False)
    visualize_folder = getattr(projector, "visualize_folder", None) if visualize else None

    # 列出所有 PNG（关联后的灰度图）
    mask_files = sorted(
        f for f in os.listdir(associated_mask_folder)
        if f.lower().endswith(".png")
    )
    if len(mask_files) == 0:
        print(f"[filter_masks_by_view_count] No PNG masks found in {associated_mask_folder}.")
        return

    # 先尝试从 projector 中拿 num_mask，如果为 0，则从现有灰度图推断最大 label 作为 num_mask
    num_mask = projector.get_num_mask
    if num_mask == 0:
        inferred_max = 0
        tmp_dtype = None
        for fname in mask_files:
            path = os.path.join(associated_mask_folder, fname)
            object_mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if object_mask is None:
                continue
            if tmp_dtype is None:
                tmp_dtype = object_mask.dtype
            if object_mask.size == 0:
                continue
            max_label = int(object_mask.max())
            if max_label > inferred_max:
                inferred_max = max_label

        num_mask = int(inferred_max)
        projector.num_mask = num_mask

        if num_mask == 0:
            print("[filter_masks_by_view_count] All masks are background (max label = 0). Nothing to filter.")
            return

        print(f"[filter_masks_by_view_count] Inferred num_mask={num_mask} from existing masks.")

    # 统计每个 label 出现于多少个视角（按视角计数，不按像素）
    mask_view_count = np.zeros(num_mask + 1, dtype=np.int32)
    base_dtype = None

    for fname in mask_files:
        path = os.path.join(associated_mask_folder, fname)
        object_mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if object_mask is None:
            continue

        if base_dtype is None:
            base_dtype = object_mask.dtype

        unique_labels = np.unique(object_mask)
        unique_labels = unique_labels[unique_labels > 0]  # 去掉背景 0
        if unique_labels.size == 0:
            continue

        mask_view_count[unique_labels] += 1

    if base_dtype is None:
        print("[filter_masks_by_view_count] Could not determine mask dtype, skip filtering.")
        return

    # 决定保留哪些 label：只保留在 >= min_views 个视角中出现的 label
    keep_labels = np.where(mask_view_count[1:] >= min_views)[0] + 1  # +1 还原为 1-based
    original_num = num_mask
    new_num = int(keep_labels.size)

    if new_num == 0:
        print(f"[filter_masks_by_view_count] Warning: no masks meet min_views={min_views}, skip filtering.")
        return

    if new_num == original_num:
        print(f"[filter_masks_by_view_count] No rare masks to filter (min_views={min_views}).")
        # 仍然写一个 info.json
        info = {
            "num_mask": int(original_num),
            "raw_mask_folder": projector.raw_mask_folder,
            "associated_mask_folder": projector.associated_mask_folder,
            "front_percentage": projector.front_percentage,
            "iou_threshold": projector.iou_threshold,
            "num_patch": projector.num_patches,
            "min_views": int(min_views),
        }
        info_path = os.path.join(associated_mask_folder, "info.json")
        json.dump(info, open(info_path, "w"))
        return

    print(f"[filter_masks_by_view_count] Filtering masks by views: {original_num} -> {new_num} (min_views={min_views})")

    # 构建 1-based label 的 remap：index = 旧 label，value = 新 label
    # 0 保持 0（背景），被丢弃的旧 label 映射为 0
    remap = np.zeros(num_mask + 1, dtype=base_dtype)
    for new_label, old_label in enumerate(keep_labels.tolist(), start=1):
        remap[old_label] = new_label

    # 更新 projector.num_mask，用于后续可视化
    projector.num_mask = new_num

    # 第二遍：重写所有灰度图 & 可视化图
    for fname in mask_files:
        path = os.path.join(associated_mask_folder, fname)
        object_mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if object_mask is None:
            continue

        # remap: shape = (num_mask+1,)；object_mask 像素值 in [0..num_mask]
        new_mask = remap[object_mask]
        cv2.imwrite(path, new_mask)

        if visualize and visualize_folder is not None:
            vis = projector.visualize_mask_association(new_mask)
            vis_path = os.path.join(visualize_folder, fname)
            cv2.imwrite(vis_path, vis)

    # 最后写出 info.json（最终版本）
    info = {
        "num_mask": int(projector.num_mask),
        "raw_mask_folder": projector.raw_mask_folder,
        "associated_mask_folder": projector.associated_mask_folder,
        "front_percentage": projector.front_percentage,
        "iou_threshold": projector.iou_threshold,
        "num_patch": projector.num_patches,
        "min_views": int(min_views),
    }
    info_path = os.path.join(associated_mask_folder, "info.json")
    json.dump(info, open(info_path, "w"))
    print(f"[filter_masks_by_view_count] Done. Final num_mask = {projector.num_mask}")


if __name__ == "__main__":
    parser = ArgumentParser()
    # model = ModelParams(parser, sentinel=True)
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--seg_method", default="sam", type=str)
    parser.add_argument("--front_percentage", "-fp", type=float, default=0.2)
    parser.add_argument("--overlap_threshold", "-ot", type=float, default=0.1)
    parser.add_argument("--num_patch", "-np", type=int, default=32)
    parser.add_argument("--visualize", "-v", action="store_true")
    # 控制“一个 mask 至少在多少个视角中出现才保留”，默认 5
    parser.add_argument("--min_views", type=int, default=5,
                        help="Minimum number of views a global mask id must appear in to be kept.")

    args = get_combined_args(parser)

    hyper_params = {
        "front_percentage": args.front_percentage,
        "overlap_threshold": args.overlap_threshold,
        "num_patch": args.num_patch,
        "seg_method": args.seg_method,
        "visualize": args.visualize
    }

    with torch.no_grad():
        # 先初始化 projector（会创建 <seg_method>_mask 文件夹，但不会删内容）
        projector = GaussianProjector(model.extract(args), pipeline.extract(args), args.iteration, hyper_params)

        # 检测是否已经存在关联好的 mask（即 <seg_method>_mask 里是否有 PNG）
        existing_mask_files = [
            f for f in os.listdir(projector.associated_mask_folder)
            if f.lower().endswith(".png")
        ]

        if len(existing_mask_files) == 0:
            # 没有现成的 mask，先做跨视角关联
            print(f"[main] No existing masks found in {projector.associated_mask_folder}, building association...")
            projector.build_mask_association()
        else:
            # 已经有 sam_mask / seg_method_mask 的灰度图，跳过重建
            print(f"[main] Found {len(existing_mask_files)} existing masks in "
                  f"{projector.associated_mask_folder}, skip building association.")

        # 统一读取关联后的灰度图，按视角数过滤 mask，并更新灰度图 / 可视化 / info.json
        filter_masks_by_view_count(projector, min_views=args.min_views)
