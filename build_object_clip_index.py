#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
根据跨视角一致的灰度实例 mask（支持 16-bit），计算每个物体 ID 的 CLIP 特征（跨视角平均），
用 mask 把 RGB 图抠出来后再送 CLIP。支持 mask 与 RGB 分辨率不一致的情况（自动缩放）。

输出：一个 .npz 文件，包含
    - ids:       [N]  int64，每个物体的灰度 id
    - features:  [N,D] float32，每个物体的平均 CLIP 图像特征（单位向量）
    - counts:    [N]  int32，每个 id 被多少个视角贡献过特征
    - model_name: str，使用的 CLIP 模型名
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import clip  # pip install git+https://github.com/openai/CLIP.git


# -------------------- 工具函数 -------------------- #

def find_corresponding_image(mask_path: Path, image_dir: Path):
    """
    根据 mask 文件名，在 image_dir 中寻找对应的 RGB 图像。

    规则：
    1. 先尝试：完全同名（含 _D）+ 原后缀；
    2. 再尝试：stem（含/不含 _D）+ 常见小写/大写扩展名。
    """
    exts = [
        mask_path.suffix,       # 原来的后缀
        ".png", ".jpg", ".jpeg",
        ".PNG", ".JPG", ".JPEG",
    ]

    stems = [mask_path.stem]

    # 如果名字以 "_D" 结尾，增加一个去掉 "_D" 的 stem
    if mask_path.stem.endswith("_D"):
        stems.append(mask_path.stem[:-2])

    # 1) 完全同名
    first_candidate = image_dir / mask_path.name
    if first_candidate.exists():
        return first_candidate

    # 2) stem + ext 组合
    candidates = []
    for stem in stems:
        for ext in exts:
            candidates.append(image_dir / f"{stem}{ext}")

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"Cannot find corresponding image for mask {mask_path} in {image_dir}"
    )


def get_instance_bbox(mask_np: np.ndarray, instance_id: int):
    """
    从 mask numpy 数组中获取指定 instance_id 的 bounding box。
    返回 (y_min, y_max, x_min, x_max)，都为 int。
    若该 id 不存在，返回 None。
    """
    ys, xs = np.where(mask_np == instance_id)
    if len(xs) == 0 or len(ys) == 0:
        return None

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    return int(y_min), int(y_max), int(x_min), int(x_max)


# -------------------- 主功能：计算 CLIP 特征 -------------------- #

def compute_clip_features_for_dataset(
    mask_dir: str,
    image_dir: str,
    output_path: str,
    model_name: str = "ViT-B/16",
    device: str = None,
    min_pixels: int = 10,
    skip_ids=(0,),
    bg_value: int = 127,
):
    """
    遍历 mask_dir 中所有灰度实例 mask（16-bit / 8-bit 都可），
    对每个实例 id：
        - 用 mask 在 RGB 上抠出该实例（其他位置用 bg_value 填充）；
        - 送入 CLIP 提取图像特征；
        - 跨视角做平均，得到每个 id 的语言特征（图像侧 CLIP embedding）。

    参数：
        mask_dir: 灰度实例 mask 目录
        image_dir: RGB 图像目录
        output_path: 输出 .npz 文件路径
        model_name: CLIP 模型名，默认 "ViT-B/16"
        device: "cuda" 或 "cpu"，默认自动检测
        min_pixels: 一个实例最少多少像素，否则忽略（太小的噪声）
        skip_ids: 要跳过的 ID，例如 (0,) 代表把 0 当背景
        bg_value: 抠图时背景填充的灰度值（0~255），默认 127 中灰
    """

    mask_dir = Path(mask_dir)
    image_dir = Path(image_dir)
    output_path = Path(output_path)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Loading CLIP model: {model_name}")
    model, preprocess = clip.load(model_name, device=device)

    # id -> 累加特征向量，和出现次数
    id_to_feat_sum = {}
    id_to_count = defaultdict(int)

    mask_paths = sorted(
        [p for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    )

    if not mask_paths:
        raise RuntimeError(f"No mask images found in {mask_dir}")

    print(f"Found {len(mask_paths)} mask images.")

    for mask_path in tqdm(mask_paths, desc="Processing masks"):
        # 找对应的 RGB 图
        try:
            image_path = find_corresponding_image(mask_path, image_dir)
        except FileNotFoundError as e:
            print(e)
            continue

        # 读 RGB 图
        rgb_img = Image.open(image_path).convert("RGB")
        rgb_w, rgb_h = rgb_img.size

        # 读 mask（保持原始位深），并自动缩放到与 RGB 一样大
        mask_img = Image.open(mask_path)
        if mask_img.size != rgb_img.size:
            # 用最近邻缩放，防止 id 被插值破坏
            mask_img = mask_img.resize(rgb_img.size, Image.NEAREST)

        mask_np = np.array(mask_img)
        if not np.issubdtype(mask_np.dtype, np.integer):
            mask_np = mask_np.astype(np.int32)

        # 当前视角所有实例 ID
        unique_ids = np.unique(mask_np)

        for inst_id in unique_ids:
            inst_id_int = int(inst_id)

            # 跳过背景等
            if inst_id_int in skip_ids:
                continue

            # 实例像素数
            inst_mask = (mask_np == inst_id_int)
            num_pixels = int(inst_mask.sum())
            if num_pixels < min_pixels:
                continue

            # bbox
            bbox = get_instance_bbox(mask_np, inst_id_int)
            if bbox is None:
                continue

            y_min, y_max, x_min, x_max = bbox

            # 裁剪 RGB patch
            patch = rgb_img.crop((x_min, y_min, x_max + 1, y_max + 1))
            patch_np = np.array(patch).copy()  # [h, w, 3]

            # 对应的 mask patch
            inst_mask_patch = inst_mask[y_min:y_max + 1, x_min:x_max + 1]  # [h, w] bool

            # 用 bg_value 填充非实例区域，实现“抠图”
            # inst_mask_patch 为 True 的地方保留原像素，False 的地方填充背景
            bg_val = np.array([bg_value, bg_value, bg_value], dtype=patch_np.dtype)
            patch_np[~inst_mask_patch] = bg_val

            # 转回 PIL Image 送入 CLIP
            patch_masked = Image.fromarray(patch_np)

            image_input = preprocess(patch_masked).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.encode_image(image_input)  # [1, D]
                feat = feat.squeeze(0)                  # [D]
                # 先归一化到单位向量
                feat = feat / feat.norm(dim=-1, keepdim=True)

            # 累加特征
            if inst_id_int not in id_to_feat_sum:
                id_to_feat_sum[inst_id_int] = feat.clone().cpu()
            else:
                id_to_feat_sum[inst_id_int] += feat.cpu()

            id_to_count[inst_id_int] += 1

    # 计算每个 id 的平均特征
    all_ids = sorted(id_to_feat_sum.keys())
    if not all_ids:
        raise RuntimeError("No instance features were computed. "
                           "Check mask directory / image directory / min_pixels / skip_ids.")

    features = []
    counts = []
    for inst_id in all_ids:
        sum_feat = id_to_feat_sum[inst_id]       # [D]
        count = id_to_count[inst_id]
        avg_feat = sum_feat / float(count)       # 简单均值
        # 再归一化
        avg_feat = avg_feat / avg_feat.norm(dim=-1, keepdim=True)

        features.append(avg_feat.numpy())
        counts.append(count)

    features = np.stack(features, axis=0)        # [N_obj, D]
    ids = np.array(all_ids, dtype=np.int64)      # 16-bit id 也能完整存下
    counts = np.array(counts, dtype=np.int32)

    # 保存索引
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        ids=ids,
        features=features,
        counts=counts,
        model_name=model_name,
    )

    print(f"Saved {len(ids)} object features to {output_path}")
    print("You can later query by object id (gray value) using this file.")


# -------------------- Demo 查询（可选） -------------------- #

def demo_query(index_path: str, query_id: int):
    """
    简单演示：从生成好的 npz 里查某个 id 的特征信息。
    """
    data = np.load(index_path)
    ids = data["ids"]
    features = data["features"]
    counts = data["counts"]

    if query_id not in ids:
        print(f"ID {query_id} not found in index.")
        return

    idx = int(np.where(ids == query_id)[0][0])
    feat = features[idx]
    cnt = int(counts[idx])

    print(f"Object id: {query_id}")
    print(f"  seen in {cnt} views.")
    print(f"  feature dim: {feat.shape[0]}")
    print(f"  first 10 dims: {feat[:10]}")


# -------------------- CLI -------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute per-object CLIP features across views (16-bit mask + masked RGB)."
    )
    parser.add_argument(
        "--mask-dir",
        type=str,
        default="/workspace/Gaga/dataset/subset_building1_29_copy/fused_mask",
        help="Directory of grayscale instance masks (can be 16-bit).",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="/workspace/Gaga/dataset/subset_building1_29_copy/images",
        help="Directory of RGB images.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/workspace/Gaga/dataset/subset_building1_29_copy/object_clip_index.npz",
        help="Path to output .npz file storing features.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="ViT-B/16",
        help="CLIP model name (e.g., 'ViT-B/16', 'ViT-B/32').",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch device (e.g., 'cuda' or 'cpu'). If None, auto-detect.",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=10,
        help="Minimum number of pixels per instance to be considered.",
    )
    parser.add_argument(
        "--skip-id",
        type=int,
        nargs="*",
        default=[0],
        help="Instance ids (gray values) to skip, e.g. background 0.",
    )
    parser.add_argument(
        "--bg-value",
        type=int,
        default=127,
        help="Background gray value (0-255) used to fill non-instance pixels.",
    )
    parser.add_argument(
        "--query-id",
        type=int,
        default=None,
        help="If set, will run a demo query after feature extraction.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    compute_clip_features_for_dataset(
        mask_dir=args.mask_dir,
        image_dir=args.image_dir,
        output_path=args.output,
        model_name=args.model_name,
        device=args.device,
        min_pixels=args.min_pixels,
        skip_ids=tuple(args.skip_id),
        bg_value=args.bg_value,
    )

    if args.query_id is not None:
        print("\n===== Demo query =====")
        demo_query(args.output, args.query_id)
