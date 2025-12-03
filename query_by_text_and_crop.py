#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
通过文本（例如 "flag"）查询物体，
在 object_clip_index.npz 中找到最相似的物体 ID（灰度值，支持 16-bit mask），
然后在所有原图中把这些物体的区域裁剪出来保存。

用法示例：
    python query_by_text_and_crop.py \
        --index /workspace/Gaga/dataset/subset_building1_29_copy/object_clip_index.npz \
        --mask-dir /workspace/Gaga/dataset/subset_building1_29_copy/fused_mask \
        --image-dir /workspace/Gaga/dataset/subset_building1_29_copy/images \
        --output-dir /workspace/Gaga/dataset/subset_building1_29_copy/query_results \
        --text "flag" \
        --top-k 5 \
        --max-crops-per-id 10
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import clip  # pip install git+https://github.com/openai/CLIP.git


# -------------------- 基础工具函数 -------------------- #

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


def load_index(index_path: str):
    """
    读取 object_clip_index.npz
    返回：
        ids: [N] int64，每个物体的灰度 id
        features: [N, D] float32，每个物体的平均 CLIP 图像特征（已归一化）
        counts: [N] int32，物体在多少视角中出现
        model_name: CLIP 模型名字
    """
    data = np.load(index_path)
    ids = data["ids"]                     # int64
    features = data["features"].astype(np.float32)
    counts = data["counts"]
    model_name = str(data["model_name"]) if "model_name" in data else "ViT-B/16"
    return ids, features, counts, model_name


def text_to_feature(text: str, model, device: str):
    """
    使用 CLIP 文本编码器，将文本转为特征向量 [D]，并归一化。
    """
    with torch.no_grad():
        tokens = clip.tokenize([text]).to(device)
        text_feat = model.encode_text(tokens)  # [1, D]
        text_feat = text_feat.squeeze(0)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    return text_feat.cpu().numpy()  # [D]


def compute_similarity(text_feat: np.ndarray, features: np.ndarray):
    """
    计算文本特征与所有物体特征的相似度（cosine）。
    假设 features 已经是单位向量，text_feat 也归一化过。
    """
    # [N, D] dot [D] -> [N]
    sims = features @ text_feat
    return sims  # 越大越相似


def select_top_k(ids, sims, counts, top_k: int, min_views: int = 1):
    """
    从所有物体中选出 top_k 个相似度最高的 ID。
    可以设置 min_views 过滤出现次数太少的物体。
    返回：
        selected_ids, selected_sims, selected_counts （都按相似度从高到低排序）
    """
    mask = counts >= min_views
    valid_ids = ids[mask]
    valid_sims = sims[mask]
    valid_counts = counts[mask]

    if valid_ids.size == 0:
        raise RuntimeError("No valid objects after filtering by min_views.")

    order = np.argsort(-valid_sims)  # 降序
    order = order[:top_k]

    return valid_ids[order], valid_sims[order], valid_counts[order]


# -------------------- 裁剪部分（支持 16-bit mask + 尺寸自动对齐） -------------------- #

def crop_instances_for_ids(
    selected_ids,
    mask_dir: str,
    image_dir: str,
    output_dir: str,
    min_pixels: int = 10,
    max_crops_per_id: int = 20,
):
    """
    对 selected_ids 中的每个 id：
        遍历所有 mask 图，
        找到该 id 的区域，裁剪对应的 RGB patch，
        保存到 output_dir/id_xxxx/ 下。

    参数：
        selected_ids: 物体灰度 id 列表（1D numpy 或 list），通常为 int64
        mask_dir: 灰度 mask 目录（16-bit / 8-bit 都支持）
        image_dir: RGB 图像目录
        output_dir: 输出目录
        min_pixels: 单个实例最少像素，否则跳过
        max_crops_per_id: 每个 id 最多保存多少个 crop（防止太多）
    """
    mask_dir = Path(mask_dir)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_paths = sorted(
        [p for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    )

    if not mask_paths:
        raise RuntimeError(f"No mask images found in {mask_dir}")

    # 为每个 id 计数，避免保存过多
    selected_ids = [int(i) for i in selected_ids]
    saved_count = {int(i): 0 for i in selected_ids}

    for mask_path in tqdm(mask_paths, desc="Cropping instances from images"):
        # 找对应的 RGB 图
        try:
            image_path = find_corresponding_image(mask_path, image_dir)
        except FileNotFoundError as e:
            print(e)
            continue

        # 先读 RGB 图
        rgb_img = Image.open(image_path).convert("RGB")
        rgb_w, rgb_h = rgb_img.size

        # ✅ 保留 mask 的原始位深（支持 16-bit），并自动缩放到与 RGB 一样大
        mask_img = Image.open(mask_path)
        if mask_img.size != rgb_img.size:
            # 最近邻缩放，避免 id 被插值改坏
            mask_img = mask_img.resize(rgb_img.size, Image.NEAREST)

        mask_np = np.array(mask_img)
        if not np.issubdtype(mask_np.dtype, np.integer):
            mask_np = mask_np.astype(np.int32)

        unique_ids_in_img = np.unique(mask_np)
        ids_in_img = set(int(x) for x in unique_ids_in_img.tolist())

        # 只关心 selected_ids 中出现在当前图里的那些 id
        intersect_ids = [i for i in selected_ids if i in ids_in_img]
        if not intersect_ids:
            continue

        for inst_id_int in intersect_ids:
            # 如果已经到达最大 crop 数量，跳过
            if saved_count[inst_id_int] >= max_crops_per_id:
                continue

            num_pixels = np.sum(mask_np == inst_id_int)
            if num_pixels < min_pixels:
                continue

            bbox = get_instance_bbox(mask_np, inst_id_int)
            if bbox is None:
                continue

            y_min, y_max, x_min, x_max = bbox
            patch = rgb_img.crop((x_min, y_min, x_max + 1, y_max + 1))

            # 保存路径：output_dir/id_XXXXX/idXXXXX_<maskname>.png
            id_dir = output_dir / f"id_{inst_id_int:05d}"
            id_dir.mkdir(parents=True, exist_ok=True)

            out_name = f"id{inst_id_int:05d}_{mask_path.stem}.png"
            out_path = id_dir / out_name
            patch.save(out_path)

            saved_count[inst_id_int] += 1

    print("\nCrop summary:")
    for inst_id in selected_ids:
        print(f"  id {int(inst_id)} -> {saved_count[int(inst_id)]} crops saved.")


# -------------------- CLI 部分 -------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description="Query objects by text with CLIP and crop patches (16-bit mask friendly).")
    parser.add_argument(
        "--index",
        type=str,
        required=True,
        help="Path to object_clip_index.npz generated previously.",
    )
    parser.add_argument(
        "--mask-dir",
        type=str,
        required=True,
        help="Directory of grayscale instance masks (can be 16-bit).",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Directory of RGB images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save cropped patches.",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help='Text query, e.g., "flag".',
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many top objects (ids) to select by similarity.",
    )
    parser.add_argument(
        "--min-views",
        type=int,
        default=1,
        help="Only keep objects that appear in at least this many views.",
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
        help="Minimum number of pixels per instance to be cropped.",
    )
    parser.add_argument(
        "--max-crops-per-id",
        type=int,
        default=20,
        help="Maximum number of crops to save per object id.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 1) 读取索引文件（物体平均图像特征）
    ids, features, counts, model_name = load_index(args.index)
    print(f"Loaded {len(ids)} objects from index.")
    print(f"CLIP model in index: {model_name}")

    # 2) 加载 CLIP 模型（用和 index 一致的模型名）
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")
    print(f"Loading CLIP model: {model_name}")
    model, _ = clip.load(model_name, device=device)

    # 3) 文本 -> 特征
    print(f"Encoding text query: '{args.text}'")
    text_feat = text_to_feature(args.text, model, device)  # [D]

    # 4) 文本特征与所有物体特征的相似度
    sims = compute_similarity(text_feat, features)  # [N]
    print("Similarity stats: min={:.4f}, max={:.4f}, mean={:.4f}".format(
        float(sims.min()), float(sims.max()), float(sims.mean())
    ))

    # 5) 选出 top-k 个最相似的物体 ID
    selected_ids, selected_sims, selected_counts = select_top_k(
        ids, sims, counts, top_k=args.top_k, min_views=args.min_views
    )

    print("\nTop-{} matched object ids:".format(args.top_k))
    for rank, (oid, sim, cnt) in enumerate(zip(selected_ids, selected_sims, selected_counts), start=1):
        print(f"  #{rank}: id={int(oid)}, sim={sim:.4f}, views={int(cnt)}")

    # 6) 在原图中把这些 id 的区域裁剪出来
    crop_instances_for_ids(
        selected_ids=selected_ids,
        mask_dir=args.mask_dir,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        min_pixels=args.min_pixels,
        max_crops_per_id=args.max_crops_per_id,
    )

    print("\nDone.")
