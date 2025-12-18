import os
import cv2
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from typing import Optional, Tuple, Dict

import torch
import clip
from PIL import Image


CLIP_MODEL_NAME = "ViT-B/32"


def id_to_color(idx: int) -> Tuple[int, int, int]:
    """
    给定一个实例 id（uint16），生成一个确定性的 RGB 颜色。
    这样同一个 id 在不同视角生成的颜色是一致的。
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
    自动从 base_path_no_ext.* 读取 mask，支持：
      - .npy
      - .png / .jpg / .jpeg / .bmp / .tif / .tiff

    返回 HxW 的整型 mask（0 = 背景, >0 = 实例 id），找不到则返回 None。
    """
    # 优先 npy
    npy_path = base_path_no_ext + ".npy"
    if os.path.exists(npy_path):
        m = np.load(npy_path)
        if m is not None:
            if m.ndim == 3:
                m = m[..., 0]
            if not np.issubdtype(m.dtype, np.integer):
                m = m.astype(np.int64)
            return m

    # 再尝试常见图像格式
    exts = [".png", ".PNG",
            ".jpg", ".JPG",
            ".jpeg", ".JPEG",
            ".bmp", ".BMP",
            ".tif", ".tiff", ".TIF", ".TIFF"]
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
    从 <scene>/images 下自动读取对应的 RGB 图像（BGR 格式，cv2 风格）。
    """
    img_dir = os.path.join(scene_folder, "images")
    exts = [".png", ".PNG",
            ".jpg", ".JPG",
            ".jpeg", ".JPEG",
            ".bmp", ".BMP"]
    for ext in exts:
        p = os.path.join(img_dir, stem + ext)
        if os.path.exists(p):
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is not None:
                return img
    return None


def compute_mask_clip_feature(
    clip_model,
    clip_preprocess,
    device: torch.device,
    rgb_bgr: np.ndarray,
    mask: np.ndarray,
) -> Optional[np.ndarray]:
    """
    给定 BGR 图像和 2D 实例 mask（H x W, bool 或 int），计算该实例的 CLIP 特征（L2 归一化）。
    """
    if rgb_bgr is None:
        return None

    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")

    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return None

    H, W = mask_bool.shape
    ys, xs = np.where(mask_bool)
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    pad = 4
    y1 = max(0, y1 - pad)
    x1 = max(0, x1 - pad)
    y2 = min(H - 1, y2 + pad)
    x2 = min(W - 1, x2 + pad)

    crop = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)[y1:y2 + 1, x1:x2 + 1, :].copy()
    m_crop = mask_bool[y1:y2 + 1, x1:x2 + 1]

    if crop.size == 0 or m_crop.size == 0:
        return None

    crop[~m_crop] = 255  # 用白色填充背景

    pil_img = Image.fromarray(crop)
    with torch.no_grad():
        inp = clip_preprocess(pil_img).unsqueeze(0).to(device)
        feat = clip_model.encode_image(inp)
        feat = feat / feat.norm(dim=-1, keepdim=True)

    feat_np = feat[0].detach().cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(feat_np) + 1e-6
    return feat_np / norm


def main():
    parser = ArgumentParser(
        description=(
            "Fuse CityGML and SAM instance masks for a scene.\n"
            "Pixel-wise priority: SAM > GML.\n"
            "GML IDs are kept, SAM IDs are offset to avoid conflicts.\n"
            "Also output colorful visualization and fused CLIP features."
        )
    )
    parser.add_argument(
        "--scene",
        "-s",
        type=str,
        required=True,
        help="scene name under dataset/, e.g. 'subset_building1_16'",
    )
    args = parser.parse_args()

    # 当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.join(script_dir, "dataset")

    scene_folder = os.path.join(dataset_root, args.scene)
    assert os.path.exists(scene_folder), f"scene folder not found: {scene_folder}"

    gml_dir = os.path.join(scene_folder, "gml_mask")
    sam_dir = os.path.join(scene_folder, "sam_mask")
    out_gray_dir = os.path.join(scene_folder, "fused_mask")      # 保存 .npy
    out_vis_dir = os.path.join(scene_folder, "fused_mask_vis")   # 保存 .png 可视化

    assert os.path.exists(gml_dir), f"gml_mask folder not found: {gml_dir}"
    assert os.path.exists(sam_dir), f"sam_mask folder not found: {sam_dir}"
    os.makedirs(out_gray_dir, exist_ok=True)
    os.makedirs(out_vis_dir, exist_ok=True)

    print(f"[INFO] scene folder: {scene_folder}")
    print(f"[INFO] gml_mask dir: {gml_dir}")
    print(f"[INFO] sam_mask dir: {sam_dir}")
    print(f"[INFO] gray output dir (.npy) : {out_gray_dir}")
    print(f"[INFO] vis  output dir (.png) : {out_vis_dir}")

    # SAM 使用固定偏移，避免和 GML 冲突
    sam_id_offset = 10000
    print(f"[INFO] Using fixed SAM id offset: {sam_id_offset}")

    # ====== 读取 SAM 的 CLIP 特征（如果存在） ======
    sam_clip_feat_path = os.path.join(sam_dir, "clip_features.npy")
    sam_clip_feats = None
    if os.path.exists(sam_clip_feat_path):
        sam_clip_feats = np.load(sam_clip_feat_path)  # [N_sam+1, D]
        print(f"[INFO] Loaded SAM clip features: {sam_clip_feat_path}, shape={sam_clip_feats.shape}")
    else:
        print("[WARN] No SAM clip_features.npy found, SAM 部分将没有 CLIP 特征。")

    # ====== 初始化 CLIP 模型（用于 GML 部分） ======
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Loading CLIP model {CLIP_MODEL_NAME} on {device} ...")
    clip_model, clip_preprocess = clip.load(CLIP_MODEL_NAME, device=device)
    clip_model.eval()

    # 用于 GML id 的跨视角平均
    gml_feat_sum: Dict[int, np.ndarray] = {}
    gml_feat_count: Dict[int, int] = {}

    # 用于统计 fused 最大 id
    max_gml_id = 0
    max_sam_id = 0

    # SAM 输入是 npy 文件
    sam_files = sorted(
        f for f in os.listdir(sam_dir)
        if f.lower().endswith(".npy")
    )
    print(f"[INFO] Found {len(sam_files)} SAM mask (.npy) files")

    for name in tqdm(sam_files, desc="Fusing masks"):
        stem, _ = os.path.splitext(name)
        sam_path = os.path.join(sam_dir, name)

        # 读取 SAM mask (npy)
        sam = np.load(sam_path)
        if sam is None:
            print(f"[WARN] cannot read SAM mask: {sam_path}, skip.")
            continue
        if sam.ndim == 3:
            sam = sam[..., 0]
        if not np.issubdtype(sam.dtype, np.integer):
            sam = sam.astype(np.int64)

        # 读取 GML mask（自动 npy / 图像）
        gml_base = os.path.join(gml_dir, stem)
        gml = load_mask_auto(gml_base)
        if gml is None:
            # 没有对应的 GML → 用全 0
            print(f"[WARN] GML mask missing for {name}, use empty GML.")
            H_sam, W_sam = sam.shape[:2]
            gml = np.zeros((H_sam, W_sam), dtype=np.int64)

        # 尺寸对齐：选更小的一边
        h_gml, w_gml = gml.shape[:2]
        h_sam, w_sam = sam.shape[:2]
        target_h = min(h_gml, h_sam)
        target_w = min(w_gml, w_sam)
        target_size = (target_w, target_h)  # cv2 是 (w, h)

        if (h_gml, w_gml) != (target_h, target_w):
            gml = cv2.resize(gml.astype(np.int32), target_size, interpolation=cv2.INTER_NEAREST)
        if (h_sam, w_sam) != (target_h, target_w):
            sam = cv2.resize(sam.astype(np.int32), target_size, interpolation=cv2.INTER_NEAREST)

        gml = gml.astype(np.int64)
        sam = sam.astype(np.int64)

        H, W = target_h, target_w

        # 更新最大 id 统计（此时 sam 还没偏移）
        if gml.size > 0:
            max_gml_id = max(max_gml_id, int(gml.max()))
        if sam.size > 0:
            max_sam_id = max(max_sam_id, int(sam.max()))

        # SAM ID 偏移
        sam_nonzero = sam != 0
        if sam_nonzero.any():
            sam[sam_nonzero] = sam[sam_nonzero] + sam_id_offset

        # 融合：优先 SAM
        fused = np.zeros((H, W), dtype=np.uint16)
        fused[sam != 0] = sam[sam != 0].astype(np.uint16)
        fused[(sam == 0) & (gml != 0)] = gml[(sam == 0) & (gml != 0)].astype(np.uint16)

        # 生成彩色可视化图
        vis = np.zeros((H, W, 3), dtype=np.uint8)
        unique_ids = np.unique(fused)
        unique_ids = unique_ids[unique_ids != 0]
        for uid in unique_ids:
            r, g, b = id_to_color(int(uid))
            vis[fused == uid] = (b, g, r)  # BGR

        # 保存 fused 灰度为 .npy，彩色为 .png
        out_gray_path = os.path.join(out_gray_dir, stem + ".npy")
        out_vis_path = os.path.join(out_vis_dir, stem + ".png")
        np.save(out_gray_path, fused)
        cv2.imwrite(out_vis_path, vis)

        # ====== 为 GML 计算 CLIP 特征（跨视角平均） ======
        rgb = load_rgb_image(scene_folder, stem)
        if rgb is not None:
            gml_ids = np.unique(gml)
            gml_ids = gml_ids[gml_ids != 0]
            for gid in gml_ids:
                mask_g = (gml == gid)
                feat = compute_mask_clip_feature(
                    clip_model, clip_preprocess, device, rgb, mask_g
                )
                if feat is None:
                    continue
                gid_int = int(gid)
                if gid_int not in gml_feat_sum:
                    gml_feat_sum[gid_int] = feat.copy()
                    gml_feat_count[gid_int] = 1
                else:
                    gml_feat_sum[gid_int] += feat
                    gml_feat_count[gid_int] += 1

    print("[INFO] Fusion done for all images.")

    # ====== 生成 fused CLIP 特征 ======
    fused_clip_path = os.path.join(scene_folder, "clip_features_fused.npy")

    # 确定最大 id（GML + SAM 偏移后）
    fused_max_id = max(max_gml_id, max_sam_id + sam_id_offset)
    if fused_max_id <= 0:
        print("[WARN] No valid ids found for CLIP features, skip saving fused clip features.")
        return

    # 特征维度：优先从 SAM 的 clip_features 中取，否则从 GML 的第一个特征取
    feat_dim = None
    if sam_clip_feats is not None and sam_clip_feats.ndim == 2:
        feat_dim = sam_clip_feats.shape[1]
    else:
        for v in gml_feat_sum.values():
            feat_dim = v.shape[0]
            break

    if feat_dim is None:
        print("[WARN] No CLIP features for SAM nor GML, skip saving fused clip features.")
        return

    fused_feats = np.zeros((fused_max_id + 1, feat_dim), dtype=np.float32)

    # 1) SAM 部分：把 sam_clip_feats 的 [i] 挪到 offset 后的位置 i+sam_id_offset
    if sam_clip_feats is not None and sam_clip_feats.ndim == 2:
        N_sam = sam_clip_feats.shape[0] - 1  # index 0 是背景
        for sid in range(1, N_sam + 1):
            new_id = sid + sam_id_offset
            if new_id <= fused_max_id:
                fused_feats[new_id] = sam_clip_feats[sid]

    # 2) GML 部分：对每个 id 做平均并归一化
    for gid, feat_sum in gml_feat_sum.items():
        cnt = gml_feat_count.get(gid, 0)
        if cnt <= 0:
            continue
        avg_feat = feat_sum / float(cnt)
        norm = np.linalg.norm(avg_feat) + 1e-6
        fused_feats[gid] = (avg_feat / norm).astype(np.float32)

    np.save(fused_clip_path, fused_feats)
    print(f"[INFO] Saved fused CLIP features to {fused_clip_path}, shape={fused_feats.shape}")


if __name__ == "__main__":
    main()
