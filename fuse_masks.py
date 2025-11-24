import os
import cv2
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm


def compute_global_gml_max_id(gml_dir):
    """
    在整个 gml_mask 文件夹中，统计所有图像中的最大 ID（>0）。
    若目录为空或都为 0，则返回 0。
    """
    max_id = 0
    if not os.path.exists(gml_dir):
        print(f"[WARN] gml_mask dir not found: {gml_dir}")
        return 0

    files = sorted(os.listdir(gml_dir))
    for name in files:
        if not name.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
            continue
        path = os.path.join(gml_dir, name)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARN] cannot read gml mask: {path}")
            continue
        cur_max = int(np.max(img))
        if cur_max > max_id:
            max_id = cur_max

    print(f"[INFO] Global max GML id in {gml_dir}: {max_id}")
    return max_id


def id_to_color(idx: int):
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


def fuse_one_image(
    gml_path,
    sam_path,
    out_gray_path,
    out_vis_path,
    sam_id_offset: int,
):
    """
    只根据灰度 mask 融合：
    - 像素优先 sam_mask（非0即用）
    - sam 为0时，用 gml
    - GML 的 ID 保持不变
    - SAM 的 ID 全部加上 sam_id_offset，避免和 gml 冲突

    同时生成一张彩色可视化图，保存到 out_vis_path。
    """
    gml = cv2.imread(gml_path, cv2.IMREAD_UNCHANGED)
    sam = cv2.imread(sam_path, cv2.IMREAD_UNCHANGED)

    if gml is None:
        print(f"[WARN] cannot read gml mask: {gml_path}")
        return
    if sam is None:
        print(f"[WARN] cannot read sam mask: {sam_path}")
        return

    # ---------- 尺寸对齐：选更小的一边 ----------
    h_gml, w_gml = gml.shape[:2]
    h_sam, w_sam = sam.shape[:2]

    target_h = min(h_gml, h_sam)
    target_w = min(w_gml, w_sam)
    target_size = (target_w, target_h)  # cv2 是 (w, h)

    if (h_gml, w_gml) != (target_h, target_w):
        gml = cv2.resize(gml, target_size, interpolation=cv2.INTER_NEAREST)
    if (h_sam, w_sam) != (target_h, target_w):
        sam = cv2.resize(sam, target_size, interpolation=cv2.INTER_NEAREST)

    H, W = gml.shape[:2]

    # SAM ID 整体偏移，避免和 GML ID 冲突
    sam = sam.astype(np.uint32)  # 防止加偏移溢出
    sam_nonzero = sam != 0
    sam[sam_nonzero] = sam[sam_nonzero] + sam_id_offset

    # 融合：优先 sam_mask
    fused = np.zeros((H, W), dtype=np.uint16)

    # sam 非0的像素 → 用 sam（已经偏移）
    fused[sam != 0] = sam[sam != 0].astype(np.uint16)
    # sam 为0 的像素 → 用 gml
    fused[(sam == 0) & (gml != 0)] = gml[(sam == 0) & (gml != 0)].astype(np.uint16)

    # ---------- 生成彩色可视化图 ----------
    vis = np.zeros((H, W, 3), dtype=np.uint8)
    unique_ids = np.unique(fused)
    unique_ids = unique_ids[unique_ids != 0]  # 跳过背景

    for uid in unique_ids:
        r, g, b = id_to_color(int(uid))
        # OpenCV 是 BGR
        vis[fused == uid] = (b, g, r)

    # 保存
    os.makedirs(os.path.dirname(out_gray_path), exist_ok=True)
    os.makedirs(os.path.dirname(out_vis_path), exist_ok=True)

    cv2.imwrite(out_gray_path, fused)
    cv2.imwrite(out_vis_path, vis)


def main():
    parser = ArgumentParser(
        description="Fuse GML and SAM instance masks for a scene. "
                    "Pixel-wise priority: SAM > GML. "
                    "GML IDs are kept, SAM IDs are offset to avoid conflicts. "
                    "Also output colorful visualization."
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
    # dataset 就在脚本所在目录下
    dataset_root = os.path.join(script_dir, "dataset")

    scene_folder = os.path.join(dataset_root, args.scene)
    assert os.path.exists(scene_folder), f"scene folder not found: {scene_folder}"

    gml_dir = os.path.join(scene_folder, "gml_mask")
    sam_dir = os.path.join(scene_folder, "sam_mask")
    out_gray_dir = os.path.join(scene_folder, "fused_mask")
    out_vis_dir = os.path.join(scene_folder, "fused_mask_vis")

    assert os.path.exists(gml_dir), f"gml_mask folder not found: {gml_dir}"
    assert os.path.exists(sam_dir), f"sam_mask folder not found: {sam_dir}"
    os.makedirs(out_gray_dir, exist_ok=True)
    os.makedirs(out_vis_dir, exist_ok=True)

    print(f"[INFO] scene folder: {scene_folder}")
    print(f"[INFO] gml_mask dir: {gml_dir}")
    print(f"[INFO] sam_mask dir: {sam_dir}")
    print(f"[INFO] gray output dir : {out_gray_dir}")
    print(f"[INFO] vis  output dir : {out_vis_dir}")

    # 先在整个 gml_mask 下统计最大 ID，给 SAM 做全局偏移
    global_max_gml_id = compute_global_gml_max_id(gml_dir)
    sam_id_offset = global_max_gml_id

    # 遍历 sam_mask 文件夹，用相同文件名在 gml_mask 里找
    sam_files = sorted(os.listdir(sam_dir))
    print(f"[INFO] Found {len(sam_files)} SAM mask files")

    for name in tqdm(sam_files, desc="Fusing masks"):
        if not name.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
            continue

        stem, ext = os.path.splitext(name)
        sam_path = os.path.join(sam_dir, name)

        # gml 同名
        gml_path = os.path.join(gml_dir, stem + ".png")
        if not os.path.exists(gml_path):
            # 如果确实没有对应的 gml，就构造一个全0图（按 sam 尺寸）
            print(f"[WARN] GML mask missing for {name}, use empty GML.")
            sam_img = cv2.imread(sam_path, cv2.IMREAD_UNCHANGED)
            if sam_img is None:
                print(f"[WARN] cannot read SAM mask: {sam_path}, skip.")
                continue
            H, W = sam_img.shape[:2]
            gml = np.zeros((H, W), dtype=np.uint16)
            # 临时文件只在本次调用中使用，不会被再次读取
            tmp_gml_path = os.path.join(out_gray_dir, "__tmp_empty_gml__.png")
            cv2.imwrite(tmp_gml_path, gml)
            gml_path_use = tmp_gml_path
        else:
            gml_path_use = gml_path

        out_gray_path = os.path.join(out_gray_dir, stem + ".png")
        out_vis_path = os.path.join(out_vis_dir, stem + ".png")

        fuse_one_image(
            gml_path_use,
            sam_path,
            out_gray_path,
            out_vis_path,
            sam_id_offset=sam_id_offset,
        )

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
