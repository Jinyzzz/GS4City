import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import clip
from PIL import Image


# ==== 你可以在这里直接改默认路径 ====
DEFAULT_IMAGES_DIR    = "/workspace/Gaga/dataset/subset_building1_16_gml/images"
DEFAULT_CITY_DIR      = "/workspace/Gaga/dataset/subset_building1_16_gml/sam_mask"
DEFAULT_SAM_DIR       = "/workspace/Gaga/dataset/subset_building1_16/sam_mask"
DEFAULT_OUT_GRAY_DIR  = "/workspace/Gaga/dataset/subset_building1_16_fused/sam_mask"
DEFAULT_OUT_RGB_DIR   = "/workspace/Gaga/dataset/subset_building1_16_fused/sam_mask_vis"
# ====================================


def load_clip_model(device, model_name="ViT-B/32"):
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    return model, preprocess


def encode_texts(model, device, texts):
    with torch.no_grad():
        text_tokens = clip.tokenize(texts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features  # [N, D]


def classify_sam_instance(
    model,
    preprocess,
    device,
    image_rgb,
    mask_bool,
    text_feats_city,
    text_feats_fore,
):
    """用 CLIP 判断一个 SAM 实例更像建筑还是前景"""
    ys, xs = np.where(mask_bool)
    if ys.size == 0:
        return "fore"

    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    crop = image_rgb[y1: y2 + 1, x1: x2 + 1, :]
    mask_crop = mask_bool[y1: y2 + 1, x1: x2 + 1]
    crop = crop.copy()
    crop[~mask_crop] = 255  # 把非mask部分涂白

    pil_img = Image.fromarray(crop)
    with torch.no_grad():
        clip_in = preprocess(pil_img).unsqueeze(0).to(device)
        img_feat = model.encode_image(clip_in)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        sim_city = (img_feat @ text_feats_city.T).max().item()
        sim_fore = (img_feat @ text_feats_fore.T).max().item()

    # ✅ 给前景一点优势，防止全部被分成 city
    if sim_fore >= sim_city - 0.03:
        return "fore"
    else:
        return "city"


def id_to_color(idx: int):
    """
    给定一个最终的实例 id（uint16），生成一个确定性的 RGB 颜色。
    这样同一个 id 在不同视角生成的颜色是一致的。
    """
    h = (idx * 2654435761) & 0xFFFFFFFF
    r = (h & 0xFF)
    g = ((h >> 8) & 0xFF)
    b = ((h >> 16) & 0xFF)
    if r == 0 and g == 0 and b == 0:
        r = 50
    return (r, g, b)


def fuse_one_image(
    img_path: Path,
    city_path: Path,
    sam_path: Path,
    out_gray_path: Path,
    out_rgb_path: Path,
    model,
    preprocess,
    device,
    text_feats_city,
    text_feats_fore,
    min_sam_pixels=50,
):
    # 读取图像
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] cannot read image: {img_path}")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 读取两个 mask
    city = cv2.imread(str(city_path), cv2.IMREAD_UNCHANGED)
    sam = cv2.imread(str(sam_path), cv2.IMREAD_UNCHANGED)

    if city is None or sam is None:
        print(f"[WARN] cannot read masks for {img_path.name}")
        return

    # ---------- 尺寸对齐：选尺寸更小的那个 ----------
    h_city, w_city = city.shape[:2]
    h_sam, w_sam = sam.shape[:2]

    target_h = min(h_city, h_sam)
    target_w = min(w_city, w_sam)
    target_size = (target_w, target_h)  # cv2 是 (w, h)

    if (h_city, w_city) != (target_h, target_w):
        city = cv2.resize(city, target_size, interpolation=cv2.INTER_NEAREST)
    if (h_sam, w_sam) != (target_h, target_w):
        sam = cv2.resize(sam, target_size, interpolation=cv2.INTER_NEAREST)
    if img_rgb.shape[:2] != (target_h, target_w):
        img_rgb = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LINEAR)
    # ---------- 尺寸对齐结束 ----------

    H, W = city.shape[:2]

    # 输出图：灰度 uint16
    fused = np.zeros((H, W), dtype=np.uint16)
    # 输出图：RGB 可视化
    vis = np.zeros((H, W, 3), dtype=np.uint8)

    # 先把所有 sam 实例识别出来
    sam_ids = np.unique(sam)
    sam_ids = sam_ids[sam_ids != 0]  # 去掉背景

    # 本图中 city 已经占用的 id
    city_ids = np.unique(city)
    city_ids = set(int(x) for x in city_ids if x != 0)

    # 每个 sam id 的语义分类
    sam_id_to_sem = {}
    for sid in sam_ids:
        mask_bool = (sam == sid)
        if mask_bool.sum() < min_sam_pixels:
            sam_id_to_sem[sid] = "fore"
            continue

        sem = classify_sam_instance(
            model,
            preprocess,
            device,
            img_rgb,
            mask_bool,
            text_feats_city,
            text_feats_fore,
        )
        sam_id_to_sem[sid] = sem

    # ✅ 给所有 SAM 实例分配“新的、不和 city 冲突的”ID，从 1 开始
    sam_id_old2new = {}
    next_free_id = 1
    for sid in sam_ids:
        # 找到一个不在 city_ids 里的下一个 id
        while next_free_id in city_ids:
            next_free_id += 1
        sam_id_old2new[sid] = next_free_id
        next_free_id += 1

    # 开始按像素融合
    for y in range(H):
        for x in range(W):
            c = int(city[y, x])
            s = int(sam[y, x])

            if c != 0 and s != 0:
                # 两边都有，看 sam 被判成啥
                sem = sam_id_to_sem.get(s, "fore")
                if sem == "city":
                    final_id = c  # 保留 citygml id
                else:
                    final_id = sam_id_old2new[s]  # 用重新分配的、不冲突的 SAM id
            elif c != 0 and s == 0:
                # 只有 city
                final_id = c
            elif c == 0 and s != 0:
                # 只有 sam，无论 city/fore 判定，只能用 sam 的新 id
                final_id = sam_id_old2new[s]
            else:
                final_id = 0  # 背景

            fused[y, x] = final_id

            if final_id == 0:
                vis[y, x] = (0, 0, 0)
            else:
                r, g, b = id_to_color(final_id)
                vis[y, x] = (b, g, r)

    # 保存
    out_gray_path.parent.mkdir(parents=True, exist_ok=True)
    out_rgb_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_gray_path), fused)
    cv2.imwrite(str(out_rgb_path), vis)


def main():
    parser = argparse.ArgumentParser(
        description="Fuse CityGML and SAM masks using CLIP semantic guidance, "
                    "keep CityGML IDs, remap SAM IDs to start from 1 and avoid CityGML IDs, output uint16 gray + RGB"
    )

    parser.add_argument("--images_dir", default=DEFAULT_IMAGES_DIR, help="Path to images directory")
    parser.add_argument("--city_dir",   default=DEFAULT_CITY_DIR,   help="Path to CityGML masks directory")
    parser.add_argument("--sam_dir",    default=DEFAULT_SAM_DIR,    help="Path to SAM masks directory")
    parser.add_argument("--out_gray_dir", default=DEFAULT_OUT_GRAY_DIR, help="Path to save fused gray masks (uint16)")
    parser.add_argument("--out_rgb_dir",  default=DEFAULT_OUT_RGB_DIR,  help="Path to save fused RGB visualization")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min_sam_pixels", type=int, default=50)
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[INFO] cuda not available, fallback to cpu")
        device = "cpu"

    print(f"[INFO] Loading CLIP model on {device} ...")
    model, preprocess = load_clip_model(device)

    city_texts = [
        "a building",
        "a house",
        "a wall of a building",
        "a window of a building",
        "a door of a building",
        "a roof of a building",
        "a balcony of a building",
        "a facade of a building",
    ]
    fore_texts = [
        "a person",
        "a car",
        "a bus",
        "a truck",
        "a tree",
        "vegetation",
        "a traffic sign",
        "a pole",
    ]

    text_feats_city = encode_texts(model, device, city_texts)
    text_feats_fore = encode_texts(model, device, fore_texts)

    img_dir = Path(args.images_dir)
    city_dir = Path(args.city_dir)
    sam_dir = Path(args.sam_dir)
    out_gray_dir = Path(args.out_gray_dir)
    out_rgb_dir = Path(args.out_rgb_dir)

    img_files = sorted(img_dir.glob("*.*"))
    print(f"[INFO] Found {len(img_files)} images to process")

    def find_mask(base_dir, stem):
        for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
            p = base_dir / f"{stem}{ext}"
            if p.exists():
                return p
        return None

    for img_path in img_files:
        name = img_path.name
        stem = img_path.stem

        city_path = find_mask(city_dir, stem)
        sam_path = find_mask(sam_dir, stem)
        gray_out_path = out_gray_dir / f"{stem}.png"
        rgb_out_path  = out_rgb_dir / f"{stem}.png"

        if city_path is None or sam_path is None:
            print(f"[WARN] Missing masks for {name}, skipping")
            continue

        print(f"[INFO] Fusing {name}")
        fuse_one_image(
            img_path,
            city_path,
            sam_path,
            gray_out_path,
            rgb_out_path,
            model,
            preprocess,
            device,
            text_feats_city,
            text_feats_fore,
            min_sam_pixels=args.min_sam_pixels,
        )


if __name__ == "__main__":
    main()
