import os
import sys
import cv2
import torch
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
import json
from typing import Dict, Optional

import clip
from PIL import Image


# ===================== 全局参数 / Prompt 定义（方便以后统一进配置） =====================

# ---- CLIP 模型 ----
CLIP_MODEL_NAME = "ViT-B/32"

# 建筑类文本 prompt
CITY_TEXT_PROMPTS = [
    "a building",
    "a house",
    "a wall of a building",
    "a window of a building",
    "a door of a building",
    "a roof of a building",
    "a balcony of a building",
    "a facade of a building",
]

# 前景类文本 prompt
FORE_TEXT_PROMPTS = [
    # 树 / 植被
    "a tree",
    "a big tree",
    "a thin tree with branches",
    "vegetation",
    "bushes",
    "leaves and branches",
    "a street tree",
    "a tree with a building behind it",
    # 动态物体 / 道路元素
    "a person",
    "a car",
    "a bus",
    "a truck",
    "a traffic sign",
    "a pole",
    "a street lamp",
]

# ---- SAM + CLIP 相关默认阈值 ----
DEFAULT_SCALE = 0.25           # 图像缩放比例
DEFAULT_OVERLAP_CLIP_THRESH = 0.8
DEFAULT_CLIP_MARGIN = 0.05
DEFAULT_MIN_SAM_PIXELS = 50    # SAM 实例最小像素数

DEFAULT_CONFIDENCE_THRESHOLD = 0.5  # config.json 中没有时兜底

# ---- 可视化颜色相关 ----
COLOR_SEED = 0


# ===================== CLIP 辅助函数 =====================

def load_clip_model(device, model_name: str = CLIP_MODEL_NAME):
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    return model, preprocess


def encode_texts(model, device, texts):
    with torch.no_grad():
        tokens = clip.tokenize(texts).to(device)
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats  # [N, D]


def classify_sam_instance_with_clip(
    model,
    preprocess,
    device,
    image_rgb,       # H x W x 3, uint8, RGB
    mask_bool_np,    # H x W, bool (numpy)
    text_feats_city,
    text_feats_fore,
    margin: float = DEFAULT_CLIP_MARGIN,
) -> str:
    """
    用 CLIP 判断一个 SAM 实例更像建筑 (city) 还是前景 (fore)。
    margin > 0: 只有当前景相似度明显高于建筑时才判为 fore。
    """
    ys, xs = np.where(mask_bool_np)
    if ys.size == 0:
        return "city"  # 空实例直接当建筑扔掉

    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    # 稍微 pad 一点，别裁太紧
    H, W = image_rgb.shape[:2]
    pad = 4
    y1 = max(0, y1 - pad)
    x1 = max(0, x1 - pad)
    y2 = min(H - 1, y2 + pad)
    x2 = min(W - 1, x2 + pad)

    crop = image_rgb[y1:y2 + 1, x1:x2 + 1, :]
    mask_crop = mask_bool_np[y1:y2 + 1, x1:x2 + 1]
    crop = crop.copy()
    crop[~mask_crop] = 255  # 非实例区域涂白

    pil_img = Image.fromarray(crop)

    with torch.no_grad():
        clip_in = preprocess(pil_img).unsqueeze(0).to(device)
        img_feat = model.encode_image(clip_in)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        sim_city = (img_feat @ text_feats_city.T).max().item()
        sim_fore = (img_feat @ text_feats_fore.T).max().item()

    if sim_fore > sim_city + margin:
        return "fore"
    else:
        return "city"


# ===================== 可视化相关 =====================

def get_n_different_colors(n: int) -> np.ndarray:
    """生成 n 个随机颜色，用于可视化实例 mask。"""
    np.random.seed(COLOR_SEED)
    return np.random.randint(1, 256, (n, 3), dtype=np.uint8)


def visualize_mask(mask: np.ndarray) -> np.ndarray:
    """
    将实例标签图（0 表示背景，1..N 表示不同实例）转换为彩色图方便可视化。
    """
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    num_masks = int(mask.max())
    if num_masks == 0:
        return color_mask

    random_colors = get_n_different_colors(num_masks)
    for i in range(num_masks):
        color_mask[mask == (i + 1)] = random_colors[i]
    return color_mask


# ===================== SAM 模型构建 =====================

def get_seg_model(
    config: Dict,
    seg_method: str,
    device: str,
):
    """
    根据 config.json 中的配置，构建分割模型。
    所有与 SAM 相关的参数（encoder、checkpoint、points_per_side 等）
    统一从 config.json 中读取，不再从命令行传入。
    """
    if seg_method == "sam":
        from segment_anything import sam_model_registry

        # === 这里显式把 mask 目录加入 sys.path，方便从 mask/ 里导入 ===
        script_dir = os.path.dirname(os.path.abspath(__file__))
        mask_dir = os.path.join(script_dir, "mask")
        if mask_dir not in sys.path:
            sys.path.append(mask_dir)
        from automatic_mask_generator import SamAutomaticMaskGenerator

        sam = sam_model_registry[config["sam_encoder_version"]](
            checkpoint=config["sam_checkpoint_path"]
        ).to(device=device)

        # 全部从 config 中读，若缺失则给默认值
        pps = config.get("sam_num_points_per_side", 32)
        ppb = config.get("sam_num_points_per_batch", 64)
        piou = config.get("sam_pred_iou_threshold", 0.86)
        cnl = config.get("sam_crop_n_layers", 0)

        auto_sam = SamAutomaticMaskGenerator(
            sam,
            points_per_side=pps,
            points_per_batch=ppb,
            pred_iou_thresh=piou,
            crop_n_layers=cnl,
        )
        return auto_sam

    else:
        raise NotImplementedError("only sam seg_method is supported for now")


# ===================== SAM + CityGML + CLIP 前景筛选 =====================

def get_sam_mask(
    auto_sam,
    image: np.ndarray,                  # RGB, numpy
    confidence_threshold: float,
    city_mask: Optional[np.ndarray] = None,
    overlap_clip_thresh: float = DEFAULT_OVERLAP_CLIP_THRESH,
    # ==== CLIP 相关，可选 ====
    clip_model=None,
    clip_preprocess=None,
    clip_device: str = "cuda",
    text_feats_city=None,
    text_feats_fore=None,
    clip_margin: float = DEFAULT_CLIP_MARGIN,
    min_sam_pixels: int = DEFAULT_MIN_SAM_PIXELS,
    enable_clip: bool = True,          # 外部开关：是否启用 CLIP
) -> np.ndarray:
    """
    生成“前景实例分割图”，返回 numpy.uint16:
    0=背景/建筑，1..N=前景实例。
    """
    H, W = image.shape[:2]

    with torch.no_grad():
        mask_data = auto_sam.generate(image)

    # 假设 mask_data["masks"] 和 ["iou_preds"] 是 torch.Tensor 或可转成 tensor
    pred_masks = mask_data["masks"].float()   # [N, H, W]
    pred_scores = mask_data["iou_preds"]      # [N]

    # 统一 device（显式处理 CPU / GPU）
    pred_masks = pred_masks.to("cuda" if torch.cuda.is_available() else "cpu")
    pred_scores = pred_scores.to(pred_masks.device)
    device = pred_masks.device

    # 置信度筛选
    keep_idx = (pred_scores >= confidence_threshold)
    if keep_idx.sum().item() == 0:
        return np.zeros((H, W), dtype=np.uint16)

    pred_masks = pred_masks[keep_idx]  # [K, H, W]
    pred_scores = pred_scores[keep_idx]  # [K]

    # 二值 mask
    masks_bool = (pred_masks > 0.5)   # bool [K, H, W]

    # city mask → numpy bool；先不转 torch，等知道 device 后再转
    city_bin_np: Optional[np.ndarray] = None
    if city_mask is not None:
        if city_mask.ndim == 3:
            city_bin_np = (city_mask[..., :3].sum(axis=2) > 0)
        else:
            city_bin_np = (city_mask > 0)
        city_bin_np = city_bin_np.astype(np.bool_)

    # ====== 1) 向量化计算面积 + area 过滤 ======
    # areas: [K]
    areas = masks_bool.sum(dim=(1, 2))
    area_keep = (areas >= min_sam_pixels)
    if area_keep.sum().item() == 0:
        return np.zeros((H, W), dtype=np.uint16)

    masks_bool = masks_bool[area_keep]      # [K2, H, W]
    pred_scores = pred_scores[area_keep]    # [K2]
    areas = areas[area_keep]                # [K2]
    K2 = masks_bool.shape[0]

    # ====== 2) 向量化计算和 city_mask 的重叠比例 ======
    if city_bin_np is not None:
        city_bin_torch = torch.from_numpy(city_bin_np).to(device)  # [H, W], bool
        overlaps = (masks_bool & city_bin_torch).sum(dim=(1, 2))   # [K2]
        overlap_ratio = overlaps.float() / areas.float().clamp_min(1)
    else:
        city_bin_torch = None
        overlaps = torch.zeros_like(areas, dtype=torch.long, device=device)
        overlap_ratio = torch.zeros_like(areas, dtype=torch.float32, device=device)

    # 实际是否使用 CLIP
    use_clip = (
        enable_clip and
        (clip_model is not None) and
        (clip_preprocess is not None) and
        (text_feats_city is not None) and
        (text_feats_fore is not None)
    )

    # ====== 3) 决定哪些实例直接保留，哪些需要 CLIP，哪些直接丢弃 ======
    if city_bin_torch is None:
        # 没有 citygml → 全部直接保留，不用 CLIP
        keep_direct_mask = torch.ones(K2, dtype=torch.bool, device=device)
        needs_clip_mask = torch.zeros(K2, dtype=torch.bool, device=device)
    else:
        keep_direct_mask = (overlap_ratio < overlap_clip_thresh)          # overlap 小 → 直接保留
        needs_clip_mask = (overlap_ratio >= overlap_clip_thresh)          # overlap 大 → 需要 CLIP 或直接丢弃

    if not use_clip:
        # 不启用 CLIP 时，把 needs_clip 的全丢弃
        needs_clip_mask = torch.zeros_like(needs_clip_mask)

    idx_keep_direct = torch.nonzero(keep_direct_mask, as_tuple=False).flatten()
    idx_needs_clip = torch.nonzero(needs_clip_mask, as_tuple=False).flatten()

    kept_masks = []
    kept_scores = []

    # 3.1 直接保留的实例
    if idx_keep_direct.numel() > 0:
        for idx in idx_keep_direct.tolist():
            kept_masks.append(masks_bool[idx])
            kept_scores.append(pred_scores[idx])

    # 3.2 需要 CLIP 判定的实例（重叠高 & use_clip=True & 有 citygml）
    if use_clip and city_bin_torch is not None and idx_needs_clip.numel() > 0:
        image_rgb = image  # numpy RGB
        for idx in idx_needs_clip.tolist():
            m_tensor = masks_bool[idx]
            # 这里只在需要 CLIP 时才把 mask 转成 numpy，避免频繁来回
            m_np_bool = m_tensor.detach().to("cpu").numpy().astype(bool)
            sem = classify_sam_instance_with_clip(
                clip_model,
                clip_preprocess,
                clip_device,
                image_rgb,
                m_np_bool,
                text_feats_city,
                text_feats_fore,
                margin=clip_margin,
            )
            if sem == "fore":
                kept_masks.append(m_tensor)
                kept_scores.append(pred_scores[idx])

    if len(kept_masks) == 0:
        return np.zeros((H, W), dtype=np.uint16)

    kept_masks = torch.stack(kept_masks, dim=0)   # [K_fg, H, W], bool
    kept_scores = torch.stack(kept_scores, dim=0) # [K_fg]

    # ========= 4) 按置信度从低到高画，让高置信度最后覆盖 =========
    mask_id = torch.zeros((H, W), dtype=torch.int32, device=device)  # 临时 id

    # sort_idx: 置信度从低到高
    _, order = torch.sort(kept_scores, descending=False)

    for new_idx, idx in enumerate(order.tolist()):
        m_bool = kept_masks[idx]  # bool [H, W]，同一 device
        # new_idx+1 作为临时 id
        mask_id[m_bool] = int(new_idx + 1)

    # ========= 5) 压缩编号，去掉被覆盖得太厉害的实例 =========
    output_mask_t = torch.zeros((H, W), dtype=torch.int32, device=device)
    unique_ids = mask_id.unique()

    cur_id = 1
    for tmp_id in unique_ids.tolist():
        if tmp_id == 0:
            continue
        cur_mask = (mask_id == tmp_id)
        kept_area = int(cur_mask.sum().item())
        if kept_area < min_sam_pixels:
            continue
        output_mask_t[cur_mask] = cur_id
        cur_id += 1

    # 转 numpy uint16 输出
    output_mask = output_mask_t.to("cpu").numpy().astype(np.uint16)
    return output_mask


# ===================== 主流程 =====================

def run_segmentation(
    scene: str,
    seg_method: str = "sam",
    visualize: bool = False,
    use_gml: bool = False,
    use_clip: bool = False,
    scale: float = DEFAULT_SCALE,
) -> None:
    """
    主流程封装函数：
    - 从 <当前脚本同级>/dataset/<scene> 读取数据；
    - 从 images/ 读取输入图像；
    - 若 use_gml=True，则从 gml_mask/ 读取建筑掩码用于辅助前景筛选；
    - 若 use_clip=True，则加载 CLIP 并在与建筑高度重叠的实例上做语义筛选；
    - 使用 mask/config.json 中对应 seg_method 的配置构建模型并跑分割。
    """
    # 设备（注意：get_sam_mask 里也会处理 device，这里只是构建模型时使用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 当前脚本所在目录（现在和 mask/ 同级）
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # === 路径修改：dataset 在当前目录下，而不是 ../dataset ===
    dataset_root = os.path.normpath(os.path.join(script_dir, "dataset"))

    # 场景目录
    scene_folder = os.path.join(dataset_root, scene)
    assert os.path.exists(scene_folder), f"scene folder not found: {scene_folder}"

    # 输入图像目录：<dataset>/<scene>/images
    image_folder = os.path.join(scene_folder, "images")
    assert os.path.exists(image_folder), f"image folder not found: {image_folder}"

    # 输出目录：这里仍然叫 raw_<seg_method>_mask，但存的是 .npy
    output_folder = os.path.join(scene_folder, f"raw_{seg_method}_mask")
    os.makedirs(output_folder, exist_ok=True)

    vis_output_folder = None
    if visualize:
        vis_output_folder = os.path.join(scene_folder, f"raw_{seg_method}_mask_vis")
        os.makedirs(vis_output_folder, exist_ok=True)

    # === 路径修改：config.json 放在 mask/ 目录下 ===
    config_path = os.path.join(script_dir, "mask", "config.json")
    assert os.path.exists(config_path), f"config.json not found: {config_path}"

    full_config = json.load(open(config_path, "r"))
    assert seg_method in full_config, f"seg_method '{seg_method}' not found in config.json"
    config = full_config[seg_method]

    confidence_threshold = config.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)

    print(f"[INFO] loading {seg_method} model ...")
    seg_model = get_seg_model(
        config=config,
        seg_method=seg_method,
        device=device,
    )

    # ====== CLIP 相关：只有在 use_clip=True 时才加载 ======
    if use_clip:
        clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] loading CLIP model '{CLIP_MODEL_NAME}' on {clip_device} ...")
        clip_model, clip_preprocess = load_clip_model(clip_device)

        text_feats_city = encode_texts(clip_model, clip_device, CITY_TEXT_PROMPTS)
        text_feats_fore = encode_texts(clip_model, clip_device, FORE_TEXT_PROMPTS)
    else:
        print("[INFO] CLIP filtering disabled (use_clip = False).")
        clip_device = "cpu"  # 占个坑，不会真的用
        clip_model = None
        clip_preprocess = None
        text_feats_city = None
        text_feats_fore = None

    # citygml 目录：<dataset>/<scene>/gml_mask
    citygml_dir = None
    if use_gml:
        citygml_dir = os.path.join(scene_folder, "gml_mask")
        assert os.path.exists(citygml_dir), f"gml_mask dir not found: {citygml_dir}"
        print(f"[INFO] use citygml dir: {citygml_dir}")
    else:
        print("[INFO] GML filtering disabled (use_gml = False).")

    image_names = sorted(os.listdir(image_folder))
    print(f"[INFO] found {len(image_names)} files in {image_folder}")

    for image_name in tqdm(image_names):
        # 跳过非图像
        if not image_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            continue

        img_path = os.path.join(image_folder, image_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] cannot read image: {img_path}, skip.")
            continue

        h, w = img.shape[:2]
        if scale != 1.0:
            img_small = cv2.resize(
                img,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            img_small = img

        # SAM 期望 RGB 输入，OpenCV 是 BGR，这里需转换
        img_small_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

        # 直接按同名去 gml_mask 目录找
        city_small = None
        if citygml_dir is not None:
            city_path = os.path.join(
                citygml_dir,
                os.path.splitext(image_name)[0] + ".png"
            )
            if os.path.exists(city_path):
                # 使用 IMREAD_UNCHANGED 读取可能存在的透明通道
                city_img = cv2.imread(city_path, cv2.IMREAD_UNCHANGED)
                if city_img is not None:
                    if scale != 1.0:
                        city_small = cv2.resize(
                            city_img,
                            (int(w * scale), int(h * scale)),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    else:
                        city_small = city_img
            else:
                city_small = None

        mask_small = get_sam_mask(
            auto_sam=seg_model,
            image=img_small_rgb,
            confidence_threshold=confidence_threshold,
            city_mask=city_small,
            overlap_clip_thresh=DEFAULT_OVERLAP_CLIP_THRESH,
            clip_model=clip_model,
            clip_preprocess=clip_preprocess,
            clip_device=clip_device,
            text_feats_city=text_feats_city,
            text_feats_fore=text_feats_fore,
            clip_margin=DEFAULT_CLIP_MARGIN,
            min_sam_pixels=DEFAULT_MIN_SAM_PIXELS,
            enable_clip=use_clip,
        )

        if scale != 1.0:
            # mask_small 是 uint16 numpy，最近邻放大回原分辨率
            mask = cv2.resize(
                mask_small,
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.uint16)
        else:
            mask = mask_small.astype(np.uint16)

        # === 以 .npy 格式保存 mask，支持 >256 个类别 ===
        base_name = os.path.splitext(image_name)[0]
        save_path_npy = os.path.join(output_folder, base_name + ".npy")
        np.save(save_path_npy, mask)

        # 可视化仍然输出彩色 PNG
        if visualize and vis_output_folder is not None:
            vis_mask = visualize_mask(mask)          # RGB
            vis_path = os.path.join(
                vis_output_folder,
                base_name + ".png",
            )
            # OpenCV 期望 BGR
            cv2.imwrite(vis_path, vis_mask[:, :, ::-1])

    print("[INFO] segmentation finished.")


if __name__ == "__main__":
    parser = ArgumentParser()

    # 1) scene 名称
    parser.add_argument(
        "--scene",
        "-s",
        default="subset_building1_29_copy",
        type=str,
        help="scene name under dataset/, e.g. 'mipnerf360/room'",
    )

    # 2) 分割方法，目前只支持 sam，但保留参数以便以后扩展
    parser.add_argument(
        "--seg_method",
        "-m",
        default="sam",
        type=str,
        help="segmentation method, currently only 'sam' is supported",
    )

    # 3) 是否保存可视化的彩色 mask
    parser.add_argument(
        "--visualize",
        "-v",
        action="store_true",
        help="if set, also save colorful visualization of instance masks",
    )

    # 4) 是否启用 gml_mask 作为建筑掩码
    parser.add_argument(
        "--gml",
        "-g",
        action="store_true",
        help="if set, use dataset/<scene>/gml_mask masks to filter SAM results",
    )

    # 5) 是否启用 CLIP 语义筛选
    parser.add_argument(
        "--clip",
        "-c",
        action="store_true",
        help="if set, use CLIP to classify highly-overlapping SAM instances as foreground/building",
    )

    args = parser.parse_args()

    run_segmentation(
        scene=args.scene,
        seg_method=args.seg_method,
        visualize=args.visualize,
        use_gml=args.gml,
        use_clip=args.clip,
    )
