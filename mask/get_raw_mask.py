import os
import cv2
import torch
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
import json
from typing import Dict, Optional

import clip
from PIL import Image


# ===================== CLIP 辅助函数 =====================

def load_clip_model(device, model_name="ViT-B/32"):
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
    mask_bool,       # H x W, bool
    text_feats_city,
    text_feats_fore,
    margin: float = 0.02,
) -> str:
    """
    用 CLIP 判断一个 SAM 实例更像建筑 (city) 还是前景 (fore)。
    margin > 0: 只有当前景相似度明显高于建筑时才判为 fore。
    """
    ys, xs = np.where(mask_bool)
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
    mask_crop = mask_bool[y1:y2 + 1, x1:x2 + 1]
    crop = crop.copy()
    crop[~mask_crop] = 255  # 非实例区域涂白

    pil_img = Image.fromarray(crop)

    with torch.no_grad():
        clip_in = preprocess(pil_img).unsqueeze(0).to(device)
        img_feat = model.encode_image(clip_in)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        sim_city = (img_feat @ text_feats_city.T).max().item()
        sim_fore = (img_feat @ text_feats_fore.T).max().item()

    # 只有当前景明显强于建筑，才当 fore；否则 city
    if sim_fore > sim_city + margin:
        return "fore"
    else:
        return "city"


# ===================== 可视化相关 =====================

def get_n_different_colors(n: int) -> np.ndarray:
    """生成 n 个随机颜色，用于可视化实例 mask。"""
    np.random.seed(0)
    return np.random.randint(1, 256, (n, 3), dtype=np.uint8)


def visualize_mask(mask: np.ndarray) -> np.ndarray:
    """
    将实例标签图（0 表示背景，1..N 表示不同实例）转换为彩色图方便可视化。
    """
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    num_masks = np.max(mask)
    if num_masks == 0:
        return color_mask

    random_colors = get_n_different_colors(num_masks)
    for i in range(num_masks):
        color_mask[mask == i + 1] = random_colors[i]
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
    image: np.ndarray,                  # RGB
    confidence_threshold: float,
    city_mask: Optional[np.ndarray] = None,
    overlap_clip_thresh: float = 0.6,   # 重叠≥60% 才送 CLIP
    # ==== CLIP 相关，可选 ====
    clip_model=None,
    clip_preprocess=None,
    clip_device: str = "cuda",
    text_feats_city=None,
    text_feats_fore=None,
    clip_margin: float = 0.05,
    min_sam_pixels: int = 500,
    enable_clip: bool = True,          # 外部开关：是否启用 CLIP
) -> np.ndarray:
    """
    生成“前景实例分割图”：
    - 先用 SAM 生成实例；
    - 对每个实例，与 citygml 建筑掩码计算重叠比例：
        - overlap_ratio < overlap_clip_thresh (默认为 0.6)：认为建筑影响小 → 直接当前景保留（不送 CLIP）；
        - overlap_ratio >= overlap_clip_thresh 且 enable_clip=True：送入 CLIP 判断：
            - CLIP=前景 fore → 整个实例保留（包括盖在建筑上的部分）；
            - CLIP=建筑 city → 丢弃；
        - overlap_ratio >= overlap_clip_thresh 且 enable_clip=False：直接丢弃（当作建筑相关实例）；
    - 最终输出：HxW, uint16，0=背景/建筑，1..N=前景实例。
    """
    H, W = image.shape[:2]

    with torch.no_grad():
        mask_data = auto_sam.generate(image)

    pred_masks = mask_data["masks"].float()   # [N, H, W]
    pred_scores = mask_data["iou_preds"]      # [N]

    # 置信度筛选
    keep_idx = (pred_scores >= confidence_threshold)
    pred_masks = pred_masks[keep_idx]
    pred_scores = pred_scores[keep_idx]

    if pred_masks.numel() == 0:
        return np.zeros((H, W), dtype=np.uint16)

    K, m_H, m_W = pred_masks.shape
    assert (m_H, m_W) == (H, W), "SAM 输出尺寸要和输入 image 一致"

    masks_bool = (pred_masks > 0.5)  # [K, H, W]

    # city mask → bool
    if city_mask is not None:
        if city_mask.ndim == 3:
            city_bin = (city_mask[..., :3].sum(axis=2) > 0)
        else:
            city_bin = (city_mask > 0)
        city_bin = city_bin.astype(np.bool_)
    else:
        city_bin = None

    # 实际是否使用 CLIP：需要 enable_clip & 各种对象都存在
    use_clip = (
        enable_clip and
        (clip_model is not None) and
        (clip_preprocess is not None) and
        (text_feats_city is not None) and
        (text_feats_fore is not None)
    )

    kept_masks = []
    kept_scores = []

    image_rgb = image  # 已是 RGB

    for m_tensor, s_tensor in zip(masks_bool, pred_scores):
        m_np = m_tensor.cpu().numpy()
        sam_area = int(m_np.sum())
        if sam_area < min_sam_pixels:
            continue

        # 1) 先看与 citygml 的重叠比例
        if city_bin is not None:
            overlap = np.logical_and(m_np, city_bin).sum()
            overlap_ratio = overlap / float(sam_area)
        else:
            overlap_ratio = 0.0

        keep_this = False

        if (city_bin is None) or (overlap_ratio < overlap_clip_thresh):
            # 没有 citygml，或者与建筑重叠很少 → 直接当前景保留整个实例
            keep_this = True
        else:
            # 重叠≥阈值：如果不开 CLIP，直接丢；开了 CLIP 就用 CLIP 判定
            if not use_clip:
                keep_this = False
            else:
                sem = classify_sam_instance_with_clip(
                    clip_model,
                    clip_preprocess,
                    clip_device,
                    image_rgb,
                    m_np,
                    text_feats_city,
                    text_feats_fore,
                    margin=clip_margin,
                )
                if sem == "fore":
                    keep_this = True
                else:
                    keep_this = False  # 建筑类实例，丢掉

        if keep_this:
            kept_masks.append(m_tensor.float())
            kept_scores.append(s_tensor)

    if len(kept_masks) == 0:
        return np.zeros((H, W), dtype=np.uint16)

    kept_masks = torch.stack(kept_masks, dim=0)   # [K_fg, H, W]
    kept_scores = torch.stack(kept_scores, dim=0) # [K_fg]

    # ========= 按置信度从低到高画，让高置信度最后覆盖 =========
    mask_id = np.zeros((H, W), dtype=np.int32)

    # sort_idx: 置信度从低到高
    _, order = torch.sort(kept_scores, descending=False)

    for new_idx, idx in enumerate(order):
        idx_int = int(idx)
        m_np = (kept_masks[idx_int] > 0.5).cpu().numpy()
        # new_idx+1 作为临时 id（只是区分用，后面会压缩重排）
        mask_id[m_np] = new_idx + 1

    # ========= 压缩编号，去掉被覆盖得太厉害的实例 =========
    output_mask = np.zeros((H, W), dtype=np.uint16)
    cur_id = 1
    unique_ids = np.unique(mask_id)

    for tmp_id in unique_ids:
        if tmp_id == 0:
            continue
        cur_mask = (mask_id == tmp_id)
        kept_area = int(cur_mask.sum())
        if kept_area < min_sam_pixels:
            continue
        output_mask[cur_mask] = cur_id
        cur_id += 1

    return output_mask


# ===================== 主流程 =====================

def run_segmentation(
    scene: str,
    seg_method: str = "sam",
    visualize: bool = False,
    use_gml: bool = False,
    use_clip: bool = False,
    scale: float = 0.25,
) -> None:
    """
    主流程封装函数：
    - 自动从当前脚本的上一层目录中的 dataset/<scene> 读取数据；
    - 从 images/ 读取输入图像；
    - 若 use_gml=True，则从 gml_mask/ 读取建筑掩码用于辅助前景筛选；
    - 若 use_clip=True，则加载 CLIP 并在与建筑高度重叠的实例上做语义筛选；
    - 使用 config.json 中对应 seg_method 的配置构建模型并跑分割。
    """
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # dataset 根目录：脚本上一层目录的 dataset 文件夹
    dataset_root = os.path.normpath(os.path.join(script_dir, "..", "dataset"))

    # 场景目录
    scene_folder = os.path.join(dataset_root, scene)
    assert os.path.exists(scene_folder), f"scene folder not found: {scene_folder}"

    # 输入图像目录：<dataset>/<scene>/images
    image_folder = os.path.join(scene_folder, "images")
    assert os.path.exists(image_folder), f"image folder not found: {image_folder}"

    # 输出目录
    output_folder = os.path.join(scene_folder, f"raw_{seg_method}_mask")
    os.makedirs(output_folder, exist_ok=True)

    vis_output_folder = None
    if visualize:
        vis_output_folder = os.path.join(scene_folder, f"raw_{seg_method}_mask_vis")
        os.makedirs(vis_output_folder, exist_ok=True)

    # 加载配置
    config_path = os.path.join(script_dir, "config.json")
    assert os.path.exists(config_path), f"config.json not found: {config_path}"

    full_config = json.load(open(config_path, "r"))
    assert seg_method in full_config, f"seg_method '{seg_method}' not found in config.json"
    config = full_config[seg_method]

    print(f"[INFO] loading {seg_method} model ...")
    seg_model = get_seg_model(
        config=config,
        seg_method=seg_method,
        device=device,
    )

    # ====== CLIP 相关：只有在 use_clip=True 时才加载 ======
    if use_clip:
        clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] loading CLIP model on {clip_device} ...")
        clip_model, clip_preprocess = load_clip_model(clip_device)

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
            # 基本前景类
            "a tree",
            "a big tree",
            "a thin tree with branches",
            "vegetation",
            "bushes",
            "leaves and branches",
            "a street tree",
            "a tree with a building behind it",

            "a person",
            "a car",
            "a bus",
            "a truck",
            "a traffic sign",
            "a pole",
            "a street lamp",
        ]


        text_feats_city = encode_texts(clip_model, clip_device, city_texts)
        text_feats_fore = encode_texts(clip_model, clip_device, fore_texts)
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
            img_small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            img_small = img

        # SAM 期望 RGB 输入，OpenCV 是 BGR，这里需转换
        img_small_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

        # 直接按同名去 gml_mask 目录找
        city_small = None
        if citygml_dir is not None:
            city_path = os.path.join(citygml_dir, os.path.splitext(image_name)[0] + ".png")
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
            confidence_threshold=config["confidence_threshold"],
            city_mask=city_small,
            overlap_clip_thresh=0.8,
            clip_model=clip_model,
            clip_preprocess=clip_preprocess,
            clip_device=clip_device,
            text_feats_city=text_feats_city,
            text_feats_fore=text_feats_fore,
            clip_margin=0.05,
            min_sam_pixels=50,
            enable_clip=use_clip,
        )

        if scale != 1.0:
            mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            mask = mask_small

        save_path = os.path.join(output_folder, os.path.splitext(image_name)[0] + ".png")
        cv2.imwrite(save_path, mask)

        if visualize and vis_output_folder is not None:
            vis_mask = visualize_mask(mask)
            vis_path = os.path.join(vis_output_folder, os.path.splitext(image_name)[0] + ".png")
            cv2.imwrite(vis_path, vis_mask)


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
