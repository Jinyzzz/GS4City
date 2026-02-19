import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torchvision
import json
from PIL import Image
from typing import Optional
import clip

from arguments import ModelParams, PipelineParams
from mask.utils import (
    get_n_different_colors,
    ndc2Pixel,
    transformPoint4x4,
    convert_matched_mask,
    mask_id_to_binary_mask,
)

from scene import Scene, GaussianModel


# ===================== 配置相关 =====================
CLIP_MODEL_NAME = "ViT-B/32"  # 这个不是超参，只是模型名

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
CONFIG_SECTION = "projector"

def _load_default_projector_params() -> dict:
    """
    从 mask/config.json 的 "projector" 段读取默认参数。
    不包含 visualize / use_clip，这两个必须由用户输入控制。
    """
    assert os.path.exists(CONFIG_PATH), f"config.json not found at {CONFIG_PATH}"
    with open(CONFIG_PATH, "r") as f:
        full_cfg = json.load(f)
    assert CONFIG_SECTION in full_cfg, f"`{CONFIG_SECTION}` section not found in config.json"
    cfg = full_cfg[CONFIG_SECTION]
    assert isinstance(cfg, dict), f"`{CONFIG_SECTION}` in config.json must be a dict"
    return cfg


class GaussianProjector(torch.nn.Module):
    def __init__(
        self,
        dataset: ModelParams,
        pipeline: PipelineParams,
        iteration: int,
        params: Optional[dict] = None,
        device: torch.device = torch.device("cuda"),
    ):
        """
        params:
          - None  : 完全使用 config.json["projector"] 的默认参数（不含 visualize/use_clip）
          - dict  : 在默认参数基础上覆盖对应键；常见用法是传入:
                    {"visualize": bool, "use_clip": bool}
        """
        super(GaussianProjector, self).__init__()
        self.device = device

        # ===== 1) 读 config 默认参数 =====
        base_params = _load_default_projector_params()  # 不包含 visualize/use_clip

        # ===== 2) 用用户传入的覆盖（例如 visualize / use_clip） =====
        if params is not None:
            base_params.update(params)
        params = base_params

        # ===== 3) 超参数全从 params 里拿 =====
        # Load pre-trained Gaussians and cameras
        self.gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, self.gaussians, load_iteration=iteration, shuffle=False)
        self.gaussians_xyz = self.gaussians.get_xyz.to(self.device)
        self.viewpoint_camera = scene.getTrainCameras()
        
        print("\n[DEBUG] ===== dataset/source_path check =====")
        print("[DEBUG] dataset.source_path =", dataset.source_path)
        print("[DEBUG] images dir expected =", os.path.join(dataset.source_path, "images"))
        try:
            img_dir = os.path.join(dataset.source_path, "images")
            if os.path.isdir(img_dir):
                files = sorted(os.listdir(img_dir))
                print("[DEBUG] images file count =", len(files))
                print("[DEBUG] images sample =", files[:5])
            else:
                print("[DEBUG] images dir does NOT exist")
        except Exception as e:
            print("[DEBUG] failed to list images dir:", e)

        print("[DEBUG] train cameras returned =", len(self.viewpoint_camera))
        if len(self.viewpoint_camera) > 0:
            v0 = self.viewpoint_camera[0]
            print("[DEBUG] first camera image_name =", getattr(v0, "image_name", None))
        print("[DEBUG] =====================================\n")


        # 几何/分块相关
        self.front_percentage = float(params["front_percentage"])
        self.iou_threshold = float(params["iou_threshold"])
        self.num_patches = int(params["num_patch"])
        self.max_group_distance = float(params.get("max_group_distance", 0.0))

        # CLIP 相关（只看用户是否传了 use_clip，没有就默认为 False）
        self.use_clip = bool(params.get("use_clip", False))
        self.clip_sim_threshold = float(params["clip_sim_threshold"])
        self.min_views = int(params["min_views"])
        if self.use_clip:
            self.clip_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.clip_model, self.clip_preprocess = clip.load(CLIP_MODEL_NAME, device=self.clip_device)
            self.clip_model.eval()
        else:
            self.clip_device = torch.device("cpu")
            self.clip_model = None
            self.clip_preprocess = None
        self._warned_no_image = False

        # Paths
        self.source_path = dataset.source_path
        self.seg_method = params["seg_method"]
        self.raw_mask_folder = os.path.join(self.source_path, f"raw_{self.seg_method}_mask")
        assert os.path.exists(self.raw_mask_folder), "Mask folder does not exist."
        self.associated_mask_folder = os.path.join(self.source_path, f"{self.seg_method}_mask")
        os.makedirs(self.associated_mask_folder, exist_ok=True)

        # 是否可视化：只看用户传没传 visualize，没传就 False
        visualize_flag = bool(params.get("visualize", False))
        if visualize_flag:
            self.visualize = True
            self.visualize_folder = os.path.join(self.source_path, f"{self.seg_method}_mask_vis")
            os.makedirs(self.visualize_folder, exist_ok=True)
            self.random_colors = None  # 延迟到可视化时按需生成
        else:
            self.visualize = False


        # For mask partition
        random_mask = self.load_mask(self.viewpoint_camera[0])
        self.image_width, self.image_height = random_mask.shape[1], random_mask.shape[2]

        self.patch_width = (
            self.image_width // self.num_patches + 1
            if self.image_width % self.num_patches != 0
            else self.image_width // self.num_patches
        )
        self.patch_height = (
            self.image_height // self.num_patches + 1
            if self.image_height % self.num_patches != 0
            else self.image_height // self.num_patches
        )

        self.patch_mask = torch.zeros(
            (self.num_patches, self.num_patches, self.image_width, self.image_height),
            dtype=torch.bool,
            device=self.device,
        )
        for i in range(self.num_patches):
            for j in range(self.num_patches):
                self.patch_mask[i, j,
                                i * self.patch_width: (i + 1) * self.patch_width,
                                j * self.patch_height: (j + 1) * self.patch_height] = True
        self.flatten_patch_mask = self.patch_mask.flatten(start_dim=2)

        # For mask association
        self.gaussian_idx_bank = []      # list[Tensor]，每个全局 mask 对应的一组高斯 index
        self.num_mask = 0
        # We don't want the same Gaussian to be assigned to multiple masks
        self.assigned_gaussians = []

        # CLIP：为每个全局 mask 维护平均特征 + 计数
        self.mask_clip_feats = []        # list[np.ndarray or None]，index = 全局 mask id
        self.mask_clip_counts = []       # list[int]
        self.global_clip_features = None # 后处理时用，[num_mask+1, D]，0 行为背景

    @property
    def get_num_mask(self):
        if len(self.gaussian_idx_bank) == 0:
            self.num_mask = 0
            return 0
        self.num_mask = len(self.gaussian_idx_bank)
        return self.num_mask

    def load_image_for_viewpoint(self, viewpoint):
        """
        尝试从 <source_path>/images 下加载与 viewpoint.image_name 对应的 RGB 图像。
        """
        image_dir = os.path.join(self.source_path, "images")
        base = viewpoint.image_name
        exts = [".png", ".PNG",
                ".jpg", ".JPG",
                ".jpeg", ".JPEG",
                ".bmp", ".BMP"]

        for ext in exts:
            img_path = os.path.join(image_dir, base + ext)
            if os.path.exists(img_path):
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is not None:
                    return img
        if self.use_clip and not self._warned_no_image:
            print(f"[GaussianProjector] Warning: cannot find image for {base} in {image_dir}, "
                  f"CLIP will be disabled for this view.")
            self._warned_no_image = True
        return None

    def compute_mask_clip_feature(self, rgb_image_bgr: np.ndarray, mask_bool: torch.Tensor):
        """
        输入 BGR 图像和单个实例的 2D bool mask（H x W, torch.bool），输出 CLIP L2-normalized feature。
        注意：这里只接受 [H, W]，调用前请保证 transpose 过。
        """
        if not self.use_clip or self.clip_model is None:
            return None

        mask_np = mask_bool.detach().to("cpu").numpy().astype(bool)

        # 必须是 2D
        if mask_np.ndim != 2:
            raise ValueError(f"mask for CLIP must be 2D, got shape {mask_np.shape}")

        # 全空就不用 CLIP 了
        if not mask_np.any():
            return None

        H, W = mask_np.shape
        ys, xs = np.where(mask_np)
        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()

        pad = 4
        y1 = max(0, y1 - pad)
        x1 = max(0, x1 - pad)
        y2 = min(H - 1, y2 + pad)
        x2 = min(W - 1, x2 + pad)

        img_rgb = cv2.cvtColor(rgb_image_bgr, cv2.COLOR_BGR2RGB)
        crop = img_rgb[y1:y2 + 1, x1:x2 + 1, :].copy()
        m_crop = mask_np[y1:y2 + 1, x1:x2 + 1]

        # 裁剪出来为空，直接放弃 CLIP，防止 boolean index 报错
        if crop.size == 0 or m_crop.size == 0:
            return None

        crop[~m_crop] = 255

        pil_img = Image.fromarray(crop)

        with torch.no_grad():
            inp = self.clip_preprocess(pil_img).unsqueeze(0).to(self.clip_device)
            feat = self.clip_model.encode_image(inp)
            feat = feat / feat.norm(dim=-1, keepdim=True)

        feat_np = feat[0].detach().cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(feat_np) + 1e-6
        return feat_np / norm

    def maintain_gaussian_idx_bank(self, idx, front_gaussian_of_mask, clip_feat=None):
        """
        同时维护：
        - gaussian_idx_bank / assigned_gaussians（原逻辑）
        - mask_clip_feats / mask_clip_counts（新增：平均 CLIP 特征）
        """
        assert not idx > self.num_mask, "idx is larger than the number of masks"

        # ========== 新的全局 mask ==========
        if idx == self.num_mask:
            self.gaussian_idx_bank.append(front_gaussian_of_mask)

            if isinstance(self.assigned_gaussians, list) and len(self.assigned_gaussians) == 0:
                self.assigned_gaussians = front_gaussian_of_mask
            else:
                self.assigned_gaussians = torch.unique(
                    torch.cat([self.assigned_gaussians, front_gaussian_of_mask])
                )

            # CLIP：初始化该组团的平均特征
            if clip_feat is not None:
                self.mask_clip_feats.append(clip_feat.astype(np.float32))
                self.mask_clip_counts.append(1)
            else:
                self.mask_clip_feats.append(None)
                self.mask_clip_counts.append(0)

        # ========== 向已有的全局 mask 追加 ==========
        else:
            non_assigned_gaussians = torch.unique(
                front_gaussian_of_mask[~torch.isin(front_gaussian_of_mask, self.assigned_gaussians)]
            )
            if non_assigned_gaussians.numel() > 0:
                self.gaussian_idx_bank[idx] = torch.unique(
                    torch.cat([self.gaussian_idx_bank[idx], non_assigned_gaussians])
                )
                self.assigned_gaussians = torch.unique(
                    torch.cat([self.assigned_gaussians, non_assigned_gaussians])
                )

            # CLIP：更新该组团的平均特征
            if clip_feat is not None:
                feat = clip_feat.astype(np.float32)
                norm = np.linalg.norm(feat) + 1e-6
                feat = feat / norm

                cur_feat = self.mask_clip_feats[idx]
                cur_count = self.mask_clip_counts[idx] if idx < len(self.mask_clip_counts) else 0

                if cur_feat is None or cur_count == 0:
                    self.mask_clip_feats[idx] = feat
                    if idx >= len(self.mask_clip_counts):
                        # 理论上不会发生，但保险
                        self.mask_clip_counts.extend([0] * (idx + 1 - len(self.mask_clip_counts)))
                    self.mask_clip_counts[idx] = 1
                else:
                    new_feat = (cur_feat * cur_count + feat) / (cur_count + 1)
                    new_norm = np.linalg.norm(new_feat) + 1e-6
                    self.mask_clip_feats[idx] = (new_feat / new_norm).astype(np.float32)
                    self.mask_clip_counts[idx] = cur_count + 1

    def initialize(self, viewpoint):
        front_gaussian, mask = self.get_patch_front_gaussian_of_mask(viewpoint)

        # 每个视角只加载一次图像
        rgb = self.load_image_for_viewpoint(viewpoint) if self.use_clip else None

        # 清空已有的组团
        self.gaussian_idx_bank = []
        self.assigned_gaussians = []
        self.mask_clip_feats = []
        self.mask_clip_counts = []

        num_mask_init = len(front_gaussian)
        for m_idx in range(num_mask_init):
            front_gaussian_of_mask = front_gaussian[m_idx]

            clip_feat = None
            if self.use_clip and rgb is not None:
                # mask[m_idx] 是 [W, H]，转成 [H, W] 再给 CLIP
                mask_2d = mask[m_idx].transpose(0, 1)
                clip_feat = self.compute_mask_clip_feature(rgb, mask_2d)

            new_idx = self.get_num_mask  # 当前已有的全局 mask 数
            self.maintain_gaussian_idx_bank(new_idx, front_gaussian_of_mask, clip_feat=clip_feat)
            self.get_num_mask  # 更新计数

        labels = torch.arange(self.num_mask, dtype=torch.long, device=self.device)
        return labels

    def associate(self, viewpoint):
        """
        之后的视角：对每个新视角的局部 mask，
        先看高斯重合（IoU），再看与已有组团平均 CLIP 特征的相似度；
        两个条件都满足才归并，否则新建组团。
        """
        front_gaussian, mask = self.get_patch_front_gaussian_of_mask(viewpoint)
        num_mask_cur_view = len(front_gaussian)

        self.get_num_mask
        labels = torch.zeros(num_mask_cur_view, dtype=torch.long, device=self.device)

        rgb = self.load_image_for_viewpoint(viewpoint) if self.use_clip else None

        for m_idx in range(num_mask_cur_view):
            front_gaussian_of_mask = front_gaussian[m_idx]

            # 计算该局部实例的 CLIP 特征
            clip_feat = None
            if self.use_clip and rgb is not None:
                # mask[m_idx] 是 [W, H]，转成 [H, W]
                mask_2d = mask[m_idx].transpose(0, 1)
                clip_feat = self.compute_mask_clip_feature(rgb, mask_2d)

            # 如果当前 mask 没有任何前景高斯，或者目前还没有全局 mask，直接新建
            if front_gaussian_of_mask.numel() == 0 or self.num_mask == 0:
                selected_mask = self.num_mask  # 作为一个新全局 mask
                self.maintain_gaussian_idx_bank(selected_mask, front_gaussian_of_mask, clip_feat=clip_feat)
                labels[m_idx] = selected_mask
                self.get_num_mask
                continue

            # -------- 用 IOU 计算与已有全局 mask 的相似度 --------
            num_union = []
            num_intersection = []
            for i in range(self.num_mask):
                union_i = torch.unique(torch.cat([self.gaussian_idx_bank[i], front_gaussian_of_mask]))
                num_union.append(len(union_i))
                num_intersection.append(
                    len(self.gaussian_idx_bank[i]) + len(front_gaussian_of_mask) - len(union_i)
                )

            iou = [num_intersection[i] / (num_union[i] + 1e-8) for i in range(self.num_mask)]
            iou = torch.tensor(iou, dtype=torch.float32, device=self.device)

            # 选 IOU 最大的那个全局 mask
            selected_mask = int(torch.argmax(iou).item())
            is_new_group = False

            # 几何条件不过关 → 新组团
            if iou[selected_mask] < self.iou_threshold:
                is_new_group = True
            else:
                # 几何 OK，再看 CLIP
                if self.use_clip and clip_feat is not None:
                    if selected_mask < len(self.mask_clip_feats):
                        group_feat = self.mask_clip_feats[selected_mask]
                    else:
                        group_feat = None

                    if group_feat is not None:
                        # 两个特征都是单位向量，可以直接点积
                        cos_sim = float((group_feat * clip_feat).sum())
                        if cos_sim < self.clip_sim_threshold:
                            is_new_group = True

            if is_new_group:
                selected_mask = self.num_mask  # 新的全局 mask id

            # 维护 gaussian_idx_bank & assigned_gaussians & 平均 CLIP 特征
            self.maintain_gaussian_idx_bank(selected_mask, front_gaussian_of_mask, clip_feat=clip_feat)
            labels[m_idx] = selected_mask
            self.get_num_mask

        return labels

    def build_mask_association(self):
        """
        只负责做跨视角关联，并把关联后的 mask 以 .npy 格式和可视化图写到磁盘。
        同时在内部为每个全局 mask 维护平均 CLIP 特征（self.global_clip_features）。
        """
        for view in tqdm(self.viewpoint_camera, desc="Building mask association"):
            view = view.to(self.device)
            if self.num_mask == 0:
                labels = self.initialize(view)
            else:
                labels = self.associate(view)
                
            # 读取原始局部 mask（可以是 .npy 或 .png）
            mask = self._read_raw_mask(view.image_name)

            # 将局部 mask 的 label 映射为全局 mask id
            object_mask = convert_matched_mask(labels, mask)
            if isinstance(object_mask, torch.Tensor):
                object_mask = object_mask.cpu().numpy()
            object_mask = object_mask.astype(np.uint16)

            # === 主结果：以 .npy 格式保存，用于后续 finalize 脚本按视角数过滤等 ===
            object_mask_npy_path = os.path.join(
                self.associated_mask_folder, view.image_name + ".npy"
            )
            np.save(object_mask_npy_path, object_mask)

            # === 可视化：彩色 PNG，方便检查 ===
            if self.visualize:
                visualize_mask = self.visualize_mask_association(object_mask)
                visualize_mask_path = os.path.join(
                    self.visualize_folder, view.image_name + ".png"
                )
                cv2.imwrite(visualize_mask_path, visualize_mask)

        # ====== 构建全局平均 CLIP 特征矩阵，方便后续脚本根据 mask id 查询 ======
        self.get_num_mask
        if self.use_clip and self.num_mask > 0 and len(self.mask_clip_feats) == self.num_mask:
            # 找到特征维度
            dim = 0
            for feat in self.mask_clip_feats:
                if feat is not None:
                    dim = feat.shape[0]
                    break
            if dim > 0:
                global_feats = np.zeros((self.num_mask + 1, dim), dtype=np.float32)
                # index 0 对应背景，保持 0 向量；1..num_mask 对应全局 mask id+1
                for i, feat in enumerate(self.mask_clip_feats):
                    if feat is not None:
                        global_feats[i + 1] = feat
                self.global_clip_features = global_feats
            else:
                self.global_clip_features = None
        else:
            self.global_clip_features = None

        # 在这里不再写 info.json，过滤和最终 info 交给外部脚本

    def visualize_mask_association(self, object_mask):
        """
        object_mask: uint16, H x W, 0=背景, 1..num_mask=不同实例
        可视化时动态保证 random_colors 的长度 >= num_mask
        """
        h, w = object_mask.shape
        visualize_mask = np.zeros((h, w, 3), dtype=np.uint8)

        num_needed = int(self.num_mask)
        if num_needed <= 0:
            return visualize_mask

        # 确保颜色表够长
        if (self.random_colors is None) or (self.random_colors.shape[0] < num_needed):
            self.random_colors = get_n_different_colors(num_needed)

        # 只遍历真的出现过的 label，顺便少一点循环
        labels = np.unique(object_mask)
        labels = labels[labels > 0]  # 去掉背景 0

        for lab in labels:
            idx = int(lab) - 1  # lab 是 1-based，color 是 0-based
            if 0 <= idx < self.random_colors.shape[0]:
                visualize_mask[object_mask == lab] = self.random_colors[idx]

        visualize_mask = cv2.cvtColor(visualize_mask, cv2.COLOR_RGB2BGR)
        return visualize_mask

    def project_gaussian(self, viewpoint):
        proj_matrix = viewpoint.full_proj_transform

        p_hom = transformPoint4x4(self.gaussians_xyz, proj_matrix)
        p_hom_z = p_hom[:, 2]

        p_w = 1 / (p_hom[:, 3:] + 1e-8)
        p_proj = p_hom[:, :3] * p_w

        p_proj[:, 0] = ndc2Pixel(p_proj[:, 0], self.image_width)
        p_proj[:, 1] = ndc2Pixel(p_proj[:, 1], self.image_height)
        p_proj = torch.round(p_proj[:, :2]).long()

        # Remove the points that are outside the image
        p_proj_inside_mask = (
            (p_proj[:, 0] >= 0) & (p_proj[:, 0] < self.image_width) &
            (p_proj[:, 1] >= 0) & (p_proj[:, 1] < self.image_height) &
            (p_hom_z > 0)
        )
        p_proj_inside = p_proj[p_proj_inside_mask]
        p_proj_inside_indices = p_proj_inside_mask.nonzero().squeeze()
        p_proj_inside_reverse_mapping = {
            p_proj_inside_indices[i].item(): i for i in range(len(p_proj_inside_indices))
        }
        p_proj_flatten = p_proj_inside[:, 0] * self.image_height + p_proj_inside[:, 1]

        projected_gaussian = {
            "p_proj_flatten": p_proj_flatten,
            "p_proj_inside_indices": p_proj_inside_indices,
            "p_proj_inside_reverse_mapping": p_proj_inside_reverse_mapping,
            "p_hom_z": p_hom_z,
        }

        return projected_gaussian

    def load_mask(self, viewpoint):
        # 原始 mask：H x W，整型标签（0=背景，1..N=实例 id）
        mask = self._read_raw_mask(viewpoint.image_name)

        # 这里如果你以后想在关联阶段也用 resolution 下采样，
        # 可以在这个地方对 mask 做 cv2.resize(..., INTER_NEAREST)
        # 现在先不动，保持行为跟你原始代码一致。

        binary_mask_mask = mask_id_to_binary_mask(mask)
        # mask_id_to_binary_mask 输出 [num_obj, H, W]
        binary_mask = torch.tensor(
            binary_mask_mask, dtype=torch.bool, device=self.device
        ).transpose(1, 2)  # -> [num_obj, W, H] (你原来的约定)
        return binary_mask


    def _read_raw_mask(self, image_name: str):
        """
        从 raw_mask_folder 读取原始实例 mask。
        支持两种格式：
          1) .npy : 直接 np.load，得到 H x W 的整型标签图
          2) .png : 用 cv2.imread 读灰度/无符号整型
        """
        base = os.path.join(self.raw_mask_folder, image_name)

        # 优先 .npy（你现在就是这样）
        npy_path = base + ".npy"
        if os.path.exists(npy_path):
            mask = np.load(npy_path)
            if mask is None:
                raise ValueError(f"[GaussianProjector] Loaded None from {npy_path}")
            return mask

        # 兼容老版本：灰度 PNG
        for ext in [".png", ".tif", ".tiff"]:
            img_path = base + ext
            if os.path.exists(img_path):
                mask = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if mask is None:
                    continue
                # 如果读到的是三通道，取单通道（一般不会）
                if mask.ndim == 3:
                    mask = mask[..., 0]
                return mask

        # 两种格式都找不到 → 明确抛异常
        raise FileNotFoundError(
            f"[GaussianProjector] cannot find raw mask for '{image_name}' "
            f"in {self.raw_mask_folder} (tried .npy/.png/.tif)."
        )

    def get_patch_front_gaussian_of_mask(self, viewpoint):
        projected_gaussian = self.project_gaussian(viewpoint)

        p_proj_flatten = projected_gaussian["p_proj_flatten"]
        p_proj_inside_indices = projected_gaussian["p_proj_inside_indices"]
        p_hom_z = projected_gaussian["p_hom_z"]

        mask = self.load_mask(viewpoint)
        assert mask.shape[1] == self.image_width and mask.shape[2] == self.image_height, \
            "Mask and image have different sizes."
        mask_flatten = mask.flatten(start_dim=1)

        front_gaussian = []
        for obj_m in mask_flatten:
            front_gaussian_of_mask = []
            for i in range(self.num_patches):
                for j in range(self.num_patches):
                    patch_m = self.flatten_patch_mask[i, j]
                    m = obj_m & patch_m
                    if m.sum() == 0:
                        continue
                    gaussian_of_mask_inside = m[p_proj_flatten].nonzero().squeeze(-1)
                    if gaussian_of_mask_inside.shape[0] == 0:
                        continue

                    gaussian_of_mask = p_proj_inside_indices[gaussian_of_mask_inside]
                    p_hom_z_of_mask = p_hom_z[gaussian_of_mask]
                    num_front_gaussians = max(int(self.front_percentage * len(gaussian_of_mask)), 1)
                    front_gaussian_of_mask.append(
                        gaussian_of_mask[torch.argsort(p_hom_z_of_mask)][:num_front_gaussians]
                    )

            if len(front_gaussian_of_mask) == 0:
                front_gaussian.append(torch.tensor([], dtype=torch.long, device=self.device))
            else:
                front_gaussian.append(torch.cat(front_gaussian_of_mask))

        return front_gaussian, mask
