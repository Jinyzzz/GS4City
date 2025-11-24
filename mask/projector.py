import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torchvision
import json
from PIL import Image

from arguments import ModelParams, PipelineParams
from mask.utils import get_n_different_colors, ndc2Pixel, transformPoint4x4, convert_matched_mask, mask_id_to_binary_mask

from scene import Scene, GaussianModel

default_params = {
    "seg_method": "sam",
    "front_percentage": 0.2,
    "iou_threshold": 0.2,
    "num_patch": 32,
    "visualize": False
}


class GaussianProjector(torch.nn.Module):
    def __init__(self,
                 dataset: ModelParams,
                 pipeline: PipelineParams,
                 iteration: int,
                 params: dict = default_params,
                 device: torch.device = torch.device("cuda"),
                 ):
        super(GaussianProjector, self).__init__()
        self.device = device

        # Load pre-trained Gaussians and cameras
        self.gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, self.gaussians, load_iteration=iteration, shuffle=False)
        self.gaussians_xyz = self.gaussians.get_xyz.to(self.device)
        # Only use the training cameras for mask association
        self.viewpoint_camera = scene.getTrainCameras()

        # Key hyperparameters
        self.front_percentage = params["front_percentage"]
        self.iou_threshold = params.get("iou_threshold", params.get("overlap_threshold", 0.4))
        self.num_patches = params["num_patch"]

        # Paths
        self.source_path = dataset.source_path
        self.seg_method = params["seg_method"]
        self.raw_mask_folder = os.path.join(self.source_path, f"raw_{self.seg_method}_mask")
        assert os.path.exists(self.raw_mask_folder), "Mask folder does not exist."
        self.associated_mask_folder = os.path.join(self.source_path, f"{self.seg_method}_mask")
        os.makedirs(self.associated_mask_folder, exist_ok=True)

        if params["visualize"]:
            self.visualize = True
            self.visualize_folder = os.path.join(self.source_path, f"{self.seg_method}_mask_vis")
            os.makedirs(self.visualize_folder, exist_ok=True)
            self.random_colors = get_n_different_colors(1000)
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
            device=self.device
        )
        for i in range(self.num_patches):
            for j in range(self.num_patches):
                self.patch_mask[i, j,
                                i * self.patch_width: (i + 1) * self.patch_width,
                                j * self.patch_height: (j + 1) * self.patch_height] = True
        self.flatten_patch_mask = self.patch_mask.flatten(start_dim=2)

        # For mask association
        self.gaussian_idx_bank = []
        self.num_mask = 0
        # We don't want the same Gaussian to be assigned to multiple masks
        self.assigned_gaussians = []

    @property
    def get_num_mask(self):
        if len(self.gaussian_idx_bank) == 0:
            self.num_mask = 0
            return 0
        self.num_mask = len(self.gaussian_idx_bank)
        return self.num_mask

    def maintain_gaussian_idx_bank(self, idx, front_gaussian_of_mask):
        assert not idx > self.num_mask, "idx is larger than the number of masks"
        if idx == self.num_mask:
            # 新的全局 mask
            self.gaussian_idx_bank.append(front_gaussian_of_mask)
            if isinstance(self.assigned_gaussians, list) and len(self.assigned_gaussians) == 0:
                self.assigned_gaussians = front_gaussian_of_mask
            else:
                self.assigned_gaussians = torch.unique(
                    torch.cat([self.assigned_gaussians, front_gaussian_of_mask])
                )
        else:
            # 向已有的全局 mask 追加新高斯（避免重复）
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

    def initialize(self, viewpoint):
        front_gaussian, mask = self.get_patch_front_gaussian_of_mask(viewpoint)
        self.gaussian_idx_bank.extend(front_gaussian)
        self.assigned_gaussians = torch.unique(torch.cat(front_gaussian))

        self.get_num_mask
        labels = torch.arange(self.num_mask, dtype=torch.long, device=self.device)
        return labels

    def associate(self, viewpoint):
        front_gaussian, mask = self.get_patch_front_gaussian_of_mask(viewpoint)
        num_mask_cur_view = len(front_gaussian)

        self.get_num_mask
        labels = torch.zeros(num_mask_cur_view, dtype=torch.long, device=self.device)

        for m_idx in range(num_mask_cur_view):
            front_gaussian_of_mask = front_gaussian[m_idx]

            # 如果当前 mask 没有任何前景高斯，直接新建一个空的全局 mask
            if front_gaussian_of_mask.numel() == 0 or self.num_mask == 0:
                selected_mask = self.num_mask  # 作为一个新全局 mask
                self.maintain_gaussian_idx_bank(selected_mask, front_gaussian_of_mask)
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
            selected_mask = torch.argmax(iou)
            # 如果最大 IOU 还小于阈值，则认为是一个新实例
            if iou[selected_mask] < self.iou_threshold:
                selected_mask = self.num_mask  # 新的全局 mask id

            # 维护 gaussian_idx_bank & assigned_gaussians
            self.maintain_gaussian_idx_bank(selected_mask, front_gaussian_of_mask)
            labels[m_idx] = selected_mask

            # 更新 self.num_mask
            self.get_num_mask

        return labels

    def build_mask_association(self):
        """
        只负责做跨视角关联，并把关联后的灰度图和可视化图写到磁盘。
        不在这里做“按视角数过滤”；过滤逻辑放在外部脚本中统一读取灰度图再处理。
        """
        for view in tqdm(self.viewpoint_camera, desc="Building mask association"):
            view = view.to(self.device)
            if self.num_mask == 0:
                labels = self.initialize(view)
            else:
                labels = self.associate(view)

            # 写入关联后的灰度图
            mask_path = os.path.join(self.raw_mask_folder, view.image_name + ".png")
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

            # ===== 关键修改：确保灰度图是 uint16，支持超过 255 种可能性 =====
            object_mask = convert_matched_mask(labels, mask)
            if isinstance(object_mask, torch.Tensor):
                object_mask = object_mask.cpu().numpy()
            # 强制转换为 uint16（PNG 会保存为 16bit 灰度）
            object_mask = object_mask.astype(np.uint16)
            # ========================================================

            object_mask_path = os.path.join(self.associated_mask_folder, view.image_name + ".png")
            cv2.imwrite(object_mask_path, object_mask)

            # 可视化（这里仍然是 uint8 RGB，不影响 16bit 灰度图）
            if self.visualize:
                visualize_mask = self.visualize_mask_association(object_mask)
                visualize_mask_path = os.path.join(self.visualize_folder, view.image_name + ".png")
                cv2.imwrite(visualize_mask_path, visualize_mask)

        # 在这里不再写 info.json，改由外部脚本在过滤之后写入最终信息

    def visualize_mask_association(self, object_mask):
        # object_mask 现在是 uint16，但比较时不受影响
        h, w = object_mask.shape
        visualize_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(self.num_mask):
            visualize_mask[object_mask == (i + 1)] = self.random_colors[i]
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
            "p_hom_z": p_hom_z
        }

        return projected_gaussian

    def load_mask(self, viewpoint):
        mask_path = os.path.join(self.raw_mask_folder, viewpoint.image_name + ".png")
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        binary_mask_mask = mask_id_to_binary_mask(mask)
        binary_mask = torch.tensor(binary_mask_mask, dtype=torch.bool, device=self.device).transpose(1, 2)
        return binary_mask

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

        # 这里也顺便帮你把后面那句中文注释从代码里拿掉，避免语法错误
        return front_gaussian, mask
