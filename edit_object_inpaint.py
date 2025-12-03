# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified for Gaga-style editing:
# - JSON 中指定：
#     * target_ids        : 目标区域的「原始 mask ID」列表
#     * template_ids      : 作为 inpaint 模板的「原始 mask ID」列表
#     * inpaint_mode      : "texture"（随机纹理采样）或 "replace"（同类替换）
#     * removal_thresh    : 3D 分类阈值
#     * lambda_dlpips     : LPIPS 权重
#     * finetune_iteration: 微调迭代数
# ------------------------------------------------------------------------

import os
import json
from random import randint
from os import makedirs

import torch
import torchvision
import lpips
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from argparse import ArgumentParser
from scipy.spatial import KDTree

from scene import Scene
from gaussian_renderer import render, GaussianModel
from utils.general_utils import safe_state
from utils.loss_utils import masked_l1_loss
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args

from render import feature_to_rgb, visualize_obj
from edit_object_removal import points_inside_convex_hull


# ----------------------------
# 一些 2D / patch 工具函数
# ----------------------------

def mask_to_bbox(mask: torch.Tensor):
    """
    mask: [H, W] bool 或 0/1
    返回 (xmin, ymin, xmax, ymax)
    """
    rows = torch.any(mask, dim=1)
    cols = torch.any(mask, dim=0)
    idx_rows = torch.where(rows)[0]
    idx_cols = torch.where(cols)[0]
    if idx_rows.numel() == 0 or idx_cols.numel() == 0:
        # 空 mask，就返回整个图像范围
        return 0, 0, mask.shape[1] - 1, mask.shape[0] - 1
    ymin, ymax = idx_rows[[0, -1]]
    xmin, xmax = idx_cols[[0, -1]]
    return int(xmin), int(ymin), int(xmax), int(ymax)


def crop_using_bbox(image: torch.Tensor, bbox):
    """
    image: [C, H, W]
    bbox: (xmin, ymin, xmax, ymax)
    """
    xmin, ymin, xmax, ymax = bbox
    return image[:, ymin:ymax + 1, xmin:xmax + 1]


def divide_into_patches(image: torch.Tensor, K: int):
    """
    image: [B, C, H, W]
    把图像均匀切成 K x K 个 patch，返回 [B, K*K, C, patch_h, patch_w]
    """
    B, C, H, W = image.shape
    patch_h, patch_w = H // K, W // K
    patches = torch.nn.functional.unfold(image, (patch_h, patch_w), stride=(patch_h, patch_w))
    patches = patches.view(B, C, patch_h, patch_w, -1)
    return patches.permute(0, 4, 1, 2, 3)


def build_2d_mask_from_ids(objects_2d: torch.Tensor, target_ids_origin):
    """
    objects_2d: [H, W]，里面是「原始 ID」
    target_ids_origin: python list[int]，原始 ID 列表
    返回: [H, W] bool mask，表示「属于这些 ID 的区域」
    """
    mask = torch.zeros_like(objects_2d, dtype=torch.bool, device=objects_2d.device)
    for oid in target_ids_origin:
        mask |= (objects_2d == int(oid))
    return mask


# ----------------------------
# 3D 区域 & 特征初始化
# ----------------------------

def compute_3d_masks_from_ids(
    gaussians: GaussianModel,
    classifier: torch.nn.Conv2d,
    id_map: dict,
    target_ids_origin,
    template_ids_origin,
    removal_thresh: float,
):
    """
    使用 classifier 的权重，在 3D object feature 上做一次 chunk 化分类，
    得到：
      - mask_target3d  : 目标区域的 3D mask
      - mask_template3d: 模板区域的 3D mask
    其中 target_ids_origin/template_ids_origin 是「原始 ID」，需要先用 id_map 映射成紧凑 ID。
    """
    device = gaussians._xyz.device

    # ---------- 1. 原始 ID -> 紧凑 ID ----------
    def map_ids(origin_ids):
        compact_ids = []
        for oid in origin_ids:
            oid_int = int(oid)
            if oid_int in id_map:
                compact_ids.append(id_map[oid_int])
            else:
                print(f"[Inpaint][WARN] origin id {oid_int} not in id_map, 忽略该 id")
        return compact_ids

    target_ids_compact = map_ids(target_ids_origin)
    template_ids_compact = map_ids(template_ids_origin)

    if len(target_ids_compact) == 0:
        raise RuntimeError("[Inpaint][ERROR] 没有任何有效的 target 紧凑 ID")
    if len(template_ids_compact) == 0:
        raise RuntimeError("[Inpaint][ERROR] 没有任何有效的 template 紧凑 ID")

    target_ids_compact = torch.tensor(target_ids_compact, dtype=torch.long, device=device)
    template_ids_compact = torch.tensor(template_ids_compact, dtype=torch.long, device=device)

    # ---------- 2. 把 3D 特征整理成 [N, C] ----------
    feat3d = gaussians._objects_dc.permute(2, 0, 1).contiguous()
    # feat3d 可能是 [C, N, 1]，去掉最后一维
    if feat3d.dim() == 3 and feat3d.shape[-1] == 1:
        feat3d = feat3d.squeeze(-1)  # [C, N]
    # 统一成 [N, C]
    if feat3d.shape[0] < feat3d.shape[1]:
        feat3d = feat3d.transpose(0, 1).contiguous()  # [N, C]
    N3d, C3d = feat3d.shape

    # ---------- 3. 从 classifier 中取出线性层权重 ----------
    W = classifier.weight.view(classifier.out_channels, -1)  # [num_classes, C3d]
    b = classifier.bias                                      # [num_classes]
    num_classes = W.shape[0]

    # ---------- 4. 按 chunk 计算 logits & softmax ----------
    CHUNK_3D = 100_000  # 显存紧张可以改小
    mask_target3d = torch.zeros(N3d, dtype=torch.bool, device=device)
    mask_template3d = torch.zeros(N3d, dtype=torch.bool, device=device)

    for start in range(0, N3d, CHUNK_3D):
        end = min(start + CHUNK_3D, N3d)
        f_part = feat3d[start:end, :]                 # [chunk, C3d]
        logits_part = f_part @ W.T + b                # [chunk, num_classes]
        prob_part = torch.softmax(logits_part, dim=1) # [chunk, num_classes]

        # 目标区域：属于 target_ids_compact 的概率 > 阈值
        prob_target = prob_part[:, target_ids_compact]       # [chunk, K_t]
        mask_target_chunk = (prob_target > removal_thresh).any(dim=1)

        # 模板区域：属于 template_ids_compact 的概率 > 阈值
        prob_template = prob_part[:, template_ids_compact]   # [chunk, K_temp]
        mask_template_chunk = (prob_template > removal_thresh).any(dim=1)

        mask_target3d[start:end] |= mask_target_chunk
        mask_template3d[start:end] |= mask_template_chunk

    return mask_target3d, mask_template3d


def init_target_features_from_template(
    gaussians: GaussianModel,
    mask_target3d: torch.Tensor,
    mask_template3d: torch.Tensor,
    mode: str,
):
    """
    根据 inpaint 模式，用模板区域的特征初始化目标区域 Gaussians 的 appearance。
    mode:
      - "texture":   随机采样模板特征，给目标区域上“纹理”
      - "replace":   用 3D 最近邻匹配，把最相近的模板特征拷贝到对应 target 上
    注意：这里只改 feature 相关（颜色/高频系数/identity embedding），几何位置不动。
    """

    device = gaussians._xyz.device

    with torch.no_grad():
        idx_target = torch.nonzero(mask_target3d, as_tuple=False).squeeze(-1)
        idx_template = torch.nonzero(mask_template3d, as_tuple=False).squeeze(-1)

        if idx_target.numel() == 0:
            raise RuntimeError("[Inpaint][ERROR] 3D 中没有任何 target 点")
        if idx_template.numel() == 0:
            raise RuntimeError("[Inpaint][ERROR] 3D 中没有任何 template 点")

        # 取出模板的特征
        feat_dc_temp = gaussians._features_dc[idx_template].detach().clone()
        feat_rest_temp = gaussians._features_rest[idx_template].detach().clone()
        obj_dc_temp = gaussians._objects_dc[idx_template].detach().clone()

        if mode == "texture":
            # 纯随机纹理采样：对每个 target 点随机挑一个模板点
            num_temp = idx_template.numel()
            rand_ids = torch.randint(0, num_temp, (idx_target.numel(),), device=device)

            gaussians._features_dc[idx_target] = feat_dc_temp[rand_ids]
            gaussians._features_rest[idx_target] = feat_rest_temp[rand_ids]
            gaussians._objects_dc[idx_target] = obj_dc_temp[rand_ids]

        elif mode == "replace":
            # 最近邻匹配：用 3D 位置最近的模板点特征来替换
            xyz = gaussians._xyz.detach()
            xyz_target = xyz[idx_target].cpu().numpy()
            xyz_template = xyz[idx_template].cpu().numpy()

            from scipy.spatial import KDTree
            kdt = KDTree(xyz_template)
            _, nn_idx = kdt.query(xyz_target, k=1)
            nn_idx = torch.tensor(nn_idx, dtype=torch.long, device=device)

            gaussians._features_dc[idx_target] = feat_dc_temp[nn_idx]
            gaussians._features_rest[idx_target] = feat_rest_temp[nn_idx]
            gaussians._objects_dc[idx_target] = obj_dc_temp[nn_idx]

        else:
            raise ValueError(f"[Inpaint] Unknown inpaint_mode: {mode}")


# ----------------------------
# Finetune / Inpainting 主过程
# ----------------------------

def finetune_inpaint(
    opt,
    model_path,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    classifier,
    id_map: dict,
    target_ids_origin,
    template_ids_origin,
    removal_thresh: float,
    finetune_iteration: int,
    inpaint_mode: str,
):
    """
    主 inpainting 流程：
      1. 通过 3D classifier + id_map 得到 target/template 的 3D 区域
      2. 用模板区域特征初始化 target 区域（两种模式）
      3. 只对 target 区域做 finetune（LPIPS + masked L1）
    """
    device = gaussians._xyz.device

    # 1) 得到 3D 区域 mask
    with torch.no_grad():
        mask_target3d, mask_template3d = compute_3d_masks_from_ids(
            gaussians,
            classifier,
            id_map,
            target_ids_origin,
            template_ids_origin,
            removal_thresh,
        )

        # 对 target 再做一个凸包扩张，让区域更完整
        mask_target3d_convex = points_inside_convex_hull(
            gaussians._xyz.detach(),
            mask_target3d,
            outlier_factor=1.0,
        )
        mask_target3d = torch.logical_or(mask_target3d, mask_target3d_convex)

    # 2) 用模板特征初始化 target 区域 appearance
    init_target_features_from_template(
        gaussians,
        mask_target3d,
        mask_template3d,
        mode=inpaint_mode,
    )

    # 3) 对 target 区域做 finetune：只允许 target 区域的点更新
    mask3d_for_finetune = mask_target3d.float()[:, None, None]  # [N,1,1]
    gaussians.finetune_setup(opt, mask3d_for_finetune)

    # LPIPS
    LPIPS_net = lpips.LPIPS(net='vgg').to(device)
    for p in LPIPS_net.parameters():
        p.requires_grad = False

    iterations = finetune_iteration
    progress_bar = tqdm(range(iterations), desc="Inpainting finetune progress")

    for it in range(iterations):
        # 随机选一个视角
        viewpoint_stack = views.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render(viewpoint_cam, gaussians, pipeline, background)
        image = render_pkg["render"]                     # [3, H, W]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()   # [3, H_gt, W_gt]
        objects_2d = viewpoint_cam.objects.to(device)    # [H_mask, W_mask] 原始 ID

        # 2D 目标 mask（原始 ID，先在 GT 分辨率上构建）
        mask2d_target = build_2d_mask_from_ids(objects_2d, target_ids_origin)  # [H_mask, W_mask] bool

        # --- 把 mask resize 到渲染图像的分辨率 ---
        H_im, W_im = image.shape[-2], image.shape[-1]    # 渲染出来的分辨率
        if mask2d_target.shape[0] != H_im or mask2d_target.shape[1] != W_im:
            mask_float = mask2d_target.float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            mask_resized = torch.nn.functional.interpolate(
                mask_float,
                size=(H_im, W_im),
                mode="nearest",
            )
            mask2d_target = (mask_resized[0, 0] > 0.5)   # [H_im, W_im] bool

        # 同样需要把 gt_image 也 resize 到和 image 一样大，避免它和 image 本身不一致
        if gt_image.shape[-2] != H_im or gt_image.shape[-1] != W_im:
            gt_image = torch.nn.functional.interpolate(
                gt_image.unsqueeze(0),
                size=(H_im, W_im),
                mode="area",   # 或 "bilinear"，随便，你之前 pipeline 用啥就用啥
            )[0]

        # 只在「非 target 区域」上约束 L1（背景 + 其他物体）
        # 只在「非 target 区域」上约束 L1（背景 + 其他物体）
        Ll1 = masked_l1_loss(image, gt_image, ~mask2d_target)

        # ---------- 在 target 区域做 LPIPS（如果区域太小就放大或跳过） ----------
        if mask2d_target.any():
            # 1) 先取 bbox（注意：用的是 resize 后的 mask）
            bbox = mask_to_bbox(mask2d_target)
            cropped_image = crop_using_bbox(image, bbox)
            cropped_gt_image = crop_using_bbox(gt_image, bbox)

            # 2) 保证 LPIPS 的输入区域不要太小
            # VGG 下采样 5 次，尺度因子 32；
            # 我们再切 K=2 的 patch，每个 patch 至少要 >= 32 像素，
            # 所以裁剪区域至少 >= 64 比较安全。
            MIN_SIZE = 64
            ch, cw = cropped_image.shape[-2], cropped_image.shape[-1]
            new_h = max(MIN_SIZE, ch)
            new_w = max(MIN_SIZE, cw)

            if new_h != ch or new_w != cw:
                cropped_image = torch.nn.functional.interpolate(
                    cropped_image.unsqueeze(0),
                    size=(new_h, new_w),
                    mode="bilinear",
                    align_corners=False,
                )[0]
                cropped_gt_image = torch.nn.functional.interpolate(
                    cropped_gt_image.unsqueeze(0),
                    size=(new_h, new_w),
                    mode="bilinear",
                    align_corners=False,
                )[0]

            # 3) 切 patch，再算 LPIPS
            K = 2
            rendering_patches = divide_into_patches(cropped_image[None, ...], K)  # [1, K*K, C, ph, pw]
            gt_patches = divide_into_patches(cropped_gt_image[None, ...], K)

            B, KK, C, PH, PW = rendering_patches.shape
            rendering_patches_flat = rendering_patches.view(B * KK, C, PH, PW)
            gt_patches_flat = gt_patches.view(B * KK, C, PH, PW)

            lpips_loss = LPIPS_net(
                rendering_patches_flat * 2 - 1,
                gt_patches_flat * 2 - 1
            ).mean()
        else:
            # 这一帧没有目标区域，LPIPS 就当 0
            lpips_loss = torch.tensor(0.0, device=device)


        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * lpips_loss

        loss.backward()

        with torch.no_grad():
            # 可以根据需要保留或删除 densify 部分，这里简单保留
            if it < 5000:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if it % 300 == 0:
                    size_threshold = 20
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        viewpoint_cam.cameras_extent if hasattr(viewpoint_cam, "cameras_extent") else 1.0,
                        size_threshold,
                    )

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

        if it % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{loss.item():.7f}"})
            progress_bar.update(10)

    progress_bar.close()

    # 保存 inpaint 后的高斯
    point_cloud_path = os.path.join(model_path, f"point_cloud_object_inpaint/iteration_{iteration}")
    makedirs(point_cloud_path, exist_ok=True)
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    return gaussians


# ----------------------------
# 渲染可视化
# ----------------------------

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, classifier):
    render_path = os.path.join(model_path, name, f"ours{iteration}", "renders")
    gts_path = os.path.join(model_path, name, f"ours{iteration}", "gt")
    colormask_path = os.path.join(model_path, name, f"ours{iteration}", "objects_feature16")
    gt_colormask_path = os.path.join(model_path, name, f"ours{iteration}", "gt_objects_color")
    pred_obj_path = os.path.join(model_path, name, f"ours{iteration}", "objects_pred")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(colormask_path, exist_ok=True)
    makedirs(gt_colormask_path, exist_ok=True)
    makedirs(pred_obj_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc=f"Rendering {name} set")):
        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]
        rendering_obj = results["render_seg"] if "render_seg" in results else results.get("render_object")

        if rendering_obj.dim() == 3:
            rendering_obj = rendering_obj.unsqueeze(0)

        logits = classifier(rendering_obj)
        pred_obj = torch.argmax(logits, dim=1)[0]
        pred_obj_mask = visualize_obj(pred_obj.cpu().numpy().astype(np.uint8))

        gt_objects = view.objects
        gt_rgb_mask = visualize_obj(gt_objects.cpu().numpy().astype(np.uint8))

        rgb_mask = feature_to_rgb(rendering_obj.squeeze(0))
        Image.fromarray(rgb_mask).save(os.path.join(colormask_path, f"{idx:05d}.png"))
        Image.fromarray(gt_rgb_mask).save(os.path.join(gt_colormask_path, f"{idx:05d}.png"))
        Image.fromarray(pred_obj_mask).save(os.path.join(pred_obj_path, f"{idx:05d}.png"))

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{idx:05d}.png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, f"{idx:05d}.png"))

    out_path = os.path.join(render_path[:-8], 'concat')
    makedirs(out_path, exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*'DIVX')
    size = (gt.shape[-1] * 5, gt.shape[-2])
    fps = float(5) if 'train' in out_path else float(1)
    writer = cv2.VideoWriter(os.path.join(out_path, 'result.mp4'), fourcc, fps, size)

    for file_name in sorted(os.listdir(gts_path)):
        gt_img = np.array(Image.open(os.path.join(gts_path, file_name)))
        rgb = np.array(Image.open(os.path.join(render_path, file_name)))
        gt_obj_img = np.array(Image.open(os.path.join(gt_colormask_path, file_name)))
        render_obj_img = np.array(Image.open(os.path.join(colormask_path, file_name)))
        pred_obj_img = np.array(Image.open(os.path.join(pred_obj_path, file_name)))

        result = np.hstack([gt_img, rgb, gt_obj_img, pred_obj_img, render_obj_img]).astype('uint8')
        Image.fromarray(result).save(os.path.join(out_path, file_name))
        writer.write(result[:, :, ::-1])

    writer.release()


# ----------------------------
# 顶层 inpaint 封装
# ----------------------------

def inpaint(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    opt: OptimizationParams,
    target_ids_origin,
    template_ids_origin,
    removal_thresh: float,
    finetune_iteration: int,
    inpaint_mode: str,
):
    """
    主入口：
      - target_ids_origin   : 目标区域「原始 mask ID」列表（要被填补/替换的区域）
      - template_ids_origin : 模板区域「原始 mask ID」列表（提供 appearance 的区域）
      - inpaint_mode        : "texture" 或 "replace"
    """
    device = torch.device("cuda")

    # 1. 加载 Gaussians & Scene
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    # 2. 读取 id_mapping.json（old_id -> compact_id）
    matched_mask_path = os.path.join(dataset.source_path, dataset.object_path)
    id_map_path = os.path.join(matched_mask_path, "id_mapping.json")
    if not os.path.exists(id_map_path):
        raise RuntimeError(
            f"id_mapping.json not found at {id_map_path}. "
            "请先完成带全局 ID 映射的训练（会自动生成该文件）。"
        )

    with open(id_map_path, "r") as f:
        raw_id_map = json.load(f)
    id_map = {int(k): int(v) for k, v in raw_id_map.items()}
    print(f"[Inpaint] id_map size = {len(id_map)}")

    # 3. 从 classifier.pth 推断 num_classes & in_channels
    ckpt_cls_path = os.path.join(
        dataset.model_path,
        "point_cloud",
        f"iteration_{scene.loaded_iter}",
        "classifier.pth",
    )
    print(f"[Inpaint] Loading classifier state_dict from: {ckpt_cls_path}")
    state_dict = torch.load(ckpt_cls_path, map_location=device)
    w = state_dict["weight"]  # [out_channels, in_channels, 1, 1]
    num_classes = w.shape[0]
    in_ch = w.shape[1]
    print(f"[Inpaint] num_classes = {num_classes}, in_channels = {in_ch}")

    classifier = torch.nn.Conv2d(in_ch, num_classes, kernel_size=1).to(device)
    classifier.load_state_dict(state_dict)

    # 背景色
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    # 4. 调整 inpaint（3D 初始化 + finetune）
    gaussians = finetune_inpaint(
        opt,
        dataset.model_path,
        scene.loaded_iter,
        scene.getTrainCameras(),
        gaussians,
        pipeline,
        background,
        classifier,
        id_map,
        target_ids_origin,
        template_ids_origin,
        removal_thresh,
        finetune_iteration,
        inpaint_mode,
    )

    # 5. 直接用当前 scene 的相机 + 修改后的 gaussians 渲染（不再新建 Scene）
    with torch.no_grad():
        if not skip_train:
            render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                classifier,
            )

        if (not skip_test) and (len(scene.getTestCameras()) > 0):
            render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
                classifier,
            )


# ----------------------------
# CLI 入口
# ----------------------------

if __name__ == "__main__":
    parser = ArgumentParser(description="3D inpainting script with target/template masks")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument(
        "--config_file",
        type=str,
        default="config/object_inpaint/example.json",
        help="Path to the configuration file",
    )

    args = get_combined_args(parser)
    print("Inpainting on model_path:", args.model_path)

    # 非 lift 模式：直接使用 output/... 下的语义高斯
    args.lift = False

    # 从 JSON 读取配置
    try:
        with open(args.config_file, "r") as file:
            config = json.load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config_file}' not found.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse the JSON configuration file: {e}")
        exit(1)

    # 这些是你要在 JSON 里指定的字段：
    # {
    #   "target_ids": [1657],            # 目标区域的「原始 mask id」
    #   "template_ids": [1234],          # 模板区域的「原始 mask id」
    #   "inpaint_mode": "texture",       # 或 "replace"
    #   "removal_thresh": 0.3,
    #   "lambda_dlpips": 0.5,
    #   "finetune_iteration": 10000,
    #   "images": "images",
    #   "object_path": "object_mask",
    #   "r": 1
    # }
    args.target_ids = config.get("target_ids", [1657])
    args.template_ids = config.get("template_ids", [1657])
    args.inpaint_mode = config.get("inpaint_mode", "texture")  # "texture" or "replace"
    args.removal_thresh = config.get("removal_thresh", 0.3)
    # args.images = config.get("images", "images")
    # args.object_path = config.get("object_path", "object_mask")
    # args.resolution = config.get("r", 1)
    args.lambda_dssim = config.get("lambda_dlpips", 0.5)
    args.finetune_iteration = config.get("finetune_iteration", 10_000)

    # 初始化 RNG
    safe_state(args.quiet)

    inpaint(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        opt.extract(args),
        args.target_ids,
        args.template_ids,
        args.removal_thresh,
        args.finetune_iteration,
        args.inpaint_mode,
    )
