# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified for Gaga-style global ID mapping & semantic inpainting:
# - JSON 中的 select_obj_id 是「原始 ID」
# - 使用 id_mapping.json (old_id -> compact_id) 做映射
# - num_classes 从 classifier.pth 中自动推断
# - 3D 分类按 chunk 线性推理，避免 OOM
# ------------------------------------------------------------------------

import os
import json
import numpy as np
from tqdm import tqdm
from os import makedirs
from random import randint

import torch
import torch.nn.functional as F
import torchvision
import cv2
from PIL import Image
import lpips

from scene import Scene
from gaussian_renderer import render, GaussianModel
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from utils.loss_utils import masked_l1_loss

from render import feature_to_rgb, visualize_obj
from edit_object_removal import points_inside_convex_hull


def mask_to_bbox(mask: torch.Tensor):
    """mask: [H, W] bool → 返回 (xmin, ymin, xmax, ymax)"""
    rows = torch.any(mask, dim=1)
    cols = torch.any(mask, dim=0)
    ymin, ymax = torch.where(rows)[0][[0, -1]]
    xmin, xmax = torch.where(cols)[0][[0, -1]]
    return xmin, ymin, xmax, ymax


def crop_using_bbox(image: torch.Tensor, bbox):
    """image: [C, H, W]"""
    xmin, ymin, xmax, ymax = bbox
    return image[:, ymin:ymax + 1, xmin:xmax + 1]


def divide_into_patches(image: torch.Tensor, K: int):
    """
    image: [B, C, H, W]
    返回: [B, K*K, C, H//K, W//K]
    """
    B, C, H, W = image.shape
    patch_h, patch_w = H // K, W // K
    patches = torch.nn.functional.unfold(image, (patch_h, patch_w), stride=(patch_h, patch_w))
    patches = patches.view(B, C, patch_h, patch_w, -1)
    return patches.permute(0, 4, 1, 2, 3)


def finetune_inpaint(opt,
                     model_path,
                     base_iteration,
                     views,
                     gaussians,
                     pipeline,
                     background,
                     classifier,
                     selected_obj_ids_compact,
                     cameras_extent,
                     removal_thresh,
                     finetune_iteration):
    """
    selected_obj_ids_compact: 已通过 id_mapping 映射好的紧凑 ID 列表（new_id）
    """
    device = gaussians._xyz.device
    selected_obj_ids_compact = torch.tensor(selected_obj_ids_compact, dtype=torch.long, device=device)

    # ========= 1. 在 3D 上用 classifier 做分类，找出要 inpaint 的高斯 =========
    with torch.no_grad():
        # 把 objects_dc 整理成 [N, C]
        feat3d = gaussians._objects_dc.permute(2, 0, 1).contiguous()  # [C, N, 1] or [C, N]
        if feat3d.dim() == 3 and feat3d.shape[-1] == 1:
            feat3d = feat3d.squeeze(-1)  # [C, N]
        if feat3d.shape[0] < feat3d.shape[1]:
            feat3d = feat3d.transpose(0, 1).contiguous()  # [N, C]

        N3d, C3d = feat3d.shape

        # 从 classifier 里取出权重当线性层：Conv2d(in_ch=C3d, out_ch=num_classes, 1x1)
        W = classifier.weight.view(classifier.out_channels, -1)  # [num_classes, C3d]
        b = classifier.bias                                      # [num_classes]
        num_classes = W.shape[0]

        mask3d = torch.zeros(N3d, dtype=torch.bool, device=device)

        CHUNK_3D = 100_000  # 显存紧就调小，比如 50_000 / 20_000
        for start in range(0, N3d, CHUNK_3D):
            end = min(start + CHUNK_3D, N3d)
            f_part = feat3d[start:end, :]               # [chunk, C3d]

            # 线性分类: [chunk, C] @ [C, num_cls] -> [chunk, num_cls]
            logits = f_part @ W.T + b                   # [chunk, num_cls]
            prob = torch.softmax(logits, dim=1)         # [chunk, num_cls]

            prob_sel = prob[:, selected_obj_ids_compact]      # [chunk, K]
            mask_chunk = (prob_sel > removal_thresh).any(dim=1)  # [chunk]

            mask3d[start:end] |= mask_chunk

        # 凸包扩张
        mask3d_convex = points_inside_convex_hull(
            gaussians._xyz.detach(), mask3d, outlier_factor=1.0
        )
        mask3d = torch.logical_or(mask3d, mask3d_convex)   # [N]
        mask3d = mask3d.float()[:, None, None]             # [N,1,1]

    # ========= 2. 根据 mask 做 inpaint 设置（只优化新点） =========
    gaussians.inpaint_setup(opt, mask3d)

    iterations = finetune_iteration
    progress_bar = tqdm(range(iterations), desc="Finetuning (inpaint)")

    # LPIPS 感知损失
    LPIPS = lpips.LPIPS(net='vgg').to(device)
    for p in LPIPS.parameters():
        p.requires_grad = False

    for it in range(iterations):
        # 随机挑一个训练视角
        viewpoint_stack = views.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render(viewpoint_cam, gaussians, pipeline, background)
        image = render_pkg["render"]  # [3, H_render, W_render]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        H_render, W_render = image.shape[-2:]

        # --- 对齐 GT 图像分辨率 ---
        gt_image = viewpoint_cam.original_image.cuda()  # [3, H_gt, W_gt]
        if gt_image.shape[-2:] != (H_render, W_render):
            gt_image = F.interpolate(
                gt_image.unsqueeze(0),
                size=(H_render, W_render),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        # --- 对齐 mask 分辨率 ---
        mask2d = (viewpoint_cam.objects.cuda() > 128)   # [H_mask, W_mask]
        if mask2d.shape != (H_render, W_render):
            mask2d = F.interpolate(
                mask2d.unsqueeze(0).unsqueeze(0).float(),
                size=(H_render, W_render),
                mode="nearest",
            ).squeeze(0).squeeze(0).bool()

        # 若当前视角中完全没有该目标，跳过这一轮
        if not mask2d.any():
            gaussians.optimizer.zero_grad(set_to_none=True)
            continue

        # 未被 inpaint 覆盖区域，用 masked L1 保持整体一致
        Ll1 = masked_l1_loss(image, gt_image, ~mask2d)

        # 在 bbox 内做 LPIPS 感知损失（只关注 inpaint 区域的视觉质量）
        bbox = mask_to_bbox(mask2d)
        cropped_image = crop_using_bbox(image, bbox)
        cropped_gt_image = crop_using_bbox(gt_image, bbox)

        K = 2
        rendering_patches = divide_into_patches(cropped_image[None, ...], K)
        gt_patches = divide_into_patches(cropped_gt_image[None, ...], K)
        lpips_loss = LPIPS(
            rendering_patches.squeeze() * 2 - 1,
            gt_patches.squeeze() * 2 - 1
        ).mean()

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * lpips_loss
        loss.backward()

        with torch.no_grad():
            if it < 5000:
                # 只统计梯度，不真正 densify
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                # 不调用 densify_and_prune

        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)


        if it % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{loss.item():.7f}"})
            progress_bar.update(10)

    progress_bar.close()

    # 保存 inpaint 后的高斯（便于之后单独加载）
    final_inner_iter = iterations - 1
    point_cloud_path = os.path.join(
        model_path,
        f"point_cloud_object_inpaint/iteration_{final_inner_iter}"
    )
    makedirs(point_cloud_path, exist_ok=True)
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    return gaussians


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

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]

        # 优先使用 render_seg，若没有则 fallback 到 render_object
        if "render_seg" in results:
            rendering_obj = results["render_seg"]
        elif "render_object" in results:
            rendering_obj = results["render_object"]
        else:
            raise KeyError("render result has neither 'render_seg' nor 'render_object'.")

        # 统一为 [1, C, H, W]
        if rendering_obj.dim() == 3:
            rendering_obj = rendering_obj.unsqueeze(0)

        logits = classifier(rendering_obj)           # [1, num_cls, H, W]
        pred_obj = torch.argmax(logits, dim=1)[0]   # [H, W]
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
        gt_obj = np.array(Image.open(os.path.join(gt_colormask_path, file_name)))
        render_obj = np.array(Image.open(os.path.join(colormask_path, file_name)))
        pred_obj_img = np.array(Image.open(os.path.join(pred_obj_path, file_name)))

        result = np.hstack([gt_img, rgb, gt_obj, pred_obj_img, render_obj]).astype('uint8')
        Image.fromarray(result).save(os.path.join(out_path, file_name))
        writer.write(result[:, :, ::-1])

    writer.release()


def inpaint(dataset: ModelParams,
            iteration: int,
            pipeline: PipelineParams,
            skip_train: bool,
            skip_test: bool,
            opt: OptimizationParams,
            selected_obj_ids_origin,
            removal_thresh: float,
            finetune_iteration: int):
    """
    selected_obj_ids_origin: JSON 中给出的原始 ID 列表（old_id）
    """
    device = torch.device("cuda")

    # 1. 加载语义高斯 & 场景
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    # 2. 读取 id_mapping.json（old_id -> compact_id）
    matched_mask_path = os.path.join(dataset.source_path, dataset.object_path)
    id_map_path = os.path.join(matched_mask_path, "id_mapping.json")
    if not os.path.exists(id_map_path):
        raise RuntimeError(
            f"id_mapping.json not found at {id_map_path}. "
            "请先完成语义训练并生成该映射文件。"
        )

    with open(id_map_path, "r") as f:
        raw_id_map = json.load(f)
    id_map = {int(k): int(v) for k, v in raw_id_map.items()}
    print(f"[Inpaint] id_map size = {len(id_map)}")

    # 3. 从 classifier.pth 推断 num_classes / in_channels
    ckpt_cls_path = os.path.join(
        dataset.model_path,
        "point_cloud",
        f"iteration_{scene.loaded_iter}",
        "classifier.pth",
    )
    print(f"[Inpaint] Loading classifier from: {ckpt_cls_path}")
    state_dict = torch.load(ckpt_cls_path, map_location=device)

    w = state_dict["weight"]  # [out_ch, in_ch, 1, 1]
    num_classes = w.shape[0]
    in_ch = w.shape[1]
    print(f"[Inpaint] num_classes (from classifier) = {num_classes}, in_channels = {in_ch}")

    classifier = torch.nn.Conv2d(in_ch, num_classes, kernel_size=1).to(device)
    classifier.load_state_dict(state_dict)

    # 4. 把原始 ID 映射为紧凑 ID
    selected_obj_ids_compact = []
    for oid in selected_obj_ids_origin:
        oid_int = int(oid)
        if oid_int in id_map:
            nid = id_map[oid_int]
            if nid >= num_classes:
                print(f"[Inpaint][WARN] compact id {nid} (from origin {oid_int}) >= num_classes {num_classes}, 忽略")
                continue
            selected_obj_ids_compact.append(nid)
            print(f"[Inpaint] origin id {oid_int} -> compact id {nid}")
        else:
            print(f"[Inpaint][WARN] origin id {oid_int} not in id_map, 忽略")

    if len(selected_obj_ids_compact) == 0:
        print("[Inpaint][ERROR] 没有任何有效的紧凑 ID，需要 inpaint 的对象为空，直接退出。")
        return

    # 5. 背景色
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    # 6. 针对选定对象做 3D inpaint 微调
    gaussians = finetune_inpaint(
        opt,
        dataset.model_path,
        scene.loaded_iter,
        scene.getTrainCameras(),
        gaussians,
        pipeline,
        background,
        classifier,
        selected_obj_ids_compact,
        scene.cameras_extent,
        removal_thresh,
        finetune_iteration,
    )

    # 7. 用 inpaint 后的 gaussians + 原 scene 相机构造渲染
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


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Inpainting script parameters (Gaga style)")
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
        default="config/object_inpaint/bear.json",
        help="Path to the configuration file",
    )

    # 用 cfg_args + 命令行合并参数
    args = get_combined_args(parser)
    print("Inpainting on model_path:", args.model_path)

    # 和 render / remove 一样，语义模型在 output 下，所以关掉 lift
    args.lift = False

    # 读 JSON 配置（只改行为参数，不改路径）
    try:
        with open(args.config_file, 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config_file}' not found.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse the JSON configuration file: {e}")
        exit(1)

    args.removal_thresh = config.get("removal_thresh", 0.3)
    args.select_obj_id = config.get("select_obj_id", [34])          # 原始 ID
    args.lambda_dssim = config.get("lambda_dlpips", 0.5)
    args.finetune_iteration = config.get("finetune_iteration", 10_000)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    inpaint(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        opt.extract(args),
        args.select_obj_id,          # 原始 ID 列表
        args.removal_thresh,
        args.finetune_iteration,
    )
