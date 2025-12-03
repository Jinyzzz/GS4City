# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified for Gaga-style global ID mapping:
# - JSON 中的 select_obj_id 是 "原始 ID"
# - 使用 id_mapping.json (old_id -> compact_id) 做映射
# - num_classes 由 classifier.pth 决定，不从 JSON 读取
# ------------------------------------------------------------------------

import os
import json
import numpy as np
from tqdm import tqdm
from os import makedirs

import torch
import torchvision
import cv2
from PIL import Image
from scipy.spatial import Delaunay

from scene import Scene
from gaussian_renderer import render, GaussianModel
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args

from render import feature_to_rgb, visualize_obj


def points_inside_convex_hull(point_cloud, mask, remove_outliers=True, outlier_factor=1.0):
    """
    给定点云和一个 mask（表示子集），计算这批点的凸包，
    然后判断原点云中哪些点在这个凸包内部。
    """
    device = point_cloud.device
    masked_points = point_cloud[mask].detach().cpu().numpy()
    if masked_points.shape[0] == 0:
        return torch.zeros(point_cloud.shape[0], dtype=torch.bool, device=device)

    if remove_outliers:
        Q1 = np.percentile(masked_points, 25, axis=0)
        Q3 = np.percentile(masked_points, 75, axis=0)
        IQR = Q3 - Q1
        outlier_mask = (masked_points < (Q1 - outlier_factor * IQR)) | (masked_points > (Q3 + outlier_factor * IQR))
        filtered_masked_points = masked_points[~np.any(outlier_mask, axis=1)]
        if filtered_masked_points.shape[0] == 0:
            filtered_masked_points = masked_points
    else:
        filtered_masked_points = masked_points

    delaunay = Delaunay(filtered_masked_points)
    points_inside_hull_mask = delaunay.find_simplex(point_cloud.detach().cpu().numpy()) >= 0
    return torch.tensor(points_inside_hull_mask, device=device)


def removal_setup(opt,
                  model_path,
                  iteration,
                  views,
                  gaussians,
                  pipeline,
                  background,
                  classifier,
                  selected_obj_ids_compact,
                  cameras_extent,
                  removal_thresh):
    """
    根据紧凑 ID（new_id）做 3D 删除：
    1) 用 classifier 的权重，在 3D object feature 上做线性分类（按 chunk 分批算）
    2) 选出属于 selected_obj_ids_compact 的点（> 阈值）
    3) 用这些点的凸包扩张区域
    4) 调用 gaussians.removal_setup 真正删除
    """
    device = gaussians._xyz.device
    selected_obj_ids_compact = torch.tensor(
        selected_obj_ids_compact, dtype=torch.long, device=device
    )

    with torch.no_grad():
        # ======== 1. 把 3D 特征整理成 [N, C] ========
        feat3d = gaussians._objects_dc.permute(2, 0, 1).contiguous()
        # 形状可能是 [C, N, 1]，压掉最后一个 1
        if feat3d.dim() == 3 and feat3d.shape[-1] == 1:
            feat3d = feat3d.squeeze(-1)          # [C, N]
        # 我们要 [N, C]
        if feat3d.shape[0] < feat3d.shape[1]:
            feat3d = feat3d.transpose(0, 1).contiguous()   # [N, C]

        N3d, C3d = feat3d.shape

        # ======== 2. 从 classifier 取出权重当线性层用 ========
        # classifier: Conv2d(in_ch=C3d, out_ch=num_classes, kernel_size=1)
        W = classifier.weight.view(classifier.out_channels, -1)  # [num_classes, C3d]
        b = classifier.bias                                      # [num_classes]
        num_classes = W.shape[0]

        # 3D 上的整体 mask，初始全 False
        mask3d = torch.zeros(N3d, dtype=torch.bool, device=device)

        # ======== 3. 按 chunk 计算 softmax 概率，只保留我们关心的类别 ========
        CHUNK_3D = 100_000  # 你显存再紧可以再往下调，比如 50_000
        for start in range(0, N3d, CHUNK_3D):
            end = min(start + CHUNK_3D, N3d)
            f_part = feat3d[start:end, :]                 # [chunk, C3d]

            # [chunk, C] @ [C, num_classes] -> [chunk, num_classes]
            logits_part = f_part @ W.T + b                # 在 GPU 上算

            # softmax 得到概率 [chunk, num_classes]
            prob_part = torch.softmax(logits_part, dim=1)

            # 只关心我们要删的那些类的概率 [chunk, K]
            prob_sel = prob_part[:, selected_obj_ids_compact]

            # 这 chunk 里属于这些类、且概率 > 阈值 的点
            mask_chunk = (prob_sel > removal_thresh).any(dim=1)  # [chunk]

            # 写回全局 mask
            mask3d[start:end] |= mask_chunk

        # ======== 4. 凸包扩张 ========
        mask3d_convex = points_inside_convex_hull(
            gaussians._xyz.detach(), mask3d, outlier_factor=1.0
        )
        mask3d = torch.logical_or(mask3d, mask3d_convex)        # [N]

        # 传给 GaussianModel.removal_setup 需要 [N,1,1] 的 float mask
        mask3d_tensor = mask3d.float()[:, None, None]

    # 真正删除高斯
    gaussians.removal_setup(opt, mask3d_tensor)

    # 保存删完的 ply
    point_cloud_path = os.path.join(
        model_path, f"point_cloud_object_removal/iteration_{iteration}"
    )
    makedirs(point_cloud_path, exist_ok=True)
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    return gaussians


def render_set(model_path,
               name,
               iteration,
               views,
               gaussians,
               pipeline,
               background,
               classifier):
    """
    渲染删除后的结果：
    - 删除后 RGB 渲染
    - GT RGB
    - feature PCA 可视化
    - classifier 预测的彩色 mask（紧凑 ID 上色）
    """
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
        rendering_obj = results["render_seg"]  # Gaga 里用的是 render_seg

        if rendering_obj.dim() == 3:
            rendering_obj = rendering_obj.unsqueeze(0)  # [1, C, H, W]

        # classifier 预测（紧凑 ID）
        logits = classifier(rendering_obj)              # [1, num_classes, H, W]
        pred_obj = torch.argmax(logits, dim=1)[0]       # [H, W]
        pred_obj_mask = visualize_obj(pred_obj.cpu().numpy().astype(np.uint8))

        # GT 对象（这里是原始 ID，直接可视化）
        gt_objects = view.objects
        gt_rgb_mask = visualize_obj(gt_objects.cpu().numpy().astype(np.uint8))

        # 特征 PCA 可视化
        rgb_mask = feature_to_rgb(rendering_obj.squeeze(0))
        Image.fromarray(rgb_mask).save(os.path.join(colormask_path, f"{idx:05d}.png"))

        # GT 彩色 mask
        Image.fromarray(gt_rgb_mask).save(os.path.join(gt_colormask_path, f"{idx:05d}.png"))

        # 预测彩色 mask（紧凑 ID 上色）
        Image.fromarray(pred_obj_mask).save(os.path.join(pred_obj_path, f"{idx:05d}.png"))

        # 保存 RGB 渲染 + RGB GT
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{idx:05d}.png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, f"{idx:05d}.png"))

    # 拼接视频
    out_path = os.path.join(render_path[:-8], "concat")
    makedirs(out_path, exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*"DIVX")
    size = (gt.shape[-1] * 5, gt.shape[-2])
    fps = float(5) if "train" in out_path else float(1)
    writer = cv2.VideoWriter(os.path.join(out_path, "result.mp4"), fourcc, fps, size)

    for file_name in sorted(os.listdir(gts_path)):
        gt_img = np.array(Image.open(os.path.join(gts_path, file_name)))
        rgb = np.array(Image.open(os.path.join(render_path, file_name)))
        gt_obj = np.array(Image.open(os.path.join(gt_colormask_path, file_name)))
        render_obj = np.array(Image.open(os.path.join(colormask_path, file_name)))
        pred_obj_img = np.array(Image.open(os.path.join(pred_obj_path, file_name)))

        result = np.hstack([gt_img, rgb, gt_obj, pred_obj_img, render_obj]).astype("uint8")
        Image.fromarray(result).save(os.path.join(out_path, file_name))
        writer.write(result[:, :, ::-1])

    writer.release()


def removal(dataset: ModelParams,
            iteration: int,
            pipeline: PipelineParams,
            skip_train: bool,
            skip_test: bool,
            opt: OptimizationParams,
            selected_obj_ids_origin,
            removal_thresh: float):
    """
    selected_obj_ids_origin: JSON 中给出的「原始 ID 列表」（old_id）
    不从 JSON 读取 num_classes，num_classes 由 classifier.pth 的 shape 决定。
    """
    device = torch.device("cuda")

    # 1. 加载 gaussians & 场景
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

    print(f"[Removal] id_map size = {len(id_map)}")

    # 3. 从 classifier.pth 推断 num_classes（out_channels）
    ckpt_cls_path = os.path.join(
        dataset.model_path,
        "point_cloud",
        f"iteration_{scene.loaded_iter}",
        "classifier.pth",
    )
    print(f"[Removal] Loading classifier state_dict from: {ckpt_cls_path}")
    state_dict = torch.load(ckpt_cls_path, map_location=device)

    # Conv2d 保存出来一般有 'weight' 和 'bias'
    w = state_dict["weight"]   # [out_channels, in_channels, 1, 1]
    num_classes = w.shape[0]
    in_ch = w.shape[1]
    print(f"[Removal] num_classes (from classifier) = {num_classes}, in_channels = {in_ch}")

    classifier = torch.nn.Conv2d(in_ch, num_classes, kernel_size=1).to(device)
    classifier.load_state_dict(state_dict)

    # 4. 把「原始 ID」映射成「紧凑 ID」
    selected_obj_ids_compact = []
    for oid in selected_obj_ids_origin:
        oid_int = int(oid)
        if oid_int in id_map:
            nid = id_map[oid_int]
            if nid >= num_classes:
                print(
                    f"[Removal][WARN] compact id {nid} (from origin {oid_int}) "
                    f">= num_classes {num_classes}, 忽略该 id"
                )
                continue
            selected_obj_ids_compact.append(nid)
            print(f"[Removal] origin id {oid_int} -> compact id {nid}")
        else:
            print(f"[Removal][WARN] origin id {oid_int} not in id_map, 忽略该 id")

    if len(selected_obj_ids_compact) == 0:
        print("[Removal][ERROR] 没有任何有效的紧凑 ID，可以删除的对象为空，直接退出。")
        return

    # 5. 背景色
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    # 6. 调用 removal_setup 删除指定对象（用紧凑 ID）
    gaussians = removal_setup(
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
    )

    # 7. 不再新建 Scene，直接用原有 scene 的相机 + 修改后的 gaussians 渲染
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
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Object removal script with global ID mapping (Gaga style)")
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
        default="config/object_removal/bear.json",
        help="Path to the configuration file (origin id, thresh, etc.)",
    )

    # ✅ 用 cfg_args 合并参数
    args = get_combined_args(parser)
    print("Removal on model_path:", args.model_path)

    # 🔴 关键一行：语义删除要从 output 模型读，所以必须关掉 lift
    args.lift = False

    # 从 JSON 只读取：删除的原始 ID + 阈值
    try:
        with open(args.config_file, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config_file}' not found.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON config: {e}")
        exit(1)

    args.removal_thresh = config.get("removal_thresh", 0.3)
    # 注意：这里是「原始 ID」
    args.select_obj_id = config.get("select_obj_id", [34])

    safe_state(args.quiet)

    removal(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        opt.extract(args),
        args.select_obj_id,      # 原始 ID 列表
        args.removal_thresh,
    )
