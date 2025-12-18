#
# Copyright (C) 2024, Gaga
# Gaga research group, https://github.com/weijielyu/Gaga
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping

import torch
torch.backends.cudnn.enabled = False

import os
import sys
import uuid
import json
import shutil
from random import randint
from argparse import ArgumentParser, Namespace

from tqdm import tqdm
import wandb

from utils.loss_utils import l1_loss, loss_cls_3d
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args


def training(dataset, opt, pipe,
             testing_iterations, saving_iterations,
             checkpoint_iterations, checkpoint,
             debug_from, use_wandb):
    """
    dataset: 已经经过 ModelParams.extract(args) 并在 main 里修正过路径的对象
    opt    : OptimizationParams.extract(args)
    pipe   : PipelineParams.extract(args)
    """

    first_iter = 0

    # 设置输出目录 & 写 cfg_args（此时 dataset.model_path 已经是 output 目录）
    prepare_output_and_logger(dataset)

    # ======== 拷贝语义相关文件到 output/<output>/<scene> ========
    copy_semantic_files_to_output(dataset)

    # ======== 初始化高斯和场景（从 pretrained 3DGS 加载） ========
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=True)
    gaussians.training_seg_only_setup(opt)

    # ---- 只保留有 objects 的训练相机（关键修复点） ----
    all_train_cams = scene.getTrainCameras()
    train_cams = [cam for cam in all_train_cams
                  if getattr(cam, "objects", None) is not None]

    if len(train_cams) == 0:
        raise RuntimeError(
            "No cameras have 'objects' masks for supervision. "
            "Check that fused_mask/*.npy are correctly generated and loaded in Scene."
        )

    # 读 mask 关联信息（只用于找 object_path）
    matched_mask_path = os.path.join(dataset.source_path, dataset.object_path)

    # ===== 全局一致的 mask id -> 紧凑 id 映射（跨视角一致）=====
    # ⚠️ 注意：这里用的是 fused_mask 目录下的 id_mapping.json（数值 id 映射）
    # scene 根目录那个 id_mapping.json 是 citygml 语义映射，训练阶段不要碰它
    id_map_path = os.path.join(matched_mask_path, "id_mapping.json")

    if os.path.exists(id_map_path):
        with open(id_map_path, "r") as f:
            raw = json.load(f)

        id_map = {}
        for k, v in raw.items():
            try:
                old_id = int(k)
                new_id = int(v)
            except (TypeError, ValueError):
                # 如果有奇怪的 key（比如 "DEBY_LOD2_..."），直接跳过
                continue
            id_map[old_id] = new_id

        print(f"[Global-ID-Mapping] Loaded mapping from {id_map_path}, "
              f"{len(id_map)} foreground ids.")

        # 再扫一遍「有 objects 的训练相机」，看看有没有新的前景 id 没在映射里
        unique_fg = set()
        for cam in train_cams:
            ids = torch.unique(cam.objects).cpu().tolist()
            unique_fg.update([int(x) for x in ids if x > 0])

        missing_ids = [x for x in unique_fg if x not in id_map]
        if missing_ids:
            start_idx = max(id_map.values()) + 1 if id_map else 1
            for i, new_id in enumerate(sorted(missing_ids)):
                id_map[new_id] = start_idx + i

            os.makedirs(matched_mask_path, exist_ok=True)
            with open(id_map_path, "w") as f:
                json.dump(id_map, f)
            print(f"[Global-ID-Mapping] Updated {len(missing_ids)} new ids into {id_map_path}")
        else:
            print("[Global-ID-Mapping] No new ids found. Mapping unchanged.")
    else:
        # 首次：扫描所有「有 objects 的训练相机」，统计全局前景 id（>0）
        unique_fg = set()
        for cam in train_cams:
            ids = torch.unique(cam.objects).cpu().tolist()
            for x in ids:
                xi = int(x)
                if xi > 0:
                    unique_fg.add(xi)

        non_bg_sorted = sorted(unique_fg)
        id_map = {old_id: i + 1 for i, old_id in enumerate(non_bg_sorted)}  # 背景保留 0，前景从 1 开始

        os.makedirs(matched_mask_path, exist_ok=True)
        with open(id_map_path, "w") as f:
            json.dump(id_map, f)

        print(f"[Global-ID-Mapping] Built & saved mapping to {id_map_path}, "
              f"{len(id_map)} foreground ids.")

    # 根据映射确定类别数（背景0 + 前景K）
    num_classes = (max(id_map.values()) if len(id_map) > 0 else 0) + 1
    print(f"[Global-ID-Mapping] num_classes (with background) = {num_classes}")

    # 为快速 remap 构建查找表（lookup）；背景0 -> 0；未出现在映射表的 id 也会默认为 0
    max_old_id = max(id_map.keys()) if len(id_map) > 0 else 0
    lookup = torch.zeros(max_old_id + 1, dtype=torch.long, device="cuda")
    for old_id, new_id in id_map.items():
        lookup[old_id] = new_id

    # 分类头
    classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1).cuda()
    cls_criterion = torch.nn.CrossEntropyLoss(reduction="none")
    cls_optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-4)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # 2D 分块大小，可以按显存再调
    CHUNK_2D = 512

    # 提前缓存 log(num_classes)，避免每步创建 tensor
    log_num_classes = torch.log(torch.tensor(num_classes, device="cuda"))

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, \
                    pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam is not None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # 随机拿一张「有 objects 的训练相机」
        if not viewpoint_stack:
            viewpoint_stack = train_cams.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # 渲染
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        objects_2d = render_pkg["render_seg"]        # [C, H, W]

        if getattr(viewpoint_cam, "objects", None) is None:
            raise RuntimeError("viewpoint_cam.objects is None, but camera was filtered as having masks.")

        gt_obj = viewpoint_cam.objects.cuda().long() # [H_gt, W_gt]

        # ===== 使用全局一致映射进行重映射 =====
        gt_obj_clamped = gt_obj.clone()
        gt_obj_clamped[gt_obj_clamped > max_old_id] = 0
        gt_obj = lookup[gt_obj_clamped]

        # 统一 2D 分辨率
        objects_2d = objects_2d.unsqueeze(0)         # [1, C, H, W]
        _, C2d, H2d, W2d = objects_2d.shape
        H_gt, W_gt = gt_obj.shape
        if (H_gt != H2d) or (W_gt != W2d):
            gt_obj_resized = torch.nn.functional.interpolate(
                gt_obj.unsqueeze(0).unsqueeze(0).float(),
                size=(H2d, W2d),
                mode="nearest",
            ).long().squeeze(0).squeeze(0)
        else:
            gt_obj_resized = gt_obj

        # ---------- 2D 分块分类 ----------
        objects_flat = objects_2d.view(1, C2d, -1)   # [1, C, N]
        gt_flat = gt_obj_resized.view(-1)           # [N]
        N2d = H2d * W2d

        loss_2d_sum = 0.0
        cnt_2d = 0

        for start in range(0, N2d, CHUNK_2D):
            end = min(start + CHUNK_2D, N2d)
            part = objects_flat[:, :, start:end].view(1, C2d, 1, end - start)  # [1, C, 1, chunk]
            out_part = classifier(part)                                        # [1, num_cls, 1, chunk]
            out_part = out_part.view(1, num_classes, end - start)              # [1, num_cls, chunk]

            target_part = gt_flat[start:end].view(1, end - start)              # [1, chunk]

            loss_part = cls_criterion(out_part, target_part).mean()
            loss_2d_sum = loss_2d_sum + loss_part
            cnt_2d += 1

        loss_obj = loss_2d_sum / max(cnt_2d, 1)
        # 用 log(num_classes) 正则化
        loss_obj = loss_obj / log_num_classes

        # ---------- 3D 正则：采样 + 分批线性 ----------
        loss_obj_3d = None
        if iteration % opt.reg3d_interval == 0:
            # feat3d 原始形状一般是 [C, N, 1] 或 [C, N]
            feat3d = gaussians._objects_dc.permute(2, 0, 1).contiguous()
            if feat3d.dim() == 3 and feat3d.shape[-1] == 1:
                feat3d = feat3d.squeeze(-1)          # -> [C, N]

            # 需要 [N, C]
            if feat3d.shape[0] < feat3d.shape[1]:
                feat3d = feat3d.transpose(0, 1).contiguous()   # -> [N, C]

            assert feat3d.dim() == 2, f"Unexpected feat3d shape: {feat3d.shape}"

            N3d, C3d = feat3d.shape  # N3d = 高斯数

            # 这里用配置里的 max_points 做子集
            max_pts = getattr(opt, "reg3d_max_points", 300000)
            if N3d > max_pts:
                # 随机选 max_pts 个点
                idx = torch.randperm(N3d)[:max_pts]
                feat3d = feat3d[idx]
                xyz_for_reg = gaussians._xyz.squeeze()[idx]
            else:
                xyz_for_reg = gaussians._xyz.squeeze()

            N3d_sub = feat3d.shape[0]

            # 从 classifier 拿权重做线性层
            W = classifier.weight.view(classifier.out_channels, -1)  # [num_classes, C3d]
            b = classifier.bias                                      # [num_classes]

            # 全部搬到 CPU 算，最省显存
            feat3d_cpu = feat3d.detach().cpu()   # [N3d_sub, C3d]
            W_cpu = W.detach().cpu()
            b_cpu = b.detach().cpu()

            chunk3d = 256  # 再不行就 128
            logits_list = []

            for start in range(0, N3d_sub, chunk3d):
                end = min(start + chunk3d, N3d_sub)
                f_part = feat3d_cpu[start:end, :]              # [chunk, C3d]
                logits_part = f_part @ W_cpu.T + b_cpu         # [chunk, num_classes]
                logits_list.append(logits_part)

            logits3d_cpu = torch.cat(logits_list, dim=0)       # [N3d_sub, num_classes]
            prob_obj3d = torch.softmax(logits3d_cpu, dim=1).to("cuda")

            # 用采样后的 xyz 做 3d 正则
            loss_obj_3d = loss_cls_3d(
                xyz_for_reg.detach(),
                prob_obj3d,
                opt.reg3d_k,
                opt.reg3d_lambda_val,
                opt.reg3d_max_points,
                opt.reg3d_sample_size,
            )

            loss = loss_obj + loss_obj_3d
        else:
            loss = loss_obj

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            try:
                training_report(
                    iteration,
                    loss_obj,
                    loss,
                    l1_loss,
                    iter_start.elapsed_time(iter_end),
                    testing_iterations,
                    scene,
                    render,
                    (pipe, background),
                    loss_obj_3d,
                    use_wandb,
                )
            except Exception:
                pass

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)
                torch.save(
                    classifier.state_dict(),
                    os.path.join(
                        scene.model_path,
                        f"point_cloud/iteration_{iteration}",
                        "classifier.pth",
                    ),
                )

            try:
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                    cls_optimizer.step()
                    cls_optimizer.zero_grad()
            except Exception:
                print("Error in optimizer step")
                pass

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )


def prepare_output_and_logger(dataset_params):
    """
    dataset_params: ModelParams.extract(args) 的结果
    这里的 model_path 被当作“输出模型路径”使用。
    """
    if not dataset_params.model_path:
        # 理论上在 main 里已经用 --output 设过 model_path 了，这里只是兜底
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        dataset_params.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(dataset_params.model_path))
    os.makedirs(dataset_params.model_path, exist_ok=True)
    with open(os.path.join(dataset_params.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(dataset_params))))


def copy_semantic_files_to_output(dataset_params):
    """
    在 scene 根目录下查找:
      - city_semantics.json
      - id_mapping.json
      - clip_features_fused.npy
    然后复制到:
      <output_root>/<output>/<scene>/ 目录下
    其中 clip_features_fused.npy 重命名为 clip_semantics.npy
    """
    scene_root = dataset_params.source_path              # ./dataset/<scene>
    output_root = dataset_params.model_path              # ./output/<output> （已经在 main 里修成了）
    scene_name = os.path.basename(scene_root.rstrip("/"))
    out_scene_dir = os.path.join(output_root, scene_name)
    os.makedirs(out_scene_dir, exist_ok=True)

    # 1) city_semantics.json
    src_city = os.path.join(scene_root, "city_semantics.json")
    if os.path.exists(src_city):
        dst_city = os.path.join(out_scene_dir, "city_semantics.json")
        shutil.copy2(src_city, dst_city)
        print(f"[Semantic Copy] city_semantics.json -> {dst_city}")
    else:
        print(f"[Semantic Copy] city_semantics.json not found in {scene_root}, skip.")

    # 2) id_mapping.json
    src_idmap = os.path.join(scene_root, "id_mapping.json")
    if os.path.exists(src_idmap):
        dst_idmap = os.path.join(out_scene_dir, "id_mapping.json")
        shutil.copy2(src_idmap, dst_idmap)
        print(f"[Semantic Copy] id_mapping.json -> {dst_idmap}")
    else:
        print(f"[Semantic Copy] id_mapping.json not found in {scene_root}, skip.")

    # 3) clip_features_fused.npy  -> clip_semantics.npy
    src_clip = os.path.join(scene_root, "clip_features_fused.npy")
    if os.path.exists(src_clip):
        dst_clip = os.path.join(out_scene_dir, "clip_semantics.npy")
        shutil.copy2(src_clip, dst_clip)
        print(f"[Semantic Copy] clip_features_fused.npy -> {dst_clip}")
    else:
        print(f"[Semantic Copy] clip_features_fused.npy not found in {scene_root}, skip.")


def training_report(iteration, loss_obj, loss, l1_loss,
                    elapsed, testing_iterations, scene: Scene,
                    renderFunc, renderArgs, loss_obj_3d, use_wandb):

    if use_wandb:
        if loss_obj_3d is not None:
            wandb.log({
                "train_loss_patches/total_loss": loss.item(),
                "train_loss_patches/loss_obj": loss_obj.item(),
                "train_loss_patches/loss_obj_3d": loss_obj_3d.item(),
                "iter_time": elapsed,
                "iter": iteration,
            })
        else:
            wandb.log({
                "train_loss_patches/total_loss": loss.item(),
                "train_loss_patches/loss_obj": loss_obj.item(),
                "iter_time": elapsed,
                "iter": iteration,
            })

    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")

    # ========= Gaga 参数组 =========
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    # ========= 额外通用参数 =========
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[10_000])
    parser.add_argument("--use_wandb", action='store_true', default=False, help="Use wandb to record loss value")
    parser.add_argument("--my_debug_tag", action='store_true', default=False, help="Debug tag for my own purpose")

    # ====== 使用 get_combined_args：支持从 pretrained model 的 cfg_args 里读默认参数 ======
    args = get_combined_args(parser)

    # 保证这些字段存在
    if not hasattr(args, "test_iterations"):
        args.test_iterations = [10_000]
    if not hasattr(args, "save_iterations"):
        args.save_iterations = [10_000]
    if not hasattr(args, "checkpoint_iterations"):
        args.checkpoint_iterations = [10_000]
    if not hasattr(args, "quiet"):
        args.quiet = False

    args.save_iterations.append(args.iterations)

    # 临时方案：原始实现里强行打开 lift
    assert args.lift is False
    args.lift = True

    # ====== 自动读取 config/train.json ======
    project_root = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(project_root, "config", "train.json")
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
        print(f"[Config] Loaded training config from {config_path}")
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse the JSON configuration file: {e}")
        sys.exit(1)

    # 将 JSON 中的 3D 正则参数写回 args（不再用 args.num_classes）
    args.densify_until_iter = config.get("densify_until_iter", 15000)
    args.reg3d_interval = config.get("reg3d_interval", 2)
    args.reg3d_k = config.get("reg3d_k", 5)
    args.reg3d_lambda_val = config.get("reg3d_lambda_val", 2)
    args.reg3d_max_points = config.get("reg3d_max_points", 300000)
    args.reg3d_sample_size = config.get("reg3d_sample_size", 1000)

    # ====== 通过 ModelParams.extract 生成 dataset_params，并修正路径语义 ======
    dataset_params = lp.extract(args)
    dataset_params.object_path = "fused_mask"

    # 此时：
    #   dataset_params.source_path       = <repo>/dataset/<scene>
    #   dataset_params.model_path        = <repo>/model/<model>      （来自 --model）
    #   dataset_params.trained_model_path= <repo>/output/<output>    （来自 --output）

    # 我们希望：
    #   pretrained_dir = <repo>/model/<model>
    #   output_dir     = <repo>/output/<output>
    pretrained_dir = dataset_params.model_path
    output_dir     = dataset_params.trained_model_path

    if not pretrained_dir:
        print("[Error] Pretrained model path is empty. Please specify --model.")
        sys.exit(1)
    if not output_dir:
        print("[Error] Output path is empty. Please specify --output.")
        sys.exit(1)

    # 修正给 Scene 使用的语义：
    #   Scene 会用 trained_model_path 去找 point_cloud（预训练）
    #   scene.save() 会用 model_path 作为保存目录（输出）
    dataset_params.trained_model_path = pretrained_dir
    dataset_params.model_path = output_dir

    print(f"[Path] source_path        = {dataset_params.source_path}")
    print(f"[Path] pretrained_model   = {pretrained_dir}")
    print(f"[Path] output(model_path) = {output_dir}")

    print("Optimizing (pretrained) from " + pretrained_dir)
    print("Saving new model to        " + output_dir)

    if args.use_wandb:
        wandb.init(project="Gaga")
        wandb.config.args = args
        run_name = "_".join(output_dir.split("/")[1:])
        wandb.run.name = run_name

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # 注意：resolution 仍然来自 ModelParams，用户可以继续用 --resolution/-r
    training(
        dataset_params,
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        checkpoint=None,          # ← 不再用 args.start_checkpoint
        debug_from=args.debug_from,
        use_wandb=args.use_wandb,
    )

    print("\nTraining complete.")
