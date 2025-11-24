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
from random import randint
from utils.loss_utils import l1_loss, loss_cls_3d
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import wandb
import json

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, use_wandb):
    first_iter = 0
    prepare_output_and_logger(dataset)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=True)
    gaussians.training_seg_only_setup(opt)

    # 读 mask 关联信息
    matched_mask_path = os.path.join(dataset.source_path, dataset.object_path)

    # ===== 全局一致的 SAM id -> 紧凑 id 映射（跨视角一致）=====
    id_map_path = os.path.join(matched_mask_path, "id_mapping.json")
    if os.path.exists(id_map_path):
        with open(id_map_path, "r") as f:
            id_map = {int(k): int(v) for k, v in json.load(f).items()}
        print(f"[Global-ID-Mapping] Loaded mapping from {id_map_path}, "
              f"{len(id_map)} foreground ids.")
        unique_fg = set()
        for cam in scene.getTrainCameras():
            ids = torch.unique(cam.objects).cpu().tolist()
            unique_fg.update([int(x) for x in ids if x > 0])

        # 检查是否有新 ID
        missing_ids = [x for x in unique_fg if x not in id_map]
        if missing_ids:
            start_idx = max(id_map.values()) + 1 if id_map else 1
            for i, new_id in enumerate(sorted(missing_ids)):
                id_map[new_id] = start_idx + i

            # 保存更新后的映射文件
            with open(id_map_path, "w") as f:
                json.dump(id_map, f)
            print(f"[Global-ID-Mapping] Updated {len(missing_ids)} new ids into {id_map_path}")
        else:
            print("[Global-ID-Mapping] No new ids found. Mapping unchanged.")
    else:
        # 扫描所有训练相机，统计全局前景 id（>0）
        unique_fg = set()
        for cam in scene.getTrainCameras():
            ids = torch.unique(cam.objects).cpu().tolist()
            for x in ids:
                xi = int(x)
                if xi > 0:
                    unique_fg.add(xi)
        non_bg_sorted = sorted(unique_fg)
        id_map = {old_id: i + 1 for i, old_id in enumerate(non_bg_sorted)}  # 背景保留 0，前景从 1 开始
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

    # 这两个可以按显存再往下调
    CHUNK_2D = 512

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
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

        # 随机拿一张训练相机
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # 渲染
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        objects_2d = render_pkg["render_seg"]        # [C, H, W]
        gt_obj = viewpoint_cam.objects.cuda().long() # [H_gt, W_gt]

        # ===== 使用全局一致映射进行重映射 =====
        gt_obj = lookup[gt_obj] 

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
        gt_flat = gt_obj_resized.view(-1)            # [N]
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
        loss_obj = loss_obj / torch.log(torch.tensor(num_classes, device="cuda"))

        # ---------- 3D 正则：采样 + 分批线性 ----------
        loss_obj_3d = None
        if iteration % opt.reg3d_interval == 0:
            # 原始特征：你现在这里其实会得到 [C, N] 那种形状
            feat3d = gaussians._objects_dc.permute(2, 0, 1).contiguous()
            # 有些版本是 [C, N, 1]，压掉最后一个 1
            if feat3d.dim() == 3 and feat3d.shape[-1] == 1:
                feat3d = feat3d.squeeze(-1)          # -> [C, N]

            # 我们要 [N, C]
            if feat3d.shape[0] < feat3d.shape[1]:
                feat3d = feat3d.transpose(0, 1).contiguous()   # -> [N, C]

            N3d, C3d = feat3d.shape  # N3d = 高斯数，可能几百万

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
                # [chunk, C] @ [C, num_classes] -> [chunk, num_classes]
                logits_part = f_part @ W_cpu.T + b_cpu
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
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
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
            except:
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
            except:
                print("Error in optimizer step")
                pass

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


def training_report(iteration, loss_obj, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, loss_obj_3d, use_wandb):

    if use_wandb:
        if loss_obj_3d:
            wandb.log({"train_loss_patches/total_loss": loss.item(), "train_loss_patches/loss_obj": loss_obj.item(), "train_loss_patches/loss_obj_3d": loss_obj_3d.item(), "iter_time": elapsed, "iter": iteration})
        else:
            wandb.log({"train_loss_patches/total_loss": loss.item(), "train_loss_patches/loss_obj": loss_obj.item(), "iter_time": elapsed, "iter": iteration})
   
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[10_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # Add an argument for the configuration file
    parser.add_argument("--config_file", type=str, default="config/train.json", help="Path to the configuration file")
    parser.add_argument("--use_wandb", action='store_true', default=False, help="Use wandb to record loss value")
    parser.add_argument("--my_debug_tag", action='store_true', default=False, help="Debug tag for my own purpose")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # temproray solution
    assert args.lift == False
    args.lift = True

    # Read and parse the configuration file
    try:
        with open(args.config_file, 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config_file}' not found.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse the JSON configuration file: {e}")
        exit(1)

    args.densify_until_iter = config.get("densify_until_iter", 15000)
    args.num_classes = config.get("num_classes", 200)
    args.reg3d_interval = config.get("reg3d_interval", 2)
    args.reg3d_k = config.get("reg3d_k", 5)
    args.reg3d_lambda_val = config.get("reg3d_lambda_val", 2)
    args.reg3d_max_points = config.get("reg3d_max_points", 300000)
    args.reg3d_sample_size = config.get("reg3d_sample_size", 1000)
    
    print("Optimizing " + args.model_path)

    if args.use_wandb:
        wandb.init(project="Gaga")
        wandb.config.args = args
        run_name = "_".join(args.model_path.split("/")[1:])
        wandb.run.name = run_name
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.use_wandb)

    # All done
    print("\nTraining complete.")