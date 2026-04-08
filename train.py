#
# Copyright (C) 2026, GS4City
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaga
# Gaga research group, https://github.com/weijielyu/Gaga
#

import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

import os
import sys
import uuid
import json
import shutil
import time
import subprocess
from pathlib import Path
from random import randint
from argparse import ArgumentParser, Namespace

from tqdm import tqdm
import wandb

from utils.loss_utils import l1_loss, loss_cls_3d
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def _bytes_to_mb(x: int) -> float:
    return float(x) / (1024.0 ** 2)

def _dir_size_bytes(path: str) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total

def _nvidia_smi_mem_used_mb(gpu_index: int = 0):
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return float(out.splitlines()[0])
    except Exception:
        return None
    
def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    use_wandb,
):
    first_iter = 0
    train_wall_start = time.time()
    train_perf_start = time.perf_counter()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    tb_writer = prepare_output_and_logger(dataset)

    use_clip_semantics = (getattr(dataset, "object_path", "") == "fused_mask")
    copy_semantic_files_to_output(dataset, use_clip_semantics=use_clip_semantics)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=True)
    gaussians.training_seg_only_setup(opt)

    try:
        test_cams = scene.getTestCameras()
        test_names = []
        for cam in test_cams:
            name = getattr(cam, "image_name", None)
            if name is None:
                path = getattr(cam, "image_path", None)
                if path is not None:
                    name = os.path.splitext(os.path.basename(path))[0]
                else:
                    name = str(getattr(cam, "uid", "unknown"))
            test_names.append(name)

        test_names = sorted(set(test_names))
        print("\n================ FINAL TEST SET (NOT USED FOR TRAINING) ================")
        print(f"Test images: {len(test_names)}")
        if len(test_names) > 0:
            for n in test_names:
                print("  ", n)
        print("========================================================================\n")
    except Exception as e:
        print(f"[Warn] Failed to print/save test set list: {e}")

    all_train_cams = scene.getTrainCameras()
    train_cams = [cam for cam in all_train_cams if getattr(cam, "objects", None) is not None]

    if len(train_cams) == 0:
        raise RuntimeError(
            "No cameras have 'objects' masks for supervision. "
            "Check that fused_mask/*.npy (or sam_mask/*.npy) are correctly generated and loaded in Scene."
        )

    matched_mask_path = os.path.join(dataset.source_path, dataset.object_path)
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
                continue
            id_map[old_id] = new_id

        print(f"[Global-ID-Mapping] Loaded mapping from {id_map_path}, {len(id_map)} foreground ids.")

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
        unique_fg = set()
        for cam in train_cams:
            ids = torch.unique(cam.objects).cpu().tolist()
            for x in ids:
                xi = int(x)
                if xi > 0:
                    unique_fg.add(xi)

        non_bg_sorted = sorted(unique_fg)
        id_map = {old_id: i + 1 for i, old_id in enumerate(non_bg_sorted)}

        os.makedirs(matched_mask_path, exist_ok=True)
        with open(id_map_path, "w") as f:
            json.dump(id_map, f)

        print(f"[Global-ID-Mapping] Built & saved mapping to {id_map_path}, {len(id_map)} foreground ids.")

    num_classes = (max(id_map.values()) if len(id_map) > 0 else 0) + 1
    print(f"[Global-ID-Mapping] num_classes (with background) = {num_classes}")

    max_old_id = max(id_map.keys()) if len(id_map) > 0 else 0
    lookup = torch.zeros(max_old_id + 1, dtype=torch.long, device="cuda")
    for old_id, new_id in id_map.items():
        lookup[old_id] = new_id

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

    CHUNK_2D = 16384
    CHUNK_3D = getattr(opt, "reg3d_chunk", 8192)

    log_num_classes = torch.log(torch.tensor(num_classes, device="cuda"))

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = (
                    network_gui.receive()
                )
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

        if not viewpoint_stack:
            viewpoint_stack = train_cams.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        objects_2d = render_pkg["render_seg"]

        if getattr(viewpoint_cam, "objects", None) is None:
            raise RuntimeError("viewpoint_cam.objects is None, but camera was filtered as having masks.")

        gt_obj = viewpoint_cam.objects.cuda().long()

        gt_obj_clamped = gt_obj.clone()
        gt_obj_clamped[gt_obj_clamped > max_old_id] = 0
        gt_obj = lookup[gt_obj_clamped]

        objects_2d = objects_2d.unsqueeze(0)
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

        objects_flat = objects_2d.view(1, C2d, -1)
        gt_flat = gt_obj_resized.view(-1)
        N2d = H2d * W2d

        loss_2d_sum = 0.0
        cnt_2d = 0

        for start in range(0, N2d, CHUNK_2D):
            end = min(start + CHUNK_2D, N2d)
            part = objects_flat[:, :, start:end].view(1, C2d, 1, end - start)
            out_part = classifier(part)
            out_part = out_part.view(1, num_classes, end - start)

            target_part = gt_flat[start:end].view(1, end - start)
            loss_part = cls_criterion(out_part, target_part).mean()

            loss_2d_sum = loss_2d_sum + loss_part
            cnt_2d += 1

        loss_obj = loss_2d_sum / max(cnt_2d, 1)
        loss_obj = loss_obj / log_num_classes

        loss_obj_3d = None
        if iteration % opt.reg3d_interval == 0:
            feat3d = gaussians._objects_dc.permute(2, 0, 1).contiguous()

            if feat3d.dim() != 3:
                raise RuntimeError(f"Unexpected _objects_dc permuted shape: {feat3d.shape}. Expected 3D (C,H,W).")

            N3d = feat3d.shape[1]

            max_pts = getattr(opt, "reg3d_max_points", 300000)
            if N3d > max_pts:
                idx = torch.randperm(N3d, device=feat3d.device)[:max_pts]
                feat3d = feat3d[:, idx, :]
                xyz_for_reg = gaussians._xyz.squeeze()[idx]
            else:
                xyz_for_reg = gaussians._xyz.squeeze()

            logits_list = []
            N3d_sub = feat3d.shape[1]
            for start in range(0, N3d_sub, CHUNK_3D):
                end = min(start + CHUNK_3D, N3d_sub)
                feat_part = feat3d[:, start:end, :]
                logits_part = classifier(feat_part)
                logits_list.append(logits_part)

            logits3d = torch.cat(logits_list, dim=1)
            prob_obj3d = torch.softmax(logits3d, dim=0).squeeze(-1).permute(1, 0)

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
                    tb_writer,
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

    train_wall_end = time.time()
    train_perf_end = time.perf_counter()
    if tb_writer:
        tb_writer.add_scalar("perf/train_wall_time_s", train_wall_end - train_wall_start, 0)
        tb_writer.add_scalar("perf/train_perf_time_s", train_perf_end - train_perf_start, 0)
        tb_writer.flush()
    
    if tb_writer:
        tb_writer.close()


def prepare_output_and_logger(dataset_params):
    if not dataset_params.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        dataset_params.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(dataset_params.model_path))
    os.makedirs(dataset_params.model_path, exist_ok=True)
    with open(os.path.join(dataset_params.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(dataset_params))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(dataset_params.model_path)
    else:
        print("Tensorboard not available: not logging progress")

    return tb_writer


def copy_semantic_files_to_output(dataset_params, use_clip_semantics: bool = True):
    scene_root = dataset_params.source_path
    output_root = dataset_params.model_path
    scene_name = os.path.basename(scene_root.rstrip("/"))
    out_scene_dir = os.path.join(output_root, scene_name)
    os.makedirs(out_scene_dir, exist_ok=True)

    src_city = os.path.join(scene_root, "city_semantics.json")
    if os.path.exists(src_city):
        dst_city = os.path.join(out_scene_dir, "city_semantics.json")
        shutil.copy2(src_city, dst_city)
        print(f"[Semantic Copy] city_semantics.json -> {dst_city}")
    else:
        print(f"[Semantic Copy] city_semantics.json not found in {scene_root}, skip.")

    src_idmap = os.path.join(scene_root, "id_mapping.json")
    if os.path.exists(src_idmap):
        dst_idmap = os.path.join(out_scene_dir, "id_mapping.json")
        shutil.copy2(src_idmap, dst_idmap)
        print(f"[Semantic Copy] id_mapping.json -> {dst_idmap}")
    else:
        print(f"[Semantic Copy] id_mapping.json not found in {scene_root}, skip.")

    if use_clip_semantics:
        src_clip = os.path.join(scene_root, "clip_features_fused.npy")
        if os.path.exists(src_clip):
            dst_clip = os.path.join(out_scene_dir, "clip_semantics.npy")
            shutil.copy2(src_clip, dst_clip)
            print(f"[Semantic Copy] clip_features_fused.npy -> {dst_clip}")
        else:
            print(f"[Semantic Copy] clip_features_fused.npy not found in {scene_root}, skip.")


def training_report(
    tb_writer,
    iteration,
    loss_obj,
    loss,
    l1_loss,
    elapsed,  # CUDA event elapsed_time: ms
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    loss_obj_3d,
    use_wandb,
):
    LOG_GPU_EVERY = 200
    LOG_SMI_EVERY = 2000
    LOG_STORAGE_EVERY = 5000

    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/loss_obj_2d", loss_obj.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        if loss_obj_3d is not None:
            tb_writer.add_scalar("train_loss_patches/loss_obj_3d", loss_obj_3d.item(), iteration)

        # time (ms -> also log seconds)
        tb_writer.add_scalar("perf/iter_time_ms", elapsed, iteration)
        tb_writer.add_scalar("perf/iter_time_s", elapsed / 1000.0, iteration)

        # GPU memory (PyTorch)
        if (iteration % LOG_GPU_EVERY == 0) and torch.cuda.is_available():
            tb_writer.add_scalar("gpu/memory_allocated_mb",
                                 _bytes_to_mb(torch.cuda.memory_allocated()),
                                 iteration)
            tb_writer.add_scalar("gpu/memory_reserved_mb",
                                 _bytes_to_mb(torch.cuda.memory_reserved()),
                                 iteration)
            tb_writer.add_scalar("gpu/peak_memory_allocated_mb",
                                 _bytes_to_mb(torch.cuda.max_memory_allocated()),
                                 iteration)
            tb_writer.add_scalar("gpu/peak_memory_reserved_mb",
                                 _bytes_to_mb(torch.cuda.max_memory_reserved()),
                                 iteration)

        # GPU memory (nvidia-smi, slow)
        if iteration % LOG_SMI_EVERY == 0:
            smi_used = _nvidia_smi_mem_used_mb(gpu_index=0)
            if smi_used is not None:
                tb_writer.add_scalar("gpu/nvidia_smi_memory_used_mb", smi_used, iteration)

        # storage (slow)
        if iteration % LOG_STORAGE_EVERY == 0:
            model_path = getattr(scene, "model_path", None)
            if model_path is not None:
                tb_writer.add_scalar("storage/model_path_mb",
                                     _bytes_to_mb(_dir_size_bytes(model_path)),
                                     iteration)

                pc_dir = os.path.join(model_path, "point_cloud")
                tb_writer.add_scalar("storage/point_cloud_dir_mb",
                                     _bytes_to_mb(_dir_size_bytes(pc_dir)),
                                     iteration)

                # semantic subdir (contains clip_semantics.npy etc.)
                try:
                    subdirs = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
                    if len(subdirs) > 0:
                        sem_dir = os.path.join(model_path, subdirs[0])
                        tb_writer.add_scalar("storage/semantic_dir_mb",
                                             _bytes_to_mb(_dir_size_bytes(sem_dir)),
                                             iteration)
                except Exception:
                    pass

                # optional candidates
                candidates = [
                    os.path.join(model_path, "identity_features"),
                    os.path.join(model_path, "feature_bank"),
                    os.path.join(model_path, "features"),
                ]
                for c in candidates:
                    tag = "storage/" + os.path.basename(c) + "_mb"
                    tb_writer.add_scalar(tag, _bytes_to_mb(_dir_size_bytes(c)), iteration)

    if use_wandb:
        log_dict = {
            "train_loss_patches/total_loss": loss.item(),
            "train_loss_patches/loss_obj": loss_obj.item(),
            "perf/iter_time_ms": elapsed,
            "perf/iter_time_s": elapsed / 1000.0,
            "iter": iteration,
        }
        if loss_obj_3d is not None:
            log_dict["train_loss_patches/loss_obj_3d"] = loss_obj_3d.item()

        if (iteration % LOG_GPU_EVERY == 0) and torch.cuda.is_available():
            log_dict["gpu/memory_allocated_mb"] = _bytes_to_mb(torch.cuda.memory_allocated())
            log_dict["gpu/memory_reserved_mb"] = _bytes_to_mb(torch.cuda.memory_reserved())
            log_dict["gpu/peak_memory_allocated_mb"] = _bytes_to_mb(torch.cuda.max_memory_allocated())
            log_dict["gpu/peak_memory_reserved_mb"] = _bytes_to_mb(torch.cuda.max_memory_reserved())

        if iteration % LOG_SMI_EVERY == 0:
            smi_used = _nvidia_smi_mem_used_mb(gpu_index=0)
            if smi_used is not None:
                log_dict["gpu/nvidia_smi_memory_used_mb"] = smi_used

        wandb.log(log_dict)

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[10_000])
    parser.add_argument("--use_wandb", action="store_true", default=False, help="Use wandb to record loss value")
    parser.add_argument("--my_debug_tag", action="store_true", default=False, help="Debug tag")

    args = get_combined_args(parser)

    if not hasattr(args, "test_iterations"):
        args.test_iterations = [10_000]
    if not hasattr(args, "save_iterations"):
        args.save_iterations = [10_000]
    if not hasattr(args, "checkpoint_iterations"):
        args.checkpoint_iterations = [10_000]
    if not hasattr(args, "quiet"):
        args.quiet = False

    args.save_iterations.append(args.iterations)

    assert args.lift is False
    args.lift = True

    project_root = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(project_root, "config", "train.json")
    try:
        with open(config_path, "r") as file:
            config = json.load(file)
        print(f"[Config] Loaded training config from {config_path}")
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse the JSON configuration file: {e}")
        sys.exit(1)

    args.densify_until_iter = config.get("densify_until_iter", 15000)
    args.reg3d_interval = config.get("reg3d_interval", 2)
    args.reg3d_k = config.get("reg3d_k", 5)
    args.reg3d_lambda_val = config.get("reg3d_lambda_val", 2)
    args.reg3d_max_points = config.get("reg3d_max_points", 300000)
    args.reg3d_sample_size = config.get("reg3d_sample_size", 1000)

    dataset_params = lp.extract(args)

    fused_mask_dir = os.path.join(dataset_params.source_path, "fused_mask")
    if os.path.isdir(fused_mask_dir):
        dataset_params.object_path = "fused_mask"
        print(f"[Mask] Using fused_mask: {fused_mask_dir}")
    else:
        dataset_params.object_path = "sam_mask"
        print(f"[Mask] fused_mask not found. Using sam_mask instead: {os.path.join(dataset_params.source_path, 'sam_mask')}")

    pretrained_dir = dataset_params.model_path
    output_dir = dataset_params.trained_model_path

    if not pretrained_dir:
        print("[Error] Pretrained model path is empty. Please specify --model.")
        sys.exit(1)
    if not output_dir:
        print("[Error] Output path is empty. Please specify --output.")
        sys.exit(1)

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

    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        dataset_params,
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        checkpoint=None,
        debug_from=args.debug_from,
        use_wandb=args.use_wandb,
    )

    print("\nTraining complete.")