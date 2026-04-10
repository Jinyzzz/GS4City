# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import os
import sys
import json
import colorsys

import torch
torch.backends.cudnn.enabled = False

from PIL import Image
from tqdm import tqdm
from os import makedirs
from argparse import ArgumentParser, Namespace

import torchvision
import numpy as np
import cv2
from sklearn.decomposition import PCA

from scene import Scene
from gaussian_renderer import render
from utils.general_utils import safe_state
from utils.graphics_utils import getWorld2View2
from utils.pose_utils import generate_ellipse_path, generate_spiral_path
from arguments import ModelParams, PipelineParams, RenderParams
from gaussian_renderer import GaussianModel


# ============ Rendering helpers ============

def feature_to_rgb(features):
    # Input feature shape: (C, H, W) or (1, C, H, W)

    H, W = features.shape[-2], features.shape[-1]

    if features.dim() == 3:
        # [C, H, W] -> [H*W, C]
        features_reshaped = features.view(features.shape[0], -1).T
    elif features.dim() == 4:
        # [1, C, H, W] -> [H*W, C]
        features_reshaped = features.view(features.shape[1], -1).T
    else:
        raise ValueError(f"Unexpected feature dim: {features.shape}")

    # PCA to 3 components for visualization
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_reshaped.cpu().numpy())

    # Reshape back to (H, W, 3)
    pca_result = pca_result.reshape(H, W, 3)

    # Normalize to [0, 255]
    pca_normalized = 255 * (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())

    rgb_array = pca_normalized.astype('uint8')

    return rgb_array


def id2rgb(id, max_num_obj=256):
    """
    Map an integer ID to a stable color.
    No longer enforces id <= max_num_obj; only requires id >= 0.
    """
    if id < 0:
        raise ValueError("ID should be non-negative")

    golden_ratio = 1.6180339887
    h = ((id * golden_ratio) % 1)   # hue in [0, 1)
    s = 0.5 + (id % 2) * 0.5        # alternate 0.5 / 1.0 saturation
    l = 0.5

    rgb = np.zeros((3,), dtype=np.uint8)
    if id == 0:  # background / invalid region
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r * 255), int(g * 255), int(b * 255)

    return rgb


def visualize_obj(objects):
    """
    objects: numpy array [H, W], values are object IDs (expected to be original IDs)
    """
    rgb_mask = np.zeros((*objects.shape[-2:], 3), dtype=np.uint8)
    all_obj_ids = np.unique(objects)
    for id in all_obj_ids:
        colored_mask = id2rgb(int(id))
        rgb_mask[objects == id] = colored_mask
    return rgb_mask


def render_video_func(source_path,
                      model_path,
                      iteration,
                      views,
                      gaussians,
                      pipeline,
                      background,
                      classifier,
                      inverse_lookup,
                      fps=30):
    """
    Render a video:
      - left: RGB render
      - right: segmentation visualization (mapped back to original IDs)
    """
    render_path = os.path.join(model_path, 'video', "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)
    view = views[0]

    if source_path.find('llff') != -1:
        render_poses = generate_spiral_path(np.load(source_path + '/poses_bounds.npy'))
    else:
        render_poses = generate_ellipse_path(views)

    size = (view.original_image.shape[2] * 2, view.original_image.shape[1])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    final_video = cv2.VideoWriter(os.path.join(render_path, 'final_video.mp4'), fourcc, fps, size)

    for idx, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        view.world_view_transform = torch.tensor(
            getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)
        ).transpose(0, 1).cuda()
        view.full_proj_transform = (
            view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]
        rendering = render(view, gaussians, pipeline, background)

        # RGB rendering
        img = torch.clamp(rendering["render"], min=0., max=1.).cpu()  # [3, H, W]

        # Segmentation features
        rendering_obj = rendering["render_seg"]  # usually [C, H, W]
        if rendering_obj.dim() == 3:
            rendering_obj = rendering_obj.unsqueeze(0)  # -> [1, C, H, W]

        # Predict compact IDs
        logits = classifier(rendering_obj)              # [1, num_classes, H, W]
        pred_compact = torch.argmax(logits, dim=1)[0]  # [H, W], long

        # Map back to original IDs
        pred_original = inverse_lookup[pred_compact]   # [H, W], long
        pred_original_np = pred_original.cpu().numpy().astype(np.int32)

        # Color visualization using original IDs
        pred_obj_mask = visualize_obj(pred_original_np) / 255.0
        pred_obj_mask = torch.clamp(torch.tensor(pred_obj_mask), min=0., max=1.).permute(2, 0, 1)

        # Concatenate: RGB | Seg
        combined_img = torch.cat([img, pred_obj_mask], dim=2)
        torchvision.utils.save_image(combined_img, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        video_img = (combined_img.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)[..., ::-1]
        final_video.write(video_img)

    final_video.release()


def render_set(model_path,
               name,
               iteration,
               views,
               gaussians,
               pipeline,
               background,
               classifier,
               inverse_lookup):
    """
    Save:
      - RGB renders
      - GT masks (gray + color) if available
      - predicted masks (original IDs: gray + color)
      - feature PCA visualization
    """
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    colormask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "objects_feature16")

    if name == "train":
        gt_colormask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_objects_color")
        makedirs(gt_colormask_path, exist_ok=True)
        gt_object_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_objects")
        makedirs(gt_object_path, exist_ok=True)

    pred_obj_path = os.path.join(model_path, name, "ours_{}".format(iteration), "objects_pred")   # color predictions
    test_obj_path = os.path.join(model_path, name, "ours_{}".format(iteration), "objects_test")   # gray predictions (original IDs)

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(colormask_path, exist_ok=True)
    makedirs(pred_obj_path, exist_ok=True)
    makedirs(test_obj_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]
        rendering_obj = results["render_seg"]  # [C, H, W] or [1, C, H, W]

        # Ensure [1, C, H, W]
        if rendering_obj.dim() == 3:
            rendering_obj = rendering_obj.unsqueeze(0)

        # ---------- Predict compact IDs ----------
        logits = classifier(rendering_obj)              # [1, num_classes, H, W]
        pred_compact = torch.argmax(logits, dim=1)[0]  # [H, W]

        # ---------- Map back to original IDs ----------
        pred_original = inverse_lookup[pred_compact]             # [H, W], long
        pred_original_np = pred_original.cpu().numpy().astype(np.int32)

        # Color visualization using original IDs
        pred_obj_mask = visualize_obj(pred_original_np)

        # ---------- GT (train only, if available) ----------
        gt_np = None
        gt_rgb_mask = None

        if name == "train":
            if view.objects is not None:
                gt_objects = view.objects
                gt_np = gt_objects.cpu().numpy().astype(np.int32)
                gt_rgb_mask = visualize_obj(gt_np)
            else:
                # Some train views might not have GT masks
                pass

        # ---------- Feature PCA visualization ----------
        rgb_mask = feature_to_rgb(rendering_obj.squeeze(0))

        Image.fromarray(rgb_mask).save(os.path.join(colormask_path, '{}'.format(view.image_name) + ".png"))

        if name == "train":
            if gt_np is not None:
                Image.fromarray(gt_np.astype(np.uint16)).save(
                    os.path.join(gt_object_path, '{}'.format(view.image_name) + ".png")
                )
            if gt_rgb_mask is not None:
                Image.fromarray(gt_rgb_mask).save(
                    os.path.join(gt_colormask_path, '{}'.format(view.image_name) + ".png")
                )

        # Predicted color (original IDs)
        Image.fromarray(pred_obj_mask).save(
            os.path.join(pred_obj_path, '{}'.format(view.image_name) + ".png")
        )

        # Predicted gray (original IDs)
        Image.fromarray(pred_original_np.astype(np.uint16)).save(
            os.path.join(test_obj_path, '{}'.format(view.image_name) + ".png")
        )

        # Save RGB render and GT RGB image
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{}'.format(view.image_name) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{}'.format(view.image_name) + ".png"))


def render_sets(dataset: ModelParams, pipeline: PipelineParams, render_params: RenderParams):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=render_params.iteration, shuffle=False)

        # ========= Same num_classes logic as training =========
        matched_mask_path = os.path.join(dataset.source_path, dataset.object_path)

        # Print info.json if present (for reference only)
        info_path = os.path.join(matched_mask_path, "info.json")
        if os.path.exists(info_path):
            info = json.load(open(info_path))
            print("Info of the mask association process: ", info)
        else:
            print(f"[WARN] info.json not found at {info_path}")

        # Load id_mapping.json produced during training (old_id -> new_id)
        id_map_path = os.path.join(matched_mask_path, "id_mapping.json")
        if not os.path.exists(id_map_path):
            raise RuntimeError(
                f"id_mapping.json not found at {id_map_path}. "
                f"Please make sure you have run training and generated this mapping."
            )

        with open(id_map_path, "r") as f:
            raw_id_map = json.load(f)
        id_map = {int(k): int(v) for k, v in raw_id_map.items()}

        if len(id_map) > 0:
            max_new_id = max(id_map.values())
        else:
            max_new_id = 0

        num_classes = max_new_id + 1  # background 0 + K foreground
        print(f"[Global-ID-Mapping@render] num_classes (with background) = {num_classes}")

        # ========= Build inverse lookup: compact ID -> original ID =========
        inverse_lookup = torch.zeros(num_classes, dtype=torch.long, device="cuda")
        inverse_lookup[0] = 0
        for old_id, new_id in id_map.items():
            if 0 <= new_id < num_classes:
                inverse_lookup[new_id] = int(old_id)

        # ========= Build classifier and load trained weights =========
        classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1).cuda()

        ckpt_path = os.path.join(
            dataset.model_path,
            "point_cloud",
            "iteration_" + str(scene.loaded_iter),
            "classifier.pth"
        )
        print(f"[Render] Loading classifier from: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cuda")
        classifier.load_state_dict(state_dict)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # FPS for video (default 30)
        fps = getattr(render_params, "fps", 30)

        # Render video
        if render_params.render_video:
            render_video_func(
                dataset.source_path,
                dataset.model_path,
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                classifier,
                inverse_lookup,
                fps=fps
            )

        # Render train set
        if not render_params.skip_train:
            render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                classifier,
                inverse_lookup
            )

        # Render test set
        if (not render_params.skip_test) and (len(scene.getTestCameras()) > 0):
            render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
                classifier,
                inverse_lookup
            )


# ============ main: use output folder name only ============

if __name__ == "__main__":

    # ====== 1) Parse CLI args (render-related args + output_name only) ======
    parser = ArgumentParser(description="Testing (render) script parameters")

    # These are only used to register CLI flags like resolution/skip_train/skip_test/render_video/fps, etc.
    _mp = ModelParams(parser, sentinel=True)
    _pp = PipelineParams(parser)
    _rp = RenderParams(parser)

    parser.add_argument("--quiet", action="store_true")

    # New: only input the output folder name (no -o to avoid conflicts with object_path)
    parser.add_argument(
        "--output_name",
        type=str,
        required=True,
        help="Name of folder under ./output that contains the trained semantic model."
    )

    args_cmd = parser.parse_args(sys.argv[1:])

    # ====== 2) Build output_dir from output_name and load cfg_args ======
    repo_root = os.path.dirname(os.path.abspath(__file__))
    output_root = os.path.join(repo_root, "output")
    output_dir = os.path.join(output_root, args_cmd.output_name)

    print(f"Rendering from output folder: {output_dir}")

    cfg_path = os.path.join(output_dir, "cfg_args")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"cfg_args not found at {cfg_path}")

    print(f"Loading cfg_args from {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg_string = f.read()
    # cfg_args is stored as a Namespace(...) string
    cfg_ns = eval(cfg_string)

    # ====== 3) Merge cfg defaults and CLI overrides ======
    merged_dict = vars(cfg_ns).copy()
    for k, v in vars(args_cmd).items():
        if k == "output_name":
            continue
        if v is not None:
            merged_dict[k] = v

    # Force model_path to output_dir for rendering
    merged_dict["model_path"] = os.path.abspath(output_dir)

    # Rendering only needs point_cloud from output_dir; do not use lift branch
    merged_dict["lift"] = False

    # Prevent ModelParams.extract from recomputing paths from name-only fields
    for key in ("scene", "model", "output"):
        if key in merged_dict:
            merged_dict[key] = ""

    full_args = Namespace(**merged_dict)

    # ====== 4) Use a dummy parser + ParamGroup.extract to split args into groups ======
    dummy_parser = ArgumentParser()
    mp = ModelParams(dummy_parser, sentinel=True)
    pp = PipelineParams(dummy_parser)
    rp = RenderParams(dummy_parser)

    dataset_params = mp.extract(full_args)
    pipeline_params = pp.extract(full_args)
    render_params = rp.extract(full_args)

    # Ensure model_path points to output_dir (safety)
    dataset_params.model_path = os.path.abspath(output_dir)

    print("Final model_path for rendering:", dataset_params.model_path)
    print("Source path (dataset):", dataset_params.source_path)

    # Init RNG/state
    safe_state(getattr(full_args, "quiet", False))

    # Start rendering
    render_sets(dataset_params, pipeline_params, render_params)
