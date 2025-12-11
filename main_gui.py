#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import json

import torch

from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams

from semantic_viewer import SemanticGaussianGUI, estimate_focus_from_gaussians

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description="Gaga Semantic Gaussian GUI")

    lp = ModelParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument(
        "--iteration",
        type=int,
        default=-1,
        help="要加载的迭代号，-1 表示自动加载最新 iteration",
    )
    parser.add_argument(
        "--gui_width",
        type=int,
        default=800,
        help="渲染宽度（像素）",
    )
    parser.add_argument(
        "--gui_height",
        type=int,
        default=600,
        help="渲染高度（像素）",
    )
    parser.add_argument(
        "--gui_radius",
        type=float,
        default=0.0,
        help="初始相机半径（>0 时强制使用；=0 时自动估计）",
    )

    args = parser.parse_args()

    # 构建 dataset / pipe / gaussians / scene
    dataset = lp.extract(args)
    pipe = pp.extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    # 确定 iteration
    loaded_iter = getattr(scene, "loaded_iter", None)
    if loaded_iter is None and args.iteration > 0:
        loaded_iter = args.iteration

    classifier = None
    num_classes = 0
    inverse_lookup = None

    # ========= 训练用 id_mapping.json: old_id(灰度) -> new_id(紧凑) =========
    try:
        matched_mask_path = os.path.join(dataset.source_path, dataset.object_path)
        train_id_map_path = os.path.join(matched_mask_path, "id_mapping.json")
        if os.path.exists(train_id_map_path):
            with open(train_id_map_path, "r") as f:
                raw_id_map = json.load(f)
            id_map = {int(k): int(v) for k, v in raw_id_map.items()}

            if len(id_map) > 0:
                max_new_id = max(id_map.values())
            else:
                max_new_id = 0

            num_classes = max_new_id + 1
            print(f"[GUI] Loaded train id_mapping.json from {train_id_map_path}")
            print(f"[GUI] num_classes (with background) = {num_classes}")

            inverse_lookup = torch.zeros(num_classes, dtype=torch.long, device=device)
            inverse_lookup[0] = 0
            for old_id, new_id in id_map.items():
                if 0 <= new_id < num_classes:
                    inverse_lookup[new_id] = int(old_id)
        else:
            print(f"[GUI] 未找到训练用 id_mapping.json：{train_id_map_path}")
            num_classes = 0
            inverse_lookup = None
    except Exception as e:
        print(f"[GUI] 读取训练用 id_mapping.json 失败: {e}")
        num_classes = 0
        inverse_lookup = None

    # ========= classifier =========
    if loaded_iter is not None and num_classes > 0:
        try:
            classifier_net = torch.nn.Conv2d(
                in_channels=gaussians.num_objects,
                out_channels=num_classes,
                kernel_size=1,
                bias=True,
            ).to(device)

            ckpt_path = os.path.join(
                dataset.model_path,
                "point_cloud",
                f"iteration_{loaded_iter}",
                "classifier.pth",
            )
            print(f"[GUI] 从 {ckpt_path} 加载 classifier")
            state = torch.load(ckpt_path, map_location=device)
            classifier_net.load_state_dict(state)
            classifier_net.eval()

            classifier = classifier_net
            print(f"[GUI] classifier: in_channels={gaussians.num_objects}, num_classes={num_classes}")
        except Exception as e:
            print(f"[GUI] 加载 classifier 失败: {e}")
            classifier = None
    else:
        if loaded_iter is None:
            print("[GUI] Scene 未加载任何 iteration，无法加载 classifier；仅显示 RGB。")
        elif num_classes == 0:
            print("[GUI] num_classes=0（可能缺少训练 id_mapping.json），无法构造 classifier；仅显示 RGB。")

    # ========= 读取 city_semantics.json & model_path/id_mapping.json =========
    grayid_to_cityobject = {}
    city_semantics = {}

    model_path = dataset.model_path
    sem_path = os.path.join(model_path, "city_semantics.json")
    gray_id_map_path = os.path.join(model_path, "id_mapping.json")

    try:
        if os.path.exists(sem_path):
            with open(sem_path, "r", encoding="utf-8") as f:
                city_semantics = json.load(f)
            print(f"[GUI] Loaded city_semantics.json from {sem_path} "
                  f"(num objects = {len(city_semantics)})")
        else:
            print(f"[GUI] 未找到 city_semantics.json：{sem_path}")
    except Exception as e:
        print(f"[GUI] 读取 city_semantics.json 失败: {e}")
        city_semantics = {}

    # 3) building_function.json: 功能编号 -> 名称
    building_function_map = {}
    func_path = os.path.join(model_path, "building_function.json")
    try:
        if os.path.exists(func_path):
            with open(func_path, "r", encoding="utf-8") as f:
                building_function_map = json.load(f)
            print(f"[GUI] Loaded building_function.json from {func_path} "
                  f"(num = {len(building_function_map)})")
        else:
            print(f"[GUI] 未找到 building_function.json：{func_path}")
    except Exception as e:
        print(f"[GUI] 读取 building_function.json 失败: {e}")
        building_function_map = {}

    # 4) building_rooftype.json: 屋顶编号 -> 名称
    building_rooftype_map = {}
    roof_path = os.path.join(model_path, "building_rooftype.json")
    try:
        if os.path.exists(roof_path):
            with open(roof_path, "r", encoding="utf-8") as f:
                building_rooftype_map = json.load(f)
            print(f"[GUI] Loaded building_rooftype.json from {roof_path} "
                  f"(num = {len(building_rooftype_map)})")
        else:
            print(f"[GUI] 未找到 building_rooftype.json：{roof_path}")
    except Exception as e:
        print(f"[GUI] 读取 building_rooftype.json 失败: {e}")
        building_rooftype_map = {}

    try:
        if os.path.exists(gray_id_map_path):
            with open(gray_id_map_path, "r", encoding="utf-8") as f:
                raw_gray_id_map = json.load(f)
            for k, v in raw_gray_id_map.items():
                try:
                    gid = int(k)
                except ValueError:
                    print(f"[GUI] model_path/id_mapping.json 的键无法转换为 int: {k}")
                    continue
                grayid_to_cityobject[gid] = str(v)
            print(f"[GUI] Loaded gray-id->cityobject id_mapping.json from {gray_id_map_path} "
                  f"(num mapped ids = {len(grayid_to_cityobject)})")
        else:
            print(f"[GUI] 未找到 model_path/id_mapping.json：{gray_id_map_path}")
    except Exception as e:
        print(f"[GUI] 读取 model_path/id_mapping.json 失败: {e}")
        grayid_to_cityobject = {}

    # 自动估计主体中心和半径
    focus_center, focus_radius = estimate_focus_from_gaussians(gaussians)

    cameras_extent = getattr(scene, "cameras_extent", 1.0)
    auto_radius = max(focus_radius, cameras_extent * 1.2)

    if args.gui_radius > 0:
        init_radius = float(args.gui_radius)
    else:
        init_radius = auto_radius

    gui = SemanticGaussianGUI(
        scene=scene,
        gaussians=gaussians,
        pipe=pipe,
        background=background,
        classifier=classifier,
        num_classes=num_classes,
        inverse_lookup=inverse_lookup,
        grayid_to_cityobject=grayid_to_cityobject,
        city_semantics=city_semantics,
        building_function_map=building_function_map,
        building_rooftype_map=building_rooftype_map,
        width=args.gui_width,
        height=args.gui_height,
        radius=init_radius,
        model_root=dataset.model_path,   
    )

    gui.set_initial_center(focus_center, radius=init_radius)
    gui.render_loop()


if __name__ == "__main__":
    main()
