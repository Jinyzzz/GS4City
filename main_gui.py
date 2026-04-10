#
# Copyright (C) 2026, GS4City
# All rights reserved.
#

import os
import argparse
import json

import torch

from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams

from semantic_viewer import SemanticGaussianGUI, estimate_focus_from_gaussians

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description="GS4City Semantic Gaussian GUI")

    lp = ModelParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument(
        "--iteration",
        type=int,
        default=-1,
        help="Iteration to load (-1 loads the latest iteration automatically).",
    )
    parser.add_argument(
        "--gui_width",
        type=int,
        default=800,
        help="GUI render width in pixels.",
    )
    parser.add_argument(
        "--gui_height",
        type=int,
        default=600,
        help="GUI render height in pixels.",
    )
    parser.add_argument(
        "--gui_radius",
        type=float,
        default=0.0,
        help="Initial camera radius (>0 forces this value; 0 uses auto-estimation).",
    )

    args = parser.parse_args()

    # Build dataset / pipeline / gaussians / scene
    dataset = lp.extract(args)
    pipe = pp.extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    # Resolve loaded iteration
    loaded_iter = getattr(scene, "loaded_iter", None)
    if loaded_iter is None and args.iteration > 0:
        loaded_iter = args.iteration

    classifier = None
    num_classes = 0
    inverse_lookup = None

    # Training id_mapping.json: old_id (gray) -> new_id (compact)
    try:
        matched_mask_path = os.path.join(dataset.source_path, dataset.object_path)
        train_id_map_path = os.path.join(matched_mask_path, "id_mapping.json")
        if os.path.exists(train_id_map_path):
            with open(train_id_map_path, "r") as f:
                raw_id_map = json.load(f)
            id_map = {int(k): int(v) for k, v in raw_id_map.items()}

            max_new_id = max(id_map.values()) if len(id_map) > 0 else 0
            num_classes = max_new_id + 1

            print(f"[GUI] Loaded train id_mapping.json from {train_id_map_path}")
            print(f"[GUI] num_classes (with background) = {num_classes}")

            inverse_lookup = torch.zeros(num_classes, dtype=torch.long, device=device)
            inverse_lookup[0] = 0
            for old_id, new_id in id_map.items():
                if 0 <= new_id < num_classes:
                    inverse_lookup[new_id] = int(old_id)
        else:
            print(f"[GUI] Training id_mapping.json not found: {train_id_map_path}")
            num_classes = 0
            inverse_lookup = None
    except Exception as e:
        print(f"[GUI] Failed to read training id_mapping.json: {e}")
        num_classes = 0
        inverse_lookup = None

    # Load classifier if available
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
            print(f"[GUI] Loading classifier from {ckpt_path}")
            state = torch.load(ckpt_path, map_location=device)
            classifier_net.load_state_dict(state)
            classifier_net.eval()

            classifier = classifier_net
            print(f"[GUI] classifier: in_channels={gaussians.num_objects}, num_classes={num_classes}")
        except Exception as e:
            print(f"[GUI] Failed to load classifier: {e}")
            classifier = None
    else:
        if loaded_iter is None:
            print("[GUI] No iteration loaded, classifier will not be used (RGB only).")
        elif num_classes == 0:
            print("[GUI] num_classes=0 (missing training id_mapping.json?), classifier disabled (RGB only).")

    # Load CityGML semantics and mappings from model_path
    grayid_to_cityobject = {}
    city_semantics = {}

    model_path = dataset.model_path
    sem_path = os.path.join(model_path, "city_semantics.json")
    gray_id_map_path = os.path.join(model_path, "id_mapping.json")

    try:
        if os.path.exists(sem_path):
            with open(sem_path, "r", encoding="utf-8") as f:
                city_semantics = json.load(f)
            print(f"[GUI] Loaded city_semantics.json from {sem_path} (num objects = {len(city_semantics)})")
        else:
            print(f"[GUI] city_semantics.json not found: {sem_path}")
    except Exception as e:
        print(f"[GUI] Failed to read city_semantics.json: {e}")
        city_semantics = {}

    # building_function.json: code -> name
    building_function_map = {}
    func_path = os.path.join(model_path, "building_function.json")
    try:
        if os.path.exists(func_path):
            with open(func_path, "r", encoding="utf-8") as f:
                building_function_map = json.load(f)
            print(f"[GUI] Loaded building_function.json from {func_path} (num = {len(building_function_map)})")
        else:
            print(f"[GUI] building_function.json not found: {func_path}")
    except Exception as e:
        print(f"[GUI] Failed to read building_function.json: {e}")
        building_function_map = {}

    # building_rooftype.json: code -> name
    building_rooftype_map = {}
    roof_path = os.path.join(model_path, "building_rooftype.json")
    try:
        if os.path.exists(roof_path):
            with open(roof_path, "r", encoding="utf-8") as f:
                building_rooftype_map = json.load(f)
            print(f"[GUI] Loaded building_rooftype.json from {roof_path} (num = {len(building_rooftype_map)})")
        else:
            print(f"[GUI] building_rooftype.json not found: {roof_path}")
    except Exception as e:
        print(f"[GUI] Failed to read building_rooftype.json: {e}")
        building_rooftype_map = {}

    # model_path/id_mapping.json: gray_id -> CityObject id
    try:
        if os.path.exists(gray_id_map_path):
            with open(gray_id_map_path, "r", encoding="utf-8") as f:
                raw_gray_id_map = json.load(f)
            for k, v in raw_gray_id_map.items():
                try:
                    gid = int(k)
                except ValueError:
                    print(f"[GUI] Non-integer key in model_path/id_mapping.json: {k}")
                    continue
                grayid_to_cityobject[gid] = str(v)
            print(f"[GUI] Loaded gray-id->cityobject mapping from {gray_id_map_path} (num mapped ids = {len(grayid_to_cityobject)})")
        else:
            print(f"[GUI] model_path/id_mapping.json not found: {gray_id_map_path}")
    except Exception as e:
        print(f"[GUI] Failed to read model_path/id_mapping.json: {e}")
        grayid_to_cityobject = {}

    # Auto-estimate focus center and radius
    focus_center, focus_radius = estimate_focus_from_gaussians(gaussians)
    cameras_extent = getattr(scene, "cameras_extent", 1.0)
    auto_radius = max(focus_radius, cameras_extent * 1.2)

    init_radius = float(args.gui_radius) if args.gui_radius > 0 else auto_radius

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
