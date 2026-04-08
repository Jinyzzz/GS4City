import os
import json
import shutil
import cv2
import numpy as np
from argparse import ArgumentParser

VIS_IMAGE_EXT = ".png"


def visualize_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape={mask.shape}")

    h, w = mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    labels = np.unique(mask)
    labels = labels[labels > 0]

    for lab in labels.tolist():
        color = np.array([
            (lab * 37) % 256,
            (lab * 67) % 256,
            (lab * 97) % 256,
        ], dtype=np.uint8)
        vis[mask == lab] = color

    return vis


def filter_masks_by_min_views(
    input_mask_folder: str,
    output_mask_folder: str,
    min_views: int,
    visualize: bool = False,
    visualize_folder: str = None,
):
    mask_files = sorted(
        f for f in os.listdir(input_mask_folder)
        if f.lower().endswith(".npy") and f != "clip_features.npy"
    )

    if len(mask_files) == 0:
        print(f"[filter] No .npy masks found in {input_mask_folder}.")
        return

    os.makedirs(output_mask_folder, exist_ok=True)

    if visualize:
        if visualize_folder is None:
            visualize_folder = os.path.join(output_mask_folder, "visualization")
        os.makedirs(visualize_folder, exist_ok=True)

    view_count_dict = {}
    max_label = 0
    base_dtype = None

    for fname in mask_files:
        path = os.path.join(input_mask_folder, fname)
        mask = np.load(path)

        if mask.ndim != 2:
            print(f"[filter] Skip non-2D npy file in first pass: {fname}, shape={mask.shape}")
            continue

        if not np.issubdtype(mask.dtype, np.integer):
            mask = mask.astype(np.int64)

        if base_dtype is None:
            base_dtype = mask.dtype

        if mask.size == 0:
            continue

        max_lab_local = int(mask.max())
        if max_lab_local == 0:
            continue

        max_label = max(max_label, max_lab_local)

        counts = np.bincount(mask.ravel(), minlength=max_lab_local + 1)
        labels = np.nonzero(counts)[0]
        labels = labels[labels > 0]

        if labels.size == 0:
            continue

        for lab in labels.tolist():
            view_count_dict[lab] = view_count_dict.get(lab, 0) + 1

    if base_dtype is None or max_label == 0 or len(view_count_dict) == 0:
        print("[filter] All masks are background or empty. Nothing to filter.")
        return

    num_mask = int(max_label)

    view_counts = np.zeros(num_mask + 1, dtype=np.int32)
    for lab, cnt in view_count_dict.items():
        if lab <= num_mask:
            view_counts[lab] = cnt

    keep_mask = (view_counts >= min_views)
    keep_mask[0] = False

    keep_labels = np.where(keep_mask[1:])[0] + 1
    original_num = num_mask
    new_num = int(keep_labels.size)

    if new_num == 0:
        print(f"[filter] Warning: no masks meet min_views={min_views}.")
        return

    print(f"[filter] Filtering by views: {original_num} -> {new_num} (min_views={min_views})")

    remap = np.zeros(num_mask + 1, dtype=np.int32)
    old_to_new = {}
    for new_label, old_label in enumerate(keep_labels.tolist(), start=1):
        remap[old_label] = new_label
        old_to_new[int(old_label)] = int(new_label)

    for fname in mask_files:
        in_path = os.path.join(input_mask_folder, fname)
        out_path = os.path.join(output_mask_folder, fname)

        mask = np.load(in_path)

        if mask.ndim != 2:
            print(f"[filter] Skip non-2D npy file in second pass: {fname}, shape={mask.shape}")
            continue

        if not np.issubdtype(mask.dtype, np.integer):
            mask = mask.astype(np.int64)

        new_mask = remap[mask]
        np.save(out_path, new_mask)

        if visualize:
            vis = visualize_mask(new_mask)
            vis_path = os.path.join(
                visualize_folder,
                os.path.splitext(fname)[0] + VIS_IMAGE_EXT,
            )
            cv2.imwrite(vis_path, vis)

    input_clip_path = os.path.join(input_mask_folder, "clip_features.npy")
    if os.path.exists(input_clip_path):
        old_feats = np.load(input_clip_path)
        if isinstance(old_feats, np.ndarray) and old_feats.ndim == 2:
            if old_feats.shape[0] < num_mask + 1:
                print(
                    f"[filter] clip_features.npy first dim too small: "
                    f"{old_feats.shape[0]} < {num_mask + 1}, skip saving CLIP features."
                )
            else:
                D = old_feats.shape[1]
                clip_feats_new = np.zeros((new_num + 1, D), dtype=old_feats.dtype)
                clip_feats_new[0] = 0.0
                for old_label in keep_labels.tolist():
                    new_label = int(remap[old_label])
                    clip_feats_new[new_label] = old_feats[old_label]

                out_clip_path = os.path.join(output_mask_folder, "clip_features.npy")
                np.save(out_clip_path, clip_feats_new)
                print(f"[filter] Saved remapped CLIP features to {out_clip_path}")
        else:
            print("[filter] clip_features.npy has unexpected shape, skip saving CLIP features.")
    else:
        print("[filter] No clip_features.npy found in input folder, skip CLIP features.")

    for fname in os.listdir(input_mask_folder):
        src = os.path.join(input_mask_folder, fname)
        dst = os.path.join(output_mask_folder, fname)

        if not os.path.isfile(src):
            continue
        if fname == "clip_features.npy":
            continue
        if fname == "info.json":
            continue
        if fname.lower().endswith(".npy"):
            continue

        try:
            shutil.copy2(src, dst)
        except Exception as e:
            print(f"[filter] Warning: failed to copy extra file {fname}: {e}")

    info = {
        "input_mask_folder": input_mask_folder,
        "output_mask_folder": output_mask_folder,
        "num_mask_before": int(original_num),
        "num_mask_after": int(new_num),
        "min_views": int(min_views),
        "visualize": bool(visualize),
        "visualize_folder": visualize_folder if visualize else "",
        "old_to_new": old_to_new,
    }

    info_path = os.path.join(output_mask_folder, "info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"[filter] Done. Final num_mask = {new_num}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_mask_folder", type=str, required=True)
    parser.add_argument("--output_mask_folder", type=str, required=True)
    parser.add_argument("--min_views", type=int, required=True)
    parser.add_argument("--visualize", "-v", action="store_true")
    parser.add_argument("--visualize_folder", type=str, default=None)
    args = parser.parse_args()

    filter_masks_by_min_views(
        input_mask_folder=args.input_mask_folder,
        output_mask_folder=args.output_mask_folder,
        min_views=args.min_views,
        visualize=args.visualize,
        visualize_folder=args.visualize_folder,
    )