import os
import numpy as np
import cv2
from pathlib import Path

# 配置路径
CITY_DIR = Path("/workspace/Gaga/dataset/subset_building1_16/depth_map")
GS_DIR   = Path("/workspace/Gaga/model/8cc490ce-1/train/ours_30000/depth_npy")
OUT_DIR  = Path("/workspace/Gaga/model/8cc490ce-1/train/ours_30000/building_mask")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 非对称容差范围（单位与深度一致）
TOL_NEG = -30  # 容许GS比City更近（负方向）
TOL_POS = 100    # 容许GS比City更远（正方向）

def load_depth(path: Path) -> np.ndarray:
    d = np.load(str(path))
    if d.ndim == 3 and d.shape[0] == 1:
        d = d[0]
    return d

def resize_to(img: np.ndarray, target_hw):
    th, tw = target_hw
    h, w = img.shape[:2]
    if (h, w) == (th, tw):
        return img
    return cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)

def main():
    city_files = sorted([p for p in CITY_DIR.glob("*.npy")])
    gs_files   = sorted([p for p in GS_DIR.glob("*.npy")])

    city_names = {p.name for p in city_files}
    gs_names   = {p.name for p in gs_files}

    common = sorted(list(city_names & gs_names))
    print(f"[info] citygml depth npy: {len(city_files)}")
    print(f"[info] 3dgs depth npy   : {len(gs_files)}")
    print(f"[info] matched by name  : {len(common)}")

    for name in common:
        city_path = CITY_DIR / name
        gs_path   = GS_DIR / name

        city_depth = load_depth(city_path).astype(np.float32)
        gs_depth   = load_depth(gs_path).astype(np.float32)

        Hc, Wc = city_depth.shape[:2]
        gs_depth_resized = resize_to(gs_depth, (Hc, Wc)).astype(np.float32)

        valid_city = np.isfinite(city_depth) & (city_depth > 0)
        valid_gs   = np.isfinite(gs_depth_resized) & (gs_depth_resized > 0)
        valid = valid_city & valid_gs

        # 差值（正负方向）
        diff = gs_depth_resized - city_depth

        # 在 [-20, +5] 范围内认为一致
        building_mask = (valid & (diff >= TOL_NEG) & (diff <= TOL_POS)).astype(np.uint8)

        np.save(str(OUT_DIR / name), building_mask)

        png_name = name.replace(".npy", ".png")
        mask_png = (building_mask * 255).astype(np.uint8)
        cv2.imwrite(str(OUT_DIR / png_name), mask_png)

    print("[done] masks saved to", OUT_DIR)

if __name__ == "__main__":
    main()
