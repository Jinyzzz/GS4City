# GS4City

**Author:** Jinyu Zhu

> **Developed based on the following projects**
>
> * [Gaga](https://github.com/weijielyu/Gaga)
> * [Gaussian Grouping](https://github.com/lkeab/gaussian-grouping)
> * [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)

A semantic processing pipeline based on **CityGML + multi-view images + Gaussian**, including:

* **SAM mask generation**
* **Cross-view mask association** (unified instance IDs)
* **Mask fusion** (SAM + CityGML)
* **Semantic training**
* **Rendering result export**
* **GUI visualization**

---

## 1. Preprocessing (Before You Start)

Before running this project, complete the following **3 preprocessing steps**.

### Preprocessing 1: SfM (Reconstruct a Sparse Scene from Images)

Use an SfM tool (e.g., **COLMAP**) to reconstruct a sparse scene from multi-view images, and obtain:

* `sparse/0/cameras.bin`
* `sparse/0/frames.bin`
* `sparse/0/images.bin`
* `sparse/0/points3D.bin`

---

### Preprocessing 2: Train a 3DGS Scene with the Original Gaussian-Splatting Code

Use the **original Gaussian-Splatting code** to train a 3DGS scene, and place the trained model in:

* `model/<pretrained_model_name>/`

---

### Preprocessing 3: Obtain Masks and Semantic Files from CityGML

Generate (or export) the following files from CityGML data:

* `gml_mask/` (per-view `.npy` masks)
* `gml_mask_vis/` (corresponding visualization images)
* `city_semantics.json`
* `id_mapping.json`

---

## 2. Project Directory Setup (Before Running This Project)

The project root is recommended to look like this (including the `dataset/<scene_name>/` structure):

```text
your_project/
├─ dataset/
│  └─ <scene_name>/
│     ├─ images/
│     ├─ gml_mask/
│     ├─ gml_mask_vis/
│     ├─ sparse/
│     │  └─ 0/
│     │     ├─ cameras.bin
│     │     ├─ frames.bin
│     │     ├─ images.bin
│     │     └─ points3D.bin
│     ├─ city_semantics.json
│     └─ id_mapping.json
├─ model/                    # Pretrained 3DGS models
├─ weight/                   # SAM pretrained weights
├─ output/                   # Empty output directory (training/rendering results will be written here)
└─ ...
```

### Requirements

* Filenames in `images/`, `gml_mask/`, and `gml_mask_vis/` must correspond **one-to-one** (same basename, different extensions).

---

## 3. Quick Start (Recommended Order)

> The focus below is on:
> **what each step generates (outputs)**, **how to run it**, **common parameters**, and **where default parameters are defined**.

---

### 3.1 Generate SAM Masks

```bash
python get_sam_mask.py --scene <scene_name> --gml --clip --visualize
```

#### Outputs

* `dataset/<scene_name>/raw_sam_mask/`
* `dataset/<scene_name>/raw_sam_mask_vis/` (when `--visualize` is enabled)

#### Common parameters

* `--scene`: scene name
* `--gml`: use `gml_mask` for filtering
* `--clip`: enable CLIP-assisted classification
* `--visualize`: output visualization results

#### Default parameter location

* `mask/config.json` (SAM / CLIP related)

---

### 3.2 Cross-View Mask Association (Unified Instance IDs)

```bash
python associate.py --scene <scene_name> --model <pretrained_model_name> --visualize --clip
```

#### Outputs

* `dataset/<scene_name>/sam_mask/`
* `dataset/<scene_name>/sam_mask_vis/` (when `--visualize` is enabled)

#### Common parameters

* `--scene`: scene name
* `--model`: pretrained model name (under `model/`)
* `--visualize`: output association visualization
* `--clip`: enable CLIP-assisted matching

#### Default parameter location

* `mask/config.json` (projector related)
* `arguments.py` (data/pipeline parameters)

---

### 3.3 Fuse Masks (SAM + CityGML) and Compute Object CLIP Features

```bash
python fuse_masks.py --scene <scene_name>
```

#### Outputs

* `dataset/<scene_name>/fused_mask/`
* `dataset/<scene_name>/fused_mask_vis/`
* `dataset/<scene_name>/object_clip_index.npz`

#### Common parameters

* `--scene`: scene name

#### Default parameter location

* `argparse` inside `fuse_masks.py`

---

### 3.4 Semantic Training

```bash
python train.py \
  --scene <scene_name> \
  --model <pretrained_model_name> \
  --output <output_name> \
  --resolution 8 \
  --iterations 10000
```

#### Outputs

* `output/<output_name>/`

  * `cfg_args`
  * `point_cloud/iteration_xxx/classifier.pth`
  * checkpoint (optional)
  * copied scene semantic files (if present in source data)

> The training script prioritizes `fused_mask/`; if it does not exist, it falls back to `sam_mask/`.

#### Common parameters

* `--scene`: scene name
* `--model`: pretrained 3DGS model name (under `model/`)
* `--output`: training output directory name (under `output/`)
* `--resolution`: training resolution level
* `--iterations`: number of training iterations

#### Default parameter location

* `config/train.json`
* `arguments.py` (`ModelParams / OptimizationParams / PipelineParams`)

---

### 3.5 Rendering Export (Images / Video)

```bash
python render.py --output_name <output_name> --render_video
```

#### Outputs

* `output/<output_name>/train/ours_<iter>/...`
* `output/<output_name>/test/ours_<iter>/...`
* `output/<output_name>/video/ours_<iter>/final_video.mp4` (when `--render_video` is enabled)

#### Common parameters

* `--output_name`: training output directory name (under `output/`)
* `--render_video`: export video results

#### Default parameter location

* `RenderParams` in `arguments.py`
* `output/<output_name>/cfg_args`

---

## 4. GUI Visualization (Important: Additional Input Files)

```bash
python main_gui.py \
  -s dataset/<scene_name> \
  --model_path output/<output_name> \
  --iteration 10000 \
  --gui_width 1024 \
  --gui_height 768
```

### GUI inputs (must be checked)

* Scene directory: `dataset/<scene_name>`
* Training output directory: `output/<output_name>`

### Files that must be copied into `output/<output_name>/` before visualization (very important)

Please copy the following files into `output/<output_name>/` (or the actual model directory path read by the script):

* `city_semantics.json` (from **Preprocessing 3: CityGML**)
* `id_mapping.json` (from **Preprocessing 3: CityGML**)
* `object_clip_index.npz` (from **Step 3.3**)

### Outputs

* Interactive GUI window (usually no fixed files are generated unless the GUI has its own export functionality)

### Common parameters

* `-s`: scene path (e.g., `dataset/<scene_name>`)
* `--model_path`: training output directory path (e.g., `output/<output_name>`)
* `--iteration`: iteration number to load
* `--gui_width`: GUI window width
* `--gui_height`: GUI window height

### Default parameter location

* `argparse` inside `main_gui.py`
* `arguments.py` (`ModelParams / PipelineParams`)

---

## 5. Configuration File Locations (Overview)

* `mask/config.json`: preprocessing-stage defaults (SAM / CLIP / projector)
* `config/train.json`: extra training-stage parameters (especially 3D regularization)
* `arguments.py`: common defaults (data, training, rendering, pipeline)

---

## 6. Full Pipeline (Short Version)

```text
Preprocessing:
1) SfM (generate sparse/0/*.bin)
2) Original Gaussian-Splatting training for 3DGS (place model in model/)
3) CityGML generates gml_mask / gml_mask_vis / city_semantics.json / id_mapping.json

Project pipeline:
4) get_sam_mask.py   -> raw_sam_mask / raw_sam_mask_vis
5) associate.py      -> sam_mask / sam_mask_vis
6) fuse_masks.py     -> fused_mask / fused_mask_vis / object_clip_index.npz
7) train.py          -> output/<output_name>
8) render.py         -> export images/video
9) Copy city_semantics.json + id_mapping.json + object_clip_index.npz into the output path
10) main_gui.py      -> interactive visualization
```

---