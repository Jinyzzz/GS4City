# GS4City

**Author:** Jinyu Zhu

## Overview

**GS4City** is a semantic processing pipeline that integrates **CityGML**, **multi-view imagery**, and **3D Gaussian Splatting (3DGS)** to enable high-quality semantic reconstruction and visualization of urban scenes.

The pipeline includes:

* SAM-based mask generation
* Cross-view instance association
* Multi-source mask fusion (SAM + CityGML)
* Semantic-aware Gaussian training
* Rendering and export of results
* Interactive GUI visualization

This project builds upon the following prior works:

* Gaga
* Gaussian Grouping
* Gaussian Splatting

---

## 1. Preprocessing

Before running the pipeline, complete the following three preprocessing steps.

### 1.1 Structure-from-Motion (SfM)

Use an SfM tool (e.g., COLMAP) to reconstruct a sparse scene from multi-view images. The following files must be generated:

```
sparse/0/cameras.bin
sparse/0/frames.bin
sparse/0/images.bin
sparse/0/points3D.bin
```

---

### 1.2 3D Gaussian Splatting Pretraining

Train a 3DGS model using the original Gaussian Splatting implementation. Place the trained model under:

```
model/<pretrained_model_name>/
```

---

### 1.3 CityGML Semantic Data Preparation

From CityGML data, generate or export:

* `gml_mask/` (per-view `.npy` masks)
* `gml_mask_vis/` (visualization images)
* `city_semantics.json`
* `id_mapping.json`

---

## 2. Project Directory Structure

The project directory should follow this structure:

```
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
├─ model/
├─ weight/
├─ output/
```

### Requirements

* Files in `images/`, `gml_mask/`, and `gml_mask_vis/` must correspond one-to-one (same filename, different extensions).

---

## 3. Pipeline Execution

### 3.1 SAM Mask Generation

```bash
python get_sam_mask.py --scene <scene_name> --gml --clip --visualize
```

**Outputs:**

* `dataset/<scene_name>/raw_sam_mask/`
* `dataset/<scene_name>/raw_sam_mask_vis/` (if visualization enabled)

**Key Parameters:**

* `--scene`: scene identifier
* `--gml`: enable filtering using CityGML masks
* `--clip`: enable CLIP-assisted classification
* `--visualize`: save visualization results

**Default Configuration:**

* `mask/config.json`

---

### 3.2 Cross-View Mask Association

```bash
python associate.py --scene <scene_name> --model <pretrained_model_name> --visualize --clip
```

**Outputs:**

* `dataset/<scene_name>/sam_mask/`
* `dataset/<scene_name>/sam_mask_vis/`

**Key Parameters:**

* `--scene`: scene identifier
* `--model`: pretrained 3DGS model name
* `--visualize`: enable visualization
* `--clip`: enable CLIP-based matching

**Default Configuration:**

* `mask/config.json`
* `arguments.py`

---

### 3.3 Mask Fusion and CLIP Feature Extraction

```bash
python fuse_masks.py --scene <scene_name>
```

**Outputs:**

* `dataset/<scene_name>/fused_mask/`
* `dataset/<scene_name>/fused_mask_vis/`
* `dataset/<scene_name>/object_clip_index.npz`

**Key Parameters:**

* `--scene`: scene identifier

**Default Configuration:**

* Defined within `fuse_masks.py`

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

**Outputs:**

* `output/<output_name>/`

  * `cfg_args`
  * `point_cloud/iteration_xxx/classifier.pth`
  * checkpoints (optional)
  * copied semantic files (if available)

**Notes:**

* The training pipeline prioritizes `fused_mask/`; if unavailable, it falls back to `sam_mask/`.

**Key Parameters:**

* `--scene`: scene identifier
* `--model`: pretrained model
* `--output`: output directory name
* `--resolution`: training resolution level
* `--iterations`: number of iterations

**Default Configuration:**

* `config/train.json`
* `arguments.py`

---

### 3.5 Rendering and Export

```bash
python render.py --output_name <output_name> --render_video
```

**Outputs:**

* Rendered images:

  * `output/<output_name>/train/ours_<iter>/`
  * `output/<output_name>/test/ours_<iter>/`
* Video (optional):

  * `output/<output_name>/video/ours_<iter>/final_video.mp4`

**Key Parameters:**

* `--output_name`: training output directory
* `--render_video`: enable video export

**Default Configuration:**

* `arguments.py`
* `cfg_args` in output directory

---

## 4. GUI Visualization

```bash
python main_gui.py \
  -s dataset/<scene_name> \
  --model_path output/<output_name> \
  --iteration 10000 \
  --gui_width 1024 \
  --gui_height 768
```

### Required Inputs

* Scene directory:
  `dataset/<scene_name>`

* Model output directory:
  `output/<output_name>`

### Required Files (must be copied into output directory)

* `city_semantics.json`
* `id_mapping.json`
* `object_clip_index.npz`

### Parameters

* `-s`: scene path
* `--model_path`: model output path
* `--iteration`: checkpoint iteration
* `--gui_width`, `--gui_height`: window size

### Output

* Interactive visualization interface (no mandatory file output unless explicitly exported)

---

## 5. Configuration Files

* `mask/config.json`: preprocessing parameters (SAM, CLIP, projection)
* `config/train.json`: training-specific parameters (e.g., regularization)
* `arguments.py`: shared configuration (data, model, rendering, pipeline)

---