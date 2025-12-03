#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gaga 语义高斯可视化 GUI（DearPyGUI 版，带层级语义交互）

用法示例：
    python semantic_gui_dpg.py \
        -s data/scene_name \
        --model_path output/scene_name \
        --iteration 10000 \
        --gui_width 1024 \
        --gui_height 768

依赖：
    pip install dearpygui
"""

import os
import math
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
torch.backends.cudnn.enabled = False

import dearpygui.dearpygui as dpg

from gaussian_renderer import render
from scene import Scene, GaussianModel
from scene.cameras import Camera
from utils.graphics_utils import fov2focal, focal2fov
from arguments import ModelParams, PipelineParams

from scipy.spatial.transform import Rotation as R


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================
# 相机：OrbitCamera
# =====================
class OrbitCamera:
    def __init__(self, W, H, r=2.0, fovy=60.0):
        self.W = W
        self.H = H
        self.radius = r
        self.center = np.array([0, 0, 0], dtype=np.float32)

        self.rot = R.from_quat([0, 0, 0, 1])

        self.up = np.array([0, 1, 0], dtype=np.float32)
        self.right = np.array([1, 0, 0], dtype=np.float32)
        self.fovy = fovy
        self.translate = np.array([0, 0, self.radius])
        self.scale_f = 1.0

        # 1：以相机中心绕场景旋转
        self.rot_mode = 1

    @property
    def pose_movecenter(self):
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius

        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res

        res[:3, 3] -= self.center

        # 转换到 Gaussian-Splatting 使用的 [R | -R^T t] 约定
        res[:3, 3] = -rot[:3, :3].transpose() @ res[:3, 3]

        return res

    @property
    def pose_objcenter(self):
        res = np.eye(4, dtype=np.float32)

        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res

        res[2, 3] += self.radius
        res[:3, 3] -= self.center

        res[:3, :3] = rot[:3, :3].transpose()

        return res

    # 当前统一用 movecenter
    @property
    def pose(self):
        if self.rot_mode == 1:
            return self.pose_movecenter
        else:
            return self.pose_objcenter

    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2])

    def orbit(self, dx, dy):
        if self.rot_mode == 1:
            up = self.rot.as_matrix()[:3, 1]
            side = self.rot.as_matrix()[:3, 0]
        else:
            up = -self.up
            side = -self.right

        rotvec_x = up * np.radians(0.01 * dx)
        rotvec_y = side * np.radians(0.01 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        # 非线性指数缩放，滚轮每 +1 / -1 就乘以 1.2（更敏感一些）
        self.radius *= 1.2 ** (-delta)
        # 限制一下最小/最大半径，避免穿模或飞到太远
        self.radius = float(np.clip(self.radius, 0.05, 1e3))

    def pan(self, dx, dy, dz=0.0):
        if self.rot_mode == 1:
            self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, -dy, dz])
        else:
            self.center += 0.0005 * np.array([-dx, dy, dz])


# =====================
# 工具函数：颜色映射 & 自动对焦
# =====================

def build_color_map(n_cls: int):
    """为每个类别生成固定随机颜色（0 类为黑）。"""
    rng = np.random.RandomState(0)
    colors = rng.rand(n_cls, 3).astype(np.float32)
    if n_cls > 0:
        colors[0] = 0.0
    return colors


def estimate_focus_from_gaussians(gaussians: GaussianModel, top_ratio: float = 0.05):
    """
    从高斯里估计一个“主体中心”和合适的观察距离：
    - 用 opacity 排序，取前 top_ratio 部分的点（默认 5%）作为主体（忽略低 opacity 噪点）
    - 以这些点的均值作为中心
    - 用 90% 距离分位数估计一个合适的半径
    """
    try:
        xyz = gaussians._xyz.detach().cpu()
        op = gaussians._opacity.detach().view(-1).cpu()

        N = xyz.shape[0]
        if N == 0:
            raise RuntimeError("No gaussians in model.")

        # 取 opacity 较高的那一部分点
        k = max(int(N * top_ratio), 1000)  # 至少 1000 个，防止太少
        k = min(k, N)
        topk_vals, topk_idx = torch.topk(op, k)
        xyz_focus = xyz[topk_idx]

        center = xyz_focus.mean(0)  # [3]
        # 用 90% 的距离分位数估计场景尺度
        dist = torch.norm(xyz_focus - center, dim=1)
        radius = dist.quantile(0.9).item()

        # 给一点安全余量
        radius = max(radius * 2.0, 0.1)

        return center.numpy().astype(np.float32), float(radius)
    except Exception as e:
        print(f"[GUI] estimate_focus_from_gaussians 失败，退回默认值: {e}")
        return np.zeros(3, dtype=np.float32), 2.0


# =====================
# GUI 主类
# =====================

class SemanticGaussianGUI:
    def __init__(self, scene: Scene, gaussians: GaussianModel,
                 pipe, background: torch.Tensor,
                 classifier: torch.nn.Module = None,
                 num_classes: int = 0,
                 inverse_lookup: torch.Tensor = None,
                 grayid_to_cityobject: dict = None,
                 city_semantics: dict = None,
                 width: int = 800,
                 height: int = 600,
                 radius: float = 2.0):
        self.scene = scene
        self.gaussians = gaussians
        self.pipe = pipe
        self.background = background
        self.classifier = classifier
        self.num_classes = num_classes

        # new_id -> old_id（原始灰度值），形状 [num_classes]
        self.inverse_lookup = inverse_lookup  # torch.long, on device

        # 原始灰度 id -> CityObject ID（model_path/id_mapping.json）
        self.grayid_to_cityobject = grayid_to_cityobject or {}
        # CityObject ID -> 语义信息（city_semantics.json）
        self.city_semantics = city_semantics or {}

        self.width = width
        self.height = height
        self.window_width = width
        self.window_height = height

        self.camera = OrbitCamera(self.width, self.height, r=radius)

        # Orbit / Train camera 切换
        self.use_train_cam = False  # 是否使用训练相机视角

        # 收集所有训练相机
        self.train_cameras = []
        self.train_cam_names = []
        for idx, cam in enumerate(self.scene.getTrainCameras()):
            name = getattr(cam, "image_name", None) or f"cam_{idx}"
            self.train_cameras.append(cam)
            self.train_cam_names.append(f"{idx}: {name}")
        self.active_train_cam_idx = 0

        # 记录用于 Reset View 的初始值（Orbit 模式）
        self.init_center = self.camera.center.copy()
        self.init_radius = float(radius)

        self.render_buffer = np.zeros((self.height, self.width, 3), dtype=np.float32)
        self.mode = "RGB"  # "RGB" / "Segmentation" / "Overlay"

        # 类别颜色（紧凑 id 的调色板）
        self.class_colors = build_color_map(self.num_classes) if self.num_classes > 0 else None

        # 当前帧紧凑 id & 原始 id label 图
        self.label_map_compact = None   # np.int32 [H_src, W_src]，紧凑 id
        self.label_map_orig = None      # np.int32 [H_src, W_src]，原始灰度 id
        self.label_H = None
        self.label_W = None

        # 选中状态：紧凑 id + 原始灰度 id + CityObject id
        self.selected_new_id = None     # 紧凑 id（只内部用来记忆，被高亮集合替代）
        self.selected_orig_id = None    # 原始灰度 id（灰度图灰度值）
        self.selected_cityobject_id = None
        self.selected_semantic_info = None

        # 层级选择
        self.mask_level = 0  # 0 = 当前对象（叶子），1 = 父，2 = 父的父 ...
        self.current_hierarchy_chain = None  # [leaf, parent, parent_of_parent, ...]
        self.highlight_gray_ids = None       # 用于高亮的灰度 id 集合

        # 基于 city_semantics 和 grayid_to_cityobject 构建父子关系与灰度聚合
        self.city_children = {}       # parent_id -> [child_ids]
        self.city_to_grayids = {}     # city_id  -> [gray_ids]
        self.city_descendants_cache = {}

        for cid, data in (self.city_semantics or {}).items():
            parent = data.get("parent", None)
            if parent is not None:
                self.city_children.setdefault(parent, []).append(cid)

        for gid, cid in (self.grayid_to_cityobject or {}).items():
            self.city_to_grayids.setdefault(cid, []).append(gid)

        self.highlight_color = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # 高亮颜色（红）
        self.highlight_alpha = 0.6      # 叠加权重

        self.load_model = True

        # 鼠标相关
        self.moving = False
        self.moving_middle = False
        self.mouse_pos = (0, 0)

        # 初始化 dpg
        dpg.create_context()
        self.register_dpg()

    def __del__(self):
        try:
            dpg.destroy_context()
        except Exception:
            pass

    def set_initial_center(self, center_np: np.ndarray, radius: float = None):
        """
        从外部把估计好的中心和半径塞到相机里，并更新 Reset View 的基准。
        """
        try:
            center_np = np.asarray(center_np, dtype=np.float32).reshape(3,)
            self.camera.center = center_np
            self.init_center = center_np.copy()
            if radius is not None:
                self.camera.radius = float(radius)
                self.init_radius = float(radius)
            print(f"[GUI] 初始相机中心设置为 {center_np}, 半径={self.init_radius:.3f}")
        except Exception as e:
            print(f"[GUI] set_initial_center 失败: {e}")

    # --------- 给 UI 用的简单封装 ---------
    def set_render_mode(self, mode: str):
        self.mode = mode

    def reset_view_orbit(self):
        self.camera.center = self.init_center.copy()
        self.camera.radius = float(self.init_radius)
        self.use_train_cam = False
        print("[GUI] 视角已重置到 Orbit 初始视角")

    def set_use_train_cam(self, flag: bool):
        self.use_train_cam = bool(flag)
        print(f"[GUI] 使用训练相机: {self.use_train_cam}")

    def zoom_step(self, delta: int):
        # delta>0 = 拉远, delta<0 = 拉近
        self.camera.scale(delta)

    def orbit_step(self, dx_deg: float, dy_deg: float):
        # 简单用 dx, dy 控制绕场景的旋转
        self.camera.orbit(dx_deg, dy_deg)

    def pan_step(self, dx: float, dy: float):
        self.camera.pan(dx, dy)

    def on_select_train_cam(self, display_name: str):
        """
        display_name 类似 '0: DJI_2024_...'，解析前面的 index。
        """
        try:
            idx_str = display_name.split(":", 1)[0]
            idx = int(idx_str)
        except Exception:
            print(f"[GUI] 解析训练相机索引失败: {display_name}")
            return

        if 0 <= idx < len(self.train_cameras):
            self.active_train_cam_idx = idx
            self.use_train_cam = True
            print(f"[GUI] 切换到训练相机 #{idx}: {display_name}")
        else:
            print(f"[GUI] 训练相机索引越界: {idx}")

    def set_mask_level(self, level: int):
        self.mask_level = max(0, int(level))
        print(f"[GUI] 更新 mask 层级: {self.mask_level}")
        # 如果当前已经有 selection，则重新根据层级更新高亮
        self._update_highlight_from_current_selection()

    def search_and_focus(self, query: str):
        # 预留：现在先简单打印
        if not query:
            print("[GUI] search_and_focus: 空查询")
            return
        print(f"[GUI] [TODO] search_and_focus: {query}")

    def clear_selection(self):
        self.selected_new_id = None
        self.selected_orig_id = None
        self.selected_cityobject_id = None
        self.selected_semantic_info = None
        self.current_hierarchy_chain = None
        self.highlight_gray_ids = None

        dpg.set_value("_building_info_text", "No building selected.")
        dpg.set_value("_selection_info_text", "No selection yet.")

    # ------- 层级/高亮相关内部工具 -------

    def _get_hierarchy_chain(self, leaf_city_id: str):
        """从叶子一直往 parent 走，得到 [leaf, parent, parent_of_parent, ...]"""
        if leaf_city_id is None or leaf_city_id not in self.city_semantics:
            return None

        chain = []
        visited = set()
        cid = leaf_city_id
        while cid is not None and cid not in visited:
            chain.append(cid)
            visited.add(cid)
            data = self.city_semantics.get(cid, {})
            cid = data.get("parent", None)
        return chain  # leaf -> root

    def _get_descendants(self, cid: str):
        """得到 cid 的所有子孙节点（不含自身），用 DFS + cache。"""
        if cid in self.city_descendants_cache:
            return self.city_descendants_cache[cid]

        res = set()
        for child in self.city_children.get(cid, []):
            res.add(child)
            res |= self._get_descendants(child)
        self.city_descendants_cache[cid] = res
        return res

    def _update_highlight_for_cityobject(self, city_id: str):
        """根据一个 city_id，计算需要高亮的 gray_id 集合（自己 + 所有子孙）"""
        if city_id is None:
            self.highlight_gray_ids = None
            return

        all_cities = {city_id}
        all_cities |= self._get_descendants(city_id)

        gray_ids = set()
        for cid in all_cities:
            for gid in self.city_to_grayids.get(cid, []):
                gray_ids.add(gid)

        self.highlight_gray_ids = gray_ids if gray_ids else None

    def _update_highlight_from_current_selection(self):
        """
        当 mask_level 改变时，如果已经有 current_hierarchy_chain，
        重新计算要高亮的层级。
        """
        if not self.current_hierarchy_chain:
            return
        level = min(self.mask_level, len(self.current_hierarchy_chain) - 1)
        target_city = self.current_hierarchy_chain[level]
        self._update_highlight_for_cityobject(target_city)

    # ----------------- DPG 注册 -----------------
    def register_dpg(self):
        # 注册纹理
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.width,
                self.height,
                self.render_buffer.flatten(),
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        # 主窗口：显示图像
        with dpg.window(tag="_primary_window", width=self.window_width, height=self.window_height):
            dpg.add_image("_texture")

        dpg.set_primary_window("_primary_window", True)

        # 控制窗口：顶部对象信息 + 折叠面板
        with dpg.window(
            label="Control",
            tag="_control_window",
            width=320,
            height=400,
            pos=[self.window_width + 10, 0],
        ):
            # 顶部：对象信息（固定，不可折叠）
            dpg.add_text("🏠 Building Info")
            dpg.add_input_text(
                tag="_building_info_text",
                multiline=True,
                readonly=True,
                default_value="No building selected.",
                width=300,
                height=110,
            )

            dpg.add_spacer(height=4)
            dpg.add_text("🎯 Selection Info")
            dpg.add_input_text(
                tag="_selection_info_text",
                multiline=True,
                readonly=True,
                default_value="No selection yet.",
                width=300,
                height=80,
            )

            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Clear Selection",
                    callback=lambda: self.clear_selection()
                )

            dpg.add_separator()

            # ========== 折叠面板 1：View & Camera ==========
            with dpg.collapsing_header(label="1. View & Camera", default_open=True):
                # 渲染模式
                dpg.add_text("Render Mode")
                dpg.add_radio_button(
                    items=["RGB", "Segmentation", "Overlay"],
                    default_value=self.mode,
                    callback=lambda s, a: self.set_render_mode(a),
                    tag="_mode_radio",
                )

                dpg.add_spacer(height=4)
                dpg.add_separator()

                # Orbit 相机控制
                dpg.add_text("Orbit Camera")

                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="Reset View",
                        callback=lambda: self.reset_view_orbit(),
                    )
                    dpg.add_button(
                        label="Use Orbit",
                        callback=lambda: self.set_use_train_cam(False),
                    )

                dpg.add_spacer(height=2)
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="Zoom In",
                        callback=lambda: self.zoom_step(-1),
                        width=70,
                    )
                    dpg.add_button(
                        label="Zoom Out",
                        callback=lambda: self.zoom_step(+1),
                        width=70,
                    )

                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="↑ Tilt Up",
                        callback=lambda: self.orbit_step(0, -10),
                        width=80,
                    )
                    dpg.add_button(
                        label="↓ Tilt Down",
                        callback=lambda: self.orbit_step(0, +10),
                        width=80,
                    )

                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="← Pan Left",
                        callback=lambda: self.pan_step(-10, 0),
                        width=80,
                    )
                    dpg.add_button(
                        label="→ Pan Right",
                        callback=lambda: self.pan_step(+10, 0),
                        width=80,
                    )

                dpg.add_spacer(height=4)
                dpg.add_separator()

                # Train Cameras
                dpg.add_text("Train Cameras")

                if self.train_cam_names:
                    dpg.add_combo(
                        label="Camera View",
                        items=self.train_cam_names,
                        default_value=self.train_cam_names[self.active_train_cam_idx],
                        callback=lambda s, a: self.on_select_train_cam(a),
                        width=280,
                        tag="_train_cam_combo",
                    )

                    dpg.add_checkbox(
                        label="Use Train Camera",
                        default_value=self.use_train_cam,
                        callback=lambda s, a: self.set_use_train_cam(a),
                    )
                else:
                    dpg.add_text("No train cameras found.", color=(200, 200, 200))

            dpg.add_separator()

            # ========== 折叠面板 2：交互状态 ==========
            with dpg.collapsing_header(label="2. Interaction State", default_open=True):
                dpg.add_text("Mouse position: ", tag="pos_item")

                dpg.add_spacer(height=4)
                dpg.add_text("Hierarchy / Mask Level")
                dpg.add_text("0 = 当前对象, 1 = 父, 2 = 父的父 ...")

                dpg.add_slider_int(
                    label="Mask Level",
                    tag="_mask_level_slider",
                    default_value=self.mask_level,
                    min_value=0,
                    max_value=5,
                    callback=lambda s, a: self.set_mask_level(a),
                    width=280,
                )

            dpg.add_separator()

            # ========== 折叠面板 3：Search（预留） ==========
            with dpg.collapsing_header(label="3. Search (Reserved)", default_open=False):
                dpg.add_text("Search panel is reserved for future use.")
                dpg.add_input_text(
                    label="Search CityObject ID / name",
                    tag="_search_input",
                    width=250,
                )
                dpg.add_button(
                    label="Search & Focus (TODO)",
                    callback=lambda: self.search_and_focus(
                        dpg.get_value("_search_input")
                    ),
                )

        # 全局无 padding 主题（避免滚动条）
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        dpg.bind_item_theme("_primary_window", theme_no_padding)

        # 鼠标交互
        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            # 在训练相机视角下，不用滚轮控制距离（保持一模一样）
            if self.use_train_cam:
                return
            delta = app_data
            self.camera.scale(delta)

        def toggle_moving_left():
            # 训练相机视角下不允许 Orbit 旋转
            if self.use_train_cam:
                return
            self.moving = not self.moving

        def toggle_moving_middle():
            # 训练相机视角下不允许平移
            if self.use_train_cam:
                return
            self.moving_middle = not self.moving_middle

        def move_handler(sender, pos, user):
            if not dpg.is_item_focused("_primary_window"):
                self.mouse_pos = pos
                return

            # 训练相机视角下禁止 Orbit / Pan
            if self.use_train_cam:
                self.mouse_pos = pos
                return

            dx = self.mouse_pos[0] - pos[0]
            dy = self.mouse_pos[1] - pos[1]

            if self.moving:
                if dx != 0.0 or dy != 0.0:
                    self.camera.orbit(-dx * 50, dy * 50)

            if self.moving_middle:
                if dx != 0.0 or dy != 0.0:
                    self.camera.pan(-dx * 40, dy * 40)

            self.mouse_pos = pos

        def change_pos(sender, app_data):
            xy = dpg.get_mouse_pos(local=False)
            dpg.set_value("pos_item", f"Mouse position = ({xy[0]:.1f}, {xy[1]:.1f})")

        # 点击拾取：右键点击，选中该像素，映射层级，并更新 Building/Selection 信息 + 高亮
        def pick_instance_id():
            # 没有 label_map 或没有 classifier 时，没法选
            if self.label_map_compact is None:
                return

            # 只在主窗口聚焦时响应
            if not dpg.is_item_focused("_primary_window"):
                return

            # 全局鼠标坐标
            mx, my = dpg.get_mouse_pos(local=False)
            # 主窗口左上角坐标
            win_x, win_y = dpg.get_item_pos("_primary_window")

            # 转成 GUI 内图像像素坐标
            ix_gui = int(mx - win_x)
            iy_gui = int(my - win_y)

            if 0 <= ix_gui < self.width and 0 <= iy_gui < self.height and \
               self.label_W is not None and self.label_H is not None:

                # 将 GUI 像素映射回 label_map 的分辨率（训练相机视角下是原图大小）
                x_src = int(ix_gui / self.width * self.label_W)
                y_src = int(iy_gui / self.height * self.label_H)
                x_src = max(0, min(self.label_W - 1, x_src))
                y_src = max(0, min(self.label_H - 1, y_src))

                # 紧凑 id
                new_id = int(self.label_map_compact[y_src, x_src])
                # 原始灰度 id
                if self.label_map_orig is not None:
                    orig_id = int(self.label_map_orig[y_src, x_src])
                else:
                    orig_id = new_id

                self.selected_new_id = new_id
                self.selected_orig_id = orig_id

                # 叶子 cityobject（最底层对象，灰度 id -> cityobject id）
                leaf_city_id = self.grayid_to_cityobject.get(orig_id, None)
                building_id = None

                # 1. 计算 hierarchy chain
                if leaf_city_id is not None:
                    chain = self._get_hierarchy_chain(leaf_city_id)
                    self.current_hierarchy_chain = chain

                    if chain:
                        # 按 slider 的层级选中一个 cityobject 用于 mask
                        level = min(self.mask_level, len(chain) - 1)
                        target_city = chain[level]
                    else:
                        target_city = leaf_city_id
                else:
                    chain = None
                    self.current_hierarchy_chain = None
                    target_city = None

                # 2. 找到所属建筑（chain 里第一个 type == 'Building' 的对象）
                if leaf_city_id is not None and leaf_city_id in self.city_semantics:
                    cid = leaf_city_id
                    while cid is not None:
                        data = self.city_semantics.get(cid, {})
                        if data.get("type", "") == "Building":
                            building_id = cid
                            break
                        cid = data.get("parent", None)

                # 3. 更新高亮（target_city：根据层级 slider 选出来的层级）
                self._update_highlight_for_cityobject(target_city)

                # 4. 更新 GUI 顶部信息面板
                # 建筑信息板块
                if building_id is not None:
                    bdata = self.city_semantics.get(building_id, {})
                    attrs = bdata.get("attributes", {}) or {}
                    lines = [
                        f"Building ID: {building_id}",
                        f"Height: {attrs.get('measuredHeight', 'N/A')}",
                        f"Storeys: {attrs.get('storeysAboveGround', 'N/A')}",
                        f"Function: {attrs.get('function', 'N/A')}",
                        f"RoofType: {attrs.get('roofType', 'N/A')}",
                    ]
                    dpg.set_value("_building_info_text", "\n".join(lines))
                else:
                    dpg.set_value("_building_info_text", "No building info for this selection.")

                # 选取物体板块（根据当前层级 target_city）
                if target_city is not None:
                    sdata = self.city_semantics.get(target_city, {})
                    slines = [
                        f"Object ID: {target_city}",
                        f"Type: {sdata.get('type', 'N/A')}",
                        f"Gray ID (clicked): {orig_id}",
                        f"Mask Level: {self.mask_level}",
                    ]
                    dpg.set_value("_selection_info_text", "\n".join(slines))
                else:
                    dpg.set_value("_selection_info_text", f"No semantic object for gray id {orig_id}.")

                print(
                    f"[GUI] Click pick: gui_pixel=({ix_gui}, {iy_gui}), "
                    f"src_pixel=({x_src}, {y_src}), "
                    f"compact_id={new_id}, gray_id={orig_id}, leaf_city={leaf_city_id}, "
                    f"target_city={target_city}, building_id={building_id}"
                )
            else:
                # 点在图像外，就不处理
                pass

        with dpg.handler_registry():
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_click_handler(dpg.mvMouseButton_Left, callback=lambda: toggle_moving_left())
            dpg.add_mouse_release_handler(dpg.mvMouseButton_Left, callback=lambda: toggle_moving_left())
            dpg.add_mouse_click_handler(dpg.mvMouseButton_Middle, callback=lambda: toggle_moving_middle())
            dpg.add_mouse_release_handler(dpg.mvMouseButton_Middle, callback=lambda: toggle_moving_middle())
            dpg.add_mouse_move_handler(callback=lambda s, a, u: move_handler(s, a, u))
            dpg.add_mouse_click_handler(callback=change_pos)

            # 右键点击：拾取当前像素
            dpg.add_mouse_click_handler(
                button=dpg.mvMouseButton_Right,
                callback=lambda sender, app_data: pick_instance_id()
            )

        dpg.create_viewport(
            title="Gaga Semantic Gaussian Viewer",
            width=self.window_width + 340,
            height=self.window_height,
            resizable=False,
        )

        dpg.setup_dearpygui()
        dpg.show_viewport()

    # ----------------- 构造相机（Orbit 模式） -----------------
    def construct_camera(self) -> Camera:
        pose = self.camera.pose
        R_c2w = pose[:3, :3]
        t_c2w = pose[:3, 3]

        ss = math.pi / 180.0
        fovy = self.camera.fovy * ss

        fy = fov2focal(fovy, self.height)
        fovx = focal2fov(fy, self.width)

        cam = Camera(
            colmap_id=0,
            R=R_c2w,
            T=t_c2w,
            FoVx=fovx,
            FoVy=fovy,
            image=torch.zeros([3, self.height, self.width]),
            gt_alpha_mask=None,
            image_name=None,
            uid=0,
        )
        cam.feature_height, cam.feature_width = self.height, self.width
        return cam

    # ----------------- 渲染一帧 -----------------
    @torch.no_grad()
    def fetch_and_render(self, view_camera: Camera):
        render_pkg = render(view_camera, self.gaussians, self.pipe, self.background)

        # RGB（原始分辨率）
        rgb = render_pkg["render"].permute(1, 2, 0).cpu().numpy().astype(np.float32)
        rgb = np.clip(rgb, 0.0, 1.0)
        src_H, src_W = rgb.shape[:2]

        seg_rgb = None

        # 语义预测：紧凑 id -> 映射回原始 id
        if self.classifier is not None and "render_seg" in render_pkg:
            objects_2d = render_pkg["render_seg"]  # [C, H_src, W_src]
            feat = objects_2d.unsqueeze(0).to(device)  # [1, C, H_src, W_src]
            logits = self.classifier(feat)            # [1, num_classes, H_src, W_src]
            labels_compact = torch.argmax(logits, dim=1)[0]  # [H_src, W_src]，紧凑 id

            # 保存紧凑 id label 图（numpy）
            self.label_map_compact = labels_compact.cpu().numpy().astype(np.int32)
            self.label_H, self.label_W = self.label_map_compact.shape

            # 映射回原始灰度 id：inverse_lookup[new_id] = old_id
            if self.inverse_lookup is not None:
                labels_orig = self.inverse_lookup[labels_compact]  # [H_src, W_src]，原始 id
                self.label_map_orig = labels_orig.cpu().numpy().astype(np.int32)
            else:
                self.label_map_orig = self.label_map_compact.copy()

            # 上色：仍然用紧凑 id 调色，保证和训练时 color_map 一致
            labels_clamped = np.clip(self.label_map_compact, 0, self.num_classes - 1)
            seg_rgb = self.class_colors[labels_clamped]  # [H_src, W_src, 3]
        else:
            self.label_map_compact = None
            self.label_map_orig = None
            self.label_H, self.label_W = src_H, src_W

        # 根据模式组合（仍在原分辨率下）
        if self.mode == "RGB" or seg_rgb is None:
            out = rgb
        elif self.mode == "Segmentation":
            out = seg_rgb
        elif self.mode == "Overlay":
            alpha = 0.5
            out = alpha * seg_rgb + (1.0 - alpha) * rgb
            out = np.clip(out, 0.0, 1.0)
        else:
            out = rgb

        # 高亮选中的对象（根据灰度 id 集合）
        if (self.highlight_gray_ids is not None) and (self.label_map_orig is not None):
            mask = np.isin(self.label_map_orig, list(self.highlight_gray_ids))
            if mask.any():
                out = out.copy()
                alpha_h = float(self.highlight_alpha)
                out[mask] = alpha_h * self.highlight_color + (1.0 - alpha_h) * out[mask]
                out = np.clip(out, 0.0, 1.0)

        # 把原分辨率 out（H_src, W_src）缩放到 GUI 分辨率 (self.height, self.width)
        if src_H != self.height or src_W != self.width:
            out_t = torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0)  # [1,3,H_src,W_src]
            out_t = F.interpolate(out_t, size=(self.height, self.width), mode="nearest")
            out = out_t.squeeze(0).permute(1, 2, 0).cpu().numpy()

        self.render_buffer = out.reshape(-1).astype(np.float32)
        dpg.set_value("_texture", self.render_buffer)

    # ----------------- 主循环 -----------------
    def render_loop(self):
        while dpg.is_dearpygui_running():
            if self.load_model:
                if self.use_train_cam and self.train_cameras:
                    cam = self.train_cameras[self.active_train_cam_idx]
                else:
                    cam = self.construct_camera()
                self.fetch_and_render(cam)
            dpg.render_dearpygui_frame()


# =====================
# main
# =====================

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

    # 确定实际 iteration
    loaded_iter = getattr(scene, "loaded_iter", None)
    if loaded_iter is None and args.iteration > 0:
        loaded_iter = args.iteration

    classifier = None
    num_classes = 0
    inverse_lookup = None

    # ========= 从训练用 id_mapping.json 构建 new_id -> old_id =========
    # 这个在 dataset.source_path / dataset.object_path 下：old_id(灰度) -> new_id(紧凑)
    try:
        matched_mask_path = os.path.join(dataset.source_path, dataset.object_path)
        train_id_map_path = os.path.join(matched_mask_path, "id_mapping.json")
        if os.path.exists(train_id_map_path):
            with open(train_id_map_path, "r") as f:
                raw_id_map = json.load(f)
            id_map = {int(k): int(v) for k, v in raw_id_map.items()}  # old_id -> new_id

            if len(id_map) > 0:
                max_new_id = max(id_map.values())
            else:
                max_new_id = 0

            num_classes = max_new_id + 1  # 背景0 + K个前景
            print(f"[GUI] Loaded train id_mapping.json from {train_id_map_path}")
            print(f"[GUI] num_classes (with background) = {num_classes}")

            # new_id -> old_id 查表（tensor）
            inverse_lookup = torch.zeros(num_classes, dtype=torch.long, device=device)
            inverse_lookup[0] = 0
            for old_id, new_id in id_map.items():
                if 0 <= new_id < num_classes:
                    inverse_lookup[new_id] = int(old_id)
        else:
            print(f"[GUI] 未找到训练用 id_mapping.json：{train_id_map_path}，将无法精确映射回原始灰度 id")
            num_classes = 0
            inverse_lookup = None
    except Exception as e:
        print(f"[GUI] 读取训练用 id_mapping.json 失败: {e}")
        num_classes = 0
        inverse_lookup = None

    # ========= 构造 classifier，并加载训练好的权重 =========
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

    # ========= 从 model_path 读取 city_semantics.json 和 id_mapping.json（灰度 id -> CityObject id） =========
    grayid_to_cityobject = {}
    city_semantics = {}

    model_path = dataset.model_path
    sem_path = os.path.join(model_path, "city_semantics.json")
    gray_id_map_path = os.path.join(model_path, "id_mapping.json")

    # city_semantics.json: CityObject ID -> 语义信息
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

    # model_path/id_mapping.json: 灰度 id(原始) -> CityObject ID
    try:
        if os.path.exists(gray_id_map_path):
            with open(gray_id_map_path, "r", encoding="utf-8") as f:
                raw_gray_id_map = json.load(f)
            for k, v in raw_gray_id_map.items():
                try:
                    gid = int(k)           # 灰度 id
                except ValueError:
                    print(f"[GUI] model_path/id_mapping.json 的键无法转换为 int: {k}")
                    continue
                grayid_to_cityobject[gid] = str(v)   # CityObject ID
            print(f"[GUI] Loaded gray-id->cityobject id_mapping.json from {gray_id_map_path} "
                  f"(num mapped ids = {len(grayid_to_cityobject)})")
        else:
            print(f"[GUI] 未找到 model_path/id_mapping.json：{gray_id_map_path}，CityObject 映射将不可用。")
    except Exception as e:
        print(f"[GUI] 读取 model_path/id_mapping.json 失败: {e}")
        grayid_to_cityobject = {}

    # 自动根据高 opacity 高斯估计一个“主体中心”和半径（Orbit 初始视角）
    focus_center, focus_radius = estimate_focus_from_gaussians(gaussians)

    # 根据场景大小估计一个合适的初始半径
    cameras_extent = getattr(scene, "cameras_extent", 1.0)
    auto_radius = max(focus_radius, cameras_extent * 1.2)

    # 如果用户显式给了 gui_radius 且 >0，则优先用用户的；否则用自动估计
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
        width=args.gui_width,
        height=args.gui_height,
        radius=init_radius,
    )

    # 设置基于 opacity 估计的主体中心和半径（Orbit 模式）
    gui.set_initial_center(focus_center, radius=init_radius)

    gui.render_loop()


if __name__ == "__main__":
    main()
