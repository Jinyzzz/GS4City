# semantic_viewer/dpg_gui.py
import math
import numpy as np
import torch
import torch.nn.functional as F

import dearpygui.dearpygui as dpg
from typing import Optional
from gaussian_renderer import render
from scene.cameras import Camera
from utils.graphics_utils import fov2focal, focal2fov

from .orbit_camera import OrbitCamera
from .focus_utils import build_color_map
from .hierarchy import HierarchyManager

torch.backends.cudnn.enabled = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SemanticGaussianGUI:
    def __init__(
        self,
        scene,
        gaussians,
        pipe,
        background: torch.Tensor,
        classifier: torch.nn.Module = None,
        num_classes: int = 0,
        inverse_lookup: torch.Tensor = None,
        grayid_to_cityobject: dict = None,
        city_semantics: dict = None,
        building_function_map: Optional[dict] = None,
        building_rooftype_map: Optional[dict] = None,
        width: int = 800,
        height: int = 600,
        radius: float = 2.0,
    ):
        self.scene = scene
        self.gaussians = gaussians
        self.pipe = pipe
        self.background = background
        self.classifier = classifier
        self.num_classes = num_classes

        self.inverse_lookup = inverse_lookup  # new_id -> old_id
        grayid_to_cityobject = grayid_to_cityobject or {}
        city_semantics = city_semantics or {}

        self.width = width
        self.height = height
        self.window_width = width
        self.window_height = height

        self.camera = OrbitCamera(self.width, self.height, r=radius)

        # Orbit / Train camera
        self.use_train_cam = False
        self.train_cameras = []
        self.train_cam_names = []
        for idx, cam in enumerate(self.scene.getTrainCameras()):
            name = getattr(cam, "image_name", None) or f"cam_{idx}"
            self.train_cameras.append(cam)
            self.train_cam_names.append(f"{idx}: {name}")
        self.active_train_cam_idx = 0

        # Reset View 初始值
        self.init_center = self.camera.center.copy()
        self.init_radius = float(radius)

        # 渲染缓冲
        self.render_buffer = np.zeros((self.height, self.width, 3), dtype=np.float32)
        self.mode = "RGB"  # "RGB" / "Segmentation" / "Overlay"

        # 调色板
        self.class_colors = build_color_map(self.num_classes) if self.num_classes > 0 else None

        # label maps
        self.label_map_compact = None
        self.label_map_orig = None
        self.label_H = None
        self.label_W = None

        # 语义层级管理
        self.hierarchy = HierarchyManager(
            grayid_to_cityobject=grayid_to_cityobject,
            city_semantics=city_semantics,
            mask_level=0,
            function_map=building_function_map,
            rooftype_map=building_rooftype_map,
        )

        # 高亮参数
        self.highlight_color = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.highlight_alpha = 0.6

        self.load_model = True

        # 鼠标状态
        self.moving = False
        self.moving_middle = False
        self.mouse_pos = (0, 0)

        # DPG 初始化
        dpg.create_context()
        self.register_dpg()

    def __del__(self):
        try:
            dpg.destroy_context()
        except Exception:
            pass

    # ---------- 相机 / UI 控制 ----------
    def set_initial_center(self, center_np: np.ndarray, radius: float = None):
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
        self.camera.scale(delta)

    def orbit_step(self, dx_deg: float, dy_deg: float):
        self.camera.orbit(dx_deg, dy_deg)

    def pan_step(self, dx: float, dy: float):
        self.camera.pan(dx, dy)

    def on_select_train_cam(self, display_name: str):
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
        # 把值传给 HierarchyManager，让它负责计算高亮
        self.hierarchy.set_mask_level(level)
        print(f"[GUI] 更新 mask 层级: {self.hierarchy.mask_level}")

    def search_and_focus(self, query: str):
        if not query:
            print("[GUI] search_and_focus: 空查询")
            return
        print(f"[GUI] [TODO] search_and_focus: {query}")

    def clear_selection(self):
        self.hierarchy.clear_selection()
        dpg.set_value("_building_info_text", "No building selected.")
        dpg.set_value("_selection_info_text", "No selection yet.")

    # ---------- DearPyGUI 注册 ----------
    def register_dpg(self):
        # 纹理
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.width,
                self.height,
                self.render_buffer.flatten(),
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        # 主窗口
        with dpg.window(tag="_primary_window", width=self.window_width, height=self.window_height):
            dpg.add_image("_texture")

        dpg.set_primary_window("_primary_window", True)

        # 控制窗口
        with dpg.window(
            label="Control",
            tag="_control_window",
            width=320,
            height=400,
            pos=[self.window_width + 10, 0],
        ):
            # Building Info
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

            # 1. View & Camera
            with dpg.collapsing_header(label="1. View & Camera", default_open=True):
                dpg.add_text("Render Mode")
                dpg.add_radio_button(
                    items=["RGB", "Segmentation", "Overlay"],
                    default_value=self.mode,
                    callback=lambda s, a: self.set_render_mode(a),
                    tag="_mode_radio",
                )

                dpg.add_spacer(height=4)
                dpg.add_separator()

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

            # 2. Interaction State
            with dpg.collapsing_header(label="2. Interaction State", default_open=True):
                dpg.add_text("Mouse position: ", tag="pos_item")

                dpg.add_spacer(height=4)
                dpg.add_text("Hierarchy / Mask Level")
                dpg.add_text("0 = 当前对象, 1 = 父, 2 = 父的父 ...")
                dpg.add_slider_int(
                    label="Mask Level",
                    tag="_mask_level_slider",
                    default_value=self.hierarchy.mask_level,
                    min_value=0,
                    max_value=2,   # 只要 0,1,2 三挡
                    callback=lambda s, a: self.set_mask_level(a),
                    width=280,
                )


            dpg.add_separator()

            # 3. Search
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

        # 去 padding
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        dpg.bind_item_theme("_primary_window", theme_no_padding)

        # ------- 鼠标 & 交互 handler -------
        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            if self.use_train_cam:
                return
            delta = app_data
            self.camera.scale(delta)

        def toggle_moving_left():
            if self.use_train_cam:
                return
            self.moving = not self.moving

        def toggle_moving_middle():
            if self.use_train_cam:
                return
            self.moving_middle = not self.moving_middle

        def move_handler(sender, pos, user):
            if not dpg.is_item_focused("_primary_window"):
                self.mouse_pos = pos
                return

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

        def pick_instance_id():
            if self.label_map_compact is None:
                return
            if not dpg.is_item_focused("_primary_window"):
                return

            mx, my = dpg.get_mouse_pos(local=False)
            win_x, win_y = dpg.get_item_pos("_primary_window")

            ix_gui = int(mx - win_x)
            iy_gui = int(my - win_y)

            if 0 <= ix_gui < self.width and 0 <= iy_gui < self.height and \
               self.label_W is not None and self.label_H is not None:

                x_src = int(ix_gui / self.width * self.label_W)
                y_src = int(iy_gui / self.height * self.label_H)
                x_src = max(0, min(self.label_W - 1, x_src))
                y_src = max(0, min(self.label_H - 1, y_src))

                new_id = int(self.label_map_compact[y_src, x_src])
                if self.label_map_orig is not None:
                    orig_id = int(self.label_map_orig[y_src, x_src])
                else:
                    orig_id = new_id

                # 灰度 0 = 背景，不选中
                if orig_id == 0:
                    print("[GUI] Click on background (gray id 0), ignore.")
                    return
                
                info = self.hierarchy.handle_click(orig_id)

                dpg.set_value("_building_info_text", info["building_text"])
                dpg.set_value("_selection_info_text", info["selection_text"])

                print(
                    f"[GUI] Click pick: gui_pixel=({ix_gui}, {iy_gui}), "
                    f"src_pixel=({x_src}, {y_src}), "
                    f"compact_id={new_id}, gray_id={orig_id}, "
                    f"leaf_city={info['leaf_city_id']}, "
                    f"target_city={info['target_city_id']}, "
                    f"building_id={info['building_id']}"
                )

        with dpg.handler_registry():
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_click_handler(dpg.mvMouseButton_Left, callback=lambda: toggle_moving_left())
            dpg.add_mouse_release_handler(dpg.mvMouseButton_Left, callback=lambda: toggle_moving_left())
            dpg.add_mouse_click_handler(dpg.mvMouseButton_Middle, callback=lambda: toggle_moving_middle())
            dpg.add_mouse_release_handler(dpg.mvMouseButton_Middle, callback=lambda: toggle_moving_middle())
            dpg.add_mouse_move_handler(callback=lambda s, a, u: move_handler(s, a, u))
            dpg.add_mouse_click_handler(callback=change_pos)
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

    # ---------- 构造相机 ----------
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

    # ---------- 渲染 ----------
    @torch.no_grad()
    def fetch_and_render(self, view_camera: Camera):
        render_pkg = render(view_camera, self.gaussians, self.pipe, self.background)

        rgb = render_pkg["render"].permute(1, 2, 0).cpu().numpy().astype(np.float32)
        rgb = np.clip(rgb, 0.0, 1.0)
        src_H, src_W = rgb.shape[:2]

        seg_rgb = None

        if self.classifier is not None and "render_seg" in render_pkg:
            objects_2d = render_pkg["render_seg"]  # [C, H_src, W_src]
            feat = objects_2d.unsqueeze(0).to(device)
            logits = self.classifier(feat)
            labels_compact = torch.argmax(logits, dim=1)[0]

            self.label_map_compact = labels_compact.cpu().numpy().astype(np.int32)
            self.label_H, self.label_W = self.label_map_compact.shape

            if self.inverse_lookup is not None:
                labels_orig = self.inverse_lookup[labels_compact]
                self.label_map_orig = labels_orig.cpu().numpy().astype(np.int32)
            else:
                self.label_map_orig = self.label_map_compact.copy()

            labels_clamped = np.clip(self.label_map_compact, 0, self.num_classes - 1)
            seg_rgb = self.class_colors[labels_clamped]
        else:
            self.label_map_compact = None
            self.label_map_orig = None
            self.label_H, self.label_W = src_H, src_W

        # 模式混合
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

        # 高亮
        if (self.hierarchy.highlight_gray_ids is not None) and (self.label_map_orig is not None):
            mask = np.isin(self.label_map_orig, list(self.hierarchy.highlight_gray_ids))
            if mask.any():
                out = out.copy()
                alpha_h = float(self.highlight_alpha)
                out[mask] = alpha_h * self.highlight_color + (1.0 - alpha_h) * out[mask]
                out = np.clip(out, 0.0, 1.0)



        # resize 到 GUI 分辨率
        if src_H != self.height or src_W != self.width:
            out_t = torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0)
            out_t = F.interpolate(out_t, size=(self.height, self.width), mode="nearest")
            out = out_t.squeeze(0).permute(1, 2, 0).cpu().numpy()

        self.render_buffer = out.reshape(-1).astype(np.float32)
        dpg.set_value("_texture", self.render_buffer)

    # ---------- 主循环 ----------
    def render_loop(self):
        while dpg.is_dearpygui_running():
            if self.load_model:
                if self.use_train_cam and self.train_cameras:
                    cam = self.train_cameras[self.active_train_cam_idx]
                else:
                    cam = self.construct_camera()
                self.fetch_and_render(cam)
            dpg.render_dearpygui_frame()
