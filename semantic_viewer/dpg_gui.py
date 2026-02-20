#
# Copyright (C) 2026, CityGMLGaussian
# All rights reserved.
#

import math
import numpy as np
import torch
import torch.nn.functional as F
import os

from semantic_viewer.instance_query import InstanceQueryEngine
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
        model_root: Optional[str] = None,
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

        # Initial values for "Reset View"
        self.init_center = self.camera.center.copy()
        self.init_radius = float(radius)

        # Render buffer
        self.render_buffer = np.zeros((self.height, self.width, 3), dtype=np.float32)
        self.mode = "RGB"  # "RGB" / "Segmentation" / "Overlay"

        # Color palette
        self.class_colors = build_color_map(self.num_classes) if self.num_classes > 0 else None

        # Label maps
        self.label_map_compact = None
        self.label_map_orig = None
        self.label_H = None
        self.label_W = None

        # Semantic hierarchy manager
        self.hierarchy = HierarchyManager(
            grayid_to_cityobject=grayid_to_cityobject,
            city_semantics=city_semantics,
            mask_level=0,
            function_map=building_function_map,
            rooftype_map=building_rooftype_map,
        )

        # Semantic query engine (CityGML + CLIP)
        self.query_engine: Optional[InstanceQueryEngine] = None
        self.model_root = model_root

        if self.model_root is not None:
            id_mapping_path = os.path.join(self.model_root, "id_mapping.json")
            city_semantics_path = os.path.join(self.model_root, "city_semantics.json")
            clip_index_path = os.path.join(self.model_root, "object_clip_index.npz")

            if os.path.exists(id_mapping_path) and os.path.exists(city_semantics_path):
                if os.path.exists(clip_index_path):
                    try:
                        print(f"[GUI] Initializing InstanceQueryEngine: {id_mapping_path}, {city_semantics_path}, {clip_index_path}")
                        self.query_engine = InstanceQueryEngine(
                            id_mapping_path=id_mapping_path,
                            city_semantics_path=city_semantics_path,
                            object_clip_index_path=clip_index_path,
                            device=device,
                        )
                        print("[GUI] Query engine ready (CityGML + CLIP).")
                    except Exception as e:
                        print(f"[GUI] Failed to initialize query engine: {e}")
                else:
                    try:
                        # CityGML-only fallback: you can implement a lightweight CityGML-only engine,
                        # or keep it as a warning for now.
                        print("[GUI] object_clip_index.npz not found; only CityGML type queries are available.")
                        self.query_engine = InstanceQueryEngine(
                            id_mapping_path=id_mapping_path,
                            city_semantics_path=city_semantics_path,
                            object_clip_index_path=clip_index_path,
                        )
                    except Exception as e:
                        print(f"[GUI] Failed to initialize CityGML-only query engine: {e}")
            else:
                print("[GUI] Required id_mapping.json / city_semantics.json not found; query is unavailable.")

        # Highlight parameters
        self.highlight_color = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.highlight_alpha = 0.6

        self.load_model = True

        # Mouse state
        self.moving = False
        self.moving_middle = False
        self.mouse_pos = (0, 0)

        # DPG initialization
        dpg.create_context()
        self.register_dpg()

    def __del__(self):
        try:
            dpg.destroy_context()
        except Exception:
            pass

    # ---------- Camera / UI controls ----------
    def set_initial_center(self, center_np: np.ndarray, radius: float = None):
        try:
            center_np = np.asarray(center_np, dtype=np.float32).reshape(3,)
            self.camera.center = center_np
            self.init_center = center_np.copy()
            if radius is not None:
                self.camera.radius = float(radius)
                self.init_radius = float(radius)
            print(f"[GUI] Initial camera center set to {center_np}, radius={self.init_radius:.3f}")
        except Exception as e:
            print(f"[GUI] set_initial_center failed: {e}")

    def set_render_mode(self, mode: str):
        self.mode = mode

    def reset_view_orbit(self):
        self.camera.center = self.init_center.copy()
        self.camera.radius = float(self.init_radius)
        self.use_train_cam = False
        print("[GUI] View reset to the initial Orbit view")

    def set_use_train_cam(self, flag: bool):
        self.use_train_cam = bool(flag)
        print(f"[GUI] Use train camera: {self.use_train_cam}")

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
            print(f"[GUI] Failed to parse train camera index: {display_name}")
            return

        if 0 <= idx < len(self.train_cameras):
            self.active_train_cam_idx = idx

            self.snap_orbit_to_train_cam(idx)

            self.use_train_cam = False

            try:
                cam_now = self.construct_camera()
                self.fetch_and_render(cam_now)
            except Exception as e:
                print(f"[GUI] immediate refresh failed: {e}")

            print(f"[GUI] Selected & snapped to train camera #{idx}: {display_name}")
        else:
            print(f"[GUI] Train camera index out of range: {idx}")
    def set_mask_level(self, level: int):
        # Pass the value to HierarchyManager; it computes the highlight set
        self.hierarchy.set_mask_level(level)
        print(f"[GUI] Updated mask hierarchy level: {self.hierarchy.mask_level}")

    def search_and_focus(self, query: str):
        if not query:
            print("[GUI] search_and_focus: empty query")
            return
        print(f"[GUI] [TODO] search_and_focus: {query}")

    def clear_selection(self):
        self.hierarchy.clear_selection()
        dpg.set_value("_building_info_text", "No building selected.")
        dpg.set_value("_selection_info_text", "No selection yet.")

    def run_text_query(self):
        # 1. Check prerequisites
        if self.query_engine is None:
            print("[GUI] Query engine not initialized (missing id_mapping / city_semantics / CLIP index?)")
            return

        desc = dpg.get_value("_query_description_input").strip()
        if not desc:
            print("[GUI] run_text_query: empty description")
            return

        if self.label_map_orig is None:
            print("[GUI] label_map_orig is not available for the current view (classifier not run?), cannot query.")
            return

        thr = dpg.get_value("_query_threshold_slider")
        inst_img = self.label_map_orig.astype(np.int32)

        try:
            mask, heatmap, route = self.query_engine.query_image_auto(
                instance_img=inst_img,
                description=desc,
                similarity_threshold=float(thr),
            )
        except Exception as e:
            print(f"[GUI] Query error: {e}")
            return

        # 2. Collect all matched instances (gray ids) from the mask
        selected_ids = np.unique(inst_img[mask.astype(bool)])
        selected_ids = selected_ids[selected_ids != 0]  # remove background 0

        # 3. Update highlight set (gray ids)
        if selected_ids.size == 0:
            self.hierarchy.highlight_gray_ids = None
            dpg.set_value(
                "_selection_info_text",
                f"No match for query: '{desc}' (route={route})",
            )
            print(f"[GUI] Query '{desc}' via {route}: no matched instances.")
            return

        self.hierarchy.highlight_gray_ids = set(int(i) for i in selected_ids)

        # 4. Update right-side info panel (show query stats rather than selecting one object)
        info_text = "\n".join([
            f"Query: {desc}",
            f"Route: {route}",
            f"Matched instances: {len(selected_ids)}",
        ])
        dpg.set_value("_selection_info_text", info_text)

        print(f"[GUI] Query '{desc}' via {route}: highlighted {len(selected_ids)} instances.")

    # ---------- DearPyGUI registration ----------
    def register_dpg(self):
        RIGHT_PANEL_WIDTH = 400
        RIGHT_PANEL_X = self.window_width + 10
        # Texture
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.width,
                self.height,
                self.render_buffer.flatten(),
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        # Main window
        with dpg.window(tag="_primary_window", width=self.window_width, height=self.window_height):
            dpg.add_image("_texture")

        dpg.set_primary_window("_primary_window", True)

        # --------- Window 1: Building / Selection (always expanded, no scrolling) ---------
        INFO_HEIGHT = 300

        with dpg.window(
            label="Building / Selection",
            tag="_info_window",
            width=RIGHT_PANEL_WIDTH,
            height=INFO_HEIGHT,
            pos=[self.window_width + 10, 0],
            no_move=True,
            no_resize=True,
            no_collapse=True,
            no_close=True,
            no_scrollbar=True,
        ):
            dpg.add_text("Building Info")
            dpg.add_input_text(
                tag="_building_info_text",
                multiline=True,
                readonly=True,
                default_value="No building selected.",
                width=RIGHT_PANEL_WIDTH - 20,
                height=110,
            )

            dpg.add_spacer(height=4)
            dpg.add_text("Selection Info")
            dpg.add_input_text(
                tag="_selection_info_text",
                multiline=True,
                readonly=True,
                default_value="No selection yet.",
                width=RIGHT_PANEL_WIDTH - 20,
                height=80,
            )

            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Clear Selection",
                    callback=lambda: self.clear_selection()
                )

        # --------- Window 2: Controls (View + Interaction + Search, collapsible, scrollable) ---------
        CONTROLS_HEIGHT = self.window_height - INFO_HEIGHT - 20

        with dpg.window(
            label="Controls",
            tag="_controls_window",
            width=RIGHT_PANEL_WIDTH,
            height=CONTROLS_HEIGHT,
            pos=[self.window_width + 10, INFO_HEIGHT + 10],
            no_move=True,
            no_resize=True,
            no_close=True,
        ):
            # --- View & Camera ---
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
                    label="Tilt Up",
                    callback=lambda: self.orbit_step(0, -10),
                    width=80,
                )
                dpg.add_button(
                    label="Tilt Down",
                    callback=lambda: self.orbit_step(0, +10),
                    width=80,
                )

            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Pan Left",
                    callback=lambda: self.pan_step(-10, 0),
                    width=80,
                )
                dpg.add_button(
                    label="Pan Right",
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

            else:
                dpg.add_text("No train cameras found.", color=(200, 200, 200))

            dpg.add_separator()
            dpg.add_spacer(height=4)

            # --- Interaction / Mask ---
            dpg.add_text("Mouse position: ", tag="pos_item")

            dpg.add_spacer(height=4)
            dpg.add_text("Hierarchy / Mask Level")
            dpg.add_text("0 = part, 1 = surface, 2 = building")

            dpg.add_slider_int(
                label="Mask Level",
                tag="_mask_level_slider",
                default_value=self.hierarchy.mask_level,
                min_value=0,
                max_value=2,
                callback=lambda s, a: self.set_mask_level(a),
                width=280,
            )
            dpg.add_separator()
            dpg.add_spacer(height=4)

            # --- Semantic Query ---
            dpg.add_text("Semantic Query (CityGML / CLIP)")
            dpg.add_input_text(
                label="Description",
                tag="_query_description_input",
                width=RIGHT_PANEL_WIDTH - 60,
            )
            dpg.add_slider_float(
                label="Similarity Threshold (CLIP)",
                tag="_query_threshold_slider",
                default_value=0.6,
                min_value=0.0,
                max_value=1.0,
                width=RIGHT_PANEL_WIDTH - 60,
            )

        # Remove padding
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        dpg.bind_item_theme("_primary_window", theme_no_padding)

        # ------- Mouse & interaction handlers -------
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

                # Gray id 0 = background, do not select
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
            width=self.window_width + RIGHT_PANEL_WIDTH + 10,
            height=self.window_height,
            resizable=False,
        )

        dpg.setup_dearpygui()
        dpg.show_viewport()

    def snap_orbit_to_train_cam(self, idx: int):
        if not (0 <= idx < len(self.train_cameras)):
            print(f"[GUI] snap_orbit_to_train_cam: invalid idx={idx}")
            return

        cam = self.train_cameras[idx]

        try:
            import numpy as np
            from scipy.spatial.transform import Rotation as SciRot

            R_raw = np.asarray(cam.R, dtype=np.float32)
            T_raw = np.asarray(cam.T, dtype=np.float32).reshape(3)

            print("[GUI] ===== snap_orbit_to_train_cam (exact inverse) =====")
            print(f"[GUI] cam idx = {idx}")
            print(f"[GUI] cam.R shape = {R_raw.shape}")
            print(f"[GUI] cam.T = {T_raw}")
            print(f"[GUI] current orbit center = {self.camera.center}, radius = {self.camera.radius}")

            R_c2w = R_raw
            t_c2w = T_raw

            r = float(max(self.camera.radius, 0.05))

            center = R_c2w @ (t_c2w - np.array([0.0, 0.0, r], dtype=np.float32))

            self.camera.rot = SciRot.from_matrix(R_c2w)
            self.camera.radius = r
            self.camera.center = center.astype(np.float32)

            if hasattr(cam, "FoVy") and cam.FoVy is not None:
                try:
                    self.camera.fovy = float(np.degrees(float(cam.FoVy)))
                except Exception:
                    pass

            try:
                test_pose = self.camera.pose  # c2w
                test_R = np.asarray(test_pose[:3, :3], dtype=np.float32)
                test_t = np.asarray(test_pose[:3, 3], dtype=np.float32)

                err_R = float(np.linalg.norm(test_R - R_c2w))
                err_t = float(np.linalg.norm(test_t - t_c2w))
                print(f"[GUI] verify |R-R_train|={err_R:.6f}, |t-t_train|={err_t:.6f}")
            except Exception as ve:
                print(f"[GUI] verify failed: {ve}")

            print(f"[GUI] set orbit.center = {self.camera.center}")
            print(f"[GUI] set orbit.radius = {self.camera.radius}")
            print(f"[GUI] set orbit.fovy = {self.camera.fovy}")
            print("[GUI] snap_orbit_to_train_cam done (c2w exact inverse)")
            print("[GUI] ================================================")

        except Exception as e:
            print(f"[GUI] snap_orbit_to_train_cam failed: {e}")

    # ---------- Camera construction ----------
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

    # ---------- Rendering ----------
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

        # Mode blending
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

        # Highlight
        if (self.hierarchy.highlight_gray_ids is not None) and (self.label_map_orig is not None):
            mask = np.isin(self.label_map_orig, list(self.hierarchy.highlight_gray_ids))
            if mask.any():
                out = out.copy()
                alpha_h = float(self.highlight_alpha)
                out[mask] = alpha_h * self.highlight_color + (1.0 - alpha_h) * out[mask]
                out = np.clip(out, 0.0, 1.0)

        # Resize to GUI resolution
        if src_H != self.height or src_W != self.width:
            out_t = torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0)
            out_t = F.interpolate(out_t, size=(self.height, self.width), mode="nearest")
            out = out_t.squeeze(0).permute(1, 2, 0).cpu().numpy()

        self.render_buffer = out.reshape(-1).astype(np.float32)
        dpg.set_value("_texture", self.render_buffer)

    # ---------- Main loop ----------
    def render_loop(self):
        while dpg.is_dearpygui_running():
            if self.load_model:
                if self.use_train_cam and self.train_cameras:
                    cam = self.train_cameras[self.active_train_cam_idx]
                else:
                    cam = self.construct_camera()
                self.fetch_and_render(cam)
            dpg.render_dearpygui_frame()
