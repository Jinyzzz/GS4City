# semantic_viewer/hierarchy.py
from typing import Dict, Any, Optional, Set, List

class HierarchyManager:
    """
    负责：
    - 灰度 id -> CityObject id -> 层级链
    - 计算当前 mask_level 下应高亮的灰度 id 集合
    - 生成 Building Info & Selection Info 的文本
    """

    def __init__(
        self,
        grayid_to_cityobject: Dict[int, str],
        city_semantics: Dict[str, Any],
        mask_level: int = 0,
        function_map: Optional[Dict[str, str]] = None,
        rooftype_map: Optional[Dict[str, str]] = None,
    ):
        self.grayid_to_cityobject = grayid_to_cityobject or {}
        self.city_semantics = city_semantics or {}
        self.mask_level = max(0, int(mask_level))
        self.function_map = {str(k): str(v) for k, v in (function_map or {}).items()}
        self.rooftype_map = {str(k): str(v) for k, v in (rooftype_map or {}).items()}

        self.city_children: Dict[str, list] = {}
        self.city_to_grayids: Dict[str, list] = {}
        self.city_descendants_cache: Dict[str, Set[str]] = {}

        # 当前选择状态
        self.current_hierarchy_chain: Optional[List[str]] = None
        self.highlight_gray_ids: Optional[Set[int]] = None

        # 建 parent/children
        for cid, data in self.city_semantics.items():
            parent = data.get("parent")
            if parent is not None:
                self.city_children.setdefault(parent, []).append(cid)

        # 建 city -> gray ids
        for gid, cid in self.grayid_to_cityobject.items():
            self.city_to_grayids.setdefault(cid, []).append(gid)

        self.feature_to_grayids: Dict[tuple, Set[int]] = {}
        self.feature_surface_to_grayids: Dict[tuple, Set[int]] = {}
        self.feature_surface_part_to_grayids: Dict[tuple, Set[int]] = {}

        self.last_clicked_gray_id: Optional[int] = None

        self._build_feature_maps()

    # ---------- 内部工具 ----------
    def _build_feature_maps(self):
        """从 city_semantics + city_to_grayids 预计算 feature/surface/part 对应的灰度集合。"""
        for cid, data in self.city_semantics.items():
            attrs = data.get("attributes", {}) or {}
            f = attrs.get("feature", None)
            s = attrs.get("surface", None)
            p = attrs.get("part", None)

            # 这个 cityobject 覆盖的所有灰度 id
            gids = self.city_to_grayids.get(cid, [])
            if not gids:
                continue

            if f is None:
                # 没有 feature，就没办法按 feature 分级，跳过
                continue

            # level 2: 只按 feature
            key_f = (str(f),)
            self.feature_to_grayids.setdefault(key_f, set()).update(gids)

            if s is not None:
                # level 1: feature + surface
                key_fs = (str(f), str(s))
                self.feature_surface_to_grayids.setdefault(key_fs, set()).update(gids)

                if p is not None:
                    # level 0: feature + surface + part
                    key_fsp = (str(f), str(s), str(p))
                    self.feature_surface_part_to_grayids.setdefault(key_fsp, set()).update(gids)


    def _get_hierarchy_chain(self, leaf_city_id: str):
        """从叶子一直向上：leaf -> parent -> ... -> root"""
        if leaf_city_id is None or leaf_city_id not in self.city_semantics:
            return None

        chain = []
        visited = set()
        cid = leaf_city_id
        while cid is not None and cid not in visited:
            chain.append(cid)
            visited.add(cid)
            data = self.city_semantics.get(cid, {})
            cid = data.get("parent")
        return chain

    def _get_descendants(self, cid: str) -> Set[str]:
        if cid in self.city_descendants_cache:
            return self.city_descendants_cache[cid]

        res: Set[str] = set()
        for child in self.city_children.get(cid, []):
            res.add(child)
            res |= self._get_descendants(child)
        self.city_descendants_cache[cid] = res
        return res

    def _update_highlight_for_cityobject(self, city_id: Optional[str]):
        if city_id is None:
            self.highlight_gray_ids = None
            return

        all_cities = {city_id}
        all_cities |= self._get_descendants(city_id)

        gray_ids: Set[int] = set()
        for cid in all_cities:
            for gid in self.city_to_grayids.get(cid, []):
                gray_ids.add(gid)

        self.highlight_gray_ids = gray_ids if gray_ids else None

    def _find_building_for_leaf(self, leaf_city_id: Optional[str]) -> Optional[str]:
        if leaf_city_id is None:
            return None

        cid = leaf_city_id
        while cid is not None:
            data = self.city_semantics.get(cid, {})
            if data.get("type", "") == "Building":
                return cid
            cid = data.get("parent")
        return None

    # ---------- 外部接口 ----------
    def set_mask_level(self, level: int):
        self.mask_level = max(0, int(level))
        # 如果之前点过某个像素，则换 level 的时候重新算一次高亮
        if self.last_clicked_gray_id is not None:
            self._compute_highlight_for_gray(self.last_clicked_gray_id)

    def _compute_highlight_for_gray(self, gray_id: int):
        """
        根据当前 mask_level 和点击的灰度 id，更新 self.highlight_gray_ids。

        规则（基于 CityObject 层级）：
        - gray_id == 0: 背景，不高亮
        - gray_id 无映射：高亮自己（所有 level 都一样）
        - 有映射：
          level 0: 只高亮叶子 cityobject 自身
          level 1: 高亮“所在墙面”（WallSurface）及其所有子对象
          level 2: 高亮整栋建筑（Building）及其所有子对象
        """
        if gray_id == 0:
            self.highlight_gray_ids = None
            return

        leaf_city_id = self.grayid_to_cityobject.get(gray_id, None)

        # 映射不到任何 CityObject：未知物体，所有 level 都高亮自己
        if leaf_city_id is None or leaf_city_id not in self.city_semantics:
            self.highlight_gray_ids = {gray_id}
            return

        level = self.mask_level
        selected: Set[int] = set()

        # ---------- level 0: 只叶子 ----------
        if level == 0:
            gids = self.city_to_grayids.get(leaf_city_id, [])
            if gids:
                selected.update(gids)
            else:
                selected.add(gray_id)

        # ---------- level 1: 所在墙面（WallSurface） ----------
        elif level == 1:
            wall_id = self._find_wall_for_leaf(leaf_city_id)
            if wall_id is None:
                # 找不到墙面，就退化成 level 0
                gids = self.city_to_grayids.get(leaf_city_id, [])
                if gids:
                    selected.update(gids)
                else:
                    selected.add(gray_id)
            else:
                # 墙面 + 它所有子对象
                all_cities: Set[str] = {wall_id}
                all_cities |= self._get_descendants(wall_id)
                for cid in all_cities:
                    for gid in self.city_to_grayids.get(cid, []):
                        selected.add(gid)

        # ---------- level 2: 整栋建筑 ----------
        else:
            building_id = self._find_building_for_leaf(leaf_city_id)
            if building_id is None:
                # 找不到 building，就退化成 level 1 的逻辑
                wall_id = self._find_wall_for_leaf(leaf_city_id)
                if wall_id is None:
                    gids = self.city_to_grayids.get(leaf_city_id, [])
                    if gids:
                        selected.update(gids)
                    else:
                        selected.add(gray_id)
                else:
                    all_cities: Set[str] = {wall_id}
                    all_cities |= self._get_descendants(wall_id)
                    for cid in all_cities:
                        for gid in self.city_to_grayids.get(cid, []):
                            selected.add(gid)
            else:
                # building + 全部后代
                all_cities: Set[str] = {building_id}
                all_cities |= self._get_descendants(building_id)
                for cid in all_cities:
                    for gid in self.city_to_grayids.get(cid, []):
                        selected.add(gid)

        if not selected:
            selected = {gray_id}

        self.highlight_gray_ids = selected
        print(f"[DEBUG] level={self.mask_level}, gray_id={gray_id}, highlight_size={len(self.highlight_gray_ids)}")


    def handle_click(self, gray_id: int) -> Dict[str, Any]:
        self.last_clicked_gray_id = gray_id

        # 先按当前 mask_level 计算高亮集合
        self._compute_highlight_for_gray(gray_id)

        leaf_city_id = self.grayid_to_cityobject.get(gray_id, None)
        building_id = self._find_building_for_leaf(leaf_city_id)

        # ---------- Building Info 文本（和之前一致，只是高度/层/映射那套） ----------
        if building_id is not None:
            bdata = self.city_semantics.get(building_id, {})
            attrs = bdata.get("attributes", {}) or {}

            # Height + m
            h_val = attrs.get("measuredHeight", None)
            height_str = "N/A" if h_val is None else f"{h_val} m"

            # Storeys + 层
            s_val = attrs.get("storeysAboveGround", None)
            storeys_str = "N/A" if s_val is None else f"{s_val} 层"

            # Function 映射
            raw_func = attrs.get("function", None)
            if raw_func is None:
                func_str = "N/A"
            else:
                if isinstance(raw_func, (list, tuple)):
                    mapped = []
                    for c in raw_func:
                        key = str(c)
                        mapped.append(self.function_map.get(key, key))
                    func_str = ", ".join(mapped) if mapped else "N/A"
                else:
                    key = str(raw_func)
                    func_str = self.function_map.get(key, key)

            # RoofType 映射
            raw_roof = attrs.get("roofType", None)
            if raw_roof is None:
                roof_str = "N/A"
            else:
                if isinstance(raw_roof, (list, tuple)):
                    mapped = []
                    for c in raw_roof:
                        key = str(c)
                        mapped.append(self.rooftype_map.get(key, key))
                    roof_str = ", ".join(mapped) if mapped else "N/A"
                else:
                    key = str(raw_roof)
                    roof_str = self.rooftype_map.get(key, key)

            building_text = "\n".join([
                f"Building ID: {building_id}",
                f"Height: {height_str}",
                f"Storeys: {storeys_str}",
                f"Function: {func_str}",
                f"RoofType: {roof_str}",
            ])
        else:
            building_text = "No building info for this selection."

        # ---------- Selection Info 文本 ----------
        if leaf_city_id is not None and leaf_city_id in self.city_semantics:
            sdata = self.city_semantics.get(leaf_city_id, {})
            attrs = sdata.get("attributes", {}) or {}
            f = attrs.get("feature", "N/A")
            s_ = attrs.get("surface", "N/A")
            p = attrs.get("part", "N/A")

            selection_text = "\n".join([
                f"Object ID: {leaf_city_id}",
                f"Type: {sdata.get('type', 'N/A')}",
                f"Gray ID (clicked): {gray_id}",
                f"Mask Level: {self.mask_level}",
                f"Feature: {f}",
                f"Surface: {s_}",
                f"Part: {p}",
            ])
        else:
            selection_text = "\n".join([
                f"Unknown semantic object.",
                f"Gray ID (clicked): {gray_id}",
                f"Mask Level: {self.mask_level}",
            ])

        return {
            "building_text": building_text,
            "selection_text": selection_text,
            "leaf_city_id": leaf_city_id,
            "target_city_id": leaf_city_id,  # 现在不再用 parent 层级了，简化
            "building_id": building_id,
        }
    def _find_building_for_leaf(self, leaf_city_id: Optional[str]) -> Optional[str]:
        if leaf_city_id is None:
            return None

        cid = leaf_city_id
        while cid is not None:
            data = self.city_semantics.get(cid, {})
            if data.get("type", "") == "Building":
                return cid
            cid = data.get("parent")
        return None

    def _find_wall_for_leaf(self, leaf_city_id: Optional[str]) -> Optional[str]:
        """
        从叶子一路往上找第一个 type == 'WallSurface' 的对象，
        作为“所在墙面”的代表。
        """
        if leaf_city_id is None:
            return None

        cid = leaf_city_id
        while cid is not None:
            data = self.city_semantics.get(cid, {})
            if data.get("type", "") == "WallSurface":
                return cid
            cid = data.get("parent")
        return None
    
    def clear_selection(self):
        self.current_hierarchy_chain = None
        self.highlight_gray_ids = None
        self.last_clicked_gray_id = None


