#
# Copyright (C) 2026, CityGMLGaussian
# All rights reserved.
#

from typing import Dict, Any, Optional, Set, List

class HierarchyManager:
    """
    Responsibilities:
    - gray id -> CityObject id -> hierarchy chain
    - compute the set of gray ids to highlight under the current mask_level
    - generate text for Building Info & Selection Info
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

        # Current selection state
        self.current_hierarchy_chain: Optional[List[str]] = None
        self.highlight_gray_ids: Optional[Set[int]] = None

        # Build parent/children links
        for cid, data in self.city_semantics.items():
            parent = data.get("parent")
            if parent is not None:
                self.city_children.setdefault(parent, []).append(cid)

        # Build city -> gray ids
        for gid, cid in self.grayid_to_cityobject.items():
            self.city_to_grayids.setdefault(cid, []).append(gid)

        self.feature_to_grayids: Dict[tuple, Set[int]] = {}
        self.feature_surface_to_grayids: Dict[tuple, Set[int]] = {}
        self.feature_surface_part_to_grayids: Dict[tuple, Set[int]] = {}

        self.last_clicked_gray_id: Optional[int] = None

        self._build_feature_maps()

    # ---------- Internal helpers ----------
    def _build_feature_maps(self):
        """Precompute gray-id sets for feature/surface/part from city_semantics + city_to_grayids."""
        for cid, data in self.city_semantics.items():
            attrs = data.get("attributes", {}) or {}
            f = attrs.get("feature", None)
            s = attrs.get("surface", None)
            p = attrs.get("part", None)

            # All gray ids covered by this cityobject
            gids = self.city_to_grayids.get(cid, [])
            if not gids:
                continue

            if f is None:
                # Without feature, we cannot build feature-based hierarchy; skip
                continue

            # level 2: feature only
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
        """Traverse from leaf upwards: leaf -> parent -> ... -> root."""
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

    # ---------- Public API ----------
    def set_mask_level(self, level: int):
        self.mask_level = max(0, int(level))
        # If a pixel was clicked before, recompute highlight when switching levels
        if self.last_clicked_gray_id is not None:
            self._compute_highlight_for_gray(self.last_clicked_gray_id)

    def _compute_highlight_for_gray(self, gray_id: int):
        """
        Update self.highlight_gray_ids based on the current mask_level and the clicked gray id.

        Rules (CityObject-hierarchy based):
        - gray_id == 0: background, no highlight
        - no mapping for gray_id: highlight itself (same for all levels)
        - has mapping:
          level 0: highlight only the leaf cityobject itself
          level 1: highlight the containing "wall surface" (WallSurface) and all its descendants
          level 2: highlight the whole building (Building) and all its descendants
        """
        if gray_id == 0:
            self.highlight_gray_ids = None
            return

        leaf_city_id = self.grayid_to_cityobject.get(gray_id, None)

        # No CityObject mapping: unknown object, always highlight itself
        if leaf_city_id is None or leaf_city_id not in self.city_semantics:
            self.highlight_gray_ids = {gray_id}
            return

        level = self.mask_level
        selected: Set[int] = set()

        # ---------- level 0: leaf only ----------
        if level == 0:
            gids = self.city_to_grayids.get(leaf_city_id, [])
            if gids:
                selected.update(gids)
            else:
                selected.add(gray_id)

        # ---------- level 1: containing wall surface (WallSurface) ----------
        elif level == 1:
            wall_id = self._find_wall_for_leaf(leaf_city_id)
            if wall_id is None:
                # If no wall surface found, fall back to level 0
                gids = self.city_to_grayids.get(leaf_city_id, [])
                if gids:
                    selected.update(gids)
                else:
                    selected.add(gray_id)
            else:
                # Wall surface + all descendants
                all_cities: Set[str] = {wall_id}
                all_cities |= self._get_descendants(wall_id)
                for cid in all_cities:
                    for gid in self.city_to_grayids.get(cid, []):
                        selected.add(gid)

        # ---------- level 2: whole building ----------
        else:
            building_id = self._find_building_for_leaf(leaf_city_id)
            if building_id is None:
                # If no building found, fall back to level 1 logic
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
                # Building + all descendants
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

        # First compute highlight set according to the current mask_level
        self._compute_highlight_for_gray(gray_id)

        leaf_city_id = self.grayid_to_cityobject.get(gray_id, None)
        building_id = self._find_building_for_leaf(leaf_city_id)

        # ---------- Building Info text ----------
        if building_id is not None:
            bdata = self.city_semantics.get(building_id, {})
            attrs = bdata.get("attributes", {}) or {}

            # Height in meters
            h_val = attrs.get("measuredHeight", None)
            height_str = "N/A" if h_val is None else f"{h_val} m"

            # Storeys above ground
            s_val = attrs.get("storeysAboveGround", None)
            storeys_str = "N/A" if s_val is None else f"{s_val} floors"

            # Function mapping
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

            # RoofType mapping
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

        # ---------- Selection Info text ----------
        if leaf_city_id is not None and leaf_city_id in self.city_semantics:
            sdata = self.city_semantics.get(leaf_city_id, {})

            selection_text = "\n".join([
                f"Gray ID: {gray_id}",
                f"CityObject ID: {leaf_city_id}",
                f"Type: {sdata.get('type', 'N/A')}",
            ])
        else:
            # Object without semantic mapping: only show gray id
            selection_text = "\n".join([
                f"Gray ID: {gray_id}",
                "CityObject ID: N/A",
                "Type: Unknown",
            ])

        return {
            "building_text": building_text,
            "selection_text": selection_text,
            "leaf_city_id": leaf_city_id,
            "target_city_id": leaf_city_id,
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
        Walk from the leaf upwards and return the first object whose type == 'WallSurface',
        used as the representative "containing wall surface".
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
