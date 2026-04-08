#
# Copyright (C) 2026, GS4City
# All rights reserved.
#

import json
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import open_clip
import cv2
from pathlib import Path
from collections import defaultdict

# ============================================================================
# CityGML index (only handles: instance_id <-> citygml_id <-> type / parent)
# ============================================================================

class CityGMLSemanticIndex:
    """
    - instance index (0,1,2,...) -> CityGML object id
    - CityGML object id -> type, parent, children
    """

    def __init__(
        self,
        id_mapping_path: str,
        city_semantics_path: str,
    ):
        with open(id_mapping_path, "r") as f:
            self.instance_to_city_id: Dict[str, str] = json.load(f)

        with open(city_semantics_path, "r") as f:
            self.city_semantics: Dict[str, Dict] = json.load(f)

        self.city_id_to_type: Dict[str, str] = {}
        self.city_id_to_parent: Dict[str, Optional[str]] = {}
        self.parent_to_children: Dict[str, List[str]] = defaultdict(list)

        for cid, rec in self.city_semantics.items():
            ctype = rec.get("type", "")
            parent = rec.get("parent")
            self.city_id_to_type[cid] = ctype
            self.city_id_to_parent[cid] = parent
            if parent is not None:
                self.parent_to_children[parent].append(cid)

    # --- basic helpers ---

    def get_instance_city_id(self, instance_id: int) -> Optional[str]:
        return self.instance_to_city_id.get(str(instance_id))

    def get_city_object_type(self, city_id: str) -> Optional[str]:
        return self.city_id_to_type.get(city_id)

    def _collect_descendants(self, root_id: str) -> Set[str]:
        """Return the set of all descendants of root_id (including itself)."""
        result: Set[str] = set()
        stack = [root_id]
        while stack:
            cid = stack.pop()
            if cid in result:
                continue
            result.add(cid)
            for child in self.parent_to_children.get(cid, []):
                stack.append(child)
        return result

    def get_all_types(self) -> Set[str]:
        """Set of all type strings that appear in city_semantics."""
        return set(self.city_id_to_type.values())

    def build_city_ids_for_type_with_descendants(self, type_name: str) -> Set[str]:
        """
        Given a CityGML type_name (e.g., 'WallSurface'),
        find all objects with type==type_name and all their descendants.
        """
        roots = [cid for cid, t in self.city_id_to_type.items() if t == type_name]
        all_ids: Set[str] = set()
        for root in roots:
            all_ids.update(self._collect_descendants(root))
        return all_ids

    def get_instance_ids_for_type(self, type_name: str) -> Set[int]:
        """
        Return all instance_id values that belong to this type or its descendants.
        """
        city_ids_for_type = self.build_city_ids_for_type_with_descendants(type_name)
        inst_ids: Set[int] = set()
        for inst_str, city_id in self.instance_to_city_id.items():
            if city_id in city_ids_for_type:
                inst_ids.add(int(inst_str))
        return inst_ids

# ============================================================================
# CLIP instance index (only handles: instance features + arbitrary text similarity)
# ============================================================================

class CLIPInstanceIndex:
    """
    Holds CLIP features for instances, and can compute similarity of instances
    to arbitrary text descriptions (without class_id / class_mapping).
    """

    def __init__(
        self,
        object_clip_index_path: str,
        device: Optional[torch.device] = None,
        model_name: str = "ViT-B-16",
        pretrained: str = "laion2b_s34b_b88k",
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # CLIP model
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # Load features
        npz = np.load(object_clip_index_path)

        if "features" not in npz:
            raise ValueError(
                f"{object_clip_index_path} must contain 'features'. "
                f"Available keys: {list(npz.keys())}"
            )
        self.features = npz["features"].astype(np.float32)  # (N, D)

        # Support both 'instance_ids' and 'ids'
        if "instance_ids" in npz:
            self.instance_ids = npz["instance_ids"].astype(np.int32)
        elif "ids" in npz:
            self.instance_ids = npz["ids"].astype(np.int32)
        else:
            raise ValueError(
                f"{object_clip_index_path} must contain 'instance_ids' or 'ids'. "
                f"Available keys: {list(npz.keys())}"
            )

        if self.features.ndim != 2:
            raise ValueError(f"features must be (N, D), got {self.features.shape}")
        if self.instance_ids.shape[0] != self.features.shape[0]:
            raise ValueError(
                f"instance_ids and features length mismatch: "
                f"{self.instance_ids.shape[0]} vs {self.features.shape[0]}"
            )

        # Normalize features
        feat_norm = np.linalg.norm(self.features, axis=1, keepdims=True) + 1e-8
        self.features = self.features / feat_norm

        # Mean feature for each instance
        self.instance_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, inst_id in enumerate(self.instance_ids.tolist()):
            self.instance_to_indices[inst_id].append(idx)

        self.instance_mean_features: Dict[int, np.ndarray] = {}
        for inst_id, idxs in self.instance_to_indices.items():
            self.instance_mean_features[inst_id] = self.features[idxs].mean(axis=0)

    # --- text encoding ---

    def _encode_texts(self, prompts: List[str]) -> torch.Tensor:
        """Encode prompts -> normalized CLIP embeddings, (P, D)."""
        with torch.no_grad():
            tokens = self.tokenizer(prompts).to(self.device)
            text_feat = self.model.encode_text(tokens).float()
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        return text_feat

    # --- similarity ---

    def compute_similarity_to_text_for_instances(
        self,
        description: str,
        instance_ids_subset: Optional[Set[int]] = None,
    ) -> Dict[int, float]:
        """
        Compute cosine similarity of each instance to a given text description.
        Returns:
            dict[instance_id -> similarity]
        """
        with torch.no_grad():
            text_emb = self._encode_texts([description])  # (1, D)

        feat_dim = self.features.shape[1]
        sim_dict: Dict[int, float] = {}

        available_inst_ids = set(self.instance_mean_features.keys())
        if instance_ids_subset is None:
            inst_ids = available_inst_ids
        else:
            inst_ids = set(instance_ids_subset) & available_inst_ids

        with torch.no_grad():
            for inst_id in inst_ids:
                inst_feat = self.instance_mean_features[inst_id]
                inst_feat_t = torch.from_numpy(inst_feat).to(self.device).view(1, feat_dim)
                inst_feat_t = inst_feat_t / inst_feat_t.norm(dim=-1, keepdim=True)
                sim = (inst_feat_t @ text_emb.T).item()
                sim_dict[inst_id] = float(sim)

        return sim_dict

# ============================================================================
# QueryEngine: query only, no class_id mapping
# ============================================================================

class InstanceQueryEngine:
    """
    Only for interactive querying (auto-select CityGML + CLIP):

      - If description matches a CityGML type exactly (case-insensitive):
            select instances using CityGML type, mask=1 means pixels belong to that type.
      - Otherwise:
            use CLIP to compute instance-text similarity and threshold the normalized score.
    """

    def __init__(
        self,
        id_mapping_path: str,
        city_semantics_path: str,
        object_clip_index_path: str,
        device: Optional[torch.device] = None,
        model_name: str = "ViT-B-16",
        pretrained: str = "laion2b_s34b_b88k",
    ):
        self.city_index = CityGMLSemanticIndex(id_mapping_path, city_semantics_path)
        self.clip_index = CLIPInstanceIndex(
            object_clip_index_path=object_clip_index_path,
            device=device,
            model_name=model_name,
            pretrained=pretrained,
        )

        # All CityGML types that appear (strings)
        self.all_citygml_types: Set[str] = set(
            t for t in self.city_index.city_id_to_type.values() if t
        )

    # --- CLIP route ---

    def _query_clip_route(
        self,
        instance_img: np.ndarray,
        description: str,
        similarity_threshold: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if instance_img.ndim != 2:
            raise ValueError(f"instance_img must be 2D, got {instance_img.shape}")

        h, w = instance_img.shape
        instance_img = instance_img.astype(np.int32)
        unique_inst = np.unique(instance_img)
        inst_ids_set = set(int(i) for i in unique_inst)

        similarity_dict = self.clip_index.compute_similarity_to_text_for_instances(
            description=description,
            instance_ids_subset=inst_ids_set,
        )

        sim_map = np.zeros((h, w), dtype=np.float32)

        sims = np.array(list(similarity_dict.values()), dtype=np.float32)
        if sims.size > 0:
            s_min, s_max = sims.min(), sims.max()
            denom = s_max - s_min + 1e-6
        else:
            s_min, denom = 0.0, 1.0

        for inst_id in unique_inst:
            inst_id_int = int(inst_id)
            sim = similarity_dict.get(inst_id_int, 0.0)
            norm_sim = (sim - s_min) / denom  # 0~1
            sim_map[instance_img == inst_id_int] = norm_sim

        mask = sim_map >= similarity_threshold
        return mask.astype(np.uint8), sim_map

    # --- CityGML route ---

    def _query_citygml_route(
        self,
        instance_img: np.ndarray,
        type_name: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find all instances belonging to type_name (e.g., 'WallSurface'),
        including descendants, and return mask + 0/1 heatmap.
        """
        if instance_img.ndim != 2:
            raise ValueError(f"instance_img must be 2D, got {instance_img.shape}")

        h, w = instance_img.shape
        instance_img = instance_img.astype(np.int32)

        inst_ids_for_type = self.city_index.get_instance_ids_for_type(type_name)
        mask = np.isin(instance_img, list(inst_ids_for_type)).astype(np.uint8)
        heatmap = mask.astype(np.float32)  # No similarity score in CityGML route, so use 0/1

        return mask, heatmap

    # --- auto route selection ---

    def query_image_auto(
        self,
        instance_img: np.ndarray,
        description: str,
        similarity_threshold: float,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Auto-select route:
          - If description matches a CityGML type exactly (case-insensitive), use CityGML route;
          - Otherwise use CLIP route.

        Returns:
            mask: uint8, 0/1
            heatmap: float32, 0~1
            route: "citygml" or "clip"
        """
        desc_clean = description.strip()
        desc_lower = desc_clean.lower()

        matched_type = None
        for t in self.all_citygml_types:
            if desc_lower == t.lower():
                matched_type = t
                break

        if matched_type is not None:
            mask, heatmap = self._query_citygml_route(instance_img, matched_type)
            return mask, heatmap, "citygml"
        else:
            mask, sim_map = self._query_clip_route(
                instance_img, desc_clean, similarity_threshold
            )
            return mask, sim_map, "clip"

# ============================================================================
# CLI
# ============================================================================

def _sanitize_description(desc: str) -> str:
    """Convert description into a filename-friendly string."""
    import re
    s = desc.strip().lower()
    s = s.replace(" ", "_")
    s = re.sub(r"[^a-zA-Z0-9_]+", "", s)
    if not s:
        s = "query"
    return s


def _cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="Instance-based semantic query (CityGML + CLIP, auto route, NO class_mapping)"
    )
    parser.add_argument(
        "--instance_image",
        type=str,
        required=True,
        help="Path to 16-bit grayscale instance image (objects_test/*.png)",
    )
    parser.add_argument(
        "--model_root",
        type=str,
        required=True,
        help="Path to model root (contains id_mapping.json, city_semantics.json, object_clip_index.npz)",
    )
    parser.add_argument(
        "--description",
        type=str,
        required=True,
        help=(
            "Text description to query. "
            "If it exactly matches a CityGML 'type' (e.g., WallSurface, RoofSurface, Window, Door, GroundSurface), "
            "CityGML route will be used; otherwise CLIP route."
        ),
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.6,
        help="Threshold on normalized similarity (for CLIP route). Default: 0.25",
    )

    args = parser.parse_args()

    instance_image_path = Path(args.instance_image)
    model_root = Path(args.model_root)

    id_mapping_path = model_root / "id_mapping.json"
    city_semantics_path = model_root / "city_semantics.json"
    object_clip_index_path = model_root / "object_clip_index.npz"

    for p in [id_mapping_path, city_semantics_path, object_clip_index_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    # Build query-only engine (no class_mapping / citygml_class_map)
    engine = InstanceQueryEngine(
        id_mapping_path=str(id_mapping_path),
        city_semantics_path=str(city_semantics_path),
        object_clip_index_path=str(object_clip_index_path),
    )

    # Read instance image
    inst_img = cv2.imread(str(instance_image_path), cv2.IMREAD_UNCHANGED)
    if inst_img is None:
        raise RuntimeError(f"Failed to load instance image: {instance_image_path}")
    if inst_img.ndim == 3:
        inst_img = inst_img[:, :, 0]
    inst_img = inst_img.astype(np.int32)

    mask, heatmap, route = engine.query_image_auto(
        instance_img=inst_img,
        description=args.description,
        similarity_threshold=args.similarity_threshold,
    )

    query_dir = model_root / "query"
    query_dir.mkdir(parents=True, exist_ok=True)

    desc_tag = _sanitize_description(args.description)
    img_stem = instance_image_path.stem

    mask_path = query_dir / f"{img_stem}_{desc_tag}_mask.png"
    heatmap_path = query_dir / f"{img_stem}_{desc_tag}_heatmap.png"

    cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
    heatmap_u8 = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(str(heatmap_path), heatmap_u8)

    print(f"[{route.upper()}] Saved mask to {mask_path}")
    print(f"[{route.upper()}] Saved heatmap to {heatmap_path}")


if __name__ == "__main__":
    _cli()