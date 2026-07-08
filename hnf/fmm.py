# -*- coding: utf-8 -*-
"""Part 3: Fast Multipole Method and direct propagation."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from hnf.kernel import HuygensKernel


class FastMultipoleMethod:
    """快速多极子法 (FMM) — O(N log N) 近似."""

    def __init__(
        self,
        kernel: HuygensKernel,
        max_leaf_size: int = 32,
        expansion_order: int = 8,
        use_gpu: bool = True,
    ):
        self.kernel = kernel
        self.max_leaf_size = max_leaf_size
        self.expansion_order = expansion_order
        self.use_gpu = use_gpu

    def build_tree(self, x: torch.Tensor) -> Dict:
        d = x.shape[1]
        min_coord = x.min(dim=0)[0]
        max_coord = x.max(dim=0)[0]
        center = (min_coord + max_coord) / 2
        size = (max_coord - min_coord).max()
        nodes: list[Dict] = []

        def build_node(indices: np.ndarray, center_np: np.ndarray, size_val: float, level: int, parent_idx: int = -1) -> Dict:
            node = {
                "indices": indices,
                "center": center_np,
                "size": size_val,
                "level": level,
                "parent": parent_idx,
                "children": [],
                "multipole": None,
                "local": None,
            }

            if len(indices) > self.max_leaf_size and level < 10:
                half = size_val / 2
                child_indices = [[] for _ in range(2 ** d)]
                for idx in indices:
                    pos = x[idx]
                    child_idx = 0
                    for dim in range(d):
                        if pos[dim] > center_np[dim]:
                            child_idx |= 1 << dim
                    child_indices[child_idx].append(idx)

                for ci, child_inds in enumerate(child_indices):
                    if len(child_inds) > 0:
                        child_center = center_np.copy()
                        for dim in range(d):
                            if ci & (1 << dim):
                                child_center[dim] += half
                            else:
                                child_center[dim] -= half
                        child_node = build_node(
                            np.array(child_inds),
                            child_center,
                            half,
                            level + 1,
                            parent_idx=len(nodes),
                        )
                        node["children"].append(len(nodes))
                        nodes.append(child_node)

            return node

        root = build_node(np.arange(len(x)), center.numpy(), size.item(), 0, parent_idx=-1)
        nodes.append(root)
        for idx, node in enumerate(nodes):
            for child_idx in node["children"]:
                nodes[child_idx]["parent"] = idx
        return {"nodes": nodes, "root": 0}

    def compute_multipole_expansion(self, tree: Dict, sources: torch.Tensor) -> Dict:
        nodes = tree["nodes"]
        for node_idx in reversed(range(len(nodes))):
            node = nodes[node_idx]
            if len(node["children"]) == 0:
                if len(node["indices"]) > 0:
                    idx = node["indices"]
                    center_t = torch.as_tensor(node["center"], device=sources.device, dtype=sources.dtype)
                    relative_pos = sources[idx] - center_t
                    multipole = torch.zeros(self.expansion_order, sources.shape[1], device=sources.device)
                    for n in range(self.expansion_order):
                        multipole[n] = (relative_pos ** n).sum(dim=0)
                    node["multipole"] = multipole
                else:
                    node["multipole"] = None
            else:
                combined = torch.zeros(self.expansion_order, sources.shape[1], device=sources.device)
                for child_idx in node["children"]:
                    child = nodes[child_idx]
                    if child["multipole"] is not None:
                        combined += child["multipole"]
                node["multipole"] = combined
        return tree

    def compute_local_expansion(self, tree: Dict) -> Dict:
        nodes = tree["nodes"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for node_idx in range(len(nodes)):
            node = nodes[node_idx]
            if node_idx == 0:
                node["local"] = torch.zeros(self.expansion_order, 1, device=device)
            else:
                parent = nodes[node["parent"]]
                node["local"] = parent["local"].clone() if parent["local"] is not None else None

            if node["parent"] >= 0:
                parent = nodes[node["parent"]]
                for sibling_idx in parent["children"]:
                    if sibling_idx != node_idx:
                        sibling = nodes[sibling_idx]
                        if sibling["multipole"] is not None:
                            if node["local"] is None:
                                node["local"] = torch.zeros(self.expansion_order, 1, device=sibling["multipole"].device)
                            node["local"] += sibling["multipole"][0:1]
        return tree

    def evaluate(self, tree: Dict, x: torch.Tensor, sources: Optional[torch.Tensor] = None) -> torch.Tensor:
        nodes = tree["nodes"]
        near_field = torch.zeros(len(x), sources.shape[1] if sources is not None else 1, device=x.device)

        if sources is not None:
            for node in nodes:
                if len(node["children"]) == 0 and len(node["indices"]) > 1:
                    idx = node["indices"]
                    x_sub = x[idx].unsqueeze(0)
                    s_sub = sources[idx]
                    k = self.kernel(x_sub, None, None)
                    k_real = torch.abs(k.squeeze(0))
                    near_field[idx] += torch.matmul(k_real, s_sub)

        far_field = torch.zeros(len(x), 1, device=x.device)
        for node in nodes:
            if node["local"] is not None and len(node["indices"]) > 0:
                center_t = torch.as_tensor(node["center"], device=x.device, dtype=x.dtype)
                for idx in node["indices"]:
                    relative_pos = x[idx] - center_t
                    expansion = 0
                    for n in range(self.expansion_order):
                        if n < len(node["local"]):
                            expansion += node["local"][n] * (relative_pos[n] if n < len(relative_pos) else 0)
                    far_field[idx] += expansion
        return near_field + far_field

    def forward(
        self,
        x: torch.Tensor,
        sources: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        rho: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tree = self.build_tree(x)
        tree = self.compute_multipole_expansion(tree, sources)
        tree = self.compute_local_expansion(tree)
        return self.evaluate(tree, x, sources)

    def estimate_complexity(self, n: int) -> float:
        return n * np.log(n) * self.expansion_order


class DirectPropagation:
    """直接传播 O(N^2)."""

    def __init__(self, kernel: HuygensKernel):
        self.kernel = kernel

    def forward(
        self,
        x: torch.Tensor,
        sources: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        rho: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        k = self.kernel(x.unsqueeze(0), t.unsqueeze(0) if t is not None else None, rho)
        k_real = torch.abs(k.squeeze(0))
        return torch.matmul(k_real, sources)
