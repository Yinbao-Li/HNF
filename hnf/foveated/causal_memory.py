# -*- coding: utf-8 -*-
"""Causal memory graph and gaze scheduler for foveated active perception."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from hnf.foveated.fovea_processor import FoveaOutput, SUPPORTED_WINDOW_SIZES
from hnf.foveated.peripheral_scanner import CandidateRegion
from hnf.kernel import HuygensKernel


TaskType = Literal["p_only", "ps_pick", "velocity_inversion"]


@dataclass
class CausalNode:
    """One fovea gaze result stored as a causal node."""

    node_id: int
    time_stamp: int  # gaze center (sample index)
    window_start: int
    window_end: int
    window_size: int
    phase_picks: dict[str, float]  # {"p": idx, "s": idx} (mean over batch or first)
    velocity_model: Optional[dict[str, float]] = None
    confidence: float = 0.0
    uncertainty: float = 1.0
    snr: float = 0.0
    batch_index: int = 0
    meta: dict = field(default_factory=dict)


@dataclass
class CausalEdge:
    source: int
    target: int
    strength: float
    travel_time_diff: float


@dataclass
class SchedulerDecision:
    focus_index: int
    window_size: int
    reason: str
    priority: float


class WindowSelector(nn.Module):
    """Choose gaze window size from task type + local SNR."""

    def __init__(
        self,
        supported_windows: Sequence[int] = SUPPORTED_WINDOW_SIZES,
        sample_rate_hz: float = 100.0,
    ):
        super().__init__()
        self.supported_windows = tuple(int(w) for w in supported_windows)
        self.sample_rate_hz = float(sample_rate_hz)

    def forward(
        self,
        *,
        task: TaskType = "ps_pick",
        snr: float = 10.0,
        uncertainty: float = 0.0,
    ) -> int:
        if task == "p_only":
            base = 200 if snr >= 8.0 else 400
        elif task == "velocity_inversion":
            base = 1500 if uncertainty >= 0.8 else 1200
        else:  # ps_pick
            base = 800 if snr >= 5.0 else 1200

        # High uncertainty -> enlarge window toward inversion band.
        if uncertainty >= 1.2:
            base = max(base, 1200)
        if uncertainty >= 2.0:
            base = max(base, 1500)

        return min(self.supported_windows, key=lambda w: abs(w - base))


class CausalGraph:
    """Directed causal graph over gaze nodes with Huygens association strengths."""

    def __init__(
        self,
        *,
        prune_confidence: float = 0.3,
        sample_rate_hz: float = 100.0,
        assoc_kernel: Optional[HuygensKernel] = None,
    ):
        self.prune_confidence = float(prune_confidence)
        self.sample_rate_hz = float(sample_rate_hz)
        self.nodes: dict[int, CausalNode] = {}
        self.edges: list[CausalEdge] = []
        self._next_id = 0
        self.assoc_kernel = assoc_kernel or HuygensKernel(
            gamma=0.4,
            omega=0.5,
            wave_speed=6.0,
            distance_mode="time",
            causal=True,
            use_complex=False,
            sparse_band=False,
            local_window_sec=30.0,
        )

    def __len__(self) -> int:
        return len(self.nodes)

    def clear(self) -> None:
        self.nodes.clear()
        self.edges.clear()
        self._next_id = 0

    def add_from_fovea(
        self,
        fov: FoveaOutput,
        *,
        batch_index: int = 0,
    ) -> CausalNode:
        p_idx = float(fov.p_idx_global[batch_index].detach().cpu())
        s_idx = float(fov.s_idx_global[batch_index].detach().cpu())
        conf = float(fov.confidence[batch_index].detach().cpu())
        unc = float(fov.uncertainty[batch_index].detach().cpu())
        snr = float(fov.snr[batch_index].detach().cpu())
        vel = None
        if fov.velocity_model is not None:
            vel = {
                k: float(v[batch_index].detach().mean().cpu())
                if torch.is_tensor(v)
                else float(v)
                for k, v in fov.velocity_model.items()
            }
        node = CausalNode(
            node_id=self._next_id,
            time_stamp=int(fov.focus_index),
            window_start=int(fov.window_start),
            window_end=int(fov.window_end),
            window_size=int(fov.window_size),
            phase_picks={"p": p_idx, "s": s_idx},
            velocity_model=vel,
            confidence=conf,
            uncertainty=unc,
            snr=snr,
            batch_index=batch_index,
        )
        self._next_id += 1
        self._insert_node(node)
        return node

    def _insert_node(self, node: CausalNode) -> None:
        # Associate with all historical nodes via Huygens kernel strength.
        if self.nodes:
            hist = list(self.nodes.values())
            t_new = torch.tensor(
                [[node.time_stamp / self.sample_rate_hz]], dtype=torch.float32
            )
            t_old = torch.tensor(
                [[n.time_stamp / self.sample_rate_hz for n in hist]],
                dtype=torch.float32,
            )
            # Features: [t, conf, unc] — lightweight proxy for association.
            x_new = torch.tensor(
                [[[node.time_stamp / self.sample_rate_hz, node.confidence, node.uncertainty]]],
                dtype=torch.float32,
            )
            x_old = torch.tensor(
                [
                    [
                        [
                            n.time_stamp / self.sample_rate_hz,
                            n.confidence,
                            n.uncertainty,
                        ]
                        for n in hist
                    ]
                ],
                dtype=torch.float32,
            )
            with torch.no_grad():
                k = self.assoc_kernel.forward_cross(
                    x_new,
                    x_old,
                    t_a=t_new.unsqueeze(-1),
                    t_b=t_old.unsqueeze(-1),
                    return_complex=False,
                )  # (1, 1, Nhist)
            strengths = k[0, 0].detach().cpu()
            for i, old in enumerate(hist):
                strength = float(strengths[i].item())
                dt = (node.time_stamp - old.time_stamp) / self.sample_rate_hz
                # Directed: earlier -> later when physically causal (dt>0).
                if dt >= 0:
                    self.edges.append(
                        CausalEdge(
                            source=old.node_id,
                            target=node.node_id,
                            strength=max(0.0, strength),
                            travel_time_diff=float(dt),
                        )
                    )
                else:
                    self.edges.append(
                        CausalEdge(
                            source=node.node_id,
                            target=old.node_id,
                            strength=max(0.0, strength),
                            travel_time_diff=float(-dt),
                        )
                    )
        self.nodes[node.node_id] = node
        self.prune()

    def prune(self) -> list[int]:
        """Forget low-confidence nodes. Returns removed ids."""
        remove = [
            nid
            for nid, n in self.nodes.items()
            if n.confidence < self.prune_confidence
        ]
        if not remove:
            return []
        for nid in remove:
            del self.nodes[nid]
        self.edges = [
            e
            for e in self.edges
            if e.source not in remove and e.target not in remove
        ]
        return remove

    def predict_next_arrival(
        self,
        *,
        current_node_id: Optional[int] = None,
        seq_len: int = 6000,
    ) -> Optional[tuple[int, float]]:
        """Predict the most likely next wavefront sample index.

        Returns (focus_index, score) or None.
        """
        if not self.nodes:
            return None
        if current_node_id is None:
            current = max(self.nodes.values(), key=lambda n: n.time_stamp)
        else:
            current = self.nodes.get(current_node_id)
            if current is None:
                return None

        # Prefer outgoing edges; otherwise extrapolate P->S gap / mean travel.
        outgoing = [e for e in self.edges if e.source == current.node_id]
        if outgoing:
            best = max(outgoing, key=lambda e: e.strength)
            target = self.nodes.get(best.target)
            if target is not None:
                # Project one more hop with same travel time.
                nxt = int(target.time_stamp + best.travel_time_diff * self.sample_rate_hz)
                nxt = max(0, min(seq_len - 1, nxt))
                return nxt, float(best.strength)

        # Heuristic: S after P by ~ local (s-p), else +4s wavefront step.
        p = current.phase_picks.get("p", current.time_stamp)
        s = current.phase_picks.get("s", current.time_stamp)
        gap = max(0.5, (s - p) / self.sample_rate_hz)
        nxt = int(current.time_stamp + gap * self.sample_rate_hz)
        nxt = max(0, min(seq_len - 1, nxt))
        return nxt, float(current.confidence)

    def trajectory(self) -> list[dict]:
        """Serializable gaze trajectory for visualization."""
        nodes = sorted(self.nodes.values(), key=lambda n: n.time_stamp)
        return [
            {
                "node_id": n.node_id,
                "time_stamp": n.time_stamp,
                "window_start": n.window_start,
                "window_end": n.window_end,
                "window_size": n.window_size,
                "phase_picks": n.phase_picks,
                "confidence": n.confidence,
                "uncertainty": n.uncertainty,
                "snr": n.snr,
            }
            for n in nodes
        ]

    def edge_list(self) -> list[dict]:
        return [
            {
                "source": e.source,
                "target": e.target,
                "strength": e.strength,
                "travel_time_diff": e.travel_time_diff,
            }
            for e in self.edges
        ]


class CausalMemory(nn.Module):
    """Thin nn.Module wrapper holding a CausalGraph (+ optional learnable assoc)."""

    def __init__(
        self,
        *,
        prune_confidence: float = 0.05,
        sample_rate_hz: float = 100.0,
    ):
        super().__init__()
        self.graph = CausalGraph(
            prune_confidence=prune_confidence,
            sample_rate_hz=sample_rate_hz,
        )

    def reset(self) -> None:
        self.graph.clear()

    def remember(self, fov: FoveaOutput, batch_index: int = 0) -> CausalNode:
        return self.graph.add_from_fovea(fov, batch_index=batch_index)


class Scheduler(nn.Module):
    """Decide next gaze center + window size from heatmap + causal graph."""

    def __init__(
        self,
        *,
        seq_len: int = 6000,
        sample_rate_hz: float = 100.0,
        high_heat_threshold: float = 0.7,
        cover_radius: int = 200,
        default_task: TaskType = "ps_pick",
        window_selector: Optional[WindowSelector] = None,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.sample_rate_hz = float(sample_rate_hz)
        self.high_heat_threshold = float(high_heat_threshold)
        self.cover_radius = int(cover_radius)
        self.default_task = default_task
        self.window_selector = window_selector or WindowSelector(
            sample_rate_hz=sample_rate_hz
        )
        # Optional learnable scoring head for behavior cloning (stage-1).
        self.policy_mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def coverage_mask(
        self,
        history: Sequence[CausalNode] | Sequence[SchedulerDecision],
        device: torch.device,
    ) -> torch.Tensor:
        cover = torch.zeros(self.seq_len, device=device, dtype=torch.float32)
        for h in history:
            if isinstance(h, CausalNode):
                c = h.time_stamp
                r = max(self.cover_radius, h.window_size // 2)
            else:
                c = h.focus_index
                r = max(self.cover_radius, h.window_size // 2)
            lo = max(0, c - r)
            hi = min(self.seq_len, c + r)
            cover[lo:hi] = 1.0
        return cover

    def _unverified_hotspots(
        self,
        heatmap: torch.Tensor,
        cover: torch.Tensor,
        candidates: Optional[list[CandidateRegion]] = None,
    ) -> list[tuple[int, float, str]]:
        # Returns list of (focus, score, reason)
        hotspots: list[tuple[int, float, str]] = []
        uncovered = heatmap * (1.0 - cover)
        if candidates:
            for reg in candidates:
                mid = (reg.start + reg.end) // 2
                if cover[mid] > 0.5:
                    continue
                if reg.confidence >= self.high_heat_threshold:
                    hotspots.append((mid, float(reg.confidence), "unverified_hotspot"))
        if not hotspots:
            # Fallback: argmax on uncovered heatmap above threshold.
            score, idx = uncovered.max(dim=0)
            if float(score) >= self.high_heat_threshold:
                hotspots.append((int(idx.item()), float(score), "unverified_hotspot"))
        hotspots.sort(key=lambda x: x[1], reverse=True)
        return hotspots

    def forward(
        self,
        heatmap: torch.Tensor,
        graph: CausalGraph,
        *,
        candidates: Optional[list[CandidateRegion]] = None,
        history: Optional[Sequence[CausalNode]] = None,
        task: Optional[TaskType] = None,
        snr_hint: float = 10.0,
    ) -> SchedulerDecision:
        """
        Args:
            heatmap: (T,) or (1,T) anomaly scores
            graph: current causal graph
            candidates: optional regions from PeripheralScanner
            history: optional explicit history (defaults to graph nodes)
        """
        if heatmap.dim() == 2:
            heatmap = heatmap[0]
        if heatmap.numel() != self.seq_len:
            raise ValueError(f"Expected heatmap length {self.seq_len}, got {heatmap.numel()}")
        task = task or self.default_task
        hist_nodes = list(history) if history is not None else list(graph.nodes.values())
        cover = self.coverage_mask(hist_nodes, heatmap.device)

        # Priority 1: unverified high-confidence heatmap regions.
        hotspots = self._unverified_hotspots(heatmap, cover, candidates)
        if hotspots:
            focus, score, reason = hotspots[0]
            unc = hist_nodes[-1].uncertainty if hist_nodes else 0.0
            w = self.window_selector(task=task, snr=snr_hint, uncertainty=unc)
            return SchedulerDecision(focus, w, reason, score)

        # Priority 2: causal wavefront prediction.
        pred = graph.predict_next_arrival(seq_len=self.seq_len)
        if pred is not None:
            focus, score = pred
            if cover[focus] < 0.5:
                unc = hist_nodes[-1].uncertainty if hist_nodes else 0.0
                w = self.window_selector(task=task, snr=snr_hint, uncertainty=unc)
                return SchedulerDecision(
                    focus, w, "causal_prediction", float(score) + 0.5
                )

        # Priority 3: high uncertainty at chain tip -> enlarge and re-gaze nearby.
        if hist_nodes:
            tip = max(hist_nodes, key=lambda n: n.time_stamp)
            if tip.uncertainty >= 1.0:
                focus = min(self.seq_len - 1, tip.time_stamp + tip.window_size // 4)
                w = self.window_selector(
                    task="velocity_inversion" if tip.uncertainty >= 1.5 else task,
                    snr=tip.snr,
                    uncertainty=tip.uncertainty,
                )
                w = max(w, 1200)
                return SchedulerDecision(focus, w, "uncertainty_extend", tip.uncertainty)

        # Priority 4: chronological scan of uncovered regions.
        uncovered = (1.0 - cover) * (heatmap + 0.05)
        focus = int(uncovered.argmax().item())
        # If fully covered, jump to global heatmap peak.
        if float(cover.mean()) > 0.98:
            focus = int(heatmap.argmax().item())
            reason = "global_revisit"
        else:
            reason = "sequential_scan"
        unc = hist_nodes[-1].uncertainty if hist_nodes else 0.0
        w = self.window_selector(task=task, snr=snr_hint, uncertainty=unc)
        return SchedulerDecision(focus, w, reason, float(uncovered[focus]))

    def behavior_cloning_loss(
        self,
        features: torch.Tensor,
        expert_focus_norm: torch.Tensor,
    ) -> torch.Tensor:
        """Stage-1 auxiliary loss: features (B,4) -> predict normalized focus in [0,1]."""
        pred = torch.sigmoid(self.policy_mlp(features).squeeze(-1))
        return F.mse_loss(pred, expert_focus_norm)
