# -*- coding: utf-8 -*-
"""FoveatedEngine: global scan -> local fovea -> causal memory loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hnf.foveated.causal_memory import (
    CausalMemory,
    CausalNode,
    Scheduler,
    SchedulerDecision,
    TaskType,
)
from hnf.foveated.fovea_processor import FoveaOutput, FoveaProcessor
from hnf.foveated.peripheral_scanner import CandidateRegion, PeripheralScanner


@dataclass
class GazeStep:
    step: int
    decision: SchedulerDecision
    fovea: FoveaOutput
    node: CausalNode


@dataclass
class FoveatedEngineOutput:
    heatmap: torch.Tensor  # (B, T)
    candidates: list[list[CandidateRegion]]
    p_logits: torch.Tensor  # (B, T) fused global
    s_logits: torch.Tensor  # (B, T) fused global
    p_prob: torch.Tensor
    s_prob: torch.Tensor
    p_idx: torch.Tensor  # (B,)
    s_idx: torch.Tensor  # (B,)
    gaze_trace: list[list[GazeStep]]  # per-batch steps
    trajectory: list[list[dict]]  # serializable per-batch
    edges: list[list[dict]]
    velocity_model: Optional[dict[str, torch.Tensor]] = None
    n_gazes: torch.Tensor = field(default_factory=lambda: torch.zeros(1))
    coverage: torch.Tensor | None = None  # (B, T)


class FoveatedEngine(nn.Module):
    """智子双中央凹主动感知引擎.

    Loop:
      1) PeripheralScanner -> heatmap / candidates
      2) Scheduler -> next focus + window size
      3) FoveaProcessor -> dense local HNF
      4) CausalMemory -> store node / edges
      5) repeat until max_gazes or coverage complete
      6) fuse local results into global P/S curves
    """

    def __init__(
        self,
        *,
        scanner: Optional[PeripheralScanner] = None,
        fovea: Optional[FoveaProcessor] = None,
        memory: Optional[CausalMemory] = None,
        scheduler: Optional[Scheduler] = None,
        seq_len: int = 6000,
        max_gazes: int = 8,
        coverage_complete_ratio: float = 0.92,
        task: TaskType = "ps_pick",
        fuse_temperature: float = 1.0,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.max_gazes = int(max_gazes)
        self.coverage_complete_ratio = float(coverage_complete_ratio)
        self.task = task
        self.fuse_temperature = float(fuse_temperature)

        self.scanner = scanner or PeripheralScanner(seq_len=self.seq_len)
        self.fovea = fovea or FoveaProcessor(seq_len=self.seq_len)
        sample_rate = float(getattr(self.fovea, "sample_rate_hz", 100.0))
        self.memory = memory or CausalMemory(sample_rate_hz=sample_rate)
        self.scheduler = scheduler or Scheduler(
            seq_len=self.seq_len,
            sample_rate_hz=sample_rate,
            default_task=task,
        )

    def _coverage_ratio(self, nodes: list[CausalNode]) -> float:
        if not nodes:
            return 0.0
        cover = self.scheduler.coverage_mask(nodes, device=torch.device("cpu"))
        return float(cover.mean().item())

    def _fuse_pick_maps(
        self,
        steps: list[GazeStep],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[dict[str, torch.Tensor]]]:
        """Scatter-add local logits into global length-T maps with confidence weights.

        Uncovered background stays at -4 (sigmoid≈0.02) so argmax does not collapse
        to t=0 when logits are empty/near-zero.
        """
        fill = -4.0
        p = torch.full((self.seq_len,), fill, device=device, dtype=dtype)
        s = torch.full((self.seq_len,), fill, device=device, dtype=dtype)
        wsum = torch.zeros(self.seq_len, device=device, dtype=dtype)
        best_vel: Optional[dict[str, torch.Tensor]] = None
        best_conf = -1.0

        for step in steps:
            fov = step.fovea
            conf = float(fov.confidence[0].detach())
            weight = max(1e-3, conf) ** self.fuse_temperature
            sl = slice(fov.window_start, fov.window_end)
            p[sl] = p[sl] + weight * fov.p_logits[0]
            s[sl] = s[sl] + weight * fov.s_logits[0]
            wsum[sl] = wsum[sl] + weight
            # Also stamp global peaks (shift_downsample may place picks outside aperture).
            pi = int(fov.p_idx_global[0].item())
            si = int(fov.s_idx_global[0].item())
            extras = fov.extras or {}
            p_peak = float(extras["p_peak"][0].detach()) if "p_peak" in extras else float(
                torch.sigmoid(fov.p_logits[0]).amax()
            )
            s_peak = float(extras["s_peak"][0].detach()) if "s_peak" in extras else float(
                torch.sigmoid(fov.s_logits[0]).amax()
            )
            p_logit = float(torch.logit(torch.tensor(max(1e-4, min(1 - 1e-4, p_peak)))))
            s_logit = float(torch.logit(torch.tensor(max(1e-4, min(1 - 1e-4, s_peak)))))
            if 0 <= pi < self.seq_len:
                p[pi] = max(float(p[pi]), p_logit)
            if 0 <= si < self.seq_len:
                s[si] = max(float(s[si]), s_logit)
            if fov.velocity_model is not None and conf > best_conf:
                best_conf = conf
                best_vel = {k: v[:1].detach() for k, v in fov.velocity_model.items()}

        covered = wsum > 0
        p = torch.where(covered, p / wsum.clamp_min(1e-3), p)
        s = torch.where(covered, s / wsum.clamp_min(1e-3), s)
        return p, s, best_vel

    def forward(
        self,
        waveform: torch.Tensor,
        *,
        max_gazes: Optional[int] = None,
        task: Optional[TaskType] = None,
        reset_memory: bool = True,
    ) -> FoveatedEngineOutput:
        """
        Args:
            waveform: (B, 3, T) preferred; also accepts (B, T, 3)
        """
        if waveform.dim() != 3:
            raise ValueError(f"Expected waveform (B,3,T) or (B,T,3), got {tuple(waveform.shape)}")
        if waveform.size(1) == 3:
            wave_b3t = waveform
        elif waveform.size(-1) == 3:
            wave_b3t = waveform.transpose(1, 2).contiguous()
        else:
            raise ValueError(f"Channel dimension must be 3, got {tuple(waveform.shape)}")

        b, _c, t = wave_b3t.shape
        if t != self.seq_len:
            raise ValueError(f"Expected T={self.seq_len}, got T={t}")

        device = wave_b3t.device
        dtype = wave_b3t.dtype
        n_gazes = int(max_gazes or self.max_gazes)
        task = task or self.task

        heatmap, candidates = self.scanner(wave_b3t, return_candidates=True)
        assert candidates is not None

        all_steps: list[list[GazeStep]] = [[] for _ in range(b)]
        trajectories: list[list[dict]] = [[] for _ in range(b)]
        edges_out: list[list[dict]] = [[] for _ in range(b)]
        p_all = torch.zeros(b, self.seq_len, device=device, dtype=dtype)
        s_all = torch.zeros(b, self.seq_len, device=device, dtype=dtype)
        cover_all = torch.zeros(b, self.seq_len, device=device, dtype=dtype)
        n_gaze_t = torch.zeros(b, device=device, dtype=torch.long)
        velocity_acc: Optional[dict[str, torch.Tensor]] = None

        for bi in range(b):
            if reset_memory:
                self.memory.reset()
            wave_i = wave_b3t[bi : bi + 1]
            heat_i = heatmap[bi]
            cand_i = candidates[bi]
            steps: list[GazeStep] = []

            for gi in range(n_gazes):
                nodes = list(self.memory.graph.nodes.values())
                if nodes and self._coverage_ratio(nodes) >= self.coverage_complete_ratio:
                    # Still allow early stop only if hotspots are exhausted.
                    cover = self.scheduler.coverage_mask(nodes, device=heat_i.device)
                    uncovered_peak = float((heat_i * (1.0 - cover)).amax().item())
                    if uncovered_peak < self.scheduler.high_heat_threshold:
                        break

                snr_hint = float(steps[-1].fovea.snr[0].item()) if steps else 10.0
                decision = self.scheduler(
                    heat_i,
                    self.memory.graph,
                    candidates=cand_i,
                    history=nodes,
                    task=task,
                    snr_hint=snr_hint,
                )
                # Stop if scheduler revisits an already-covered focus (no progress).
                if steps:
                    cover = self.scheduler.coverage_mask(
                        [s.node for s in steps], device=heat_i.device
                    )
                    if float(cover[decision.focus_index]) >= 0.99:
                        # Allow one more only for uncertainty_extend; otherwise halt.
                        if decision.reason not in {"uncertainty_extend", "causal_prediction"}:
                            break
                        # Also halt if we already did many revisits.
                        if gi >= 2 and all(
                            abs(s.decision.focus_index - decision.focus_index)
                            < self.scheduler.cover_radius
                            for s in steps[-2:]
                        ):
                            break

                fov = self.fovea(
                    wave_i,
                    focus_index=decision.focus_index,
                    window_size=decision.window_size,
                )
                node = self.memory.remember(fov, batch_index=0)
                step = GazeStep(step=gi, decision=decision, fovea=fov, node=node)
                steps.append(step)

            p_i, s_i, vel_i = self._fuse_pick_maps(steps, device=device, dtype=dtype)
            p_all[bi] = p_i
            s_all[bi] = s_i
            n_gaze_t[bi] = len(steps)
            cover_all[bi] = self.scheduler.coverage_mask(
                [s.node for s in steps], device=device
            ).to(dtype)
            all_steps[bi] = steps
            trajectories[bi] = self.memory.graph.trajectory()
            edges_out[bi] = self.memory.graph.edge_list()
            if vel_i is not None:
                if velocity_acc is None:
                    velocity_acc = {k: v.new_zeros((b,) + v.shape[1:]) for k, v in vel_i.items()}
                for k, v in vel_i.items():
                    velocity_acc[k][bi] = v[0]

        p_prob = torch.sigmoid(p_all)
        s_prob = torch.sigmoid(s_all)
        return FoveatedEngineOutput(
            heatmap=heatmap,
            candidates=candidates,
            p_logits=p_all,
            s_logits=s_all,
            p_prob=p_prob,
            s_prob=s_prob,
            p_idx=p_prob.argmax(dim=-1),
            s_idx=s_prob.argmax(dim=-1),
            gaze_trace=all_steps,
            trajectory=trajectories,
            edges=edges_out,
            velocity_model=velocity_acc,
            n_gazes=n_gaze_t,
            coverage=cover_all,
        )

    def gaze_efficiency_penalty(self, n_gazes: torch.Tensor, max_gazes: Optional[int] = None) -> torch.Tensor:
        """Stage-2 regularizer: encourage fewer gazes. Mean(n / max)."""
        m = float(max_gazes or self.max_gazes)
        return (n_gazes.float() / m).mean()

    def causal_consistency_loss(self, edges: list[list[dict]]) -> torch.Tensor:
        """Encourage temporally causal strong edges (source earlier than target).

        Soft surrogate: mean(relu(-direction)) on serialized edges is always 0
        by construction; instead penalize weak mean strength when edges exist.
        """
        strengths = []
        for sample_edges in edges:
            for e in sample_edges:
                strengths.append(float(e["strength"]))
        if not strengths:
            return torch.zeros((), device=next(self.parameters(), torch.tensor(0.0)).device)
        # Maximize strength => minimize negative mean strength.
        return -torch.tensor(sum(strengths) / len(strengths), dtype=torch.float32)


def visualize_trajectory_ascii(
    trajectory: list[dict],
    seq_len: int = 6000,
    width: int = 80,
) -> str:
    """Tiny text viz of gaze centers for debugging."""
    canvas = ["."] * width
    for n in trajectory:
        x = int(n["time_stamp"] / max(seq_len - 1, 1) * (width - 1))
        canvas[x] = "o"
    if trajectory:
        for i, n in enumerate(trajectory):
            x = int(n["time_stamp"] / max(seq_len - 1, 1) * (width - 1))
            canvas[x] = str(i % 10)
    return "".join(canvas)
