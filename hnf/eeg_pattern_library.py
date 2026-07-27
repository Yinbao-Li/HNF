# -*- coding: utf-8 -*-
"""EEG clinical pattern library: induce → route → policy → (optional) EMA update.

Mirrors the STEAD two-tier idea, but for AD/FTD EEG subject routing:

1. **Induce** prototypes from train-subject HNF / band / (optional) region features
   plus soft class scores from a frozen classifier.
2. **Route** a held-out subject to the nearest prototype.
3. **Apply policy** — trust head / AD↔FTD second look / abstain if uncertain.
4. **Update** (opt-in) prototype centres only with *same-manifold* features
   (never mix train-agg stats with a drifted feature space).

Router stays on cheap scalar summaries. FDR / causal-style discovery stays outside.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from hnf.pattern_library import _kmeans, _zscore_fit, features_to_vector


# Cheap subject-level routing features (fill missing keys with 0 at vectorize time).
EEG_ROUTER_FEATURES: tuple[str, ...] = (
    "rho_mean",
    "rho_std",
    "rho_p90",
    "rho_cv",
    "bp_delta",
    "bp_theta",
    "bp_alpha",
    "bp_beta",
    "theta_alpha_ratio",
    "hnf_theta_energy",
    "hnf_alpha_energy",
    "hnf_theta_alpha_ratio",
    "hnf_delta_energy",
    "region_ft_contrast",
    "region_pf_contrast",
    "prob_hc",
    "prob_ftd",
    "prob_ad",
    "max_prob",
    "ad_ftd_score",
    "pred_entropy",
)


@dataclass
class EEGPatternPolicy:
    """What to do when this clinical pattern fires."""

    name: str = "trust_head"
    # If True: among AD/FTD-ish cases, re-decide with P(AD)/(P(AD)+P(FTD)).
    ad_ftd_second_look: bool = False
    # Keep HC when P(HC) >= hc_keep_margin * max(P_FTD, P_AD).
    hc_keep_margin: float = 1.15
    # Only second-look when |P_AD - P_FTD| is below this (otherwise trust argmax).
    confusion_margin: float = 0.20
    # Require P_FTD + P_AD at least this before second-look overrides HC.
    min_disease_mass: float = 0.55
    # Soft abstention on low confidence / OOD distance.
    abstain: bool = False
    min_confidence: float = 0.50
    # Majority clinical label in the inducing cluster (HC/FTD/AD/mixed).
    majority_label: str = "mixed"


@dataclass
class EEGPatternPrototype:
    pattern_id: int
    name: str
    center: list[float]
    count: int = 0
    n_hc: int = 0
    n_ftd: int = 0
    n_ad: int = 0
    mean_max_prob: float = 0.0
    mean_ad_ftd_score: float = 0.5
    train_accuracy: float = 0.0
    policy: EEGPatternPolicy = field(default_factory=EEGPatternPolicy)
    n_confirm: int = 0
    n_reject: int = 0


@dataclass
class EEGRouteDecision:
    pattern_id: int
    name: str
    distance: float
    policy: EEGPatternPolicy
    features: dict[str, float]


def subject_router_features(subj: dict[str, Any]) -> dict[str, float]:
    """Build a finite router feature dict from an aggregated subject row."""
    ph = float(subj.get("prob_hc", 0.0))
    pf = float(subj.get("prob_ftd", 0.0))
    pa = float(subj.get("prob_ad", 0.0))
    probs = np.asarray([ph, pf, pa], dtype=np.float64)
    probs = np.clip(probs, 1e-8, 1.0)
    probs = probs / probs.sum()
    ent = float(-(probs * np.log(probs)).sum())
    ad_ftd = float(pa / max(pa + pf, 1e-8))
    out: dict[str, float] = {}
    for k in EEG_ROUTER_FEATURES:
        if k == "prob_hc":
            out[k] = float(probs[0])
        elif k == "prob_ftd":
            out[k] = float(probs[1])
        elif k == "prob_ad":
            out[k] = float(probs[2])
        elif k == "max_prob":
            out[k] = float(probs.max())
        elif k == "ad_ftd_score":
            out[k] = ad_ftd
        elif k == "pred_entropy":
            out[k] = ent
        else:
            v = subj.get(k, 0.0)
            try:
                fv = float(v)
            except (TypeError, ValueError):
                fv = 0.0
            out[k] = fv if np.isfinite(fv) else 0.0
    return out


def _policy_from_cluster_stats(
    *,
    n_hc: int,
    n_ftd: int,
    n_ad: int,
    mean_max_prob: float,
    train_acc: float,
) -> EEGPatternPolicy:
    n = max(n_hc + n_ftd + n_ad, 1)
    frac = {
        "HC": n_hc / n,
        "FTD": n_ftd / n,
        "AD": n_ad / n,
    }
    maj = max(frac, key=frac.get)
    disease = (n_ftd + n_ad) / n
    # Tight AD↔FTD pocket: both disease classes well represented, HC rare, n≥4.
    balanced_disease = (
        n >= 4
        and disease >= 0.75
        and frac["HC"] <= 0.20
        and frac["FTD"] >= 0.25
        and frac["AD"] >= 0.25
    )
    if balanced_disease:
        return EEGPatternPolicy(
            name="ad_ftd_second_look",
            ad_ftd_second_look=True,
            abstain=False,
            hc_keep_margin=1.15,
            confusion_margin=0.20,
            min_disease_mass=0.55,
            majority_label=maj if frac[maj] >= 0.45 else "mixed",
        )
    # Mixed / impure / soft-max weak → abstain gate (distance OOD also abstains).
    if frac[maj] < 0.50 or (mean_max_prob < 0.70 and train_acc < 0.85) or n <= 2:
        return EEGPatternPolicy(
            name="abstain_uncertain",
            ad_ftd_second_look=False,
            abstain=True,
            min_confidence=0.60,
            majority_label=maj if frac[maj] >= 0.4 else "mixed",
        )
    return EEGPatternPolicy(
        name=f"trust_head_{maj.lower()}",
        ad_ftd_second_look=False,
        abstain=False,
        majority_label=maj,
    )


def apply_eeg_policy(
    probs: np.ndarray,
    policy: EEGPatternPolicy,
    *,
    distance: float = 0.0,
    max_route_distance: Optional[float] = None,
) -> tuple[Optional[int], bool, str]:
    """Return ``(pred_or_None, abstained, reason)``.

    Second-look only overrides when the head is *actually confused* between AD
    and FTD (small logit gap) and disease mass is high — otherwise trust argmax.
    """
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    if p.size != 3:
        raise ValueError(f"Expected 3 probs, got {p.shape}")
    p = np.clip(p, 1e-8, None)
    p = p / p.sum()
    max_p = float(p.max())
    baseline = int(p.argmax())

    if max_route_distance is not None and distance > float(max_route_distance):
        return None, True, "ood_distance"
    if policy.abstain and max_p < float(policy.min_confidence):
        return None, True, "low_confidence"

    if policy.ad_ftd_second_look:
        disease_mass = float(p[1] + p[2])
        gap = abs(float(p[2] - p[1]))
        # Clear HC → keep.
        if p[0] >= float(policy.hc_keep_margin) * max(p[1], p[2]):
            return 0, False, "second_look_keep_hc"
        # Not enough disease mass or not confused → trust head.
        if disease_mass < float(policy.min_disease_mass) or gap > float(policy.confusion_margin):
            return baseline, False, "second_look_defer_head"
        # Ambiguous AD↔FTD → force disease binary; if still too close, abstain.
        if gap < 0.05:
            return None, True, "second_look_too_close"
        return (2 if p[2] >= p[1] else 1), False, "second_look_force"

    return baseline, False, "trust_head"


class EEGPatternLibrary:
    """Prototype bank + clinical router + optional EMA feedback."""

    def __init__(
        self,
        prototypes: list[EEGPatternPrototype],
        *,
        feature_names: Sequence[str] = EEG_ROUTER_FEATURES,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        checkpoint: str = "",
        k: int = 0,
        seed: int = 0,
        max_route_distance: Optional[float] = None,
    ):
        self.prototypes = list(prototypes)
        self.feature_names = tuple(feature_names)
        d = len(self.feature_names)
        self.mean = np.zeros(d) if mean is None else np.asarray(mean, dtype=np.float64)
        self.std = np.ones(d) if std is None else np.asarray(std, dtype=np.float64)
        self.checkpoint = checkpoint
        self.k = int(k) if k else len(self.prototypes)
        self.seed = int(seed)
        self.max_route_distance = (
            None if max_route_distance is None else float(max_route_distance)
        )

    @classmethod
    def build_from_subjects(
        cls,
        subjects: list[dict[str, Any]],
        *,
        k: int = 6,
        seed: int = 0,
        feature_names: Sequence[str] = EEG_ROUTER_FEATURES,
        checkpoint: str = "",
    ) -> "EEGPatternLibrary":
        if not subjects:
            raise ValueError("Need at least one subject to induce a library")
        feat_dicts = [subject_router_features(s) for s in subjects]
        feats = np.stack(
            [features_to_vector(f, feature_names) for f in feat_dicts], axis=0
        )
        labels_y = np.asarray([int(s["label"]) for s in subjects], dtype=np.int64)
        head_pred = np.asarray(
            [
                int(np.argmax([s.get("prob_hc", 0), s.get("prob_ftd", 0), s.get("prob_ad", 0)]))
                for s in subjects
            ],
            dtype=np.int64,
        )

        mu, sd = _zscore_fit(feats)
        z = (feats - mu) / sd
        cluster_ids, centers_z = _kmeans(z, k=k, seed=seed)
        centers = centers_z * sd + mu

        prototypes: list[EEGPatternPrototype] = []
        for j in range(centers.shape[0]):
            mask = cluster_ids == j
            n = int(mask.sum())
            if n == 0:
                continue
            sub_y = labels_y[mask]
            n_hc = int((sub_y == 0).sum())
            n_ftd = int((sub_y == 1).sum())
            n_ad = int((sub_y == 2).sum())
            mean_max = float(feats[mask, list(feature_names).index("max_prob")].mean())
            mean_af = float(feats[mask, list(feature_names).index("ad_ftd_score")].mean())
            train_acc = float((head_pred[mask] == sub_y).mean())
            policy = _policy_from_cluster_stats(
                n_hc=n_hc,
                n_ftd=n_ftd,
                n_ad=n_ad,
                mean_max_prob=mean_max,
                train_acc=train_acc,
            )
            name = f"P{j}_{policy.name}"
            prototypes.append(
                EEGPatternPrototype(
                    pattern_id=j,
                    name=name,
                    center=centers[j].tolist(),
                    count=n,
                    n_hc=n_hc,
                    n_ftd=n_ftd,
                    n_ad=n_ad,
                    mean_max_prob=mean_max,
                    mean_ad_ftd_score=mean_af,
                    train_accuracy=train_acc,
                    policy=policy,
                )
            )
        return cls(
            prototypes,
            feature_names=feature_names,
            mean=mu,
            std=sd,
            checkpoint=checkpoint,
            k=k,
            seed=seed,
        )

    def normalize(self, vec: np.ndarray) -> np.ndarray:
        return (vec - self.mean) / self.std

    def route(self, feat: dict[str, float]) -> EEGRouteDecision:
        if not self.prototypes:
            pol = EEGPatternPolicy(name="trust_head")
            return EEGRouteDecision(-1, "empty", 1e9, pol, feat)
        v = features_to_vector(feat, self.feature_names)
        z = self.normalize(v)
        best_i = 0
        best_d = 1e18
        for i, p in enumerate(self.prototypes):
            c = self.normalize(np.asarray(p.center, dtype=np.float64))
            d = float(np.linalg.norm(z - c))
            if d < best_d:
                best_d = d
                best_i = i
        proto = self.prototypes[best_i]
        return EEGRouteDecision(proto.pattern_id, proto.name, best_d, proto.policy, feat)

    def route_subject(self, subj: dict[str, Any]) -> EEGRouteDecision:
        return self.route(subject_router_features(subj))

    def route_distances(self, subjects: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray(
            [self.route_subject(s).distance for s in subjects], dtype=np.float64
        )

    def calibrate_distance_gate(
        self,
        val_subjects: list[dict[str, Any]],
        *,
        min_coverage: float = 0.70,
        percentiles: tuple[float, ...] = (0.70, 0.80, 0.85, 0.90, 0.95, 0.99),
    ) -> dict[str, Any]:
        """Pick ``max_route_distance`` on val: max kept-acc with coverage ≥ floor.

        Falls back to the 90th percentile of val distances if nothing qualifies.
        """
        if not val_subjects:
            self.max_route_distance = None
            return {"chosen": None, "reason": "empty_val"}
        dists = self.route_distances(val_subjects)
        candidates = sorted({float(np.quantile(dists, q)) for q in percentiles})
        # Also try "no gate"
        best = {
            "max_route_distance": None,
            "coverage": 1.0,
            "kept_acc": float("nan"),
            "score": -1.0,
        }
        # Evaluate no-gate first
        self.max_route_distance = None
        m0 = evaluate_routed_subjects(self, val_subjects)
        best = {
            "max_route_distance": None,
            "coverage": m0["coverage"],
            "kept_acc": m0["routed_kept_subject_acc"],
            "fill_acc": m0["routed_fill_subject_acc"],
            "score": float(m0["routed_fill_subject_acc"]),
        }
        rows = [{"percentile_like": "none", **best}]
        for thr in candidates:
            self.max_route_distance = thr
            m = evaluate_routed_subjects(self, val_subjects)
            kept = m["routed_kept_subject_acc"]
            cov = m["coverage"]
            # Prefer higher kept-acc under coverage constraint; break ties by fill.
            score = (
                (0.0 if cov < min_coverage or not np.isfinite(kept) else float(kept))
                + 0.05 * float(m["routed_fill_subject_acc"])
            )
            row = {
                "max_route_distance": thr,
                "coverage": cov,
                "kept_acc": kept,
                "fill_acc": m["routed_fill_subject_acc"],
                "score": score,
            }
            rows.append(row)
            if score > best["score"] and cov >= min_coverage:
                best = row
        # If no candidate beat no-gate under constraint, keep a mild OOD gate.
        if best["max_route_distance"] is None and len(candidates) > 0:
            self.max_route_distance = float(np.quantile(dists, 0.95))
            reason = "fallback_p95"
        else:
            self.max_route_distance = best["max_route_distance"]
            reason = "val_max_kept_acc"
        return {
            "chosen": self.max_route_distance,
            "reason": reason,
            "min_coverage": min_coverage,
            "best": best,
            "grid": rows,
        }

    def predict_subject(self, subj: dict[str, Any]) -> dict[str, Any]:
        """Route + apply policy; returns pred / abstain / baseline fields."""
        probs = np.asarray(
            [subj.get("prob_hc", 0.0), subj.get("prob_ftd", 0.0), subj.get("prob_ad", 0.0)],
            dtype=np.float64,
        )
        baseline = int(probs.argmax())
        decision = self.route_subject(subj)
        pred, abstained, reason = apply_eeg_policy(
            probs,
            decision.policy,
            distance=decision.distance,
            max_route_distance=self.max_route_distance,
        )
        return {
            "subject_id": subj.get("subject_id"),
            "label": int(subj.get("label", -1)),
            "baseline_pred": baseline,
            "routed_pred": pred,
            "abstained": bool(abstained),
            "abstain_reason": reason if abstained else "",
            "decision_reason": reason,
            "pattern_id": decision.pattern_id,
            "pattern_name": decision.name,
            "distance": decision.distance,
            "policy": decision.policy.name,
            "prob_hc": float(probs[0]),
            "prob_ftd": float(probs[1]),
            "prob_ad": float(probs[2]),
        }

    def update_from_outcome(
        self,
        pattern_id: int,
        feat: dict[str, float],
        *,
        confirmed: bool,
        ema: float = 0.05,
        update_center: bool = False,
    ) -> None:
        """Record outcome; centre EMA only when ``update_center`` and same manifold."""
        proto = next((p for p in self.prototypes if p.pattern_id == pattern_id), None)
        if proto is None:
            return
        if confirmed:
            proto.n_confirm += 1
            if update_center:
                v = features_to_vector(feat, self.feature_names)
                c = np.asarray(proto.center, dtype=np.float64)
                proto.center = ((1.0 - ema) * c + ema * v).tolist()
        else:
            proto.n_reject += 1

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": "eeg_pattern_library",
            "checkpoint": self.checkpoint,
            "k": self.k,
            "seed": self.seed,
            "max_route_distance": self.max_route_distance,
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "prototypes": [
                {
                    **{k: v for k, v in asdict(p).items() if k != "policy"},
                    "policy": asdict(p.policy),
                }
                for p in self.prototypes
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "EEGPatternLibrary":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        protos = []
        for row in data["prototypes"]:
            pol_raw = row.pop("policy")
            allowed = {f.name for f in fields(EEGPatternPolicy)}
            pol = EEGPatternPolicy(**{k: v for k, v in pol_raw.items() if k in allowed})
            # Drop unknown prototype keys from older dumps.
            proto_allowed = {f.name for f in fields(EEGPatternPrototype)}
            row = {k: v for k, v in row.items() if k in proto_allowed and k != "policy"}
            protos.append(EEGPatternPrototype(policy=pol, **row))
        return cls(
            protos,
            feature_names=data.get("feature_names", EEG_ROUTER_FEATURES),
            mean=np.asarray(data["mean"], dtype=np.float64),
            std=np.asarray(data["std"], dtype=np.float64),
            checkpoint=str(data.get("checkpoint", "")),
            k=int(data.get("k", len(protos))),
            seed=int(data.get("seed", 0)),
            max_route_distance=data.get("max_route_distance"),
        )

    def summary(self) -> list[dict[str, Any]]:
        rows = []
        for p in self.prototypes:
            rows.append(
                {
                    "id": p.pattern_id,
                    "name": p.name,
                    "count": p.count,
                    "HC": p.n_hc,
                    "FTD": p.n_ftd,
                    "AD": p.n_ad,
                    "train_acc": round(p.train_accuracy, 3),
                    "mean_max_prob": round(p.mean_max_prob, 3),
                    "policy": p.policy.name,
                    "confirm": p.n_confirm,
                    "reject": p.n_reject,
                }
            )
        return rows


def evaluate_routed_subjects(
    lib: EEGPatternLibrary,
    subjects: list[dict[str, Any]],
    *,
    online_update: bool = False,
    update_center: bool = False,
) -> dict[str, Any]:
    """Subject-level baseline vs routed metrics (+ optional online counter update)."""
    rows = []
    for s in subjects:
        r = lib.predict_subject(s)
        rows.append(r)
        if online_update and r["pattern_id"] >= 0:
            # Counters only: confirmed if non-abstain and correct.
            if r["abstained"] or r["routed_pred"] is None:
                lib.update_from_outcome(
                    r["pattern_id"],
                    subject_router_features(s),
                    confirmed=False,
                    update_center=False,
                )
            else:
                ok = int(r["routed_pred"]) == int(r["label"])
                lib.update_from_outcome(
                    r["pattern_id"],
                    subject_router_features(s),
                    confirmed=bool(ok),
                    update_center=update_center,
                )

    y = np.asarray([r["label"] for r in rows], dtype=np.int64)
    base = np.asarray([r["baseline_pred"] for r in rows], dtype=np.int64)
    routed = []
    kept = []
    for r in rows:
        if r["abstained"] or r["routed_pred"] is None:
            continue
        kept.append(r)
        routed.append(int(r["routed_pred"]))
    routed_arr = np.asarray(routed, dtype=np.int64) if routed else np.zeros(0, dtype=np.int64)
    y_kept = np.asarray([r["label"] for r in kept], dtype=np.int64) if kept else np.zeros(0, dtype=np.int64)

    def _acc(yt, yp) -> float:
        if len(yt) == 0:
            return float("nan")
        return float((yt == yp).mean())

    def _adftd_acc(yt, yp) -> float:
        m = (yt == 1) | (yt == 2)
        if not np.any(m):
            return float("nan")
        return float((yt[m] == yp[m]).mean())

    coverage = float(len(kept) / max(len(rows), 1))
    filled = np.asarray(
        [
            (r["baseline_pred"] if r["abstained"] or r["routed_pred"] is None else int(r["routed_pred"]))
            for r in rows
        ],
        dtype=np.int64,
    )
    reason_counts: dict[str, int] = {}
    for r in rows:
        key = str(r.get("decision_reason") or "")
        reason_counts[key] = reason_counts.get(key, 0) + 1
    return {
        "n_subjects": len(rows),
        "coverage": coverage,
        "n_abstain": int(sum(1 for r in rows if r["abstained"])),
        "baseline_subject_acc": _acc(y, base),
        "baseline_ad_ftd_acc": _adftd_acc(y, base),
        "routed_fill_subject_acc": _acc(y, filled),
        "routed_fill_ad_ftd_acc": _adftd_acc(y, filled),
        "routed_kept_subject_acc": _acc(y_kept, routed_arr),
        "routed_kept_ad_ftd_acc": _adftd_acc(y_kept, routed_arr),
        "max_route_distance": lib.max_route_distance,
        "policy_counts": {
            k: int(sum(1 for r in rows if r["policy"] == k))
            for k in sorted({r["policy"] for r in rows})
        },
        "reason_counts": reason_counts,
        "pattern_counts": {
            str(k): int(sum(1 for r in rows if r["pattern_id"] == k))
            for k in sorted({r["pattern_id"] for r in rows})
        },
        "rows": rows,
    }
