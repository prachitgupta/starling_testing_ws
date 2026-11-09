"""Trajectory residual calibration and Gibbs--Candes ACI (2021, update (2)).

Empirical quantiles do not claim finite-sample split-conformal coverage.
ACI controls average score miscoverage across evaluated trajectories, not flight safety.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

NO_QUANTILE = "no valid quantile found"


def empirical_quantile(scores, delta=0.20):
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be strictly between 0 and 1")
    ordered = sorted(float(s) for s in scores if math.isfinite(float(s)) and float(s) >= 0)
    if not ordered:
        return None
    return ordered[max(1, math.ceil((1.0 - delta) * len(ordered))) - 1]


def resolve_data_path(value):
    requested = Path(value).expanduser()
    candidates = [requested, Path(__file__).resolve().parents[1] / requested,
                  Path(__file__).resolve().parents[2] / requested]
    try:
        from ament_index_python.packages import get_package_share_directory
        candidates.append(Path(get_package_share_directory("llm_vision_planner")) / requested)
    except (ImportError, LookupError):
        pass
    return next((p.resolve() for p in candidates if p.is_file()), requested)


def read_scores(path, delimiter=","):
    """Only real Transformer/ideal-dynamics rows can calibrate live tubes."""
    path = resolve_data_path(path)
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            scores = []
            for row in csv.DictReader(stream, delimiter=delimiter):
                if (row.get("placeholder", "").lower() != "false"
                        or row.get("planner_provider") != "transformer"
                        or row.get("dynamics_model") != "ideal_planar_double_integrator"):
                    continue
                try:
                    score = float(row["conformity_score"])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(score) and score >= 0:
                    scores.append(score)
            return scores
    except (OSError, csv.Error, UnicodeError):
        return []


def projection_geometry():
    from postprocess_residual_calibration import ideal_metric
    from lqr import certified_metric_alpha
    b, p, k = ideal_metric()
    a = np.zeros((4, 4))
    a[0, 2] = a[1, 3] = 1.0
    rate = certified_metric_alpha(a, b, k, p)
    if rate <= 0:
        raise ValueError("residual metric is not contracting")
    c = np.eye(4)[:2]
    projected = c @ np.linalg.solve(p, c.T)
    eigenvalues, eigenvectors = np.linalg.eigh(projected)
    # Current ideal metric is XY-isotropic: its exact projection is a circle.
    if not np.allclose(eigenvalues[0], eigenvalues[1], rtol=1e-8):
        raise ValueError("anisotropic XY metric requires ellipse rendering")
    return rate, math.sqrt(float(eigenvalues[-1]))


def radius_from_quantile(quantile):
    if quantile is None or not math.isfinite(quantile) or quantile < 0:
        return None
    rate, gain = projection_geometry()
    return gain * quantile / rate


def certificate(scores, delta):
    q = empirical_quantile(scores, delta)
    return {"calibrated_quantile": q, "calibrated_radius_m": radius_from_quantile(q),
            "calibration_sample_count": len(scores), "delta": delta,
            "calibration_status": "valid" if q is not None else NO_QUANTILE}


class AdaptiveResidualCalibration:
    """One prequential error per trajectory, persisted atomically with deduplication.

    alpha is deliberately NOT clipped. Full/empty prediction sets at its
    boundaries keep alpha bounded and preserve the telescoping coverage identity.
    """
    def __init__(self, scores, delta, gamma, state_path):
        empirical_quantile(scores, delta)  # validate delta even without data
        if not math.isfinite(gamma) or gamma <= 0:
            raise ValueError("acp_gamma must be finite and positive")
        self.scores = list(scores)
        self.delta, self.gamma = float(delta), float(gamma)
        self.path = Path(state_path).expanduser()
        self.signature = hashlib.sha256(json.dumps([self.scores, delta, gamma]).encode()).hexdigest()
        self.alpha = self.delta
        self.records = {}
        if self.path.exists():
            state = json.loads(self.path.read_text())
            if state.get("signature") != self.signature:
                raise ValueError("ACP state belongs to different calibration/delta/gamma; choose a new acp_state_file")
            self.alpha = float(state["alpha"])
            self.records = state["records"]
            if not math.isfinite(self.alpha):
                raise ValueError("invalid persisted ACP alpha")

    def snapshot(self, adaptive_mode=False):
        result = certificate(self.scores, self.delta)
        active = adaptive_mode or bool(self.records)
        history = self.scores + [r["score"] for r in self.records.values()]
        q, status = None, "not initialized"
        if active and self.scores:
            if self.alpha <= 0:
                status = "full set (unbounded radius)"
            elif self.alpha >= 1:
                status = "empty set"
            else:
                q = empirical_quantile(history, self.alpha)
                status = "valid"
        result.update(adaptive_quantile=q, adaptive_radius_m=radius_from_quantile(q),
                      adaptive_status=status, acp_alpha=self.alpha, acp_gamma=self.gamma,
                      acp_updates=len(self.records), acp_signature=self.signature,
                      coverage_scope="evaluated complete trajectory residual scores")
        result["active_quantile"] = q if active else result["calibrated_quantile"]
        result["active_radius_m"] = radius_from_quantile(result["active_quantile"])
        result["active_calibration"] = "adaptive" if active else "calibrated"
        return result

    def update(self, trajectory_id, score, frozen):
        key = str(trajectory_id)
        if key in self.records:
            return self.records[key]
        if not self.scores or frozen.get("acp_signature") != self.signature:
            raise ValueError(NO_QUANTILE)
        if float(frozen["acp_alpha"]) != self.alpha or int(frozen["acp_updates"]) != len(self.records):
            raise ValueError("stale ACP trajectory snapshot")
        score = float(score)
        if not math.isfinite(score) or score < 0:
            raise ValueError("invalid trajectory score")
        error = 0 if self.alpha <= 0 else 1 if self.alpha >= 1 else int(score > frozen["adaptive_quantile"])
        next_alpha = self.alpha + self.gamma * (self.delta - error)
        record = {"trajectory_id": key, "score": score, "error": error,
                  "alpha_before": self.alpha, "alpha_after": next_alpha,
                  "quantile_used": frozen["adaptive_quantile"], "radius_used_m": frozen["adaptive_radius_m"],
                  "delta": self.delta, "gamma": self.gamma}
        records = dict(self.records, **{key: record})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps({"signature": self.signature, "alpha": next_alpha, "records": records}, allow_nan=False))
        temporary.replace(self.path)
        self.alpha, self.records = next_alpha, records
        return record
