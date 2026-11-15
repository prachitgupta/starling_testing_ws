"""Trajectory residual calibration and Gibbs--Candes ACI (2021, update (2)).

Split CP uses the finite-sample n+1 rank; ACP retains its empirical quantile.
ACI controls average score miscoverage across evaluated trajectories, not flight safety.
"""
import csv
import hashlib
import json
import math
from decimal import Decimal, ROUND_CEILING
from pathlib import Path

import numpy as np

NO_QUANTILE = "no valid quantile found"


def split_rank(n, delta):
    if not isinstance(n, int) or n < 0 or not 0.0 < delta < 1.0:
        raise ValueError("n must be nonnegative and delta strictly between 0 and 1")
    # Decimal avoids turning an exact integer boundary into the next rank.
    return int(((1 - Decimal(str(delta))) * (n + 1)).to_integral_value(rounding=ROUND_CEILING))


def split_quantile(scores, delta=0.20):
    values = [float(score) for score in scores]
    if any(not math.isfinite(score) or score < 0 for score in values):
        raise ValueError("split CP requires every scheduled score to be finite and nonnegative")
    rank = split_rank(len(values), delta)
    return sorted(values)[rank - 1] if rank <= len(values) else math.inf


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
            seen = set()
            for row in csv.DictReader(stream, delimiter=delimiter):
                if (row.get("placeholder", "").lower() != "false"
                        or row.get("planner_provider") != "transformer"
                        or row.get("dynamics_model") != "ideal_planar_double_integrator"):
                    raise ValueError(f"{path}: non-deployment score row; refusing subset calibration")
                key = (row.get("capture_id"), row.get("l1_text"), row.get("l2_text"))
                if key in seen:
                    raise ValueError(f"{path}: duplicate calibration capture")
                seen.add(key)
                try:
                    score = float(row["conformity_score"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{path}: invalid score; refusing subset calibration") from exc
                if not math.isfinite(score) or score < 0:
                    raise ValueError(f"{path}: invalid score; refusing subset calibration")
                scores.append(score)
            return scores
    except (OSError, csv.Error, UnicodeError):
        return []


def projection_geometry():
    from lqr import A_DOUBLE_INTEGRATOR, B_DOUBLE_INTEGRATOR, certified_metric_alpha, solve_care
    a, b = A_DOUBLE_INTEGRATOR, B_DOUBLE_INTEGRATOR
    p, k, _ = solve_care()
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


def certificate(scores, delta, metadata=None):
    scores = list(scores)
    q = split_quantile(scores, delta)
    unbounded = not math.isfinite(q)
    return {**(metadata or {}),
            "method": "split_cp", "rank": split_rank(len(scores), delta),
            "tracking_assumption": "perfect_tracking_ideal_double_integrator",
            "quantile_unbounded": unbounded,
            "calibrated_quantile": None if unbounded else q,
            "calibrated_radius_m": radius_from_quantile(q),
            "calibration_sample_count": len(scores), "delta": delta,
            "calibration_status": "unbounded_insufficient_samples" if unbounded else "valid"}


def certificate_from_csv(path, delta):
    path = resolve_data_path(path)
    result = certificate(read_scores(path), delta)
    provenance = path.parent / "split_certificate.json"
    if provenance.is_file():
        frozen = json.loads(provenance.read_text())
        if path.name != "Dcal.scored.csv" or Path(frozen["calibration_csv"]).resolve() != path.resolve():
            raise ValueError("only the frozen Dcal CSV may calibrate this experiment")
        if frozen["delta"] != delta:
            raise ValueError("delta differs from the value chosen before collection")
        if hashlib.sha256(path.read_bytes()).hexdigest() != frozen["calibration_dataset_sha256"]:
            raise ValueError("Dcal CSV changed after certification")
        for key in ("method", "rank", "calibrated_quantile", "calibration_sample_count"):
            if frozen.get(key) != result[key]:
                raise ValueError("frozen split certificate does not match Dcal")
        result = {**frozen, **result}
    elif result["calibration_sample_count"]:
        raise ValueError("frozen split_certificate.json is required; legacy or Dtest scores cannot calibrate split mode")
    return result


class AdaptiveResidualCalibration:
    """One prequential error per trajectory, persisted atomically with deduplication.

    alpha is deliberately NOT clipped. Full/empty prediction sets at its
    boundaries keep alpha bounded and preserve the telescoping coverage identity.
    """
    def __init__(self, scores, delta, gamma, state_path, calibration_metadata=None):
        empirical_quantile(scores, delta)  # validate delta even without data
        if not math.isfinite(gamma) or gamma <= 0:
            raise ValueError("acp_gamma must be finite and positive")
        self.scores = list(scores)
        self.delta, self.gamma = float(delta), float(gamma)
        self.path = Path(state_path).expanduser()
        self.signature = hashlib.sha256(json.dumps([self.scores, delta, gamma]).encode()).hexdigest()
        self.alpha = self.delta
        self.records = {}
        self.calibration_metadata = calibration_metadata
        self._state_loaded = False

    def _load_state(self):
        if self._state_loaded:
            return
        if self.path.exists():
            state = json.loads(self.path.read_text())
            if state.get("signature") != self.signature:
                raise ValueError("ACP state belongs to different calibration/delta/gamma; choose a new acp_state_file")
            self.alpha = float(state["alpha"])
            self.records = state["records"]
            if not math.isfinite(self.alpha):
                raise ValueError("invalid persisted ACP alpha")
        self._state_loaded = True

    def snapshot(self, adaptive_mode=False):
        result = certificate(self.scores, self.delta, self.calibration_metadata)
        active = bool(adaptive_mode)
        if active:
            self._load_state()
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
        self._load_state()
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
