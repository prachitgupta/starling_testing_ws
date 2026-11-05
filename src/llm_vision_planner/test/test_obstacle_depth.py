#!/usr/bin/env python3
"""Unit tests for obstacle_depth.py (numpy only, no ROS needed)."""

import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from obstacle_depth import (  # noqa: E402
    PRIORS,
    density_filter,
    estimate_by_extent,
    estimate_depth,
    estimate_person,
)


def synthetic_person(rng, half_w=0.18, half_d=0.12, front=2.0, spacing=0.021, noise=0.006):
    """Front surface of an elliptical torso sampled on a ToF-like pixel grid.

    Camera optical frame: x right, y down, z forward. Columns are ~2 cm apart
    at a random phase, so the outermost samples fall short of the true edge.
    """
    xs = np.arange(-half_w + rng.uniform(0.0, spacing), half_w, spacing)
    rows = np.arange(-0.9, 0.8, spacing)
    pts = []
    for y in rows:
        for x in xs:
            z = front + half_d - half_d * np.sqrt(1.0 - (x / half_w) ** 2)
            pts.append([x, y, z])
    pts = np.array(pts) + rng.normal(0.0, noise, (len(pts), 3))
    wall = np.column_stack(
        [rng.uniform(-0.3, 0.3, 300), rng.uniform(-0.9, 0.8, 300), np.full(300, front + 2.0)]
    )
    return np.vstack([pts, wall])


def test_person_depth_recovers_torso_thickness():
    for half_d in (0.10, 0.12, 0.15):
        errors = []
        for seed in range(10):
            est = estimate_person(synthetic_person(np.random.default_rng(seed), half_d=half_d))
            assert est is not None
            errors.append(est.depth_m - 2.0 * half_d)
        mean_error = float(np.mean(errors))
        assert abs(mean_error) < 0.04, (half_d, mean_error)


def test_box_front_face_is_nearest_surface_and_extends_backward():
    est = estimate_person(synthetic_person(np.random.default_rng(1)))
    z = est.corners_cam[:, 2]
    assert abs(z.min() - 2.0) < 0.03, z.min()
    assert abs((z.max() - z.min()) - est.depth_m) < 0.02
    assert abs(est.front_center_cam[2] - z.min()) < 0.01
    assert est.method == "ellipse_fit"


def test_extent_objects_use_prior_as_lower_bound_and_clamp():
    rng = np.random.default_rng(2)
    thin = np.column_stack(
        [rng.uniform(-0.2, 0.2, 400), rng.uniform(-0.4, 0.0, 400), 2.5 + rng.normal(0, 0.005, 400)]
    )
    est = estimate_by_extent(thin, "chair")
    assert est.depth_m >= PRIORS["chair"]["depth"] - 1e-9
    deep = np.column_stack(
        [rng.uniform(-0.2, 0.2, 800), rng.uniform(-0.4, 0.0, 800), rng.uniform(2.5, 3.4, 800)]
    )
    est = estimate_by_extent(deep, "chair")
    assert est.depth_m <= PRIORS["chair"]["max_depth"] + 1e-9


def test_min_width_expands_box_symmetrically():
    rng = np.random.default_rng(3)
    pts = np.column_stack(
        [rng.uniform(-0.1, 0.1, 400), rng.uniform(-0.4, 0.0, 400), rng.uniform(2.5, 2.6, 400)]
    )
    est = estimate_depth(pts, "chair", min_width=0.6, min_height=0.8)
    assert abs(est.width_m - 0.6) < 1e-6
    assert abs(est.height_m - 0.8) < 1e-6


def test_too_few_points_returns_none():
    assert estimate_depth(np.zeros((5, 3)), "person") is None


def test_density_filter_removes_flying_pixels_and_is_fast():
    rng = np.random.default_rng(4)
    surface = np.column_stack(
        [rng.uniform(-1, 1, 40000), rng.uniform(-1, 1, 40000), 2.0 + rng.normal(0, 0.005, 40000)]
    )
    flyers = np.column_stack(
        [rng.uniform(-1, 1, 50), rng.uniform(-1, 1, 50), rng.uniform(2.5, 4.0, 50)]
    )
    start = time.perf_counter()
    kept = density_filter(np.vstack([surface, flyers]))
    elapsed = time.perf_counter() - start
    assert len(kept) >= 0.99 * len(surface)
    assert np.sum(kept[:, 2] > 2.3) <= 2
    assert elapsed < 2.0, elapsed


def sign_plate(rng, yaw_deg=0.0, front=2.0, size=0.6, spacing=0.021, noise=0.01, edge_pixels=60):
    """ToF samples on a flat plate (camera frame), turned about the vertical axis,
    plus mixed pixels behind its edges."""
    yaw = np.radians(yaw_deg)
    s = np.arange(-size / 2, size / 2, spacing)
    u, v = np.meshgrid(s, s)
    x = u.ravel() * np.cos(yaw)
    z = front + size / 2 * abs(np.sin(yaw)) + u.ravel() * np.sin(yaw)
    pts = np.column_stack([x, v.ravel(), z]) + rng.normal(0.0, noise, (u.size, 3))
    mixed = pts[rng.choice(len(pts), edge_pixels)] + [0.0, 0.0, 0.0]
    mixed[:, 2] += rng.uniform(0.03, 0.4, edge_pixels)
    return np.vstack([pts, mixed])


def test_stop_sign_is_10_mm_thick_behind_its_face():
    """A stop sign is not an unknown 1.5 m object: 10 mm behind the face, with
    the face located through 1 cm ToF noise and mixed edge pixels."""
    rng = np.random.default_rng(4)
    for label in ("stop sign", "Stop Sign", "stop_sign"):
        est = estimate_depth(sign_plate(rng), label)
        assert est.method == "fixed_thickness" and abs(est.depth_m - 0.01) < 1e-9, (label, est.depth_m)
        front = float(np.min(est.corners_cam[:, 2]))
        assert abs(front - 2.0) < 0.006, front              # on the plate, not 2 sigma in front of it
    # Turned 40 deg its face recedes 0.39 m along the view: all of it is covered.
    est = estimate_depth(sign_plate(rng, yaw_deg=40.0), "stop sign")
    true_back = 2.0 + 0.6 * np.sin(np.radians(40.0))
    assert np.max(est.corners_cam[:, 2]) >= true_back - 0.01, (np.max(est.corners_cam[:, 2]), true_back)
    assert est.depth_m < 0.6 * np.sin(np.radians(40.0)) + 0.06


if __name__ == "__main__":
    test_stop_sign_is_10_mm_thick_behind_its_face()
    test_person_depth_recovers_torso_thickness()
    test_box_front_face_is_nearest_surface_and_extends_backward()
    test_extent_objects_use_prior_as_lower_bound_and_clamp()
    test_min_width_expands_box_symmetrically()
    test_too_few_points_returns_none()
    test_density_filter_removes_flying_pixels_and_is_fast()
    print("obstacle depth tests passed")
