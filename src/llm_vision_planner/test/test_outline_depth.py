#!/usr/bin/env python3
"""Tests for outline-based depth (outline_depth.py) and the bag tool.

Scenes are ray-cast through the Starling camera mounts and the hires lens
distortion (outline_scene.py). The object outline comes from the renderer, so
these tests check the geometry, not a segmentation model.
"""

import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "test"))

import bag_reader as br  # noqa: E402
import outline_depth as od  # noqa: E402
import outline_scene as scene_lib  # noqa: E402
from obstacle_depth import estimate_depth, range_window_for  # noqa: E402

_CACHE = {}


def observe(name, prims, obj, label, seed=3, mask_fn=None):
    key = (name, seed)
    if key not in _CACHE:
        scene = scene_lib.make_scene(prims)
        camera, tof_r, tof_t, ids, _, tof = scene_lib.capture(scene, np.random.default_rng(seed))
        _CACHE[key] = (scene, camera, ids, od.tof_to_camera(tof, tof_r, tof_t, camera))
    scene, camera, ids, points = _CACHE[key]
    mask = ids == obj
    if mask_fn is not None:
        mask = mask_fn(mask)
    result = od.estimate_from_outline(label, mask, camera, points)
    true_depth, _, true_back = scene_lib.true_depth_along(result.level, camera, scene.truth[obj])
    return result, true_depth, true_back - result.box_level["z1"], (camera, points, mask)


def ragged(mask, p=0.3, seed=1):
    rng = np.random.default_rng(seed)
    ring_out = od.shift_mask(mask, 1) & ~mask
    ring_in = mask & ~od.shift_mask(mask, -1)
    out = mask.copy()
    out[ring_out & (rng.random(mask.shape) < p)] = True
    out[ring_in & (rng.random(mask.shape) < p)] = False
    return out


def test_camera_projection_round_trip():
    camera, _, _ = od.starling_cameras()
    uv = np.random.default_rng(0).uniform([150, 100], [870, 660], (500, 2))
    rays, ok = camera.pixel_rays(uv[:, 0], uv[:, 1])
    back, ok2 = camera.project(rays)
    assert np.all(ok & ok2)
    assert np.max(np.abs(back - uv)) < 1e-6


def test_floor_plane_from_tof_is_accurate():
    _, _, _, (camera, points, _) = observe("person", scene_lib.person(), 2, "person")
    floor = od.fit_floor_plane(points, camera.expected_down())
    assert abs(floor.height - 0.515) < 0.005, floor.height
    tilt = math.degrees(math.acos(min(1.0, floor.normal @ camera.expected_down())))
    assert tilt < 0.3, tilt


def test_person_depth_from_tangent_rays():
    """Facing the drone: torso from the tangent rays, toes from the feet. At
    3 m one image row spans ~3.5 cm of floor, so the toes are that coarse."""
    for name, prims, tol in (("person", scene_lib.person(), 0.02),
                             ("person big", scene_lib.person(half_width=0.20, half_depth=0.15), 0.02),
                             ("person 3m", scene_lib.person(x_front=3.0), 0.035)):
        result, true_depth, back_gap, _ = observe(name, prims, 2, "person")
        assert result.decided_by == "silhouette_ellipse", (name, result.decided_by)
        assert abs(result.depth_m - true_depth) < tol, (name, result.depth_m, true_depth)
        assert abs(back_gap) < 0.02, (name, back_gap)
        assert result.alignment > 0.9


def test_outline_beats_tof_only_for_a_person():
    result, true_depth, _, (camera, points, mask) = observe("person", scene_lib.person(), 2, "person")
    floor = od.fit_floor_plane(points, camera.expected_down())
    pts = points[floor.distance_above(points) > 0.05]
    uv, ok = camera.project(pts)
    vs, us = np.nonzero(mask)
    inb = ok & (uv[:, 0] >= us.min()) & (uv[:, 0] <= us.max()) & (uv[:, 1] >= vs.min()) & (uv[:, 1] <= vs.max())
    r = np.linalg.norm(pts, axis=1)
    front = np.percentile(r[inb], 5)
    tof_only = estimate_depth(pts[inb & (r <= front + range_window_for("person"))], "person")
    assert abs(result.depth_m - true_depth) < abs(tof_only.depth_m - true_depth)


def test_footprint_methods_for_chairs_and_boxes():
    cases = [("chair 45", scene_lib.chair(yaw_deg=45), 3, "chair", 0.03),
             ("chair 180", scene_lib.chair(yaw_deg=180), 3, "chair", 0.03),
             ("office chair", scene_lib.office_chair(), 3, "chair", 0.05),
             ("cabinet 30", scene_lib.cabinet(yaw_deg=30), 4, "cabinet", 0.03)]
    for name, prims, obj, label, tol in cases:
        result, true_depth, back_gap, _ = observe(name, prims, obj, label)
        assert result.decided_by in ("floor_contacts", "footprint_completion", "visible_extent"), (name, result.decided_by)
        assert back_gap < tol, (name, back_gap, result.depth_m, true_depth)
        assert result.depth_m < true_depth + 0.1, (name, result.depth_m, true_depth)


def test_ragged_outline_does_not_create_false_contacts():
    for name, prims, obj, label in (("person", scene_lib.person(), 2, "person"),
                                    ("cabinet 30", scene_lib.cabinet(yaw_deg=30), 4, "cabinet")):
        result, true_depth, back_gap, _ = observe(name, prims, obj, label, mask_fn=ragged)
        # Ragged edges may inflate a box but must not make it too thin.
        assert back_gap < 0.03, (name, back_gap, result.decided_by)
        assert result.depth_m < true_depth + 0.2, (name, result.depth_m, true_depth, result.decided_by)


def test_straight_on_box_falls_back_to_the_prior():
    """A flat face seen square-on carries no depth information: the result is
    flagged and uses the conservative class depth (the drone never moves)."""
    result, true_depth, back_gap, _ = observe("cabinet 0", scene_lib.cabinet(yaw_deg=0), 4, "cabinet")
    assert result.decided_by == "class_prior", result.decided_by
    assert not result.depth_observed
    assert not result.methods["footprint_completion"].valid
    assert result.methods["floor_contacts"].kind == "lower_bound" or not result.methods["floor_contacts"].valid
    assert back_gap < 0.0, back_gap          # conservative: the box reaches past the true back
    assert result.depth_m >= od.FURNITURE_PRIORS["cabinet"]["max_depth"] - 1e-6, result.depth_m
    assert not any("move" in w for w in result.warnings), result.warnings


def test_front_facing_chairs():
    """A chair facing the drone shows its rear legs under the seat."""
    for name, prims in (("chair facing", scene_lib.chair(yaw_deg=0)),
                        ("chair facing far", scene_lib.chair(x_center=3.2, yaw_deg=0)),
                        ("chair facing off-centre", scene_lib.chair(y=0.6, yaw_deg=0))):
        result, true_depth, back_gap, _ = observe(name, prims, 3, "chair")
        assert result.depth_observed, name
        assert -0.12 < back_gap < 0.025, (name, back_gap, result.decided_by)


def test_front_facing_closets():
    """Square-on closets: measured when a side face is visible, otherwise
    flagged and conservative, never silently too thin."""
    for name, y, observed in (("closet centred", 0.0, False), ("closet off 0.4", 0.4, False),
                              ("closet off 0.8", 0.8, True), ("closet off 1.2", 1.2, True)):
        result, true_depth, back_gap, _ = observe(name, scene_lib.wardrobe(y=y), 4, "closet")
        assert result.depth_observed == observed, (name, result.decided_by, result.warnings)
        if observed:
            assert -0.08 < back_gap < 0.035, (name, back_gap, result.decided_by)
        else:
            assert back_gap < 0.0, (name, back_gap)
            assert result.depth_m >= od.FURNITURE_PRIORS["wardrobe"]["max_depth"] - 1e-6, (name, result.depth_m)


def test_armchair_with_hidden_backrest_is_not_trusted():
    """Seen from the front, a sofa's backrest hides its own thickness."""
    result, true_depth, back_gap, _ = observe("armchair", scene_lib.armchair(), 3, "chair")
    assert not result.depth_observed, result.decided_by
    assert back_gap < 0.01, back_gap
    result, true_depth, back_gap, _ = observe("armchair off", scene_lib.armchair(y=0.8), 3, "chair")
    assert result.depth_observed and -0.08 < back_gap < 0.03, (back_gap, result.decided_by)


def test_tof_alignment_fit_recovers_a_mount_error():
    """A ToF mounted 1.3 deg off (as measured on the first real bag) is found
    from the outline and undone."""
    # One frame pins pan/tilt well; roll less so on a chair's thin legs.
    for name, prims, obj, tol in (("person", scene_lib.person(), 2, 0.35),
                                  ("chair 45", scene_lib.chair(yaw_deg=45), 3, 0.6)):
        _, _, _, (camera, points, mask) = observe(name, prims, obj, "person")
        error = od.small_rotation(0.4, -1.3, 0.0)        # what the mount does to the points
        fit = od.fit_tof_alignment(mask, camera, points @ error.T, od.fit_floor_plane(points, camera.expected_down()))
        residual = fit["rotation"] @ error
        angle = math.degrees(math.acos(min(1.0, (np.trace(residual) - 1) / 2)))
        assert angle < tol, (name, fit["angles_deg"], angle)
        assert abs(fit["angles_deg"][1] - 1.3) < 0.25, (name, fit["angles_deg"])
        assert fit["f1_after"] > fit["f1_before"] + 0.03, (name, fit)
        # Already aligned: the fit stays near zero.
        fit0 = od.fit_tof_alignment(mask, camera, points, od.fit_floor_plane(points, camera.expected_down()))
        assert max(abs(a) for a in fit0["angles_deg"][:2]) < 0.35, (name, fit0["angles_deg"])
    # The equivalent camera mount reproduces the correction exactly.
    camera, tof_r, tof_t = od.starling_cameras()
    roll, pitch, yaw, err = od.mount_with_correction(0.0, 90.0, 90.0, fit["rotation"])
    cam2, _, _ = od.starling_cameras(dict(detection_cam_roll_deg=roll, detection_cam_pitch_deg=pitch,
                                          detection_cam_yaw_deg=yaw))
    p_body = np.random.default_rng(0).normal(size=(20, 3)) + [2.0, 0.0, 0.0]
    assert err < 1e-9
    assert np.allclose(cam2.body_to_cam(p_body), camera.body_to_cam(p_body) @ fit["rotation"].T, atol=1e-9)


def test_floor_offset_under_contacts_is_corrected():
    """Foam mats: the floor at the object is a few cm off the plane fitted near
    the drone. ToF points around the legs give the local height."""
    _, _, _, (camera, points, mask) = observe("chair facing", scene_lib.chair(yaw_deg=0), 3, "chair")
    good = od.estimate_from_outline("chair", mask, camera, points)
    floor = od.fit_floor_plane(points, camera.expected_down())
    for dh in (-0.03, 0.03):
        wrong = od.FloorPlane(normal=floor.normal, height=floor.height + dh, inliers=floor.inliers, rms=floor.rms)
        result = od.estimate_from_outline("chair", mask, camera, points, wrong)
        assert "contacts corrected" in result.methods["floor_contacts"].note, result.methods["floor_contacts"].note
        assert abs(result.depth_m - good.depth_m) < 0.04, (dh, result.depth_m, good.depth_m)
    # No floor seen anywhere near the contacts: they are not trusted.
    level = good.level
    contacts = np.array([[0.0, 2.5], [0.1, 2.6], [0.2, 2.5]])
    clusters = [dict(x=0.1, z_min=2.5, z_max=2.6, count=3, x_span=0.2)]
    near_floor_only = level.to_cam(np.column_stack([np.random.default_rng(0).uniform(-0.5, 0.5, 300),
                                                    np.full(300, level.height),
                                                    np.random.default_rng(1).uniform(0.6, 1.2, 300)]))
    _, _, note = od.check_floor_under_contacts(contacts, clusters, level, near_floor_only)
    assert note.startswith("unverified"), note


def test_outline_drawn_from_the_tof_measures_nothing_new():
    """The tof segmenter's edges are ToF edges: no ellipse or contacts from it."""
    _, _, _, (camera, points, mask) = observe("person", scene_lib.person(), 2, "person")
    result = od.estimate_from_outline("person", mask, camera, points,
                                      params=od.OutlineParams(outline_from_tof=True))
    assert not result.methods["silhouette_ellipse"].valid
    assert not result.methods["floor_contacts"].valid
    assert result.decided_by == "class_prior" and not result.depth_observed, result.decided_by


def test_box_always_holds_what_the_tof_saw():
    """Wherever a method puts the front (toes, legs), the box must reach past
    every ToF point of the object and past the decided method's own back."""
    for name, prims, obj, label in (("person", scene_lib.person(), 2, "person"),
                                    ("person yaw45", scene_lib.person(yaw_deg=45), 2, "person"),
                                    ("chair 45", scene_lib.chair(yaw_deg=45), 3, "chair"),
                                    ("chair facing", scene_lib.chair(yaw_deg=0), 3, "chair"),
                                    ("cabinet 30", scene_lib.cabinet(yaw_deg=30), 4, "cabinet"),
                                    ("closet", scene_lib.wardrobe(), 4, "closet")):
        result, _, back_gap, _ = observe(name, prims, obj, label)
        z = result.geometry["points_level"][:, 2]
        assert result.box_level["z1"] >= np.percentile(z, 98) - 1e-6, name
        assert result.box_level["z0"] <= result.front_z_m + 1e-6, name
        m = result.methods.get(result.decided_by)   # class_prior isn't a measuring method
        if m is not None:
            front = result.front_z_m if m.front is None else m.front
            assert result.box_level["z1"] >= min(front + m.depth, front + result.depth_m) - 1e-6, name
        assert abs(result.depth_m - (result.box_level["z1"] - result.box_level["z0"])) < 1e-9, name


def test_side_on_person_gets_the_full_width():
    """Side-on, a person's width is along the line of sight; the front
    curvature can't show it, the narrow outline does."""
    for name, prims in (("person yaw90", scene_lib.person(yaw_deg=90)),
                        ("person big yaw90", scene_lib.person(half_width=0.22, half_depth=0.14, yaw_deg=90)),
                        ("person yaw45", scene_lib.person(yaw_deg=45))):
        result, true_depth, back_gap, _ = observe(name, prims, 2, "person")
        assert result.methods["side_on"].valid, (name, result.methods["side_on"].note)
        assert back_gap < 0.01, (name, back_gap, result.decided_by)
    facing, _, _, _ = observe("person", scene_lib.person(), 2, "person")
    assert not facing.methods["side_on"].valid
    assert facing.decided_by == "silhouette_ellipse"


def mixed_edges(points, camera, mask, px=12, push=0.08):
    """Push ToF points near the outline's side edges back, as real ToF pixels do
    where they mix a body with the background behind it."""
    uv, ok = camera.project(points)
    out = points.copy()
    for i in np.flatnonzero(ok):
        vi = int(round(uv[i, 1]))
        if not 0 <= vi < mask.shape[0]:
            continue
        cols = np.flatnonzero(mask[vi])
        if len(cols) and cols.min() - 2 <= uv[i, 0] <= cols.max() + 2:
            d = min(abs(uv[i, 0] - cols.min()), abs(cols.max() - uv[i, 0]))
            if d < px:
                out[i] *= 1 + push * (1 - d / px)
    return out


def test_mixed_edge_pixels_dont_make_a_person_deep():
    """Frame 7 of the first lab bag: ToF edges bending back made the curvature
    ellipse narrow and deep. The body is wider than deep, so facing the drone
    the person-shaped ellipse lies flat."""
    for name, prims in (("person", scene_lib.person()),
                        ("person big", scene_lib.person(half_width=0.22, half_depth=0.13))):
        _, _, _, (camera, points, mask) = observe(name, prims, 2, "person")
        result = od.estimate_from_outline("person", mask, camera, mixed_edges(points, camera, mask))
        truth = scene_lib.make_scene(prims).truth[2]
        true_depth, _, true_back = scene_lib.true_depth_along(result.level, camera, truth)
        assert result.decided_by == "body_shape", (name, result.decided_by)
        assert true_back - result.box_level["z1"] < 0.01, name                # still covers the body
        assert result.depth_m < true_depth + 0.08, (name, result.depth_m, true_depth)


def holding_something(yaw_deg, **kw):
    """Person holding a phone or controller in front of the chest (frame 2)."""
    prims = scene_lib.person(yaw_deg=yaw_deg, **kw)
    yaw, hd = math.radians(yaw_deg), kw.get("half_depth", 0.12)
    cx, lx = 2.0 + hd, -(hd + 0.14)
    prims.append(scene_lib.EllipticCylinder(cx + math.cos(yaw) * lx, math.sin(yaw) * lx, 0.07, 0.07,
                                            0.5 - 1.40, 0.5 - 1.24, yaw, 2))
    return prims


def test_hands_held_up_dont_fool_the_orientation():
    """Frame 2: side-on, hands up holding a phone. The hands widen the shoulder
    rows, but the narrow torso still says side-on, so the body ellipse stands
    up. Facing the drone, the held object moves the box front, not the body."""
    for name, prims, side in (("side-on phone", holding_something(90), True),
                              ("big side-on phone", holding_something(90, half_width=0.23, half_depth=0.14), True),
                              ("facing phone", holding_something(0), False)):
        result, true_depth, back_gap, _ = observe(name, prims, 2, "person")
        assert result.geometry["body_ellipse"]["side_on"] == side, (name, result.methods["body_shape"].note)
        assert back_gap < 0.01, (name, back_gap, result.decided_by)


def hooded(prims):
    """A hood or big hair: the head outline nearly as wide as the shoulders,
    which on its own looks side-on."""
    t = prims[0]
    prims.append(scene_lib.EllipticCylinder(t.cx, t.cy, 0.12, 0.13, 0.5 - 1.78, 0.5 - 1.50, t.yaw, 2))
    return prims


def test_feet_side_by_side_make_a_wide_ellipse():
    """Feet side by side across the view: facing toward or away, so a wide
    ellipse, even when a hood makes the head look as wide as a side-on body."""
    for name, prims in (("person", scene_lib.person()), ("hooded person", hooded(scene_lib.person())),
                        ("facing away", scene_lib.person(yaw_deg=180))):
        result, true_depth, back_gap, _ = observe(name, prims, 2, "person")
        be = result.geometry["body_ellipse"]
        assert be["cue"] == "feet" and not be["side_on"], (name, result.methods["body_shape"].note)
        assert be["stance_deg"] < 10, (name, be["stance_deg"])
        assert back_gap < 0.01 and result.depth_m < true_depth + 0.05, (name, back_gap, result.depth_m, true_depth)


def test_slim_person_facing_is_not_called_side_on():
    """A slim person's torso is as narrow as an average one side-on; two
    shoes side by side, each only a shoe's width across, say facing."""
    for name, yaw in (("slim", 0), ("slim facing away", 180)):
        prims = scene_lib.person(half_width=0.16, half_depth=0.10, yaw_deg=yaw)
        result, true_depth, back_gap, _ = observe(name, prims, 2, "person")
        assert not result.methods["side_on"].valid, (name, result.methods["side_on"].note)
        assert back_gap < 0.01 and result.depth_m < true_depth + 0.05, (name, back_gap, result.depth_m, true_depth)
    # Turned 40 deg the shoes look longer: the narrow-torso rule stays.
    result, _, back_gap, _ = observe("slim yaw40", scene_lib.person(half_width=0.16, half_depth=0.10, yaw_deg=40),
                                     2, "person")
    assert result.methods["side_on"].valid and back_gap < 0.01


def test_steps_cover_both_feet():
    """Walking toward the drone: the front foot is well ahead of the torso and
    the back foot behind it, even when the front leg hides it."""
    for name, prims in (("step", scene_lib.person(stride=0.5)),
                        ("long step", scene_lib.person(half_width=0.22, half_depth=0.14, stride=0.6)),
                        ("step, back foot hidden", scene_lib.person(yaw_deg=20, stride=0.6))):
        result, true_depth, back_gap, (camera, _, _) = observe(name, prims, 2, "person")
        assert "a step" in result.methods["feet"].note, (name, result.methods["feet"].note)
        assert back_gap < 0.01, (name, back_gap, result.decided_by)
        # The front foot is in the box too (to within its near edge's corner).
        true_front = scene_lib.true_depth_along(result.level, camera, scene_lib.make_scene(prims).truth[2])[1]
        assert result.box_level["z0"] <= true_front + 0.05, (name, result.box_level["z0"], true_front)


def test_walking_past_stays_side_on():
    """Walking past, the feet are side by side across the view while the body
    is side-on: a step that far apart doesn't make the ellipse wide."""
    for name, prims in (("walking past", scene_lib.person(yaw_deg=90, stride=0.45)),
                        ("walking past, other way", scene_lib.person(yaw_deg=270, stride=0.45))):
        result, _, back_gap, _ = observe(name, prims, 2, "person")
        assert result.geometry["body_ellipse"]["side_on"], (name, result.methods["body_shape"].note)
        assert back_gap < 0.01, (name, back_gap)


def test_stop_sign_is_10_mm_thick():
    """Stop signs have a hard-coded 10 mm thickness instead of the 1.5 m an
    unknown class gets; turned away from the drone, the receding face counts."""
    # A turned face is found from the outline's edges: ~1 px there is ~6 mm along the view.
    for name, yaw, tol, gap in (("stop sign", 0.0, 0.0, 0.006), ("stop sign yaw35", 35.0, 0.06, 0.012)):
        result, true_depth, back_gap, _ = observe(name, scene_lib.stop_sign(yaw_deg=yaw), 5, "stop sign")
        assert result.decided_by == "fixed_thickness", (name, result.decided_by)
        assert back_gap < gap, (name, back_gap)                          # the plate is inside the box
        assert result.depth_m <= true_depth + 0.01 + tol, (name, result.depth_m, true_depth)
        if yaw == 0.0:
            assert abs(result.depth_m - 0.01) < 1e-9, result.depth_m


def test_outline_edge_sensitivity_is_reported():
    result, _, _, _ = observe("person", scene_lib.person(), 2, "person")
    assert 0.005 < result.edge_sensitivity_m_per_px < 0.04, result.edge_sensitivity_m_per_px


def test_cdr_round_trip():
    registry = br.TypeRegistry([ROOT / "test" / "msg_fixtures"])
    header = {"stamp": {"sec": 12, "nanosec": 34}, "frame_id": "world"}
    cloud = np.arange(30, dtype="<f4").reshape(10, 3)
    msg = {"header": header, "height": 1, "width": 10,
           "fields": [{"name": n, "offset": 4 * i, "datatype": 7, "count": 1} for i, n in enumerate("xyz")],
           "is_bigendian": False, "point_step": 12, "row_step": 120,
           "data": np.frombuffer(cloud.tobytes(), np.uint8), "is_dense": True}
    back = br.deserialize(br.serialize(msg, "sensor_msgs/msg/PointCloud2", registry),
                          "sensor_msgs/msg/PointCloud2", registry)
    assert back.header.frame_id == "world" and back.header.stamp.nanosec == 34
    assert np.allclose(br.cloud_xyz(back), cloud)
    det = {"timestamp_ns": 5, "class_id": 1, "frame_id": 2, "class_name": "chair", "cam": "hires",
           "class_confidence": 0.8, "detection_confidence": 0.7, "x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4}
    d = br.deserialize(br.serialize(det, "voxl_msgs/msg/Aidetection", registry), "voxl_msgs/msg/Aidetection", registry)
    label, conf, box, _ = br.detection_box(d, 1000, 500)
    assert label == "chair" and abs(conf - 0.8) < 1e-6 and np.allclose(box, (100, 100, 300, 200), atol=1e-3)


def test_detection_matching_prefers_the_wanted_label():
    """A nearer box of another class, or a weak one, must not hide the object."""
    from types import SimpleNamespace
    import outline_depth_from_bag as tool
    args = SimpleNamespace(max_detection_gap=1.0, min_confidence=0.5, label="person")
    t = 10_000_000_000
    box = (0.2, 0.1, 0.4, 0.9)
    dets = [(t + 50_000_000, "chair", 0.9, box, "hires"),        # closest, other class
            (t - 400_000_000, "person", 0.3, box, "hires"),      # nearer person, too weak
            (t - 700_000_000, "person", 0.66, box, "hires"),     # repeated burst of one frame
            (t - 690_000_000, "person", 0.62, box, "hires"),
            (t + 1_500_000_000, "person", 0.9, box, "hires")]    # too far in time
    assert [d[2] for d in tool.matching_detections(dets, t, args)] == [0.66, 0.62]
    args.label = None
    assert [d[1] for d in tool.matching_detections(dets, t, args)] == ["chair"]
    args.label, args.max_detection_gap = "person", 0.3
    assert tool.matching_detections(dets, t, args) == []


def test_synthetic_bag_end_to_end():
    import make_synthetic_bag
    with tempfile.TemporaryDirectory() as tmp:
        bag = Path(tmp) / "person_bag"
        truth = make_synthetic_bag.make_bag(bag, "person", frames=2)
        out = Path(tmp) / "out"
        cmd = [sys.executable, str(ROOT / "scripts" / "outline_depth_from_bag.py"), str(bag), "--reader", "builtin",
               "--msg-dir", str(ROOT / "test" / "msg_fixtures"), "--segmenter", "maskdir",
               "--mask-dir", str(bag / "masks"), "--label", "person", "--truth-width", str(truth["width"]),
               "--truth-depth", str(truth["depth"]), "--plots", "1", "--out", str(out)]
        sample = Path(tmp) / "sample.npz"
        subprocess.run(cmd + ["--export-sample", str(sample)], check=True, capture_output=True, text=True)
        summary = json.loads((out / "summary.json").read_text())["person"]
        # The exported sample reproduces the bag run without the bag.
        out2 = Path(tmp) / "out2"
        subprocess.run([sys.executable, str(ROOT / "scripts" / "outline_depth_from_bag.py"), str(sample),
                        "--segmenter", "maskdir", "--mask-dir", str(bag / "masks"), "--label", "person",
                        "--plots", "0", "--out", str(out2)], check=True, capture_output=True, text=True)
        again = json.loads((out2 / "summary.json").read_text())["person"]
        assert again["depth_m"] == summary["depth_m"]
        assert summary["frames"] == 2
        assert abs(summary["depth_m"]["median"] - summary["true_depth_m"]["median"]) < 0.02, summary
        assert abs(summary["true_depth_m"]["median"] - truth["depth"]) < 0.01, summary
        assert abs(summary["camera_height_m"]["median"] - summary["vicon_camera_height_m"]["median"]) < 0.01
        assert (out / "results.csv").is_file() and any(out.glob("frame_*.png"))


def test_synthetic_bag_of_a_square_on_closet_is_conservative():
    """Stationary snapshot of a closet straight ahead: every frame is flagged,
    the box reaches past the true back, and nothing asks the drone to move."""
    import make_synthetic_bag
    with tempfile.TemporaryDirectory() as tmp:
        bag = Path(tmp) / "closet_bag"
        make_synthetic_bag.make_bag(bag, "wardrobe", frames=2)
        out = Path(tmp) / "out"
        cmd = [sys.executable, str(ROOT / "scripts" / "outline_depth_from_bag.py"), str(bag), "--reader", "builtin",
               "--msg-dir", str(ROOT / "test" / "msg_fixtures"), "--segmenter", "maskdir",
               "--mask-dir", str(bag / "masks"), "--label", "closet", "--truth-width", "1.0",
               "--truth-depth", "0.6", "--plots", "1", "--out", str(out)]
        run = subprocess.run(cmd, check=True, capture_output=True, text=True)
        assert "move" not in run.stdout.lower(), run.stdout
        rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
        assert len(rows) == 2
        for r in rows:
            assert not r["depth_observed"] and r["decided_by"] == "class_prior", r
            assert -0.25 < r["back_gap_m"] < 0.0, r
            assert r["depth_ms"] > 0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print("outline depth tests passed")
