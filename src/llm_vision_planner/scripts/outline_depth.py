"""Object depth from a camera outline plus ToF points (single view, stationary).

An outline on its own cannot give depth: a person and a cardboard cutout of
that person have the same outline. What the outline adds is precise geometry
that the ToF measures poorly, and combined with the ToF and the floor it lets
the hidden back of an object be inferred with simple geometry:

  1. Cleaner points: only ToF points inside the object's outline are used, so
     background seen through gaps (chair legs) and neighbouring clutter drop out.
  2. Silhouette-constrained ellipse (rounded objects such as people, bins,
     bottles): each outline edge pixel is a camera ray that grazes the object,
     so in a horizontal slice the left and right edge rays are TANGENT to the
     cross-section. Tangency pins the width far better than the ToF's few edge
     points can, and the ToF front arc then fixes how far the surface curves
     away, which sets the depth. Only used where the front is measurably curved.
  3. Floor contacts (anything standing on the floor: chairs, tables, stands):
     the bottom boundary of the outline is where the object meets the floor.
     Each such pixel's ray hits the floor plane at a metric point, including
     the rear legs seen between the front ones, which gives the footprint depth.
  4. Footprint completion (box-like objects turned to the drone): when two faces
     are visible, the smallest rectangle containing everything observed is the
     full footprint, including the hidden back corner.
  5. Box sides: a flat-fronted object off to one side shows a side face as a
     sliver at the edge of its outline.
  6. Visible extent: the ToF front-to-back spread, a lower bound.

Everything comes from the one image + ToF cloud taken where the drone
hovers; the drone is never asked to move. A flat face seen square-on (a
closet straight ahead, a sofa's backrest) has no depth information in that
view. The result is then flagged depth_observed=False and uses the class's
largest typical depth (FURNITURE_PRIORS), so the obstacle box, and the safety
tube built around it, errs too deep rather than too shallow.

The floor plane is fitted to the ToF points, so no motion capture or EKF
height is needed. Gravity-aligned ("level") coordinates are built from it:
origin at the camera, y' pointing down along the floor normal, z' the
horizontal direction to the object, x' to the right. Depth means extent along
z', the line of sight, which is what the obstacle box needs.

Methods report estimates (could be the full depth) or lower bounds (can only
be too small). The result is the largest valid value, so a method that fails
should make a box too big rather than too small. Only numpy is required.
"""

from dataclasses import dataclass, field, replace
import math
from pathlib import Path
import sys

import numpy as np

try:
    from obstacle_depth import (DEFAULT_PRIOR, density_filter, fixed_extent_along_view, fixed_thickness,
                                plane_face_fit, prior_for)
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from obstacle_depth import (DEFAULT_PRIOR, density_filter, fixed_extent_along_view, fixed_thickness,
                                plane_face_fit, prior_for)


# Typical and largest depths (m) for furniture, used only when a single view
# can't measure the depth (a flat face seen square-on). Labels map through
# LABEL_ALIASES first; anything else uses obstacle_depth's priors.
FURNITURE_PRIORS = {
    "chair": dict(depth=0.50, min_depth=0.35, max_depth=0.90),
    "couch": dict(depth=0.90, min_depth=0.60, max_depth=1.10),
    "bed": dict(depth=2.00, min_depth=1.20, max_depth=2.30),
    "wardrobe": dict(depth=0.60, min_depth=0.35, max_depth=0.75),
    "cabinet": dict(depth=0.50, min_depth=0.30, max_depth=0.70),
    "bookcase": dict(depth=0.32, min_depth=0.20, max_depth=0.45),
    "refrigerator": dict(depth=0.72, min_depth=0.55, max_depth=0.90),
    "dining table": dict(depth=0.90, min_depth=0.60, max_depth=1.60),
    "desk": dict(depth=0.70, min_depth=0.45, max_depth=0.90),
    "bench": dict(depth=0.45, min_depth=0.25, max_depth=0.80),
    "tv": dict(depth=0.15, min_depth=0.03, max_depth=0.45),
    "suitcase": dict(depth=0.28, min_depth=0.15, max_depth=0.45),
    "box": dict(depth=0.40, min_depth=0.10, max_depth=1.00),
}
LABEL_ALIASES = {
    "closet": "wardrobe", "cupboard": "wardrobe", "armoire": "wardrobe", "locker": "wardrobe",
    "dresser": "cabinet", "drawers": "cabinet", "filing cabinet": "cabinet", "sideboard": "cabinet",
    "bookshelf": "bookcase", "shelf": "bookcase", "shelves": "bookcase", "fridge": "refrigerator",
    "sofa": "couch", "armchair": "chair", "table": "dining table", "monitor": "tv",
    "cardboard box": "box", "crate": "box",
}


def object_prior(label):
    name = LABEL_ALIASES.get(label, label)
    return FURNITURE_PRIORS.get(name, prior_for(name))


# People: a body is not a box, so the footprint/box methods are not used, and
# an adult torso is at least this wide facing the drone (5th-percentile female
# shoulder breadth): a narrower outline means it is seen side-on.
BODY_LABELS = {"person"}
MIN_FACING_TORSO_WIDTH_M = 0.33
# A body's horizontal cross-section (torso with arms down) is an ellipse this
# much deeper than wide: chest depth / shoulder width of adults is ~0.55-0.65.
BODY_DEPTH_RATIOS = (0.5, 0.75)
SIDE_ON_SHOULDER_HEAD_RATIO = 1.6   # facing/back ~2.5-3, side-on ~1.2-1.5
MAX_BODY_EXTENT_M = 0.55             # widest adult with arms down
MIN_FEET_SEPARATION_M = 0.12         # closer feet don't show a direction
FEET_SIDE_BY_SIDE_DEG = 30.0         # feet line within this of across the view: facing toward/away
FEET_ONE_BEHIND_DEG = 50.0           # ... nearer along the view: side-on (or stepping toward/away)
MAX_STANCE_WIDTH_M = 0.35            # feet further apart than this are a step, not a stance
STEP_AHEAD_M = 0.15                  # a front foot this far ahead of the torso is a step
FOOT_LENGTH_M = 0.28                 # behind a foot's visible near edge, pointing along the view
FOOT_WIDTH_M = 0.12                  # ... pointing across the view
MAX_FOOT_WIDTH_FACING_M = 0.16       # a shoe pointing at/away from the drone is narrower than this across the view


def slant_angle(slope, k, side_on):
    """Turn angle (radians) of an elliptical body (depth/width ratio k) whose
    visible surface slants by `slope` (dz/dx).

    The visible surface runs along the chord between the two outline tangent
    points, whose slope is (1 - k^2) s c / (c^2 + k^2 s^2) for a body turned
    by theta. Two angles give each slope: `side_on` picks the more side-on.
    """
    m, q = abs(float(slope)), 1.0 - k * k
    if m < 1e-6:
        return math.pi / 2 if side_on else 0.0
    disc = q * q - 4.0 * m * m * k * k
    if disc <= 0:   # steeper than this body can slant: the most-slanted pose
        return math.atan(1.0 / k)
    r = math.sqrt(disc)
    return math.atan(((q + r) if side_on else (q - r)) / (2.0 * m * k * k))


def extent_at(width, theta, k):
    """Along-view extent of an elliptical body turned by theta (radians) whose
    outline is `width` wide across the view.

    Turned by theta, the outline width is w = 2 sqrt(a^2 c^2 + b^2 s^2) and
    the extent along the view e = 2 sqrt(a^2 s^2 + b^2 c^2) with b = k a, so
    e = w sqrt(s^2 + k^2 c^2) / sqrt(c^2 + k^2 s^2): k w facing the drone (or
    facing away), w / k side-on.
    """
    s2, c2 = math.sin(theta) ** 2, math.cos(theta) ** 2
    return width * math.sqrt(s2 + k * k * c2) / math.sqrt(c2 + k * k * s2)


def body_extent(width, slope, k, side_on):
    """extent_at with the turn angle from the surface slant. Returns (extent, theta_deg)."""
    theta = slant_angle(slope, k, side_on)
    return extent_at(width, theta, k), math.degrees(theta)


# Classes whose torso-like cross-section is fitted in a belly band; everything
# else uses the middle of its visible height.
BAND_FOR_LABEL = {"person": (0.30, 0.45)}
DEFAULT_BAND = (0.25, 0.75)
HALF_DEPTH_GUESS = {"person": 0.14}


def find_feet(mask, rays_level, camera_height, fy, z_ref, min_pixels=3, max_step_m=0.8,
              min_width_m=0.04, with_width=False):
    """Floor positions (level x', z') of the feet, from the bottom of the outline.

    Only the lowest outline pixel of each column counts, and only within a
    few rows of the very bottom: the floor up to a long step (max_step_m)
    behind the nearer foot shows up that many rows higher, while bottom-edge
    pixels further up trace the inside of the legs and the crotch, not the
    feet. The pixels split into feet where columns of floor separate them or
    the edge jumps back; pieces narrower than a foot (min_width_m) are the
    sloping edges of legs. Each foot is the median of its pixels on the
    floor: the middle of its near edge.
    Returns up to two (x, z) points, biggest first ((x, z, width across the
    view) with with_width); [] without a floor.
    """
    if not np.isfinite(camera_height) or camera_height <= 0:
        return []
    cols = np.flatnonzero(mask.any(axis=0))
    if not len(cols):
        return []
    rows = mask.shape[0] - 1 - np.argmax(mask[::-1, cols], axis=0)
    px_per_m = fy * camera_height / max(float(z_ref), 0.5) ** 2   # image rows per metre of floor depth
    window = 6.0 + 1.2 * px_per_m * max_step_m
    near = (rows >= rows.max() - window) & (rows < mask.shape[0] - 1)
    u, v = cols[near], rows[near]
    if not len(u):
        return []
    # A new foot starts after a gap of floor columns, or where the bottom edge
    # jumps by more than ~6 cm of floor depth (a foot partly behind the other;
    # along one shoe the edge moves a pixel or two per column).
    jump = max(3.5, 0.06 * px_per_m)
    cuts = np.flatnonzero((np.diff(u) > 3) | (np.abs(np.diff(v)) > jump)) + 1
    feet = []
    for g in np.split(np.arange(len(u)), cuts):
        if len(g) < min_pixels:
            continue
        d, ok = rays_level(u[g].astype(float), v[g].astype(float))
        ok &= d[:, 1] > 1e-3
        if ok.sum() < min_pixels:
            continue
        hits = d[ok] * (camera_height / d[ok, 1])[:, None]
        width = float(np.ptp(hits[:, 0]))
        if width < min_width_m:
            continue
        feet.append((float(np.median(hits[:, 0])), float(np.median(hits[:, 2])), width, int(ok.sum())))
    feet.sort(key=lambda f: -f[3])
    return [(x, z, w) if with_width else (x, z) for x, z, w, _ in feet[:2]]


# ------------------------------------------------------------------ camera
@dataclass
class CameraModel:
    """Pinhole camera with OpenCV plumb_bob distortion and a body-frame mount.

    rotation_cam_to_body maps optical-frame vectors (x right, y down, z
    forward) to body FRD; translation_cam_body is the optical centre in body.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist: tuple = (0.0, 0.0, 0.0, 0.0, 0.0)
    rotation_cam_to_body: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation_cam_body: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        d = list(self.dist) + [0.0] * (5 - len(self.dist))
        self.dist = tuple(float(v) for v in d[:5])
        self.rotation_cam_to_body = np.asarray(self.rotation_cam_to_body, dtype=float)
        self.translation_cam_body = np.asarray(self.translation_cam_body, dtype=float)
        # Barrel distortion folds back on itself beyond some radius; points
        # past it would project to the wrong place, so they are rejected.
        k1, k2, _, _, k3 = self.dist
        r = np.linspace(0.0, 3.0, 3001)
        slope = 1 + 3 * k1 * r ** 2 + 5 * k2 * r ** 4 + 7 * k3 * r ** 6
        bad = np.flatnonzero(slope < 0.05)
        self.max_undistorted_radius = float(r[bad[0]] * 0.97) if len(bad) else 3.0

    def scaled_to(self, width, height):
        """Same camera at another image resolution (intrinsics scale, distortion doesn't)."""
        if (int(width), int(height)) == (self.width, self.height):
            return self
        sx, sy = width / self.width, height / self.height
        return replace(
            self, fx=self.fx * sx, fy=self.fy * sy, cx=(self.cx + 0.5) * sx - 0.5,
            cy=(self.cy + 0.5) * sy - 0.5, width=int(width), height=int(height),
        )

    def _distort(self, x, y):
        k1, k2, p1, p2, k3 = self.dist
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return xd, yd

    def project(self, points_cam):
        """Pixel coordinates (N, 2) and a validity mask for optical-frame points."""
        p = np.asarray(points_cam, dtype=float).reshape(-1, 3)
        z = p[:, 2]
        ok = z > 1e-6
        zs = np.where(ok, z, 1.0)
        x, y = p[:, 0] / zs, p[:, 1] / zs
        ok &= np.hypot(x, y) <= self.max_undistorted_radius
        xd, yd = self._distort(x, y)
        uv = np.column_stack([self.fx * xd + self.cx, self.fy * yd + self.cy])
        return uv, ok

    def pixel_rays(self, u, v):
        """Unit optical-frame rays through pixel coordinates, plus a validity mask."""
        xd = (np.asarray(u, dtype=float) - self.cx) / self.fx
        yd = (np.asarray(v, dtype=float) - self.cy) / self.fy
        x, y = xd.copy(), yd.copy()
        k1, k2, p1, p2, k3 = self.dist
        for _ in range(40):
            r2 = x * x + y * y
            radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
            dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
            dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
            radial = np.where(np.abs(radial) < 1e-6, 1e-6, radial)
            x, y = (xd - dx) / radial, (yd - dy) / radial
        xr, yr = self._distort(x, y)
        ok = (np.hypot(xr - xd, yr - yd) * self.fx < 0.05) & (np.hypot(x, y) <= self.max_undistorted_radius)
        rays = np.stack([x, y, np.ones_like(x)], axis=-1)
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        return rays, ok

    def body_to_cam(self, points_body):
        p = np.asarray(points_body, dtype=float).reshape(-1, 3)
        return (self.rotation_cam_to_body.T @ (p - self.translation_cam_body).T).T

    def cam_to_body(self, points_cam):
        p = np.asarray(points_cam, dtype=float).reshape(-1, 3)
        return self.translation_cam_body + (self.rotation_cam_to_body @ p.T).T

    def expected_down(self):
        """Body +z (down) in the optical frame, valid when the drone is level."""
        return self.rotation_cam_to_body.T @ np.array([0.0, 0.0, 1.0])


def modalai_intrinsic_xyz_matrix(roll, pitch, yaw):
    """Same convention as perception_detection.py and /etc/modalai/extrinsics.conf."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rx @ ry @ rz


def starling_cameras(config=None, dist=None):
    """Detection camera and ToF mount for the Starling 2 in this package.

    config: the semantic_obstacle_perception ros__parameters dict, or None for
    the values shipped in config/llm_vision_planner.yaml. dist: plumb_bob
    (k1, k2, p1, p2, k3); default from
    calibration/opencv_hires_small_grey_intrinsics.yml.
    """
    c = dict(
        hires_fx=501.5315609739347, hires_fy=502.8286520347511,
        hires_cx=508.1806484040712, hires_cy=380.6556051674785,
        hires_width=1024, hires_height=768,
        detection_cam_body_x_m=0.068, detection_cam_body_y_m=0.012, detection_cam_body_z_m=-0.015,
        detection_cam_roll_deg=0.0, detection_cam_pitch_deg=90.0, detection_cam_yaw_deg=90.0,
        depth_cam_body_x_m=0.066, depth_cam_body_y_m=0.009, depth_cam_body_z_m=-0.012,
        depth_cam_roll_deg=0.0, depth_cam_pitch_deg=90.0, depth_cam_yaw_deg=180.0,
    )
    c.update(config or {})
    if dist is None:
        dist = (-0.28655908953387943, 0.081789942250115405, 0.00052374055289612631,
                0.00071156224551124046, -0.010172420045783252)
    camera = CameraModel(
        fx=c["hires_fx"], fy=c["hires_fy"], cx=c["hires_cx"], cy=c["hires_cy"],
        width=int(c["hires_width"]), height=int(c["hires_height"]), dist=dist,
        rotation_cam_to_body=modalai_intrinsic_xyz_matrix(
            math.radians(c["detection_cam_roll_deg"]), math.radians(c["detection_cam_pitch_deg"]),
            math.radians(c["detection_cam_yaw_deg"])),
        translation_cam_body=np.array([c["detection_cam_body_x_m"], c["detection_cam_body_y_m"],
                                       c["detection_cam_body_z_m"]]),
    )
    tof_rotation = modalai_intrinsic_xyz_matrix(
        math.radians(c["depth_cam_roll_deg"]), math.radians(c["depth_cam_pitch_deg"]),
        math.radians(c["depth_cam_yaw_deg"]))
    tof_translation = np.array([c["depth_cam_body_x_m"], c["depth_cam_body_y_m"], c["depth_cam_body_z_m"]])
    return camera, tof_rotation, tof_translation


def tof_to_camera(points_tof, tof_rotation, tof_translation, camera):
    """Raw /tof_pc points (ToF optical frame) -> detection-camera optical frame."""
    p = np.asarray(points_tof, dtype=float).reshape(-1, 3)
    p = p[np.all(np.isfinite(p), axis=1)]
    return camera.body_to_cam(tof_translation + (tof_rotation @ p.T).T)


# --------------------------------------------------------------- alignment
def small_rotation(rx_deg, ry_deg, rz_deg):
    """Rotation about the camera's x (tilts the view), y (pans left/right) and
    z (rolls about the optical axis) axes, applied as Rz @ Ry @ Rx."""
    rx, ry, rz = (math.radians(float(a)) for a in (rx_deg, ry_deg, rz_deg))
    mx = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(rx), -math.sin(rx)], [0.0, math.sin(rx), math.cos(rx)]])
    my = np.array([[math.cos(ry), 0.0, math.sin(ry)], [0.0, 1.0, 0.0], [-math.sin(ry), 0.0, math.cos(ry)]])
    mz = np.array([[math.cos(rz), -math.sin(rz), 0.0], [math.sin(rz), math.cos(rz), 0.0], [0.0, 0.0, 1.0]])
    return mz @ my @ mx


def fit_tof_alignment(mask, camera, points_cam, floor=None, max_deg=5.0, min_points=200):
    """Small ToF-to-camera rotation that best lines the ToF up with an outline.

    The outline comes from the image alone, so it shows where the object really
    is in the camera. ToF points at the object's range should project inside
    it and nearer/farther points outside it (floor points are left out: they
    touch the object at its base). The rotation maximising that agreement
    (F1 score) is a correction to the camera/ToF mount: apply it as
    points_cam @ rotation.T. A 1 degree error is ~9 px in the hires image and
    3.5 cm at 2 m. Returns None when the outline has too few ToF points.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (camera.height, camera.width):
        camera = camera.scaled_to(mask.shape[1], mask.shape[0])
    vs, us = np.nonzero(mask)
    if not len(us):
        return None
    pts = np.asarray(points_cam, dtype=float).reshape(-1, 3)
    pts = pts[np.all(np.isfinite(pts), axis=1) & (pts[:, 2] > 0.05)]
    uv, ok = camera.project(pts)
    pad = 90
    region = ok & (uv[:, 0] > us.min() - pad) & (uv[:, 0] < us.max() + pad) \
        & (uv[:, 1] > vs.min() - pad) & (uv[:, 1] < vs.max() + pad)
    if floor is not None:
        region &= floor.distance_above(pts) > 0.05
    P = pts[region]
    r = np.linalg.norm(P, axis=1)

    def inside(points):
        q_uv, q_ok = camera.project(points)
        ui, vi = np.round(q_uv[:, 0]).astype(int), np.round(q_uv[:, 1]).astype(int)
        q_ok &= (ui >= 0) & (ui < mask.shape[1]) & (vi >= 0) & (vi < mask.shape[0])
        hit = np.zeros(len(points), dtype=bool)
        hit[q_ok] = mask[vi[q_ok], ui[q_ok]]
        return hit

    first = inside(P)
    if first.sum() < 0.5 * min_points:
        return None
    r_obj = float(np.median(r[first]))
    obj = (r > r_obj - 0.40) & (r < r_obj + 0.60)
    if obj.sum() < min_points:
        return None

    def f1(angles):
        hit = inside(P @ small_rotation(*angles).T)
        tp, fn, fp = (obj & hit).sum(), (obj & ~hit).sum(), (~obj & hit).sum()
        return 2.0 * tp / max(1, 2 * tp + fn + fp)

    scores = {}

    def visit(angles):
        key = tuple(round(float(a), 2) for a in angles)
        if key not in scores:
            scores[key] = f1(key)
        return scores[key]

    for rx in np.arange(-max_deg, max_deg + 1e-9, 0.5):
        for ry in np.arange(-max_deg, max_deg + 1e-9, 0.5):
            visit((rx, ry, 0.0))
    for _ in range(2):
        bx, by, bz = max(scores, key=scores.get)
        for rx in np.arange(bx - 0.5, bx + 0.51, 0.1):
            for ry in np.arange(by - 0.5, by + 0.51, 0.1):
                for rz in np.arange(bz - 2.0, bz + 2.01, 0.5):
                    visit((rx, ry, rz))
    best = max(scores.values())
    # The score is flat near its peak: report the middle of the plateau.
    plateau = np.array([k for k, v in scores.items() if v >= best - 0.003])
    angles = plateau.mean(axis=0)
    rotation = small_rotation(*angles)
    c = P[obj].mean(axis=0)
    shift = camera.project((rotation @ c)[None])[0][0] - camera.project(c[None])[0][0]
    return dict(rotation=rotation, angles_deg=[round(float(a), 2) for a in angles],
                f1_before=round(f1((0.0, 0.0, 0.0)), 3), f1_after=round(f1(tuple(angles)), 3),
                shift_px=[round(float(a), 1) for a in shift], n_points=int(obj.sum()),
                object_range_m=round(r_obj, 2))


def mount_with_correction(roll_deg, pitch_deg, yaw_deg, rotation):
    """Detection-camera mount angles (modalai convention, degrees) that apply
    `rotation` (a fit_tof_alignment correction) to every ToF point seen in the
    camera. Changing the camera mount realigns the ToF with the image without
    moving any ToF point in the body frame. Returns (roll, pitch, yaw, error)."""
    r_cb = modalai_intrinsic_xyz_matrix(*(math.radians(a) for a in (roll_deg, pitch_deg, yaw_deg)))
    target = r_cb @ np.asarray(rotation).T
    original = np.array([roll_deg, pitch_deg, yaw_deg], dtype=float)

    def wrap(a):
        return (np.asarray(a) + 180.0) % 360.0 - 180.0

    solutions = []
    for d_roll in range(-180, 180, 30):   # Gauss-Newton on the matrix entries from several starts
        x = np.radians(original + [d_roll, -0.5, -d_roll])
        for _ in range(100):
            f = (modalai_intrinsic_xyz_matrix(*x) - target).ravel()
            jac = np.empty((9, 3))
            for j in range(3):
                dx = x.copy()
                dx[j] += 1e-7
                jac[:, j] = ((modalai_intrinsic_xyz_matrix(*dx) - target).ravel() - f) / 1e-7
            step = np.linalg.lstsq(jac, -f, rcond=None)[0]
            x = x + step
            if np.max(np.abs(step)) < 1e-12:
                break
        err = float(np.abs(modalai_intrinsic_xyz_matrix(*x) - target).max())
        solutions.append((err, wrap(np.degrees(x))))
    best_err = min(e for e, _ in solutions)
    exact = [a for e, a in solutions if e <= max(1e-9, 10 * best_err)]
    # Of the equivalent angle sets, the one closest to the current values.
    deg = min(exact, key=lambda a: float(np.abs(wrap(a - original)).sum()))
    return float(deg[0]), float(deg[1]), float(deg[2]), best_err


# ------------------------------------------------------------------- floor
@dataclass
class FloorPlane:
    normal: np.ndarray   # unit, camera frame, pointing from the camera to the floor
    height: float        # camera height above the floor [m]
    inliers: int
    rms: float
    source: str = "tof_ransac"

    def distance_above(self, points_cam):
        """Height of points above the floor [m]."""
        return self.height - np.asarray(points_cam, dtype=float) @ self.normal


def fit_floor_plane(points_cam, expected_down, max_tilt_deg=25.0, inlier_tol=0.02,
                    min_height=0.03, iterations=400, sample=8000, seed=0):
    """RANSAC floor plane from ToF points, preferring the lowest large plane.

    The lowest plane is chosen among those with at least half the best support,
    so a big tabletop or chair seat isn't mistaken for the floor.
    """
    p = np.asarray(points_cam, dtype=float).reshape(-1, 3)
    down = np.asarray(expected_down, dtype=float)
    down = down / np.linalg.norm(down)
    p = p[np.all(np.isfinite(p), axis=1)]
    p = p[p @ down > min_height]
    if len(p) < 50:
        return None
    rng = np.random.default_rng(seed)
    if len(p) > sample:
        p = p[rng.choice(len(p), sample, replace=False)]
    cos_tilt = math.cos(math.radians(max_tilt_deg))
    candidates = []
    for _ in range(iterations):
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        if n @ down < 0:
            n = -n
        if n @ down < cos_tilt:
            continue
        d = float(n @ a)
        if d < min_height:
            continue
        count = int(np.sum(np.abs(p @ n - d) < inlier_tol))
        candidates.append((count, d, n))
    if not candidates:
        return None
    count, d, n = max(candidates, key=lambda c: c[0])
    # A clearly lower surface with comparable support is the floor; the best
    # plane may be a tabletop. Small offsets are just RANSAC noise.
    lower = [c for c in candidates if c[0] >= 0.5 * count and c[1] > d + 0.10]
    if lower:
        count, d, n = max(lower, key=lambda c: c[0])
    for tol in (3 * inlier_tol, 2 * inlier_tol, inlier_tol, 0.75 * inlier_tol):
        inl = p[np.abs(p @ n - d) < tol]   # least-squares refinement, shrinking band
        if len(inl) < 3:
            break
        centroid = inl.mean(axis=0)
        n_new = np.linalg.svd(inl - centroid, full_matrices=False)[2][2]   # never the N x N U matrix
        n = n_new if n_new @ down > 0 else -n_new
        d = float(n @ centroid)
    resid = p @ n - d
    inl = np.abs(resid) < inlier_tol
    if inl.sum() < 30:
        return None
    return FloorPlane(normal=n, height=d, inliers=int(inl.sum()), rms=float(np.sqrt(np.mean(resid[inl] ** 2))))


@dataclass
class LevelFrame:
    """Gravity-aligned frame at the camera: x' right, y' down, z' toward the object."""

    rotation: np.ndarray   # rows are the level axes in camera coordinates
    height: float          # camera height above floor, or nan

    def to_level(self, points_cam):
        return np.asarray(points_cam, dtype=float).reshape(-1, 3) @ self.rotation.T

    def to_cam(self, points_level):
        return np.asarray(points_level, dtype=float).reshape(-1, 3) @ self.rotation


def make_level_frame(down_cam, bearing_cam, height=float("nan")):
    y = np.asarray(down_cam, dtype=float)
    y = y / np.linalg.norm(y)
    z = np.asarray(bearing_cam, dtype=float)
    z = z - (z @ y) * y
    z = z / np.linalg.norm(z)
    x = np.cross(y, z)
    return LevelFrame(rotation=np.vstack([x, y, z]), height=float(height))


# -------------------------------------------------------------------- mask
def mask_row_edges(mask, center_col):
    """Per mask row: outer edges, and edges of the run containing center_col.

    Edge values are pixel boundaries (column -/+ 0.5). The central run excludes
    arms separated from the torso by a visible gap and neighbouring clutter.
    Returns rows, outer_left, outer_right, run_left, run_right (run NaN when
    the row has no run near the centre column).
    """
    rows, ol, orr, rl, rr = [], [], [], [], []
    for v in np.flatnonzero(mask.any(axis=1)):
        cols = np.flatnonzero(mask[v])
        breaks = np.flatnonzero(np.diff(cols) > 1)
        starts = np.concatenate([[cols[0]], cols[breaks + 1]])
        ends = np.concatenate([cols[breaks], [cols[-1]]])
        rows.append(v)
        ol.append(cols[0] - 0.5)
        orr.append(cols[-1] + 0.5)
        inside = np.flatnonzero((starts <= center_col) & (ends >= center_col))
        if len(inside):
            i = inside[0]
        else:
            gap = np.minimum(np.abs(starts - center_col), np.abs(ends - center_col))
            i = int(np.argmin(gap))
            if gap[i] > 3:
                rl.append(np.nan)
                rr.append(np.nan)
                continue
        rl.append(starts[i] - 0.5)
        rr.append(ends[i] + 0.5)
    return (np.array(rows, dtype=float), np.array(ol), np.array(orr), np.array(rl), np.array(rr))


def mask_bottom_boundary(mask):
    """(u, v, lowest) of the lower edge of bottom-boundary pixels; lowest marks
    the lowest outline pixel of its column.

    An isolated bottom pixel is usually a step on a near-vertical side edge,
    not a place where the object meets the floor, so it needs a neighbour in
    the next column, level or one row up/down. One row up/down keeps steep
    receding edges (the bottom of a side face seen at an angle).
    """
    below = np.zeros_like(mask)
    below[:-1] = mask[1:]
    bottom = mask & ~below
    bottom[-1] = False   # the image border is not a contact
    paired = np.zeros_like(bottom)
    for dv in (-1, 0, 1):
        shifted = np.roll(bottom, dv, axis=0)
        if dv == 1:
            shifted[0] = False
        elif dv == -1:
            shifted[-1] = False
        paired[:, 1:] |= shifted[:, :-1]
        paired[:, :-1] |= shifted[:, 1:]
    # The lowest outline pixel of every column also counts: it traces receding
    # bottom edges too steep for the neighbour test (a narrow side face).
    lowest = np.zeros_like(bottom)
    cols = np.flatnonzero(mask.any(axis=0))
    rows = mask.shape[0] - 1 - np.argmax(mask[::-1, cols], axis=0)
    keep = rows < mask.shape[0] - 1
    lowest[rows[keep], cols[keep]] = True
    v, u = np.nonzero((bottom & paired) | lowest)
    return u.astype(float), v.astype(float) + 0.5, lowest[v, u]


def shift_mask(mask, pixels):
    """Grow (pixels > 0) or shrink (< 0) a mask by whole pixels (4-neighbour)."""
    m = mask.copy()
    for _ in range(abs(int(pixels))):
        s = m.copy()
        if pixels > 0:
            s[1:] |= m[:-1]; s[:-1] |= m[1:]; s[:, 1:] |= m[:, :-1]; s[:, :-1] |= m[:, 1:]
        else:
            s[1:] &= m[:-1]; s[:-1] &= m[1:]; s[:, 1:] &= m[:, :-1]; s[:, :-1] &= m[:, 1:]
            s[0] = s[-1] = False; s[:, 0] = s[:, -1] = False
        m = s
    return m


def smooth_mask(mask, pixels=1):
    """Morphological open then close: removes 1-pixel spurs and pinholes.

    Ragged outline edges otherwise create false floor contacts, because every
    step on a side edge looks like a short bottom boundary.
    """
    if pixels <= 0:
        return mask
    opened = shift_mask(shift_mask(mask, -pixels), pixels)
    return shift_mask(shift_mask(opened, pixels), -pixels)


# ------------------------------------------------------- ellipse with rays
def _ellipse_slice_fit(u_left, u_right, xz, b_init, sigma_edge, sigma_tof, b_prior,
                       sigma_b_prior=0.25, sigma_shift=0.02):
    """Fit an ellipse to one horizontal slice from tangent rays and ToF points.

    Rays pass through the camera (origin) as x = u z. A line through the origin
    with unit normal n (pointing at the object) is tangent to the axis-aligned
    ellipse with centre c and semi-axes (a, b) when n.c = sqrt(a^2 nx^2 + b^2 nz^2).
    ToF points use the Sampson distance to the ellipse, front half only.
    Parameters: cx, cz, log a, log b, and a small lateral ToF-to-camera shift
    that absorbs extrinsic error. Returns None when the data can't support a fit.
    """
    xz = np.asarray(xz, dtype=float)
    if len(xz) < 8 or len(u_left) == 0 or len(u_right) == 0:
        return None
    z_front = float(np.percentile(xz[:, 1], 3))
    zc0 = z_front + b_init
    xl, xr = float(np.mean(u_left)) * zc0, float(np.mean(u_right)) * zc0
    if xr - xl < 0.06:
        return None
    nl = np.column_stack([np.ones_like(u_left), -u_left]) / np.sqrt(1 + u_left ** 2)[:, None]
    nr = np.column_stack([-np.ones_like(u_right), u_right]) / np.sqrt(1 + u_right ** 2)[:, None]
    normals = np.vstack([nl, nr])
    params = np.array([(xl + xr) / 2, zc0, math.log((xr - xl) / 2), math.log(b_init), 0.0])

    def unpack(p):
        return p[0], p[1], math.exp(p[2]), math.exp(p[3]), p[4]

    def residuals(p, sel, w):
        cx, cz, a, b, shift = unpack(p)
        support = np.sqrt(a * a * normals[:, 0] ** 2 + b * b * normals[:, 1] ** 2)
        r_edge = (normals @ np.array([cx, cz]) - support) / sigma_edge
        x = xz[sel, 0] - shift - cx
        z = xz[sel, 1] - cz
        f = (x / a) ** 2 + (z / b) ** 2 - 1.0
        grad = 2.0 * np.hypot(x / a ** 2, z / b ** 2)
        r_tof = np.sqrt(w) * f / np.maximum(grad, 1e-9) / sigma_tof
        r_prior = np.array([shift / sigma_shift, (b - b_prior) / sigma_b_prior])
        return np.concatenate([r_edge, r_tof, r_prior])

    weights = np.ones(len(xz))
    sel = np.ones(len(xz), dtype=bool)
    for _ in range(4):  # outer loop: front-point selection and Huber weights
        cx, cz, a, b, shift = unpack(params)
        sel = (xz[:, 1] < cz) & (np.abs(xz[:, 0] - shift - cx) < 0.92 * a)
        if sel.sum() < 6:
            return None
        w = weights[sel]
        lam = 1e-3
        cost = float(np.sum(residuals(params, sel, w) ** 2))
        for _ in range(60):  # Levenberg-Marquardt, numeric Jacobian
            r = residuals(params, sel, w)
            jac = np.empty((len(r), len(params)))
            for j in range(len(params)):
                step = 1e-6 * max(1.0, abs(params[j]))
                dp = params.copy()
                dp[j] += step
                jac[:, j] = (residuals(dp, sel, w) - r) / step
            jtj = jac.T @ jac
            g = jac.T @ r
            improved = False
            for _ in range(10):
                try:
                    delta = np.linalg.solve(jtj + lam * np.diag(np.diag(jtj) + 1e-9), -g)
                except np.linalg.LinAlgError:
                    lam *= 10
                    continue
                trial = params + delta
                trial[2:4] = np.clip(trial[2:4], math.log(0.01), math.log(2.0))
                new_cost = float(np.sum(residuals(trial, sel, w) ** 2))
                if new_cost < cost:
                    params, cost, lam, improved = trial, new_cost, lam * 0.3, True
                    break
                lam *= 10
            if not improved or np.max(np.abs(delta)) < 1e-7:
                break
        cx, cz, a, b, shift = unpack(params)
        x = xz[:, 0] - shift - cx
        z = xz[:, 1] - cz
        dist = np.abs(((x / a) ** 2 + (z / b) ** 2 - 1.0) / np.maximum(2 * np.hypot(x / a ** 2, z / b ** 2), 1e-9))
        k = 1.345 * max(sigma_tof, 1.4826 * float(np.median(dist[sel])))
        weights = np.where(dist <= k, 1.0, k / np.maximum(dist, 1e-9))

    cx, cz, a, b, shift = unpack(params)
    support = np.sqrt(a * a * normals[:, 0] ** 2 + b * b * normals[:, 1] ** 2)
    edge_rms = float(np.sqrt(np.mean((normals @ np.array([cx, cz]) - support) ** 2)))
    front = xz[sel]
    x = front[:, 0] - shift - cx
    z = front[:, 1] - cz
    tof_res = ((x / a) ** 2 + (z / b) ** 2 - 1.0) / np.maximum(2 * np.hypot(x / a ** 2, z / b ** 2), 1e-9)
    tof_rms = float(np.sqrt(np.mean(tof_res ** 2)))
    # Is the front measurably curved? A straight front (box face, chair seat
    # edge) carries no depth information, however well an ellipse fits it.
    line = np.polyfit(front[:, 0], front[:, 1], 1)
    line_rms = float(np.sqrt(np.mean((np.polyval(line, front[:, 0]) - front[:, 1]) ** 2)))
    return dict(cx=cx, cz=cz, a=a, b=b, shift=shift, edge_rms=edge_rms, tof_rms=tof_rms,
                line_rms=line_rms, n_tof=int(sel.sum()), n_left=len(u_left), n_right=len(u_right))


# ------------------------------------------------------------- footprint
def fit_rectangle(points, d0=0.01, keep_points=None, extra_angles=(), support_tol=0.06, trim=1.0):
    """Rectangle around 2-D points whose sides best line up with the points.

    The orientation maximises the closeness score of Zhang et al. (2017,
    "Efficient L-shape fitting for vehicle detection using laser scanners"):
    sum over points of 1 / max(distance to the nearest side, d0). Unlike the
    minimum-area rectangle, this is unambiguous for an L-shape (two faces of a
    box), whose minimum-area rectangles tie between two orientations.
    Bounds use the 1st-99th percentile of `points` (stray ToF returns) but
    always include every one of `keep_points` (already-filtered contacts).
    extra_angles (radians) are also tried, e.g. lines between chair legs; the
    candidate with the most observed corners wins, then the closeness score.
    Returns corners (4, 2) in order around the rectangle, or None.
    """
    p = np.asarray(points, dtype=float)
    k = np.empty((0, 2)) if keep_points is None or len(keep_points) == 0 else np.asarray(keep_points, float)
    if len(p) + len(k) < 4:
        return None

    def bounds(values, kept):
        lo, hi = (np.percentile(values, [trim, 100 - trim]) if len(values) else (np.inf, -np.inf))
        if len(kept):
            lo, hi = min(lo, kept.min()), max(hi, kept.max())
        return lo, hi

    allp = np.vstack([p, k])

    def frame(theta):
        e1 = np.array([math.cos(theta), math.sin(theta)])
        e2 = np.array([-e1[1], e1[0]])
        (a0, a1), (b0, b1) = bounds(p @ e1, k @ e1), bounds(p @ e2, k @ e2)
        return e1, e2, a0, a1, b0, b1

    def score(theta):
        e1, e2, a0, a1, b0, b1 = frame(theta)
        c1, c2 = allp @ e1, allp @ e2
        d1 = np.abs(np.minimum(c1 - a0, a1 - c1))
        d2 = np.abs(np.minimum(c2 - b0, b1 - c2))
        return float(np.sum(1.0 / np.maximum(np.minimum(d1, d2), d0)))

    def corners(theta):
        e1, e2, a0, a1, b0, b1 = frame(theta)
        return np.array([a0 * e1 + b0 * e2, a1 * e1 + b0 * e2, a1 * e1 + b1 * e2, a0 * e1 + b1 * e2])

    coarse = np.radians(np.arange(0.0, 90.0, 1.0))
    best = coarse[int(np.argmax([score(t) for t in coarse]))]
    fine = best + np.radians(np.arange(-1.0, 1.01, 0.1))
    theta = fine[int(np.argmax([score(t) for t in fine]))]
    if len(extra_angles):
        cands = [theta] + [float(a) % (math.pi / 2) for a in extra_angles]
        ranked = [(int(rectangle_support(allp, corners(t), support_tol).sum()), score(t), t) for t in cands]
        theta = max(ranked)[2]
    return corners(theta)


def rectangle_support(points, rect, tol=0.06):
    """Which rectangle corners have observed points nearby. Three supported
    corners pin the fourth (hidden) one for a rectangular footprint."""
    d = np.linalg.norm(np.asarray(points)[:, None, :] - rect[None, :, :], axis=2)
    return np.any(d < tol, axis=0)


@dataclass
class FrontFace:
    """The rectangle side facing the camera, in level (x', z') coordinates."""
    p0: np.ndarray       # a point on the front line
    along: np.ndarray    # unit vector along the face
    back: np.ndarray     # unit normal pointing away from the camera, into the object
    s0: float            # lateral span of the face along `along`, relative to p0
    s1: float
    planarity: float     # fraction of ToF footprint points within 3 cm of the line

    def depth_of(self, pts):
        return (np.asarray(pts, float) - self.p0) @ self.back

    def corners(self, depth):
        return np.array([self.p0 + s * self.along + e * self.back
                         for s in (self.s0, self.s1) for e in (0.0, depth)])


def front_face(points_xz, rect, tol=0.03):
    """Pick the side of the fitted rectangle that faces the camera (origin)."""
    if rect is None:
        return None
    center = rect.mean(axis=0)
    best = None
    for i in range(4):
        a, c = rect[i], rect[(i + 1) % 4]
        length = float(np.linalg.norm(c - a))
        if length < 0.1:
            continue
        along = (c - a) / length
        normal = np.array([-along[1], along[0]])
        mid = 0.5 * (a + c)
        if normal @ (mid - center) < 0:
            normal = -normal           # outward
        facing = float(normal @ (-mid / max(np.linalg.norm(mid), 1e-9)))
        if best is None or facing > best[0]:
            best = (facing, a, along, -normal)
    if best is None:
        return None
    _, a, along, back = best
    p = np.asarray(points_xz, float)
    e = (p - a) @ back
    on = np.abs(e) < tol
    s = (p - a) @ along
    if on.sum() >= 10:
        s0, s1 = np.percentile(s[on], [2, 98])
    else:
        span = (rect - a) @ along
        s0, s1 = span.min(), span.max()
    return FrontFace(p0=a, along=along, back=back, s0=float(s0), s1=float(s1),
                     planarity=float(np.mean(on)) if len(p) else 0.0)


def side_sliver_depth(face, rays_left, rays_right, margin=0.03):
    """Depth of a box from a side face seen as a sliver at the outline's edge.

    When the camera is beyond one end of the front face, that side face is
    visible and the outline's outer edge on that side is the face's far
    vertical edge. Its ray, intersected with the side face line (the end of
    the front face, perpendicular to it), gives the far corner.
    rays_*: (N, 2) horizontal level directions (x', z') of outline edge rays.
    Returns (depth, side) or (None, reason).
    """
    s_cam = float(-face.p0 @ face.along)
    if s_cam > face.s1 + margin:
        s_side, end = face.s1, "s1"
    elif s_cam < face.s0 - margin:
        s_side, end = face.s0, "s0"
    else:
        return None, "camera is within the face's width, so neither side face is visible"
    # Which image edge (left/right) holds that end of the face?
    right = (face.along[0] > 0) == (end == "s1")
    rays = rays_right if right else rays_left
    if len(rays) == 0:
        return None, "no outline edge rays"
    denom = rays @ face.along
    # A ray nearly parallel to the side face meets it at a badly defined point:
    # 1 cm of error in where the face ends would move the corner tens of cm.
    ok = np.abs(denom) > math.sin(math.radians(10.0))
    if not ok.any():
        return None, "side face seen too edge-on to measure"
    t = (s_side + face.p0 @ face.along) / np.where(ok, denom, 1.0)
    ok &= t > 0
    depth = face.depth_of(rays * t[:, None])
    ok &= depth > 0.01
    if ok.sum() < 3:
        return None, "outline edge doesn't reach the side face"
    return float(np.median(depth[ok])), ("right" if right else "left")


def top_sliver_depth(face, top_rays_level, top_plane_y):
    """Depth from the top face seen from above: outline top edge rays meet the
    top plane (level y' = top_plane_y, below the camera) at the far edge."""
    r = np.asarray(top_rays_level, float)
    ok = r[:, 1] > 1e-4
    if ok.sum() < 3 or top_plane_y <= 0:
        return None
    t = top_plane_y / r[ok, 1]
    pts = r[ok][:, [0, 2]] * t[:, None]
    s = (pts - face.p0) @ face.along
    inside = (s > face.s0 + 0.1 * (face.s1 - face.s0)) & (s < face.s1 - 0.1 * (face.s1 - face.s0))
    if inside.sum() < 3:
        return None
    return float(np.median(face.depth_of(pts[inside])))


def on_two_lines(points, tol=0.025, frac=0.9, iterations=300, seed=0):
    """True when the points lie along one or two straight lines (the bottom
    edges of one or two box faces), rather than at separate legs or casters."""
    p = np.asarray(points, float)
    if len(p) < 6:
        return True
    rng = np.random.default_rng(seed)

    def line_inliers(a, b, q):
        d = b - a
        n = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-9)
        return np.abs((q - a) @ n) < tol

    def best_line(q):
        found = None
        for _ in range(iterations):
            i, j = rng.choice(len(q), 2, replace=False)
            if np.linalg.norm(q[i] - q[j]) < 0.05:
                continue
            inl = line_inliers(q[i], q[j], q)
            if found is None or inl.sum() > found[0].sum():
                found = (inl, q[j] - q[i])
        return found

    first = best_line(p)
    if first is None:
        return True
    rest = p[~first[0]]
    if len(rest) < 3 or first[0].mean() >= frac:
        return True
    second = best_line(rest)
    if second is None or first[0].sum() + second[0].sum() < frac * len(p):
        return False
    # Two box faces meet at a corner; two parallel rows are front and rear legs.
    a, b = first[1] / np.linalg.norm(first[1]), second[1] / np.linalg.norm(second[1])
    return abs(a @ b) < math.cos(math.radians(45.0))


def footprint_closed(foot, contacts, rect, min_sliver_deg=1.2):
    """Does the observed evidence show the back of a footprint rectangle?

    Two ways, both in level (x', z') coordinates (camera at the origin):
      rear legs   floor contacts at the back edge, separated by a gap from the
                  contacts in front of them (legs, casters), not just the far
                  end of one receding bottom edge.
      a side      points along one side for at least half the depth, where that
                  side spans at least min_sliver_deg as seen from the camera.
                  A side seen more edge-on than that may be cut short by the
                  pixel grid, so its far end isn't trusted.
    Otherwise the back edge is just where visibility stopped: a camera-facing
    surface or a side seen too edge-on, with unknown material behind it.
    Returns (closed, reason).
    """
    face = front_face(foot, rect)
    if face is None:
        return False, "no front face"
    d_rect = float(np.max(face.depth_of(rect)))
    if d_rect < 0.08:
        return False, "only a front face was seen"
    if len(contacts):
        dc = face.depth_of(contacts)
        sc = (np.asarray(contacts, float) - face.p0) @ face.along
        rear = (np.abs(dc - d_rect) < 0.06) & (dc > 0.10)
        if rear.any():
            ahead = dc[dc < dc[rear].min()]
            gap = dc[rear].min() - (ahead.max() if len(ahead) else 0.0)
            # A leg or caster seen from the front is a short level run of
            # contacts (wide, not receding); the far end of a receding bottom
            # edge is a few pixels stepping away from the camera.
            along_span, depth_span = np.ptp(sc[rear]), np.ptp(dc[rear])
            leg_like = along_span >= 0.01 and depth_span <= max(0.06, 1.5 * along_span + 0.02)
            if gap >= 0.08 and leg_like:
                return True, "rear legs/contacts reach the back"
    best = 0.0
    for end, sign in ((face.s0, -1.0), (face.s1, 1.0)):
        near = np.abs((foot - face.p0) @ face.along - end) < 0.05
        dep = face.depth_of(foot[near])
        bins = np.unique(np.floor(dep[(dep > 0.05) & (dep < d_rect - 0.02)] / 0.05))
        if len(bins) < 2 or len(bins) * 0.05 < 0.5 * (d_rect - 0.07):
            continue
        a = face.p0 + end * face.along
        b = a + d_rect * face.back
        if sign * (-face.p0 @ face.along - end) <= 0:
            continue   # this side faces away from the camera
        ang = math.degrees(abs(math.atan2(a[0] * b[1] - a[1] * b[0], a @ b)))
        best = max(best, ang)
        if ang >= min_sliver_deg:
            return True, f"a side was seen along its depth ({ang:.1f} deg wide)"
    if best > 0:
        return False, f"side seen too edge-on ({best:.1f} deg) to trust its far end"
    return False, "back hidden behind a camera-facing surface"


def check_floor_under_contacts(contacts, clusters, level, points_cam, radius=0.20, min_points=10,
                               max_reach_m=0.5):
    """Check the floor plane where the object stands and correct contacts.

    Contacts are outline rays cut with the floor plane, so a floor that is a
    few cm lower or higher at the object (foam mats, a rug, a slightly tilted
    plane fitted near the drone) moves them by z * dh / h: 3 cm at 2 m from a
    0.5 m hover is 12 cm. ToF points close to the floor around the contacts
    give the local floor height (pass only points outside the object's
    outline: shoes and leg bottoms sit above the floor); contacts are
    rescaled to it. With too few such points and contacts well beyond the
    floor the ToF saw, the contacts are marked unverified.
    Returns (contacts, clusters, note).
    """
    if not len(contacts):
        return contacts, clusters, ""
    L = level.to_level(points_cam)
    height = level.height - L[:, 1]
    low = np.abs(height) < 0.06
    near = np.zeros(len(L), dtype=bool)
    for c in clusters:
        half = 0.5 * c["x_span"] + radius
        near |= (low & (np.abs(L[:, 0] - c["x"]) <= half)
                 & (L[:, 2] >= c["z_min"] - radius) & (L[:, 2] <= c["z_max"] + radius))
    if near.sum() >= min_points:
        dh = float(np.median(height[near]))
        if abs(dh) < 0.005:
            return contacts, clusters, ""
        scale = (level.height - dh) / level.height
        fixed = []
        for c in clusters:
            c = dict(c)
            c.update(x=c["x"] * scale, z_min=c["z_min"] * scale, z_max=c["z_max"] * scale,
                     x_span=c["x_span"] * scale)
            fixed.append(c)
        side = "below" if dh < 0 else "above"
        return contacts * scale, fixed, f"floor here {abs(dh) * 100:.1f} cm {side} the fitted plane, contacts corrected"
    on_floor = np.abs(height) < 0.02
    reach = _pct(L[on_floor, 2], 95) if on_floor.sum() >= 30 else float("nan")
    if not np.isfinite(reach) or float(np.max(contacts[:, 1])) > reach + max_reach_m:
        seen = "no floor seen" if not np.isfinite(reach) else f"ToF floor ends at {reach:.1f} m"
        return contacts, clusters, f"unverified: floor not seen near the object ({seen})"
    return contacts, clusters, ""


# ---------------------------------------------------------------- result
@dataclass
class MethodEstimate:
    name: str
    kind: str              # "estimate" (may be the full depth) or "lower_bound"
    depth: float = None
    valid: bool = False
    confidence: float = 0.0
    note: str = ""
    used: bool = False     # counted in the final (max) depth
    front: float = None    # level-frame z' of its front and back, when not
    back: float = None     # the ToF front and ToF front + depth
    range: tuple = None    # body_shape: (smallest, largest) depth over body proportions


@dataclass
class OutlineParams:
    slice_height_m: float = 0.04
    sigma_px: float = 1.0                 # outline edge noise
    edge_offset_px: float = 0.0           # grow (+) or shrink (-) the outline edges
    sigma_tof_m: float = 0.008
    floor_clearance_m: float = 0.04
    min_object_points: int = 30
    contact_cluster_m: float = 0.05
    contact_link_rows: float = 6.0        # contact pixels in neighbouring columns this many rows apart chain
    min_contact_spread_m: float = 0.10
    min_estimate_confidence: float = 0.35
    min_contact_camera_height_m: float = 0.20   # below this, floor rays graze too much
    max_depth_generic_m: float = 1.5
    use_prior_floor: bool = False         # never report less than the class prior
    edge_sensitivity: bool = True
    smooth_px: int = 1                    # open/close the outline before use
    # True when the outline was drawn from the ToF points themselves (the
    # "tof" segmenter): its edges are not independent of the ToF, so the
    # tangent-ray ellipse and floor contacts can't use them.
    outline_from_tof: bool = False
    # When one view can't measure the depth (flat face seen square-on):
    # "conservative" uses the class's largest typical depth, "typical" its usual one.
    unobserved_depth: str = "conservative"


@dataclass
class OutlineResult:
    label: str
    depth_m: float
    decided_by: str
    methods: dict
    front_z_m: float                     # level frame
    box_level: dict                      # x0, x1, z0, z1, y_top, y_bottom
    corners_cam: np.ndarray
    level: LevelFrame
    floor: FloorPlane
    alignment: float                     # fraction of nearby ToF points inside the outline
    n_object_points: int
    edge_sensitivity_m_per_px: float
    warnings: list
    geometry: dict                       # data for plots
    depth_observed: bool = True          # False: back not visible from here, class depth used


def _pct(values, q):
    return float(np.percentile(values, q)) if len(values) else float("nan")


def estimate_from_outline(label, mask, camera, points_cam, floor=None, params=None):
    """Depth along the line of sight of the object in `mask`.

    label: detector class (selects the slice band and plausible bounds).
    mask: HxW bool outline in the (distorted) camera image.
    points_cam: every ToF point of the frame in the camera optical frame.
    floor: optional FloorPlane; fitted from points_cam when None.
    Returns an OutlineResult, or None if the object can't be located.
    """
    params = params or OutlineParams()
    label = str(label).strip().lower()
    prior = object_prior(label)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (camera.height, camera.width):
        camera = camera.scaled_to(mask.shape[1], mask.shape[0])
    mask = smooth_mask(mask, params.smooth_px)
    if not mask.any():
        return None
    warnings = []
    pts = np.asarray(points_cam, dtype=float).reshape(-1, 3)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    pts = pts[np.linalg.norm(pts, axis=1) > 0.15]
    if floor is None:
        floor = fit_floor_plane(pts, camera.expected_down())
    if floor is None:
        warnings.append("no floor plane found: floor contacts disabled, level frame assumes a level drone")

    # 1. ToF points inside the outline, gated to the object's range.
    uv, ok = camera.project(pts)
    ui, vi = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    ok &= (ui >= 0) & (ui < camera.width) & (vi >= 0) & (vi < camera.height)
    in_mask = np.zeros(len(pts), dtype=bool)
    in_mask[ok] = mask[vi[ok], ui[ok]]
    above_floor = np.ones(len(pts), dtype=bool) if floor is None else (
        floor.distance_above(pts) > params.floor_clearance_m)
    ranges = np.linalg.norm(pts, axis=1)
    cand = in_mask & above_floor
    if cand.sum() < params.min_object_points:
        return None
    front_range = _pct(ranges[cand], 3)
    window = float(prior.get("max_depth", params.max_depth_generic_m)) + 0.15
    if prior is DEFAULT_PRIOR:
        window = params.max_depth_generic_m
    gated = cand & (ranges <= front_range + window)
    obj = density_filter(pts[gated])
    if len(obj) < params.min_object_points:
        return None

    # Alignment check: ToF points at the object's range near the outline's
    # bounding box should mostly land inside the outline.
    vs, us = np.nonzero(mask)
    bb = (us.min() - 5, us.max() + 5, vs.min() - 5, vs.max() + 5)
    near = ok & above_floor & (ranges >= front_range - 0.05) & (ranges <= front_range + 0.35)
    near &= (uv[:, 0] >= bb[0]) & (uv[:, 0] <= bb[1]) & (uv[:, 1] >= bb[2]) & (uv[:, 1] <= bb[3])
    alignment = float(np.mean(in_mask[near])) if near.any() else float("nan")
    if alignment < 0.7:
        warnings.append(f"only {alignment:.0%} of nearby ToF points fall inside the outline: "
                        "check camera/ToF extrinsics, timing, or the segmentation")

    # 2. Level frame toward the object.
    down = floor.normal if floor is not None else camera.expected_down()
    bearing = np.median(obj / np.linalg.norm(obj, axis=1, keepdims=True), axis=0)
    level = make_level_frame(down, bearing, floor.height if floor is not None else float("nan"))
    P = level.to_level(obj)
    z_front = _pct(P[:, 2], 3)

    # 3. Outline edges as level-frame rays.
    center_col = float(np.median(us))
    rows, ol, orr, rl, rr = mask_row_edges(mask, center_col)

    def rays_level(u, v):
        d, valid = camera.pixel_rays(u, v)
        return level.to_level(d), valid

    off = params.edge_offset_px
    ray_ol, ok_ol = rays_level(ol - off, rows)
    ray_or, ok_or = rays_level(orr + off, rows)
    ok_hull = ok_ol & ok_or & (ray_ol[:, 2] > 0) & (ray_or[:, 2] > 0)
    slope_ol = ray_ol[ok_hull, 0] / ray_ol[ok_hull, 2]
    slope_or = ray_or[ok_hull, 0] / ray_or[ok_hull, 2]
    top_ray, top_ok = rays_level(np.array([center_col]), np.array([rows.min() - 0.5]))

    methods = {}
    methods["visible_extent"] = MethodEstimate(
        "visible_extent", "lower_bound", depth=_pct(P[:, 2], 97) - z_front, valid=True,
        confidence=1.0, note="ToF front-to-back spread inside the outline")

    # 4. Silhouette-constrained ellipse slices.
    half_guess = HALF_DEPTH_GUESS.get(label, 0.5 * float(prior.get("depth", 0.4)))
    floor_y = level.height if floor is not None else _pct(P[:, 1], 99)
    top_candidates = [_pct(P[:, 1], 1)]
    if top_ok[0] and top_ray[0, 2] > 0:
        top_candidates.append(top_ray[0, 1] / top_ray[0, 2] * (z_front + half_guess))
    y_top = min(top_candidates)
    band = BAND_FOR_LABEL.get(label, DEFAULT_BAND)

    def ellipse_depth(edge_shift_px):
        good = np.isfinite(rl) & np.isfinite(rr)
        lv, ld = rays_level(rl[good] - off - edge_shift_px, rows[good])
        rv, rd = rays_level(rr[good] + off + edge_shift_px, rows[good])
        keep = ld & rd & (lv[:, 2] > 0) & (rv[:, 2] > 0)
        lv, rv = lv[keep], rv[keep]
        zc = z_front + half_guess
        ray_y = 0.5 * (lv[:, 1] / lv[:, 2] + rv[:, 1] / rv[:, 2]) * zc
        u_l, u_r = lv[:, 0] / lv[:, 2], rv[:, 0] / rv[:, 2]
        height = floor_y - y_top
        y0, y1 = y_top + band[0] * height, y_top + band[1] * height
        sigma_edge = zc * params.sigma_px / camera.fx + 0.002
        fits = []
        for ys in np.arange(y0, y1 - 1e-9, params.slice_height_m):
            in_rows = (ray_y >= ys) & (ray_y < ys + params.slice_height_m)
            in_pts = (P[:, 1] >= ys) & (P[:, 1] < ys + params.slice_height_m)
            f = _ellipse_slice_fit(u_l[in_rows], u_r[in_rows], P[in_pts][:, [0, 2]], half_guess,
                                   sigma_edge, params.sigma_tof_m, half_guess)
            if f is None:
                continue
            f["y"] = ys + 0.5 * params.slice_height_m
            curved = f["line_rms"] >= max(1.5 * f["tof_rms"], f["tof_rms"] + 0.004)
            f["accepted"] = bool(curved and f["tof_rms"] < 0.02 and f["edge_rms"] < 0.02
                                 and 0.01 < f["b"] < 1.0)
            fits.append(f)
        return fits

    fits = [] if params.outline_from_tof else ellipse_depth(0.0)
    accepted = [f for f in fits if f["accepted"]]
    sensitivity = float("nan")
    if len(accepted) >= 2:
        depths = np.array([2 * f["b"] for f in accepted])
        depth = float(np.median(depths))
        spread = float(np.median(np.abs(depths - depth))) / max(depth, 1e-6)
        rms = float(np.median([f["tof_rms"] for f in accepted]))
        conf = min(1.0, len(accepted) / 4) * math.exp(-rms / 0.015) * (1.0 if spread < 0.15 else 0.5)
        if params.edge_sensitivity:
            d_plus = [2 * f["b"] for f in ellipse_depth(1.0) if f["accepted"]]
            d_minus = [2 * f["b"] for f in ellipse_depth(-1.0) if f["accepted"]]
            if d_plus and d_minus:
                sensitivity = 0.5 * (float(np.median(d_plus)) - float(np.median(d_minus)))
        class_max = float(prior.get("max_depth", params.max_depth_generic_m))
        plausible = depth <= 1.1 * class_max
        methods["silhouette_ellipse"] = MethodEstimate(
            "silhouette_ellipse", "estimate", depth=depth,
            valid=conf >= params.min_estimate_confidence and plausible, confidence=conf,
            # The fitted body sits where it is, behind anything held in front of it.
            front=float(np.median([f["cz"] - f["b"] for f in accepted])),
            back=float(np.median([f["cz"] + f["b"] for f in accepted])),
            note=f"{len(accepted)} curved slices, ToF rms {rms * 100:.1f} cm"
                 + ("" if plausible else f"; rejected: deeper than any {label} ({class_max:.2f} m)"))
    else:
        methods["silhouette_ellipse"] = MethodEstimate(
            "silhouette_ellipse", "estimate",
            note="outline made from the ToF (no independent edges)" if params.outline_from_tof
            else f"{len(accepted)} usable curved slices (need 2)")

    body_ellipse = None
    # 4b. A person whose torso outline is narrower than any adult's shoulders
    # is seen side-on: their width, not their chest depth, is along the line
    # of sight, which the front curvature can't show.
    if label in BODY_LABELS and not params.outline_from_tof:
        good = np.isfinite(rl) & np.isfinite(rr)
        lv, ld = rays_level(rl[good] - off, rows[good])
        rv, rd = rays_level(rr[good] + off, rows[good])
        keep = ld & rd & (lv[:, 2] > 0) & (rv[:, 2] > 0)
        lv, rv = lv[keep], rv[keep]
        height_b = floor_y - y_top
        # Widths are measured at the torso's own range: something held out in
        # front (a phone, a controller) must not make the body look narrower.
        torso = (P[:, 1] >= y_top + band[0] * height_b) & (P[:, 1] <= y_top + band[1] * height_b)
        torso_front = _pct(P[torso, 2], 5) if torso.sum() >= 10 else z_front
        zc = torso_front + half_guess
        ray_y = 0.5 * (lv[:, 1] / lv[:, 2] + rv[:, 1] / rv[:, 2]) * zc
        widths = (rv[:, 0] / rv[:, 2] - lv[:, 0] / lv[:, 2]) * zc
        in_band = (ray_y >= y_top + band[0] * height_b) & (ray_y <= y_top + band[1] * height_b)
        w_band = float(np.median(widths[in_band])) if in_band.any() else float("nan")
        class_max = float(prior.get("max_depth", params.max_depth_generic_m))
        side_on = np.isfinite(w_band) and w_band < MIN_FACING_TORSO_WIDTH_M
        methods["side_on"] = MethodEstimate(
            "side_on", "lower_bound", depth=class_max if side_on else None, valid=bool(side_on),
            front=torso_front, back=torso_front + class_max if side_on else None,
            confidence=1.0 if side_on else 0.0,
            note=(f"torso outline {w_band:.2f} m wide, narrower than an adult facing the drone: "
                  f"likely side-on, so up to {class_max:.2f} m of body is along the line of sight")
            if side_on else f"torso outline {w_band:.2f} m wide")

        # 4c. Person-shaped ellipse that can turn: it is wider than deep, and
        # which way it faces comes from the outline (shoulders much wider than
        # the head when facing toward or away from the drone, about as wide
        # side-on) and from the slant of the ToF surface (turned).
        def outer_width(f0, f1):
            sel = (rows >= 0)
            lo, ldok = rays_level(ol[sel] - off, rows[sel])
            ro, rdok = rays_level(orr[sel] + off, rows[sel])
            okk = ldok & rdok & (lo[:, 2] > 0) & (ro[:, 2] > 0)
            yy = 0.5 * (lo[okk, 1] / lo[okk, 2] + ro[okk, 1] / ro[okk, 2]) * zc
            ww = (ro[okk, 0] / ro[okk, 2] - lo[okk, 0] / lo[okk, 2]) * zc
            pick = (yy >= y_top + f0 * height_b) & (yy <= y_top + f1 * height_b)
            return ww[pick]
        head_w, shoulder_w = outer_width(0.02, 0.10), outer_width(0.17, 0.25)
        ratio = (float(np.percentile(shoulder_w, 75)) / float(np.median(head_w))
                 if len(head_w) >= 3 and len(shoulder_w) >= 3 and np.median(head_w) > 0.05 else float("nan"))
        # Slant of the visible surface in the torso band, away from the outline
        # edges, where ToF pixels mix the body with the background.
        slope = 0.0
        y0b, y1b = y_top + band[0] * height_b, y_top + band[1] * height_b
        sl_l = float(np.median(lv[in_band, 0] / lv[in_band, 2])) if in_band.any() else float("nan")
        sl_r = float(np.median(rv[in_band, 0] / rv[in_band, 2])) if in_band.any() else float("nan")
        tb = (P[:, 1] >= y0b) & (P[:, 1] <= y1b)
        if np.isfinite(sl_l) and np.isfinite(sl_r):
            tb &= (P[:, 0] > sl_l * P[:, 2] + 0.03) & (P[:, 0] < sl_r * P[:, 2] - 0.03)
        if tb.sum() >= 15:
            x, z = P[tb, 0], P[tb, 2]
            for _ in range(2):   # least squares, then drop the worst residuals once
                A = np.column_stack([x, np.ones_like(x)])
                coef = np.linalg.lstsq(A, z, rcond=None)[0]
                res = z - A @ coef
                keep_pts = np.abs(res) <= max(0.01, 2.5 * 1.4826 * np.median(np.abs(res)))
                x, z = x[keep_pts], z[keep_pts]
            slope = float(coef[0])
        # Feet (the bottom of the outline on the floor, where the amber floor
        # contacts are): standing, they are side by side along the body's
        # left-right axis, so side by side across the view means the body faces
        # toward or away from the drone (a wide ellipse) and one behind the
        # other means side-on (a long one). This settles what the level ToF
        # surface can't: it looks the same either way. A step is different:
        # walking toward or away from the drone puts one foot well ahead of
        # the torso and the other behind it while the body still faces along
        # the walk, and walking past puts them side by side while the body is
        # side-on. So feet more than a stance apart, or a front foot well
        # ahead of the torso, don't decide (the "feet" bound below still
        # covers the legs), and neither do feet at an angle (turned, or one
        # foot a little ahead).
        feet, stance, feet_gap, feet_side, step = [], None, float("nan"), None, False
        foot_widths = []
        if floor is not None and floor.height >= params.min_contact_camera_height_m:
            found = find_feet(mask, rays_level, level.height, camera.fy, torso_front + half_guess,
                              with_width=True)
            feet, foot_widths = [(x, z) for x, z, _ in found], [w for _, _, w in found]
        if feet:
            # Same local floor correction as the contacts (foam mats, rugs).
            fa = np.array(feet)
            fc = [dict(x=float(x), z_min=float(z), z_max=float(z), x_span=0.10) for x, z in feet]
            fa, _, _ = check_floor_under_contacts(fa, fc, level, pts[~in_mask])
            feet = [(float(x), float(z)) for x, z in fa]
        if len(feet) == 2:
            (fx1, fz1), (fx2, fz2) = sorted(feet)
            feet_gap = math.hypot(fx2 - fx1, fz2 - fz1)
            if feet_gap >= MIN_FEET_SEPARATION_M:
                stance = math.degrees(math.atan2(abs(fz2 - fz1), abs(fx2 - fx1)))
                # Where the feet are relative to the torso hangs on the floor
                # height under them (1 cm off moves them ~5 cm at 2 m from a
                # 0.5 m hover; foam mats do that), so it only separates a step
                # toward the drone from side-on; feet side by side are a step
                # only when far apart.
                step = feet_gap > MAX_STANCE_WIDTH_M
                if stance >= FEET_ONE_BEHIND_DEG:
                    step = step or min(fz1, fz2) < torso_front - STEP_AHEAD_M
                    feet_side = None if step else True
                elif stance <= FEET_SIDE_BY_SIDE_DEG and not step:
                    feet_side = False
        # One foot well ahead of the torso is a step whose other foot is
        # hidden behind the front leg: the hips sit about midway between them.
        hidden_back = None
        if len(feet) == 1 and feet[0][1] < torso_front - STEP_AHEAD_M:
            step = True
            hidden_back = 2.0 * (torso_front + half_guess) - feet[0][1]
        if stance is None:
            feet_note = ("1 foot found, well ahead of the torso: a step, the other foot hidden behind"
                         if hidden_back is not None else
                         f"{len(feet)} foot found" if len(feet) == 1 else "no feet found" if not feet
                         else "feet too close together to show a direction")
        else:
            feet_note = (f"feet {'side by side across the view' if stance <= FEET_SIDE_BY_SIDE_DEG else 'one behind the other' if stance >= FEET_ONE_BEHIND_DEG else 'at an angle'}"
                         f" ({stance:.0f} deg, {feet_gap:.2f} m apart{', a step' if step else ''})")
        # Facing vs side-on both show a level surface. A torso outline narrower
        # than any adult facing the drone settles it (side-on) and nothing
        # overrules that; otherwise the feet do, and without them the
        # shoulders being about as wide as the head. The width comes first
        # because raised arms (holding a phone) widen the shoulder rows.
        # The one exception: two feet side by side, each only a shoe's width
        # across the view (pointing at or away from the drone; side-on, a
        # shoe shows its whole length), are a slim person facing the drone.
        if (side_on and feet_side is False and len(foot_widths) == 2
                and max(foot_widths) <= MAX_FOOT_WIDTH_FACING_M):
            side_on = False
            methods["side_on"].valid, methods["side_on"].confidence = False, 0.0
            methods["side_on"].depth = methods["side_on"].back = None
            methods["side_on"].note = (f"torso outline {w_band:.2f} m wide, but both feet point at or away "
                                       f"from the drone ({max(foot_widths):.2f} m across): a slim person facing")
        if side_on:
            side_branch, cue = True, "narrow torso"
        elif feet_side is not None:
            side_branch, cue = feet_side, "feet"
        else:
            side_branch = bool(np.isfinite(ratio) and ratio < SIDE_ON_SHOULDER_HEAD_RATIO)
            cue = "shoulders/head"
        if np.isfinite(w_band):
            fits_k = [body_extent(w_band, slope, k, side_branch) for k in BODY_DEPTH_RATIOS]
            e_lo, e_hi = min(f[0] for f in fits_k), max(f[0] for f in fits_k)
            k_hi, (e_used, theta_used) = max(zip(BODY_DEPTH_RATIOS, fits_k), key=lambda kf: kf[1][0])
            e_used = min(e_used, MAX_BODY_EXTENT_M)
            methods["body_shape"] = MethodEstimate(
                "body_shape", "estimate", depth=e_used, valid=True, confidence=0.6,
                front=torso_front, back=torso_front + e_used,
                note=f"outline {w_band:.2f} m wide, shoulders {ratio:.1f}x the head, surface slant "
                     f"{math.degrees(math.atan(slope)):+.0f} deg, {feet_note}: "
                     f"{'side-on' if side_branch else 'facing toward/away'} (by the {cue}), turned ~{theta_used:.0f} deg, "
                     f"{e_lo:.2f}-{e_hi:.2f} m for a body {BODY_DEPTH_RATIOS[0]}-{BODY_DEPTH_RATIOS[1]} as deep as wide")
            methods["body_shape"].range = (e_lo, min(e_hi, MAX_BODY_EXTENT_M))
            # Drawn ellipse: its extent along the view is the depth used.
            a_b = e_used / (2.0 * math.sqrt(math.sin(math.radians(theta_used)) ** 2
                                            + k_hi ** 2 * math.cos(math.radians(theta_used)) ** 2))
            body_ellipse = dict(cx=0.5 * (sl_l + sl_r) * (z_front + 0.5 * e_used) if np.isfinite(sl_l) else 0.0,
                                cz=torso_front + 0.5 * e_used, a=a_b, b=k_hi * a_b,
                                angle=theta_used if slope >= 0 else -theta_used, side_on=side_branch,
                                feet=feet, stance_deg=stance, cue=cue)
        else:
            methods["body_shape"] = MethodEstimate("body_shape", "estimate", note="no torso outline")
        # The feet are part of the person too: in a step the front one is
        # ahead of the torso and the far one behind it. The camera sees a
        # foot's near edge; it reaches a foot length behind that when it
        # points along the view (facing, walking toward or away), a foot
        # width when it points across (side-on).
        if feet:
            f_front = min(z for _, z in feet)
            f_back = max(z for _, z in feet) + (FOOT_WIDTH_M if feet_side else FOOT_LENGTH_M)
            if hidden_back is not None:
                f_back = max(f_back, hidden_back)
            methods["feet"] = MethodEstimate(
                "feet", "lower_bound", depth=f_back - min(f_front, torso_front), valid=True, confidence=1.0,
                front=min(f_front, torso_front), back=f_back,
                note=f"{feet_note}; the farther foot ends ~{f_back:.2f} m away")
        else:
            methods["feet"] = MethodEstimate("feet", "lower_bound", note=feet_note)

    # 5. Floor contacts from the bottom of the outline.
    contacts = np.empty((0, 2))
    unverified_contacts = np.empty((0, 2))
    clusters = []
    floor_note = ""
    if floor is not None and floor.height < params.min_contact_camera_height_m:
        warnings.append(f"camera only {floor.height:.2f} m above the floor: floor contacts disabled "
                        "(each pixel would span tens of cm of floor)")
    if floor is not None and floor.height >= params.min_contact_camera_height_m and not params.outline_from_tof:
        cu, cv, c_lowest = mask_bottom_boundary(mask)
        if len(cu) > 20000:
            pick = np.random.default_rng(0).choice(len(cu), 20000, replace=False)
            cu, cv, c_lowest = cu[pick], cv[pick], c_lowest[pick]
        # The contact lies somewhere within the boundary pixel and a corner can
        # be cut off by the pixel grid, so take the pixel's far (upper) edge:
        # contacts can then only err away from the camera, the safe direction.
        cv = cv - 1.0
        d, valid = rays_level(cu, cv)
        valid &= d[:, 1] > 1e-3
        t = level.height / np.where(valid, d[:, 1], 1.0)
        hit = d * t[:, None]
        # Contacts can sit behind everything the ToF saw (rear legs under a
        # seat), but not far behind it.
        z_seen = _pct(P[:, 2], 99)
        valid &= (hit[:, 2] > z_front - 0.25) & (hit[:, 2] < min(z_front + window, z_seen + 0.30))
        # A bottom edge with object ToF points clearly in front of its floor
        # point is an overhang (seat edge, hem), not a contact.
        obj_uv, obj_ok = camera.project(obj)
        if valid.any() and obj_ok.any():
            ouv, oz = obj_uv[obj_ok], P[obj_ok, 2]
            for i in np.flatnonzero(valid):
                close = np.abs(ouv[:, 0] - cu[i]) + np.abs(ouv[:, 1] - cv[i]) < 3
                if close.sum() >= 2 and np.max(oz[close]) < hit[i, 2] - 0.15:
                    valid[i] = False
        # Group contact pixels that touch along the outline (next column, up to
        # a few rows apart): one leg's foot, or one receding bottom edge however
        # steep, is a single chain; a stray bump from a ragged outline is a
        # chain of one or two pixels.
        cand_pts = hit[valid][:, [0, 2]]
        cand_uv = np.column_stack([cu[valid], cv[valid]])
        cand_low = c_lowest[valid]
        label_of = np.full(len(cand_pts), -1)
        groups = []
        for i in range(len(cand_pts)):
            if label_of[i] >= 0:
                continue
            label_of[i] = len(groups)
            members, queue = [i], [i]
            while queue:
                j = queue.pop()
                du = np.abs(cand_uv[:, 0] - cand_uv[j, 0])
                dv = np.abs(cand_uv[:, 1] - cand_uv[j, 1])
                # Big row steps only between column-lowest pixels (a steep
                # receding edge); a step up a leg's side doesn't chain.
                reach = np.where(cand_low & cand_low[j], params.contact_link_rows, 1.0)
                near = np.flatnonzero((label_of < 0) & (du <= 1.0) & (dv <= reach))
                label_of[near] = len(groups)
                members.extend(near.tolist())
                queue.extend(near.tolist())
            groups.append(members)
        kept = []
        for members in groups:
            group = cand_pts[members]
            if len(group) >= 3:   # one or two pixels are noise, not a leg or an edge
                kept.append(group)
                clusters.append(dict(x=float(group[:, 0].mean()), z_min=float(group[:, 1].min()),
                                     z_max=float(group[:, 1].max()), count=int(len(group)),
                                     x_span=float(np.ptp(group[:, 0]))))
        contacts = np.vstack(kept) if kept else np.empty((0, 2))
        contacts, clusters, floor_note = check_floor_under_contacts(contacts, clusters, level, pts[~in_mask])
        if floor_note.startswith("unverified"):
            unverified_contacts = contacts
            contacts, clusters = np.empty((0, 2)), []
    if len(contacts) >= 6:
        z_near = min(c["z_min"] for c in clusters)
        spread = max(c["z_max"] for c in clusters) - z_near
        # Contacts along one straight line are the bottom edge of a single face,
        # however much they recede: they don't show how deep the object is.
        centred = contacts - contacts.mean(axis=0)
        normal = np.linalg.svd(centred, full_matrices=False)[2][1]
        collinear = float(np.max(np.abs(centred @ normal))) < 0.04
        ok_spread = spread >= params.min_contact_spread_m and not collinear
        # The camera sees the front face of a rear leg or caster, not its back.
        # A compact, front-facing contact (wide but not receding) is taken to
        # be as deep as it is wide, up to 8 cm.
        backs = []
        for c in clusters:
            back = c["z_max"]
            if ok_spread and c["z_max"] - c["z_min"] < 0.3 * c["x_span"]:
                back = c["z_min"] + min(c["x_span"], 0.08)
            backs.append(back)
        z_back = max(backs)
        depth = z_back - min(z_front, z_near)
        # Contacts give the whole depth only when they reach the back as rear
        # legs/casters; the far end of a receding edge (a box turned to the
        # drone) leaves a hidden corner beyond it, so that is a lower bound.
        edges = ok_spread and on_two_lines(contacts)
        ok_spread = ok_spread and not edges
        methods["floor_contacts"] = MethodEstimate(
            "floor_contacts", "estimate" if ok_spread else "lower_bound", depth=depth,
            valid=True, confidence=0.7 if ok_spread else 0.3,
            front=min(z_front, z_near), back=z_back,
            note=f"{len(clusters)} contact groups over {spread * 100:.0f} cm"
                 + (f"; {floor_note}" if floor_note else "")
                 + (", all on one edge" if collinear else "")
                 + (", along two bottom edges (lower bound: the far corner is hidden)"
                    if (not ok_spread and not collinear and spread >= params.min_contact_spread_m) else ""))
    else:
        methods["floor_contacts"] = MethodEstimate(
            "floor_contacts", "estimate",
            note="outline made from the ToF (no independent edges)" if params.outline_from_tof else
            "no floor plane" if floor is None else (
                "camera too low" if floor.height < params.min_contact_camera_height_m
                else floor_note if floor_note.startswith("unverified")
                else f"{len(contacts)} contact pixels (need 6)"))

    # 6. Footprint completion: rectangle fitted to everything seen. With three
    # of its corners observed (two faces of a box, or three chair legs) the
    # hidden fourth corner follows for a rectangular footprint.
    foot = np.vstack([P[:, [0, 2]], contacts]) if len(contacts) else P[:, [0, 2]]
    # Lines between contact groups (chair legs, casters) are likely footprint
    # edges, so their directions are tried as rectangle orientations too.
    centroids = np.array([[c["x"], 0.5 * (c["z_min"] + c["z_max"])] for c in clusters]).reshape(-1, 2)
    leg_angles = [math.atan2(b[1] - a[1], b[0] - a[0]) for i, a in enumerate(centroids)
                  for b in centroids[i + 1:] if np.linalg.norm(b - a) > 0.15]
    rect = fit_rectangle(P[:, [0, 2]], keep_points=contacts, extra_angles=leg_angles)
    if rect is not None:
        sides = np.linalg.norm(np.roll(rect, -1, axis=0) - rect, axis=1)
        corners_seen = int(rectangle_support(foot, rect).sum())
        ok_rect = corners_seen >= 3 and sides.min() >= 0.12
        rect_depth = float(rect[:, 1].max() - min(rect[:, 1].min(), z_front))
        # Is the rectangle's back really the object's back? It is if floor
        # contacts reach it (rear legs) or a side was seen along its length.
        # Otherwise the back edge is a camera-facing surface (a sofa's
        # backrest) with unknown material behind it.
        closed, why = True, ""
        if ok_rect:
            closed, reason = footprint_closed(foot, contacts, rect)
            why = "" if closed else f", {reason}"
        # An open rectangle is reported but not used: its corners can poke past
        # the object, so it isn't a safe lower bound either.
        body = label in BODY_LABELS
        methods["footprint_completion"] = MethodEstimate(
            "footprint_completion", "estimate", depth=rect_depth, valid=ok_rect and closed and not body,
            confidence=0.6 if ok_rect and closed and not body else 0.0,
            front=min(float(rect[:, 1].min()), z_front), back=float(rect[:, 1].max()),
            note=f"{corners_seen} of 4 corners observed, sides {sides.min():.2f}x{sides.max():.2f} m{why}"
                 + ("; not used for people (a body is not a box)" if body else ""))
    else:
        methods["footprint_completion"] = MethodEstimate("footprint_completion", "estimate",
                                                         note="too few points")

    # 6b. Box model for a flat front (closets, cabinets, solid furniture). The
    # ToF sees only the front face; the depth shows up only where a side or
    # top face is visible, as a sliver at the edge of the outline.
    ell = methods["silhouette_ellipse"]
    ellipse_ok = ell.valid and ell.confidence >= params.min_estimate_confidence
    face = front_face(P[:, [0, 2]], rect)
    box_like = (face is not None and not ellipse_ok and face.planarity >= 0.5
                and face.s1 - face.s0 >= 0.2 and label not in BODY_LABELS)
    if box_like:
        foot_pts = np.vstack([P[:, [0, 2]], contacts]) if len(contacts) else P[:, [0, 2]]
        d_seen = max(_pct(face.depth_of(P[:, [0, 2]]), 98),
                     float(face.depth_of(contacts).max()) if len(contacts) else 0.0, 0.0)
        z_mid = z_front + half_guess
        hy = floor_y - y_top
        heights_l = ray_ol[:, 1] / np.where(ray_ol[:, 2] > 0, ray_ol[:, 2], 1.0) * z_mid
        heights_r = ray_or[:, 1] / np.where(ray_or[:, 2] > 0, ray_or[:, 2], 1.0) * z_mid
        mid_l = ok_hull & (heights_l > y_top + 0.2 * hy) & (heights_l < y_top + 0.8 * hy)
        mid_r = ok_hull & (heights_r > y_top + 0.2 * hy) & (heights_r < y_top + 0.8 * hy)

        def horizontal(r):
            h = r[:, [0, 2]]
            return h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-9)

        d_side, side_note = side_sliver_depth(face, horizontal(ray_ol[mid_l]), horizontal(ray_or[mid_r]))
        front_on = np.abs(face.depth_of(P[:, [0, 2]])) < 0.03
        top_plane_y = _pct(P[front_on, 1], 1) if front_on.sum() >= 10 else float("nan")
        d_top = None
        if floor is not None and np.isfinite(top_plane_y) and top_plane_y >= 0.10:
            cols = np.flatnonzero(mask.any(axis=0))
            v_top = mask[:, cols].argmax(axis=0)
            top_rays, top_valid = rays_level(cols.astype(float), v_top - 0.5)
            d_top = top_sliver_depth(face, top_rays[top_valid], top_plane_y)
        seen = [d for d in (d_side, d_top) if d is not None and 0.02 < d < 3.0]
        if seen:
            d_obj = max(d_seen, max(seen))
            corners = face.corners(d_obj)
            along = float(corners[:, 1].max() - min(corners[:, 1].min(), z_front))
            parts = ([f"side sliver ({side_note}) {d_side:.2f} m"] if d_side is not None else []) + (
                [f"top face {d_top:.2f} m"] if d_top is not None else [])
            methods["box_sides"] = MethodEstimate(
                "box_sides", "estimate", depth=along, valid=True, confidence=0.6,
                front=min(float(corners[:, 1].min()), z_front), back=float(corners[:, 1].max()),
                note=f"flat front; object depth {d_obj:.2f} m from " + ", ".join(parts))
        else:
            methods["box_sides"] = MethodEstimate(
                "box_sides", "estimate", note=f"flat front, depth not visible ({side_note})")
    else:
        d_seen = 0.0
        methods["box_sides"] = MethodEstimate(
            "box_sides", "estimate", note="curved front" if ellipse_ok else
            "not used for people" if label in BODY_LABELS else "front is not one flat face")

    # 7. Fusion. A valid curved-front ellipse is the most specific model, so
    # it is used with the lower bounds only; the footprint methods are for
    # objects whose front is not curved. Otherwise the largest valid value
    # wins, and the class prior only fills in when nothing measured the depth.
    def usable(m):
        return m.valid and m.depth is not None and (
            m.kind == "lower_bound" or m.confidence >= params.min_estimate_confidence)

    bounds = {k: m for k, m in methods.items() if m.kind == "lower_bound" and usable(m)}
    body = methods.get("body_shape")
    thickness = fixed_thickness(label)
    if thickness is not None:
        # Thin object of known thickness (a stop sign): that much behind its
        # face, which a plane fit locates without the ToF noise. A face turned
        # away from the drone adds how far it recedes along the view.
        face_front, along = fixed_extent_along_view(P, thickness)
        if along > thickness + 1e-9 and len(slope_ol):
            # Turned: the face's two ends lie on the outline's edge rays
            # (x = s z), which pin them better than the outermost ToF samples.
            (a, b, c), keep = plane_face_fit(P)
            y_mid = float(np.median(P[keep, 1]))
            ends = [(a + c * y_mid) / (1.0 - b * sl) for sl in (slope_ol.min(), slope_or.max())
                    if 1.0 - b * sl > 0.1]
            ends = [z for z in ends if 0.0 < z < z_front + 2.0]
            if ends:
                back = max(face_front + along, max(ends) + thickness)
                face_front = min(face_front, min(ends))
                along = back - face_front
        methods["fixed_thickness"] = MethodEstimate(
            "fixed_thickness", "estimate", depth=along, valid=True, confidence=1.0,
            front=face_front, back=face_front + along,
            note=f"{thickness * 1000:.0f} mm thick" + (f", face recedes {along - thickness:.2f} m"
                                                        if along > thickness else ", face square-on"))
        pool = {"fixed_thickness": methods["fixed_thickness"]}
        estimated = True
    elif body is not None and body.valid:
        # People: the front-curvature ellipse is noisy on real ToF data (0.19-0.91 m
        # for one person on the first lab bag), so it is used only when it lands
        # in the range a body this wide and this turned can have.
        lo, hi = body.range
        ell = methods["silhouette_ellipse"]
        if usable(ell) and lo - 0.02 <= ell.depth <= hi + 0.02:
            pool = {**bounds, "silhouette_ellipse": ell}
        else:
            if ell.depth is not None:
                ell.note += f"; outside {lo:.2f}-{hi:.2f} m, the range for this body: not used"
            pool = {**bounds, "body_shape": body}
        estimated = True
    elif usable(methods["silhouette_ellipse"]):
        # Contacts of a rounded object are its feet or base, which say little
        # about its depth and pick up outline noise, so only the ToF bound stays.
        pool = {**bounds, "silhouette_ellipse": methods["silhouette_ellipse"]}
        estimated = True
    else:
        estimates = {k: m for k, m in methods.items() if m.kind == "estimate" and usable(m)}
        # The side-sliver geometry is the least precise (rays near grazing),
        # so it only counts when contacts or the rectangle can't close the back.
        if "box_sides" in estimates and len(estimates) > 1:
            del estimates["box_sides"]
        pool = {**bounds, **estimates}
        estimated = bool(estimates)
    max_depth = float(prior.get("max_depth", params.max_depth_generic_m))
    if prior is DEFAULT_PRIOR:
        max_depth = params.max_depth_generic_m
    typical = float(prior.get("depth", DEFAULT_PRIOR["depth"]))
    fallback_obj = max_depth if params.unobserved_depth == "conservative" else typical

    def along_view(obj_depth):
        """Depth along the line of sight of a box this deep behind the front face.
        A wide face seen at an angle adds its own obliqueness."""
        if not box_like:
            return obj_depth
        c = face.corners(obj_depth)
        return float(c[:, 1].max() - min(c[:, 1].min(), z_front))

    if not estimated or params.use_prior_floor:
        d_fb = max(fallback_obj, d_seen) if not estimated else typical
        pool["class_prior"] = MethodEstimate(
            "class_prior", "lower_bound", depth=along_view(d_fb), valid=True,
            note=f"{params.unobserved_depth if not estimated else 'typical'} {d_fb:.2f} m behind the front")
    if not estimated:
        cls = LABEL_ALIASES.get(label, label) if prior is not DEFAULT_PRIOR else "generic object"
        why = ("outline made from the ToF, which can't show more than the ToF" if params.outline_from_tof else
               "back not visible from this position (flat face seen square-on)" if box_like else
               "depth not measurable from this view (front not measurably curved, no usable floor contacts)")
        warnings.append(f"{why}: using the {params.unobserved_depth} {cls} depth, "
                        f"{max(fallback_obj, d_seen):.2f} m")
    for name, m in methods.items():
        m.used = name in pool
    max_along = max(max_depth, along_view(max_depth))
    for name in ("side_on", "feet"):
        if name in pool:
            max_along = max(max_along, pool[name].depth)
    if body is not None and body.valid:   # a person may be up to its body range deep
        max_along = max(max_along, body.range[1])
    # Each method gives where the object starts and ends along the line of
    # sight; the box spans all of them. Methods measured from the ToF front
    # start there; contacts and rectangles carry their own front and back.
    spans = {}
    for name, m in pool.items():
        front = z_front if m.front is None else float(m.front)
        back = front + float(m.depth) if m.back is None else float(m.back)
        if back - front > max_along + 1e-9:
            warnings.append(f"{name} gave {back - front:.2f} m, clamped to {max_along:.2f} m")
            back = front + max_along
        spans[name] = (front, back)
    decided_by = max(spans, key=lambda k: spans[k][1])
    z_seen_back = _pct(P[:, 2], 99)   # never cut off what the ToF saw

    # 8. Box in the level frame: visual hull laterally, floor to top vertically.
    if thickness is not None:   # the fitted face (square-on), not the noisy ToF extremes
        z0, z1 = spans["fixed_thickness"]
        if z1 - z0 > thickness + 1e-9:   # turned: everything the ToF saw of it too
            z0, z1 = min(z0, z_front), max(z1, z_seen_back)
    else:
        z0 = min([z_front] + [f for f, _ in spans.values()])
        z1 = max([z_seen_back] + [b for _, b in spans.values()])
    depth = float(max(0.0 if thickness is not None else 0.05, z1 - z0))
    z1 = z0 + depth
    if len(slope_ol):
        x0 = float(min(slope_ol.min() * z0, slope_ol.min() * z1))
        x1 = float(max(slope_or.max() * z0, slope_or.max() * z1))
    else:
        x0, x1 = _pct(P[:, 0], 2), _pct(P[:, 0], 98)
    y_bottom = level.height if floor is not None else _pct(P[:, 1], 99)
    box = dict(x0=x0, x1=x1, z0=z0, z1=z1, y_top=float(y_top), y_bottom=float(y_bottom))
    corners_level = np.array([[x, y, z] for x in (x0, x1) for y in (y_top, y_bottom) for z in (z0, z1)])
    return OutlineResult(
        label=label, depth_m=depth, decided_by=decided_by, methods=methods, front_z_m=float(z_front),
        box_level=box, corners_cam=level.to_cam(corners_level), level=level, floor=floor,
        alignment=alignment, n_object_points=int(len(obj)), edge_sensitivity_m_per_px=sensitivity,
        warnings=warnings, depth_observed=bool(estimated),
        geometry=dict(points_level=P, contacts=contacts, clusters=clusters, fits=fits,
                      slopes_left=slope_ol, slopes_right=slope_or, rect=rect, face=face if box_like else None,
                      object_points_cam=obj, unverified_contacts=unverified_contacts,
                      body_ellipse=body_ellipse),
    )


def result_summary(result):
    """Plain dict for JSON/CSV output."""
    out = dict(label=result.label, depth_m=round(result.depth_m, 3), decided_by=result.decided_by,
               front_range_m=round(result.front_z_m, 3),
               width_m=round(result.box_level["x1"] - result.box_level["x0"], 3),
               camera_height_m=None if result.floor is None else round(result.floor.height, 3),
               alignment=None if not np.isfinite(result.alignment) else round(result.alignment, 3),
               object_points=result.n_object_points,
               edge_sensitivity_cm_per_px=None if not np.isfinite(result.edge_sensitivity_m_per_px)
               else round(result.edge_sensitivity_m_per_px * 100, 2),
               depth_observed=result.depth_observed,
               warnings=result.warnings)
    for name, m in result.methods.items():
        out[f"{name}_m"] = None if m.depth is None else round(float(m.depth), 3)
        out[f"{name}_used"] = bool(m.used)
        out[f"{name}_kind"] = m.kind
        out[f"{name}_note"] = m.note
    return out
