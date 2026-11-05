"""Front-to-back depth of detected obstacles from a single-view ToF point cloud.

A ToF sensor only sees the surfaces facing it, so a box built from the visible
points alone is as thin as the front surface (a few cm for a person). That box
then lets a planned path or safety tube pass through the body behind it.

  * person -> fit the front half of an elliptical torso cross-section in
              horizontal slices around the belly. With the silhouette width
              fixed, the front arc z = cz - b*sqrt(1-u^2) is linear in (cz, b),
              and the curvature near the left/right edges fixes b.
  * others -> open or irregular structure (chairs, tables): the visible range
              spread is only a lower bound, so take max(measured, class prior).
  * thin objects of known thickness (stop sign, FIXED_THICKNESS_M) -> that
              thickness behind the visible face, located with a plane fit so
              ToF noise doesn't shift it; a face turned away from the drone
              adds how far it recedes.

Every estimate is clamped to plausible class bounds and blended toward a class
prior when the fit is weak, because underestimating depth is the unsafe
direction for path planning.

Input points are in a camera optical frame (x right, y down, z forward),
metres, segmented to ONE object. The returned box has its front face at the
nearest visible surface and extends `depth_m` away from the camera along the
line of sight. Only numpy is required, so this module is unit-testable
without ROS.
"""

from dataclasses import dataclass

import numpy as np


# Typical dimensions along the line of sight (metres). Tune for your scene.
PRIORS = {
    "person": dict(depth=0.28, min_depth=0.18, max_depth=0.45),
    "chair": dict(depth=0.50, min_depth=0.35, max_depth=0.80),
    # max_depth only bounds which points belong to the sign (one turned edge-on).
    "stop sign": dict(depth=0.01, min_depth=0.01, max_depth=0.75),
}
DEFAULT_PRIOR = dict(depth=0.40, min_depth=0.10, max_depth=1.50)

# Known thickness (metres) behind the visible face, used instead of a class
# prior: without it a stop sign is an unknown class and gets 1.5 m.
FIXED_THICKNESS_M = {"stop sign": 0.01}
LABEL_ALIASES = {"stopsign": "stop sign", "stop_sign": "stop sign", "stop-sign": "stop sign"}
# A face receding less than this along the view is taken as square-on (plane
# fit noise); beyond it the receding part is added to the thickness.
FIXED_FACE_RECEDE_M = 0.02

# Silhouette-width correction in units of the median pixel-column spacing.
# The outermost samples sit inside the true edge by up to about one spacing.
# Tuned in simulation; re-check against measured objects on the real sensor.
EDGE_K = 0.75


@dataclass
class DepthEstimate:
    label: str
    depth_m: float               # thickness along the viewing ray
    width_m: float
    height_m: float
    front_range_m: float         # optical z of the nearest visible surface
    corners_cam: np.ndarray      # (8, 3) box corners, camera optical frame
    front_center_cam: np.ndarray  # (3,) centre of the box's front face
    method: str                  # "ellipse_fit", "extent", or "prior"
    confidence: float            # 0..1
    points_used: int


def canonical_label(label):
    name = str(label).strip().lower()
    return LABEL_ALIASES.get(name, name)


def prior_for(label):
    return PRIORS.get(canonical_label(label), DEFAULT_PRIOR)


def fixed_thickness(label):
    """Known thickness of a thin object (None for everything else)."""
    return FIXED_THICKNESS_M.get(canonical_label(label))


def plane_face_fit(p, iterations=3):
    """Least-squares plane z = a + b x + c y through a flat face, with outliers
    (a pole, mixed edge pixels) dropped. Returns ((a, b, c), inlier mask)."""
    p = np.asarray(p, dtype=float)
    A = np.column_stack([p[:, 0], p[:, 1], np.ones(len(p))])
    z = p[:, 2]
    keep = np.ones(len(p), dtype=bool)
    for _ in range(iterations):
        coef = np.linalg.lstsq(A[keep], z[keep], rcond=None)[0]
        res = z - A @ coef
        keep = np.abs(res) <= max(0.015, 3.0 * 1.4826 * float(np.median(np.abs(res[keep]))))
        if keep.sum() < 3:
            keep = np.ones(len(p), dtype=bool)
            break
    return coef, keep


def plane_face_extent(p):
    """Nearest and farthest z of a flat face, from points (x, y, z) with z along
    the view: the fitted plane at the samples, so ToF noise doesn't move it."""
    p = np.asarray(p, dtype=float)
    coef, keep = plane_face_fit(p)
    face = (np.column_stack([p[:, 0], p[:, 1], np.ones(len(p))]) @ coef)[keep]
    return float(face.min()), float(face.max())


def fixed_extent_along_view(p, thickness):
    """(front z, depth along the view) of a thin object of known thickness.

    Square-on, the box is exactly `thickness` behind the fitted face. A face
    turned away from the drone also spans what the ToF saw of it (its
    outermost samples sit up to a pixel inside the true edges).
    """
    front, back = plane_face_extent(p)
    if back - front <= FIXED_FACE_RECEDE_M:
        return front, float(thickness)
    z = np.asarray(p, dtype=float)[:, 2]
    front = min(front, float(np.percentile(z, 3)))
    back = max(back, float(np.percentile(z, 97)))
    return front, back - front + float(thickness)


def range_window_for(label, extra=0.15):
    """How far behind the nearest surface points can still belong to the object."""
    return float(prior_for(label)["max_depth"]) + float(extra)


# ---------------------------------------------------------------- filters
def density_filter(points, voxel=0.05, min_count=5):
    """Drop isolated returns such as ToF flying pixels between foreground and
    background: keep points whose 3x3x3 voxel neighbourhood holds at least
    `min_count` points. Vectorized; ~27 searchsorted passes."""
    p = np.asarray(points, dtype=float)
    if len(p) == 0:
        return p
    idx = np.floor(p / float(voxel)).astype(np.int64)
    idx -= idx.min(axis=0) - 1  # >= 1, so every neighbour index is >= 0
    dims = idx.max(axis=0) + 2

    def encode(v):
        return (v[:, 0] * dims[1] + v[:, 1]) * dims[2] + v[:, 2]

    keys, counts = np.unique(encode(idx), return_counts=True)
    total = np.zeros(len(p), dtype=np.int64)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                k = encode(idx + np.array([dx, dy, dz]))
                pos = np.clip(np.searchsorted(keys, k), 0, len(keys) - 1)
                hit = keys[pos] == k
                total[hit] += counts[pos[hit]]
    return p[total >= int(min_count)]


def _align_to_bearing(points):
    """Rotate about the camera y axis so the object's median lies on +z, making
    'depth' mean thickness along the line of sight even off-centre."""
    c = np.median(points, axis=0)
    yaw = np.arctan2(c[0], c[2])
    cy, sy = np.cos(-yaw), np.sin(-yaw)
    R = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    return points @ R.T, R


def _gate_range(p, window):
    """Drop background leaking into the detection box."""
    front = np.percentile(p[:, 2], 3)
    return p[p[:, 2] <= front + window]


# ----------------------------------------------------------- ellipse fit
def _band_width(x):
    """Silhouette half-width and centre from all belly-band points pooled,
    plus a pixel-spacing correction. 1 cm of width error moves the depth
    estimate by roughly 4 cm, so this matters."""
    sx = np.sort(np.asarray(x, dtype=float))
    if len(sx) < 8:
        return None
    gaps = np.diff(sx)
    gaps = gaps[gaps > 2e-3]
    spacing = float(np.median(gaps)) if len(gaps) else 0.0
    return (sx[-1] - sx[0]) / 2.0 + EDGE_K * spacing, (sx[-1] + sx[0]) / 2.0


def _fit_front_arc(x, z, a, cx, edge_trim=0.92, iters=6):
    """Robust (Huber IRLS) fit of z = cz - b*sqrt(1-u^2) for one slice.
    Points with |u| > edge_trim are dropped: grazing-angle returns are noisy."""
    if len(x) < 8 or a < 0.05:
        return None
    u = (np.asarray(x) - cx) / a
    m = np.abs(u) < edge_trim
    if m.sum() < 6:
        return None
    s = np.sqrt(np.clip(1.0 - u[m] ** 2, 0.0, 1.0))
    zz = np.asarray(z)[m]
    if np.ptp(s) < 0.25:  # not enough edge points to see the curvature
        return None
    A = np.column_stack([np.ones_like(s), -s])
    w = np.ones_like(s)
    for _ in range(iters):
        sw = np.sqrt(w)
        sol, *_ = np.linalg.lstsq(A * sw[:, None], zz * sw, rcond=None)
        r = zz - A @ sol
        k = 1.345 * (1.4826 * np.median(np.abs(r)) + 1e-4)
        w = np.where(np.abs(r) <= k, 1.0, k / np.maximum(np.abs(r), 1e-9))
    cz, b = sol
    if b <= 0:
        return None
    return dict(cz=float(cz), b=float(b), rms=float(np.sqrt(np.mean(r ** 2))))


# ------------------------------------------------------------ estimators
def estimate_person(points, band=(0.30, 0.45), slice_h=0.04, min_width=None, min_height=None):
    """Torso depth from elliptical fits on horizontal slices.

    band: vertical fraction of the visible height (0 = top) to use; 0.30-0.45
    is roughly the belly for a standing adult fully in view. Lower than that
    reaches the legs, which are two blobs rather than one ellipse.
    """
    pr = PRIORS["person"]
    p, R = _align_to_bearing(density_filter(points))
    if len(p) < 10:
        return None
    p = _gate_range(p, window=range_window_for("person"))
    y = p[:, 1]
    top, bottom = np.percentile(y, [2, 98])
    H = bottom - top
    y0, y1 = top + band[0] * H, top + band[1] * H
    bp = p[(y >= y0) & (y < y1)]
    bw = _band_width(bp[:, 0])

    depths, rmss = [], []
    if bw is not None:
        for ys in np.arange(y0, y1, slice_h):
            sl = bp[(bp[:, 1] >= ys) & (bp[:, 1] < ys + slice_h)]
            f = _fit_front_arc(sl[:, 0], sl[:, 2], *bw)
            if f:
                depths.append(2.0 * f["b"])
                rmss.append(f["rms"])

    if len(depths) >= 2:
        measured = float(np.median(depths))
        in_bounds = pr["min_depth"] <= measured <= pr["max_depth"]
        conf = (min(1.0, len(depths) / 5.0)
                * float(np.exp(-np.median(rmss) / 0.02))
                * (1.0 if in_bounds else 0.3))
        depth = conf * measured + (1.0 - conf) * pr["depth"]
        method = "ellipse_fit"
    else:
        depth, conf, method = pr["depth"], 0.0, "prior"
    depth = float(np.clip(depth, pr["min_depth"], pr["max_depth"]))
    return _package("person", p, R, depth, method, conf, min_width, min_height)


def estimate_by_extent(points, label="chair", min_width=None, min_height=None):
    """Depth for open/irregular objects. The visible range spread is a LOWER
    bound (occlusion hides the rest), so use max(measured, prior), clamped."""
    pr = prior_for(label)
    p, R = _align_to_bearing(density_filter(points))
    if len(p) < 10:
        return None
    p = _gate_range(p, window=range_window_for(label))
    front, back = np.percentile(p[:, 2], [3, 97])
    measured = float(back - front)
    depth = float(np.clip(max(measured, pr["depth"]), pr["min_depth"], pr["max_depth"]))
    conf = float(min(1.0, measured / pr["depth"]))
    return _package(label, p, R, depth, "extent", conf, min_width, min_height)


def estimate_fixed(points, label, min_width=None, min_height=None):
    """Thin object of known thickness (FIXED_THICKNESS_M)."""
    p, R = _align_to_bearing(density_filter(points))
    if len(p) < 10:
        return None
    p = _gate_range(p, window=range_window_for(label))
    front, depth = fixed_extent_along_view(p, fixed_thickness(label))
    return _package(label, p, R, depth, "fixed_thickness", 1.0, min_width, min_height, front=front)


def estimate_depth(points, label, min_width=None, min_height=None):
    """Dispatch on detector class name. Returns None if too few points.

    min_width / min_height: extents the box must at least cover (for example
    the detector bounding box projected to the object's range), applied
    symmetrically about the measured centre.
    """
    points = np.asarray(points, dtype=float)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 10:
        return None
    if fixed_thickness(label) is not None:
        return estimate_fixed(points, label, min_width=min_width, min_height=min_height)
    if str(label).strip().lower() == "person":
        return estimate_person(points, min_width=min_width, min_height=min_height)
    return estimate_by_extent(points, label, min_width=min_width, min_height=min_height)


def _package(label, p, R, depth, method, conf, min_width, min_height, front=None):
    x0, x1 = np.percentile(p[:, 0], [2, 98])
    y0, y1 = np.percentile(p[:, 1], [2, 98])
    if min_width is not None and (x1 - x0) < float(min_width):
        xc, half = 0.5 * (x0 + x1), 0.5 * float(min_width)
        x0, x1 = xc - half, xc + half
    if min_height is not None and (y1 - y0) < float(min_height):
        yc, half = 0.5 * (y0 + y1), 0.5 * float(min_height)
        y0, y1 = yc - half, yc + half
    z0 = float(np.percentile(p[:, 2], 3)) if front is None else float(front)
    corners = np.array(
        [[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z0 + depth)]
    ) @ R  # aligned frame -> camera frame (R orthonormal)
    front_center = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1), z0]) @ R
    return DepthEstimate(
        label=str(label),
        depth_m=float(depth),
        width_m=float(x1 - x0),
        height_m=float(y1 - y0),
        front_range_m=z0,
        corners_cam=corners,
        front_center_cam=front_center,
        method=method,
        confidence=float(conf),
        points_used=int(len(p)),
    )
