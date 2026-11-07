"""Simulated scenes for testing outline-based depth: camera image, true object
outline, and a ToF point cloud, all ray-cast through the Starling 2 camera
mounts and the hires camera's real lens distortion.

Body frame is FRD (x forward, y right, z down); the floor is at body
z = drone_height. Raw ToF points are returned in the ToF optical frame, like
/tof_pc. Shapes are simple primitives: vertical elliptic cylinders and boxes.
"""

from dataclasses import dataclass, field
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from outline_depth import starling_cameras  # noqa: E402


@dataclass
class EllipticCylinder:
    """Vertical cylinder; half_depth along the object's local x (yawed), half_width along local y."""
    cx: float
    cy: float
    half_depth: float
    half_width: float
    z_top: float      # body z (more negative = higher)
    z_bottom: float
    yaw: float = 0.0
    obj: int = 2


@dataclass
class Box:
    cx: float
    cy: float
    cz: float
    half_depth: float
    half_width: float
    half_height: float
    yaw: float = 0.0
    obj: int = 2


def _local(points, cx, cy, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    x, y = points[..., 0] - cx, points[..., 1] - cy
    return np.stack([c * x + s * y, -s * x + c * y, points[..., 2]], axis=-1)


def _rot_dirs(d, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], d[:, 2]], axis=-1)


def hit_cylinder(origin, dirs, prim):
    o = _local(origin[None, :], prim.cx, prim.cy, prim.yaw)[0]
    d = _rot_dirs(dirs, prim.yaw)
    b, a = prim.half_depth, prim.half_width
    A = (d[:, 0] / b) ** 2 + (d[:, 1] / a) ** 2
    B = 2 * (o[0] * d[:, 0] / b ** 2 + o[1] * d[:, 1] / a ** 2)
    C = (o[0] / b) ** 2 + (o[1] / a) ** 2 - 1
    disc = B * B - 4 * A * C
    t = np.full(len(d), np.inf)
    ok = (A > 1e-12) & (disc >= 0)
    sq = np.sqrt(np.where(ok, disc, 0))
    for root in ((-B - sq) / (2 * np.where(ok, A, 1)), (-B + sq) / (2 * np.where(ok, A, 1))):
        z = o[2] + root * d[:, 2]
        good = ok & (root > 1e-6) & (z >= prim.z_top) & (z <= prim.z_bottom) & (root < t)
        t = np.where(good, root, t)
    for zc in (prim.z_top, prim.z_bottom):
        dz = np.where(np.abs(d[:, 2]) > 1e-12, d[:, 2], 1e-12)
        root = (zc - o[2]) / dz
        x, y = o[0] + root * d[:, 0], o[1] + root * d[:, 1]
        good = (root > 1e-6) & ((x / b) ** 2 + (y / a) ** 2 <= 1) & (root < t)
        t = np.where(good, root, t)
    return t


def hit_box(origin, dirs, prim):
    o = _local(origin[None, :], prim.cx, prim.cy, prim.yaw)[0]
    o[2] -= prim.cz
    d = _rot_dirs(dirs, prim.yaw)
    half = np.array([prim.half_depth, prim.half_width, prim.half_height])
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / np.where(np.abs(d) < 1e-12, 1e-12, d)
        t1, t2 = (-half - o) * inv, (half - o) * inv
    tmin = np.max(np.minimum(t1, t2), axis=1)
    tmax = np.min(np.maximum(t1, t2), axis=1)
    return np.where((tmax >= tmin) & (tmin > 1e-6), tmin, np.inf)


@dataclass
class Scene:
    prims: list
    drone_height: float = 0.5          # body origin above the floor [m]
    wall_x: float = 5.0                # back wall, body x [m]
    truth: dict = field(default_factory=dict)   # obj id -> (N, 3) body points on the footprint outline

    def cast(self, origin, dirs):
        """Range and object id (0 none, 1 floor/wall, >=2 objects) of first hits."""
        t = np.full(len(dirs), np.inf)
        ids = np.zeros(len(dirs), dtype=int)
        with np.errstate(divide="ignore", invalid="ignore"):
            tf = np.where(dirs[:, 2] > 1e-9, (self.drone_height - origin[2]) / dirs[:, 2], np.inf)
            tw = np.where(dirs[:, 0] > 1e-9, (self.wall_x - origin[0]) / dirs[:, 0], np.inf)
        for tt in (tf, tw):
            better = tt < t
            t, ids = np.where(better, tt, t), np.where(better, 1, ids)
        for prim in self.prims:
            tt = hit_cylinder(origin, dirs, prim) if isinstance(prim, EllipticCylinder) else hit_box(origin, dirs, prim)
            better = tt < t
            t, ids = np.where(better, tt, t), np.where(better, prim.obj, ids)
        return t, ids


def render_camera(scene, camera):
    """Object-id image and a flat-shaded RGB image from the detection camera."""
    v, u = np.mgrid[0:camera.height, 0:camera.width]
    rays, ok = camera.pixel_rays(u.ravel().astype(float), v.ravel().astype(float))
    dirs = (camera.rotation_cam_to_body @ rays.T).T
    t, ids = scene.cast(camera.translation_cam_body, dirs)
    ids = np.where(ok & np.isfinite(t), ids, 0).reshape(camera.height, camera.width)
    hits = camera.translation_cam_body + dirs * np.where(np.isfinite(t), t, 0)[:, None]
    palette = {0: (20, 20, 20), 1: (205, 196, 180), 2: (190, 60, 50), 3: (120, 85, 55), 4: (90, 110, 150)}
    rgb = np.array([palette.get(i, (150, 150, 150)) for i in range(max(5, ids.max() + 1))], dtype=float)[ids]
    floor = (ids.ravel() == 1) & (dirs[:, 2] > 0) & (np.abs(hits[:, 2] - scene.drone_height) < 1e-3)
    grid = (np.abs((hits[:, 0] * 2) % 1 - 0.5) > 0.47) | (np.abs((hits[:, 1] * 2) % 1 - 0.5) > 0.47)
    rgb.reshape(-1, 3)[floor & grid] *= 0.75
    shade = 0.85 + 0.15 * np.clip(-rays[:, 1], -1, 1)
    rgb = np.clip(rgb * shade.reshape(camera.height, camera.width, 1), 0, 255).astype(np.uint8)
    return ids, rgb


def render_tof(scene, tof_rotation, tof_translation, rng, noise=0.006, flying=True,
               width=240, height=180, f=115.0):
    """ToF-like scan; returns points in the ToF optical frame (like raw /tof_pc)."""
    v, u = np.mgrid[0:height, 0:width]
    d_opt = np.stack([(u.ravel() - (width - 1) / 2) / f, (v.ravel() - (height - 1) / 2) / f,
                      np.ones(u.size)], axis=-1)
    d_opt /= np.linalg.norm(d_opt, axis=1, keepdims=True)
    dirs = (tof_rotation @ d_opt.T).T
    t, ids = scene.cast(tof_translation, dirs)
    t = t.reshape(height, width)
    ids = ids.reshape(height, width)
    valid = np.isfinite(t) & (t < 6.0)
    r = np.where(valid, t + rng.normal(0, noise, t.shape), np.nan)
    if flying:  # mixed pixels at object silhouettes
        rr = r.copy()
        for dv, du in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nb = np.roll(np.roll(t, dv, axis=0), du, axis=1)
            nid = np.roll(np.roll(ids, dv, axis=0), du, axis=1)
            edge = (ids >= 2) & (nid != ids) & (nb - t > 0.15) & np.isfinite(nb)
            mix = edge & (rng.random(t.shape) < 0.5)
            rr = np.where(mix, t + (nb - t) * rng.uniform(0.2, 0.8, t.shape), rr)
        r = rr
    pts = d_opt.reshape(height, width, 3) * r[..., None]
    pts = pts.reshape(-1, 3)
    return pts[np.isfinite(pts[:, 2])]


def _ellipse_outline(prim, n=72):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    lx, ly = prim.half_depth * np.cos(a), prim.half_width * np.sin(a)
    c, s = math.cos(prim.yaw), math.sin(prim.yaw)
    z = 0.5 * (prim.z_top + prim.z_bottom)
    return np.column_stack([prim.cx + c * lx - s * ly, prim.cy + s * lx + c * ly, np.full(n, z)])


def _box_outline(prim):
    c, s = math.cos(prim.yaw), math.sin(prim.yaw)
    out = []
    for lx in (-prim.half_depth, prim.half_depth):
        for ly in (-prim.half_width, prim.half_width):
            out.append([prim.cx + c * lx - s * ly, prim.cy + s * lx + c * ly, prim.cz])
    return np.array(out)


def footprint_points(prims, obj):
    pts = [(_ellipse_outline(p) if isinstance(p, EllipticCylinder) else _box_outline(p))
           for p in prims if p.obj == obj]
    return np.vstack(pts)


# ------------------------------------------------------------------ objects
def person(x_front=2.0, y=0.0, half_width=0.18, half_depth=0.12, yaw_deg=0.0, arms=None,
           drone_height=0.5, obj=2, stride=0.0, shoes=True):
    """Standing person, torso front at body x = x_front when facing the drone.

    arms: None, "touching" (against the torso) or "gap" (5 cm away).
    stride: how far one foot is ahead of the other along the way the body
    faces (0 standing; ~0.4-0.6 m walking). The legs lean from the hips.
    shoes: 26 cm long, toes 7 cm further forward than the leg's centre, so
    standing they stick out ~8 cm in front of a 24 cm deep torso.
    """
    floor = drone_height
    yaw = math.radians(yaw_deg)
    cx = x_front + half_depth
    c, s = math.cos(yaw), math.sin(yaw)

    def at(lx, ly):
        return cx + c * lx - s * ly, y + s * lx + c * ly

    prims = [EllipticCylinder(cx, y, half_depth, half_width, floor - 1.45, floor - 0.86, yaw, obj)]
    for side in (-1, 1):
        if stride:
            # Leaning leg as short stacked pieces from the hip to the foot.
            n = 8
            for i in range(n):
                f = (i + 0.5) / n
                lx, ly = at(-side * 0.5 * stride * f, side * 0.09)
                prims.append(EllipticCylinder(lx, ly, 0.08, 0.075, floor - 0.86 * (1 - i / n),
                                              floor - 0.86 * (1 - (i + 1) / n), yaw, obj))
        else:
            lx, ly = at(0.0, side * 0.09)
            prims.append(EllipticCylinder(lx, ly, 0.08, 0.075, floor - 0.86, floor, yaw, obj))
        if shoes:   # the body faces -x (the drone) at yaw 0
            sx, sy = at(-side * 0.5 * stride - 0.07, side * 0.09)
            prims.append(Box(sx, sy, floor - 0.035, 0.13, 0.05, 0.035, yaw, obj))
    prims.append(EllipticCylinder(cx, y, 0.055, 0.055, floor - 1.55, floor - 1.45, 0, obj))
    prims.append(EllipticCylinder(cx, y, 0.095, 0.08, floor - 1.76, floor - 1.55, yaw, obj))
    if arms:
        gap = 0.0 if arms == "touching" else 0.05
        for side in (-1, 1):
            lx, ly = at(0.0, side * (half_width + gap + 0.045))
            prims.append(EllipticCylinder(lx, ly, 0.05, 0.045, floor - 1.42, floor - 0.80, yaw, obj))
    return prims


def chair(x_center=2.3, y=0.0, yaw_deg=0.0, seat=0.44, drone_height=0.5, obj=3):
    """Dining chair: seat faces -x (the drone) at yaw 0; backrest at +x."""
    floor = drone_height
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    h = seat / 2

    def box(lx, ly, z0, z1, hd, hw):
        return Box(x_center + c * lx - s * ly, y + s * lx + c * ly, floor - 0.5 * (z0 + z1),
                   hd, hw, 0.5 * (z1 - z0), yaw, obj)

    prims = [box(0, 0, 0.42, 0.46, h, h), box(h - 0.02, 0, 0.46, 0.90, 0.02, h)]
    for lx in (-h + 0.02, h - 0.02):
        for ly in (-h + 0.02, h - 0.02):
            prims.append(box(lx, ly, 0.0, 0.42, 0.015, 0.015))
    return prims


def office_chair(x_center=2.3, y=0.0, yaw_deg=0.0, base_radius=0.33, drone_height=0.5, obj=3):
    """Swivel chair: five-star base wider than the seat, centre column, seat, backrest."""
    floor = drone_height
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    prims = []
    for k in range(5):
        ang = yaw + math.radians(90 + 72 * k)
        ex, ey = x_center + base_radius * math.cos(ang), y + base_radius * math.sin(ang)
        mx, my = x_center + 0.5 * base_radius * math.cos(ang), y + 0.5 * base_radius * math.sin(ang)
        prims.append(Box(ex, ey, floor - 0.03, 0.03, 0.03, 0.03, ang, obj))           # caster
        prims.append(Box(mx, my, floor - 0.09, 0.5 * base_radius, 0.025, 0.025, ang, obj))  # spoke
    prims.append(EllipticCylinder(x_center, y, 0.03, 0.03, floor - 0.45, floor - 0.09, 0, obj))
    prims.append(Box(x_center, y, floor - 0.48, 0.25, 0.25, 0.04, yaw, obj))
    bx, by = x_center + c * 0.22, y + s * 0.22
    prims.append(Box(bx, by, floor - 0.80, 0.03, 0.23, 0.28, yaw, obj))
    return prims


def cabinet(x_center=2.4, y=0.0, yaw_deg=0.0, depth=0.45, width=0.6, height=0.9, drone_height=0.5, obj=4):
    return [Box(x_center, y, drone_height - height / 2, depth / 2, width / 2, height / 2,
                math.radians(yaw_deg), obj)]


def wardrobe(x_front=2.2, y=0.0, depth=0.6, width=1.0, height=1.9, yaw_deg=0.0, legs=0.0,
             drone_height=0.5, obj=4):
    """Closet/wardrobe; front face toward the drone at yaw 0. legs > 0 raises the
    body on four 4 cm legs of that height (like a cabinet on feet)."""
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    cx = x_front + depth / 2
    body = Box(cx, y, drone_height - legs - (height - legs) / 2, depth / 2, width / 2, (height - legs) / 2, yaw, obj)
    prims = [body]
    if legs > 0:
        for lx in (-depth / 2 + 0.03, depth / 2 - 0.03):
            for ly in (-width / 2 + 0.03, width / 2 - 0.03):
                prims.append(Box(cx + c * lx - s * ly, y + s * lx + c * ly, drone_height - legs / 2,
                                 0.02, 0.02, legs / 2, yaw, obj))
    return prims


def stop_sign(x_front=2.0, y=0.0, yaw_deg=0.0, size=0.6, thickness=0.01, center_height=1.0,
              drone_height=0.5, obj=5):
    """Square stand-in for a stop-sign plate, face toward the drone at yaw 0,
    on a 5 cm post 2 cm behind it (the post is a separate object, as the
    detector's box is around the sign)."""
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    cx = x_front + thickness / 2
    plate = Box(cx, y, drone_height - center_height, thickness / 2, size / 2, size / 2, yaw, obj)
    lx = thickness / 2 + 0.045
    post = EllipticCylinder(cx + c * lx, y + s * lx, 0.025, 0.025, drone_height - center_height,
                            drone_height, 0.0, obj + 1)
    return [plate, post]


def armchair(x_front=2.1, y=0.0, depth=0.85, width=0.85, yaw_deg=0.0, drone_height=0.5, obj=3):
    """Upholstered chair/sofa: solid base to the floor, seat, arms and a thick backrest."""
    floor = drone_height
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    cx = x_front + depth / 2

    def box(lx, ly, z0, z1, hd, hw):
        return Box(cx + c * lx - s * ly, y + s * lx + c * ly, floor - 0.5 * (z0 + z1), hd, hw, 0.5 * (z1 - z0), yaw, obj)

    back = 0.2
    return [box(0, 0, 0.0, 0.42, depth / 2, width / 2),                                  # base + seat
            box(depth / 2 - back / 2, 0, 0.42, 0.9, back / 2, width / 2),                # backrest
            box(0, -width / 2 + 0.08, 0.42, 0.62, depth / 2, 0.08),                      # arms
            box(0, width / 2 - 0.08, 0.42, 0.62, depth / 2, 0.08)]


def make_scene(prims, drone_height=0.5):
    scene = Scene(prims=prims, drone_height=drone_height)
    for obj in {p.obj for p in prims}:
        scene.truth[obj] = footprint_points(prims, obj)
    return scene


def true_depth_along(level, camera, truth_body):
    """True extent along the line of sight and the true far/near z' (level frame)."""
    z = level.to_level(camera.body_to_cam(truth_body))[:, 2]
    return float(z.max() - z.min()), float(z.min()), float(z.max())


def capture(scene, rng, noise=0.006, flying=True):
    camera, tof_r, tof_t = starling_cameras()
    ids, rgb = render_camera(scene, camera)
    tof = render_tof(scene, tof_r, tof_t, rng, noise=noise, flying=flying)
    return camera, tof_r, tof_t, ids, rgb, tof
