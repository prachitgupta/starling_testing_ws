#!/usr/bin/env python3
"""Estimate object depth from camera outlines + ToF in a recorded ROS 2 bag.

For each ToF cloud paired with the nearest camera image, the detected object
is outlined (SAM, YOLO, ToF or saved masks), then outline_depth.py estimates
its front-to-back depth along the line of sight. Results are compared with
the current ToF-only estimate (obstacle_depth.py) and, when the bag has Vicon
transforms, with motion-capture ground truth.

Typical use, inside the container from ~/starling_testing_ws:
  python3 src/llm_vision_planner/scripts/outline_depth_from_bag.py bags/baggies --label person
  python3 src/llm_vision_planner/scripts/outline_depth_from_bag.py bags/baggies --label person \\
      --truth-width 0.45 --truth-depth 0.28

Writes <out>/results.csv, results.jsonl, summary.json, frame_XXXX.png and
summary.png (default out: <bag>_outline_depth next to the bag).
"""

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bag_reader as br  # noqa: E402
from obstacle_depth import estimate_depth, range_window_for  # noqa: E402
from outline_depth import (OutlineParams, estimate_from_outline, fit_floor_plane,  # noqa: E402
                           fit_tof_alignment, mount_with_correction, result_summary, small_rotation,
                           starling_cameras, tof_to_camera)
from outline_segmentation import box_iou, make_segmenter  # noqa: E402

PACKAGE = HERE.parent
if not (PACKAGE / "config").is_dir():   # installed (ros2 run): config, calibration and models are in share/
    try:
        from ament_index_python.packages import get_package_share_directory
        PACKAGE = Path(get_package_share_directory("llm_vision_planner"))
    except Exception:
        pass
METHODS = ("fixed_thickness", "silhouette_ellipse", "body_shape", "side_on", "feet", "floor_contacts",
           "footprint_completion", "box_sides",
           "visible_extent")


# ------------------------------------------------------------------ config
def load_camera(config_path, calibration_path):
    config = {}
    if config_path and not Path(config_path).is_file():
        try:   # installed with ros2 run: use the package's share folder
            from ament_index_python.packages import get_package_share_directory
            config_path = Path(get_package_share_directory("llm_vision_planner")) / "config" / "llm_vision_planner.yaml"
        except Exception:
            config_path = None
    if config_path and Path(config_path).is_file():
        import yaml
        data = yaml.safe_load(Path(config_path).read_text()) or {}
        config = data.get("semantic_obstacle_perception", {}).get("ros__parameters", {})
    dist = None
    if calibration_path and Path(calibration_path).is_file():
        text = Path(calibration_path).read_text()
        m = re.search(r"\bD:.*?data:\s*\[([^\]]*)\]", text, re.S)
        if m:
            dist = tuple(float(v) for v in m.group(1).replace("\n", " ").split(","))
    camera, tof_r, tof_t = starling_cameras({k: v for k, v in config.items() if isinstance(v, (int, float))}, dist)
    return camera, tof_r, tof_t, config


# ------------------------------------------------------------------- vicon
MARKER_FROM_BODY = {"flu": np.diag([1.0, -1.0, -1.0]), "frd": np.eye(3)}


class TransformHistory:
    """TF/TransformStamped samples by child frame, looked up by bag time."""

    def __init__(self):
        self.samples = {}   # child -> (parent, [times], [(t, q)])

    def add(self, log_ns, parent, child, translation, quaternion):
        entry = self.samples.setdefault(child, (parent, [], []))
        entry[1].append(log_ns)
        entry[2].append((translation, quaternion))

    def frames(self):
        return sorted(self.samples)

    def lookup(self, child, t_ns, root=None, max_gap_s=0.5):
        """Pose of `child` in `root` (rotation, translation) at bag time t_ns."""
        rotation, translation = np.eye(3), np.zeros(3)
        frame = child
        while frame in self.samples and frame != root:
            parent, times, values = self.samples[frame]
            i = int(np.argmin(np.abs(np.asarray(times) - t_ns)))
            if abs(times[i] - t_ns) > max_gap_s * 1e9:
                return None
            tr, q = values[i]
            r = br.quaternion_matrix(q)
            rotation, translation = r @ rotation, r @ translation + tr
            frame = parent
        if root is not None and frame != root:
            return None
        return rotation, translation


def vicon_vehicle_frame(history, args):
    return args.vicon_vehicle or next((f for f in history.frames() if "starling" in f.lower()), None)


def vicon_truth(history, args, t_ns, camera, level, est_front):
    """Ground truth in the estimate's level frame, from Vicon poses."""
    frames = history.frames()
    vehicle = vicon_vehicle_frame(history, args)
    obj = args.vicon_object or next((f for f in frames if args.label and args.label in f.lower()), None)
    if not vehicle or not obj:
        return {}
    root = history.samples[vehicle][0]
    v = history.lookup(vehicle, t_ns, root)
    o = history.lookup(obj, t_ns, root)
    if v is None or o is None:
        return {}
    r_body = v[0] @ MARKER_FROM_BODY[args.vicon_vehicle_axes]          # body -> world
    to_level = lambda p_world: level.to_level(camera.body_to_cam(r_body.T @ (p_world - v[1])))[0]  # noqa: E731
    cam_world = v[1] + r_body @ camera.translation_cam_body
    center = to_level(o[1])
    out = {"vicon_vehicle": vehicle, "vicon_object": obj,
           "vicon_camera_height_m": round(float(cam_world[2]), 3),
           "true_center_range_m": round(float(center[2]), 3),
           "implied_depth_m": round(float(2 * (center[2] - est_front)), 3)}
    if args.truth_width and args.truth_depth:
        corners = [o[1] + o[0] @ np.array([sx * args.truth_width / 2, sy * args.truth_depth / 2, 0.0])
                   for sx in (-1, 1) for sy in (-1, 1)]
        z = np.array([to_level(c)[2] for c in corners])
        x = np.array([to_level(c)[0] for c in corners])
        out.update(true_depth_m=round(float(z.max() - z.min()), 3), true_front_m=round(float(z.min()), 3),
                   true_back_m=round(float(z.max()), 3), true_footprint_x=[round(float(a), 3) for a in x],
                   true_footprint_z=[round(float(a), 3) for a in z])
    out["true_center_x"] = round(float(center[0]), 3)
    return out


# ---------------------------------------------------------------- baseline
def tof_only_depth(points_cam, camera, box, label, floor):
    """The current perception estimate: ToF points in the detector box."""
    uv, ok = camera.project(points_cam)
    if floor is not None:
        ok &= floor.distance_above(points_cam) > 0.05
    x1, y1, x2, y2 = box
    inb = ok & (uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)
    if inb.sum() < 10:
        return None
    r = np.linalg.norm(points_cam, axis=1)
    front = np.percentile(r[inb], 5)
    sel = inb & (r >= front - 0.15) & (r <= front + range_window_for(label))
    return estimate_depth(points_cam[sel], label)


# -------------------------------------------------------------------- plots
def plot_frame(path, rgb, obj_mask, res, camera, box, baseline, truth, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse, Polygon, Rectangle

    fig, (ax_img, ax_top) = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.35, 1]})
    ax_img.imshow(rgb)
    ax_img.contour(obj_mask.astype(float), levels=[0.5], colors=["#00E5FF"], linewidths=1.2)
    pts = res.geometry["object_points_cam"]
    uv, ok = camera.project(pts)
    sc = ax_img.scatter(uv[ok, 0], uv[ok, 1], c=np.linalg.norm(pts[ok], axis=1), s=2, cmap="turbo")
    fig.colorbar(sc, ax=ax_img, fraction=0.03, label="ToF range [m]")
    if res.floor is not None and len(res.geometry["contacts"]):
        c = res.geometry["contacts"]
        c_cam = res.level.to_cam(np.column_stack([c[:, 0], np.full(len(c), res.level.height), c[:, 1]]))
        cuv, cok = camera.project(c_cam)
        ax_img.scatter(cuv[cok, 0], cuv[cok, 1], s=6, c="#FFB000", label="floor contacts")
    be = res.geometry.get("body_ellipse")
    if res.floor is not None and be is not None and be.get("feet"):
        fe = np.array(be["feet"])
        f_cam = res.level.to_cam(np.column_stack([fe[:, 0], np.full(len(fe), res.level.height), fe[:, 1]]))
        fuv, fok = camera.project(f_cam)
        ax_img.scatter(fuv[fok, 0], fuv[fok, 1], s=70, marker="X", c="#E65100", ec="w", lw=0.6, zorder=5)
    if box is not None:
        ax_img.add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False,
                                   ec="w", ls="--", lw=1))
    ax_img.set_title("Outline (cyan), ToF points inside it, floor contacts (amber), feet (orange X)", fontsize=10)
    ax_img.axis("off")

    P = res.geometry["points_level"]
    ax_top.scatter(P[:, 0], P[:, 2], s=2, c="#8A8F98", label="ToF in outline")
    zmax = res.box_level["z1"] + 0.4
    if len(res.geometry["slopes_left"]):
        for s in (res.geometry["slopes_left"].min(), res.geometry["slopes_right"].max()):
            ax_top.plot([0, s * zmax], [0, zmax], color="#00ACC1", lw=0.8)
    fits = [f for f in res.geometry["fits"] if f["accepted"]]
    if fits and res.methods["silhouette_ellipse"].used:
        f = fits[len(fits) // 2]
        ax_top.add_patch(Ellipse((f["cx"], f["cz"]), 2 * f["a"], 2 * f["b"], fill=False, ec="#00ACC1",
                                 lw=1.5, ls="-", label="ellipse fit (tangent to outline)"))
    be = res.geometry.get("body_ellipse")
    if be is not None and "body_shape" in res.methods:
        # Drawn even when the curvature ellipse gave the depth (dashed), so
        # the ellipse type picked for the person is always visible.
        used = res.methods["body_shape"].used
        kind = ("long: side-on" if be.get("side_on") else
                "wide: facing toward/away" if abs(be["angle"]) < 30 else "turned")
        ax_top.add_patch(Ellipse((be["cx"], be["cz"]), 2 * be["a"], 2 * be["b"], angle=be["angle"], fill=False,
                                 ec="#7B1FA2", lw=1.5, ls="-" if used else "--",
                                 label=f"body shape ({kind}, by the {be.get('cue', '?')}"
                                       f"{'' if used else '; check only'})"))
    if len(res.geometry["contacts"]):
        c = res.geometry["contacts"]
        ax_top.scatter(c[:, 0], c[:, 1], s=6, c="#FFB000", label="floor contacts")
    if be is not None and be.get("feet"):
        fe = np.array(be["feet"])
        st = be.get("stance_deg")
        ax_top.plot(fe[:, 0], fe[:, 1], "-", color="#E65100", lw=1.2)
        ax_top.scatter(fe[:, 0], fe[:, 1], s=60, marker="X", c="#E65100", zorder=5,
                       label="feet" + ("" if st is None else f" (stance {st:.0f} deg)"))
    rect = res.geometry["rect"]
    if rect is not None and res.methods["footprint_completion"].valid:
        ax_top.add_patch(Polygon(rect, closed=True, fill=False, ec="#2E7D32", ls="--", label="footprint rectangle"))
    b = res.box_level
    ax_top.add_patch(Rectangle((b["x0"], b["z0"]), b["x1"] - b["x0"], b["z1"] - b["z0"], fill=False,
                               ec="#B0127A", lw=2, label=f"outline box, depth {res.depth_m:.2f} m"))
    if baseline is not None:
        bz = res.level.to_level(baseline.corners_cam)
        ax_top.add_patch(Rectangle((bz[:, 0].min(), bz[:, 2].min()), np.ptp(bz[:, 0]), np.ptp(bz[:, 2]),
                                   fill=False, ec="#1F6FEB", ls=":", lw=1.5,
                                   label=f"ToF-only box, depth {baseline.depth_m:.2f} m"))
    if truth.get("true_footprint_x"):
        tx, tz = np.array(truth["true_footprint_x"]), np.array(truth["true_footprint_z"])
        order = [0, 1, 3, 2]
        ax_top.add_patch(Polygon(np.column_stack([tx[order], tz[order]]), closed=True, fill=False, ec="k",
                                 ls="--", lw=1.2, label=f"Vicon truth, depth {truth['true_depth_m']:.2f} m"))
    if "true_center_range_m" in truth:
        ax_top.plot(truth["true_center_x"], truth["true_center_range_m"], "kx", ms=8, label="Vicon object centre")
    face = res.geometry.get("face")
    if face is not None:
        fc = face.corners(0.0)
        ax_top.plot(fc[:, 0], fc[:, 1], color="#2E7D32", lw=3, alpha=0.6, label="flat front face")
    if not res.depth_observed:
        note = ("Back not visible from here (flat face seen square-on):\n"
                f"box uses the conservative class depth, {res.depth_m:.2f} m.")
        ax_top.text(0.02, 0.02, note, transform=ax_top.transAxes, fontsize=8, va="bottom",
                    bbox=dict(boxstyle="round", fc="#FFF3E0", ec="#E65100"))
    ax_top.set_xlim(b["x0"] - 0.5, b["x1"] + 0.5)
    ax_top.set_ylim(b["z0"] - 0.5, zmax)
    ax_top.set_aspect("equal")
    ax_top.set_xlabel("x' right [m]")
    ax_top.set_ylabel("z' along line of sight [m]")
    ax_top.grid(alpha=0.3)
    ax_top.legend(fontsize=7, loc="upper left")
    ax_top.set_title("Top-down, level frame at the camera", fontsize=10)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_summary(path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = sorted({r["label"] for r in rows})
    fig, axes = plt.subplots(len(labels), 1, figsize=(10, 3.6 * len(labels)), squeeze=False)
    styles = {"depth_m": ("#B0127A", "-", "outline result"), "tof_only_m": ("#1F6FEB", ":", "ToF-only (current)"),
              "silhouette_ellipse_m": ("#00ACC1", "--", "ellipse"), "floor_contacts_m": ("#FFB000", "--", "contacts"),
              "footprint_completion_m": ("#2E7D32", "--", "rectangle"), "visible_extent_m": ("#8A8F98", "--", "visible extent"),
              "body_shape_m": ("#7B1FA2", "-.", "body shape"), "feet_m": ("#E65100", ":", "feet"),
              "true_depth_m": ("k", "-", "Vicon truth")}
    # "Vicon implied" (2 x centre range - front) is left out: it is only a depth
    # when the Vicon marker set is centred in the object, not on a person's head.
    for ax, label in zip(axes[:, 0], labels):
        sub = [r for r in rows if r["label"] == label]
        t = [r["time_s"] for r in sub]
        for key, (color, ls, name) in styles.items():
            y = [r.get(key) for r in sub]
            if any(v is not None for v in y):
                ax.plot(t, [np.nan if v is None else v for v in y], ls, color=color, marker=".", label=name,
                        lw=2 if key == "depth_m" else 1.2)
        ax.set_title(f"{label}: depth along the line of sight per frame")
        ax.set_xlabel("bag time [s]")
        ax.set_ylabel("depth [m]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def same_object(a, b):
    """Two boxes for one object: they overlap mostly, or one sits inside the other."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return box_iou(a, b) >= 0.6 or (smaller > 0 and ix * iy / smaller >= 0.8)


def object_motion(history, args, t_image, t_cloud):
    """How far the Vicon-tracked object moved between the image and the ToF
    frame [m], or None without a Vicon object. The outline and the ToF only
    describe the same object if it stayed put."""
    frames = history.frames()
    obj = args.vicon_object or next((f for f in frames if args.label and args.label in f.lower()), None)
    if not obj or obj not in history.samples:
        return None
    root = history.samples[obj][0]
    a, b = history.lookup(obj, t_image, root), history.lookup(obj, t_cloud, root)
    if a is None or b is None:
        return None
    return float(np.linalg.norm(np.asarray(a[1]) - np.asarray(b[1])))


def matching_detections(detections, t_ns, args):
    """Detections for the image at bag time t_ns: the detector message group
    closest in time among those with the wanted label and confidence, within
    --max-detection-gap. Filtering first matters: a nearer low-confidence or
    other-class box must not hide the object."""
    gap = args.max_detection_gap * 1e9
    cands = [d for d in detections if abs(d[0] - t_ns) <= gap and d[2] >= args.min_confidence
             and (not args.label or d[1] == args.label)]
    if not cands:
        return []
    closest = min(cands, key=lambda d: abs(d[0] - t_ns))[0]
    return [d for d in cands if abs(d[0] - closest) <= 0.05e9]


# ------------------------------------------------------------------ loading
def load_bag(bag, args):
    reader = br.open_bag(bag, args.reader, args.msg_dir)
    types = reader.topics()
    print(f"bag: {bag} ({reader.name} reader)")
    for t, ty in sorted(types.items()):
        print(f"  {t:40s} {ty}")
    image_topic = args.image_topic or pick_topic(types, None, "/Image", ("/hires_small_color", "/hires_color"),
                                                 avoid=("/tflite", "/tof_ir", "/tof_depth", "/tof_conf"))
    if image_topic is None:
        image_topic = pick_topic(types, None, "/CompressedImage")
    cloud_topic = pick_topic(types, args.cloud_topic, "/PointCloud2", ("/tof_pc",))
    det_topic = args.detection_topic or ("/tflite_data" if "/tflite_data" in types else None)
    tf_topics = [t for t, ty in types.items() if ty.endswith("TFMessage") or ty.endswith("TransformStamped")]
    if not image_topic or not cloud_topic:
        raise SystemExit("the bag needs a camera image topic and a PointCloud2 topic")
    if det_topic and not reader.can_decode(det_topic):
        print(f"warning: can't decode {det_topic} ({types[det_topic]}); pass --msg-dir with the voxl_msgs/msg "
              "folder or run inside the container. Continuing without detections.")
        det_topic = None
    tf_topics = [t for t in tf_topics if reader.can_decode(t)]
    print(f"using image {image_topic}, cloud {cloud_topic}, detections {det_topic or '-'}, "
          f"transforms {', '.join(tf_topics) or '-'}")

    # Pass 1: timing, detections and transforms (images and clouds stay undecoded).
    image_times, cloud_times, detections = [], [], []
    history = TransformHistory()
    light = [t for t in [det_topic] + tf_topics if t]
    for topic, stamp, msg in reader.messages([image_topic, cloud_topic] + light, decode=light):
        if topic == image_topic:
            image_times.append(stamp)
        elif topic == cloud_topic:
            cloud_times.append(stamp)
        elif topic == det_topic:
            detections.append((stamp,) + br.detection_fields(msg))
        else:
            for parent, child, _, tr, q in br.transforms(msg, types[topic]):
                history.add(stamp, parent, child, tr, q)
    image_times = np.array(sorted(image_times))
    if not len(image_times) or not cloud_times:
        raise SystemExit("no images or no point clouds in the bag")
    pairs = []
    for ct in sorted(cloud_times):
        i = int(np.argmin(np.abs(image_times - ct)))
        if abs(image_times[i] - ct) <= args.max_pair_gap * 1e9:
            pairs.append((ct, int(image_times[i])))
    print(f"{len(cloud_times)} clouds, {len(image_times)} images, {len(pairs)} pairs within "
          f"{args.max_pair_gap:.2f} s, {len(detections)} detections, TF frames: {history.frames() or '-'}")
    if not pairs:
        raise SystemExit("no image/point-cloud pairs close enough in time; try a larger --max-pair-gap")
    if detections and not args.box and args.segmenter != "yolo":
        # Only frames with a box to outline are useful to the box-prompted segmenters.
        with_det = [pr for pr in pairs if matching_detections(detections, pr[1], args)]
        print(f"{len(with_det)} of {len(pairs)} pairs have a {args.label or 'detector'} detection within "
              f"{args.max_detection_gap:.1f} s (confidence >= {args.min_confidence})")
        if with_det:
            pairs = with_det
        else:
            labels = sorted({d[1] for d in detections})
            print(f"warning: no pair has a matching detection (labels in the bag: {labels}); "
                  "check --label and --min-confidence, or pass --box")
    if len(pairs) > args.max_frames:
        pairs = [pairs[int(i)] for i in np.linspace(0, len(pairs) - 1, args.max_frames)]
    want_clouds = {c for c, _ in pairs}
    want_images = {i for _, i in pairs}

    # Pass 2: decode only the selected images and clouds.
    clouds, images = {}, {}
    for topic, stamp, msg in reader.messages([image_topic, cloud_topic]):
        if topic == cloud_topic and stamp in want_clouds:
            clouds[stamp] = br.cloud_xyz(msg)
        elif topic == image_topic and stamp in want_images and stamp not in images:
            images[stamp] = br.image_to_rgb(msg)
    return dict(pairs=pairs, images=images, clouds=clouds, detections=detections, history=history,
                t0=int(min(image_times[0], min(cloud_times))),
                topics=dict(image=image_topic, cloud=cloud_topic, detections=det_topic, tf=tf_topics))


def save_sample(path, data):
    """Selected frames in one compressed .npz, small enough to share."""
    pairs = data["pairs"]
    image_t = sorted({i for _, i in pairs})
    cloud_t = sorted({c for c, _ in pairs})
    clouds = [data["clouds"][t].astype(np.float32) for t in cloud_t]
    lo, hi = min(image_t + cloud_t) - 1e9, max(image_t + cloud_t) + 1e9
    tf = {}
    for child, (parent, times, values) in data["history"].samples.items():
        keep = [k for k, t in enumerate(times) if lo <= t <= hi]
        tf[child] = dict(parent=parent, t=[int(times[k]) for k in keep],
                         tr=[values[k][0].tolist() for k in keep], q=[values[k][1].tolist() for k in keep])
    dets = [dict(t=int(d[0]), label=d[1], conf=d[2], box=list(d[3]), cam=d[4]) for d in data["detections"]
            if lo <= d[0] <= hi]
    np.savez_compressed(
        path, images=np.stack([data["images"][t] for t in image_t]), image_t=np.array(image_t, np.int64),
        cloud_points=np.concatenate(clouds), cloud_sizes=np.array([len(c) for c in clouds]),
        cloud_t=np.array(cloud_t, np.int64), pairs=np.array(pairs, np.int64),
        meta=np.array(json.dumps(dict(t0=data["t0"], detections=dets, tf=tf, topics=data.get("topics", {})))))


def load_sample(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    images = {int(t): img for t, img in zip(z["image_t"], z["images"])}
    parts = np.split(z["cloud_points"], np.cumsum(z["cloud_sizes"])[:-1])
    clouds = {int(t): p.astype(float) for t, p in zip(z["cloud_t"], parts)}
    history = TransformHistory()
    for child, e in meta["tf"].items():
        for t, tr, q in zip(e["t"], e["tr"], e["q"]):
            history.add(t, e["parent"], child, np.array(tr), np.array(q))
    detections = [(d["t"], d["label"], d["conf"], tuple(d["box"]), d["cam"]) for d in meta["detections"]]
    pairs = [(int(c), int(i)) for c, i in z["pairs"]]
    print(f"sample: {path}: {len(pairs)} pairs, {len(detections)} detections, TF frames: {history.frames() or '-'}")
    return dict(pairs=pairs, images=images, clouds=clouds, detections=detections, history=history,
                t0=int(meta["t0"]), topics=meta.get("topics", {}))


# --------------------------------------------------------------------- main
def pick_topic(types, explicit, wanted_type, prefer=(), avoid=()):
    if explicit:
        if explicit not in types:
            raise SystemExit(f"topic {explicit} is not in the bag")
        return explicit
    cands = [t for t, ty in types.items() if ty.endswith(wanted_type) and t not in avoid]
    for p in prefer:
        if p in cands:
            return p
    return cands[0] if cands else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", help="bag folder (with metadata.yaml), .db3/.mcap file, or a sample .npz from --export-sample")
    ap.add_argument("--out", help="output folder (default: <bag>_outline_depth)")
    ap.add_argument("--label", default=None, help="only this detector class (e.g. person, chair)")
    ap.add_argument("--min-confidence", type=float, default=0.5)
    ap.add_argument("--segmenter", default="auto", choices=["auto", "sam", "efficientsam", "yolo", "tof", "maskdir"])
    ap.add_argument("--sam-model", default="mobile_sam.pt")
    ap.add_argument("--yolo-model", default="yolo11n-seg.pt")
    ap.add_argument("--efficientsam-model", default=str(PACKAGE / "models" / "efficientsam_ti_2025april.onnx"),
                    help="EfficientSAM-Ti ONNX file for --segmenter efficientsam")
    ap.add_argument("--mask-dir")
    ap.add_argument("--device", default=None, help="torch device for sam/yolo, e.g. cpu or 0")
    ap.add_argument("--box", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                    help="fixed object box in pixels when the bag has no detections")
    ap.add_argument("--image-topic")
    ap.add_argument("--cloud-topic")
    ap.add_argument("--detection-topic")
    ap.add_argument("--max-frames", type=int, default=40)
    ap.add_argument("--max-pair-gap", type=float, default=0.2, help="max image/cloud time gap [s]")
    ap.add_argument("--max-object-motion", type=float, default=0.10,
                    help="skip frames where the Vicon-tracked object moved more than this between the "
                         "image and the ToF frame [m]")
    ap.add_argument("--max-detection-gap", type=float, default=1.0,
                    help="max detection/image time gap [s]; detections arrive in bursts over Wi-Fi, "
                         "and a hovering snapshot tolerates a stale box (lower it for moving scenes)")
    ap.add_argument("--plots", type=int, default=8, help="how many frame plots to save")
    ap.add_argument("--config", default=str(PACKAGE / "config" / "llm_vision_planner.yaml"))
    ap.add_argument("--calibration", default=str(PACKAGE / "calibration" / "opencv_hires_small_grey_intrinsics.yml"))
    ap.add_argument("--edge-offset-px", type=float, default=0.0,
                    help="grow (+) or shrink (-) outline edges, to correct a known segmentation bias")
    ap.add_argument("--truth-width", type=float, help="measured object width (Vicon object X) [m]")
    ap.add_argument("--truth-depth", type=float, help="measured object depth (Vicon object Y) [m]")
    ap.add_argument("--vicon-vehicle", help="TF frame of the drone (default: auto, contains 'starling')")
    ap.add_argument("--vicon-object", help="TF frame of the object (default: auto, contains the label)")
    ap.add_argument("--vicon-vehicle-axes", default="flu", choices=["flu", "frd"])
    ap.add_argument("--align-tof", default="report", choices=["report", "apply", "off"],
                    help="fit the camera/ToF misalignment from the outlines: report it, or also correct it")
    ap.add_argument("--tof-rotation-deg", type=float, nargs=3, metavar=("RX", "RY", "RZ"),
                    help="known ToF-to-camera rotation correction (camera x, y, z axes) to apply first")
    ap.add_argument("--unobserved-depth", default="conservative", choices=["conservative", "typical"],
                    help="class depth to use when the object's back isn't visible (flat face seen square-on)")
    ap.add_argument("--reader", default="auto", choices=["auto", "ros", "builtin"])
    ap.add_argument("--msg-dir", action="append", default=[], help="extra folder of .msg files (builtin reader)")
    ap.add_argument("--export-sample", metavar="FILE.npz",
                    help="also save the selected frames (images, clouds, detections, transforms) to a small file")
    ap.add_argument("--export-only", action="store_true", help="with --export-sample: stop after saving it")
    args = ap.parse_args()
    if args.label:
        args.label = args.label.strip().lower()

    bag = Path(args.bag)
    name = bag.stem if bag.is_file() else bag.name
    out = Path(args.out) if args.out else bag.parent / f"{name}_outline_depth"
    out.mkdir(parents=True, exist_ok=True)
    data = load_sample(bag) if bag.suffix == ".npz" else load_bag(bag, args)
    pairs, images, clouds = data["pairs"], data["images"], data["clouds"]
    detections, history, t0_bag = data["detections"], data["history"], data["t0"]
    if args.export_sample:
        save_sample(args.export_sample, data)
        print(f"wrote sample {args.export_sample} ({Path(args.export_sample).stat().st_size / 1e6:.1f} MB): "
              f"{len(pairs)} image/cloud pairs, detections and transforms")
        if args.export_only:
            return

    base_camera, tof_r, tof_t, camera_config = load_camera(args.config, args.calibration)
    fixed_rotation = small_rotation(*args.tof_rotation_deg) if args.tof_rotation_deg else np.eye(3)
    segmenter = make_segmenter(args.segmenter, prompts_available=bool(detections) or bool(args.box),
                               sam_model=args.sam_model, yolo_model=args.yolo_model, mask_dir=args.mask_dir,
                               device=args.device, label=args.label, efficientsam_model=args.efficientsam_model)
    params = OutlineParams(edge_offset_px=args.edge_offset_px, unobserved_depth=args.unobserved_depth)
    if getattr(segmenter, "name", "") == "tof":
        print("note: the tof outline is drawn from the ToF itself, so it adds nothing the ToF doesn't "
              "already show: depths fall back to the visible extent and the class depth. Use "
              "--segmenter efficientsam, sam or yolo for a real outline")

    # Pass 1: outline every frame and check how well the ToF lines up with it.
    records = []
    for frame, (ct, it) in enumerate(pairs):
        rgb = images[it]
        h, w = rgb.shape[:2]
        camera = base_camera.scaled_to(w, h)
        pc = tof_to_camera(clouds[ct], tof_r, tof_t, camera)
        floor = fit_floor_plane(pc, camera.expected_down())
        prompts = [(label, conf, br.box_to_pixels(vals, w, h))
                   for _, label, conf, vals, _ in matching_detections(detections, it, args)]
        # The detector can repeat one object with a similar or nested box: keep the most confident.
        kept = []
        for p in sorted(prompts, key=lambda q: -q[1]):
            if all(p[0] != k[0] or not same_object(p[2], k[2]) for k in kept):
                kept.append(p)
        prompts = kept
        if not prompts and args.box:
            prompts = [(args.label or "object", 1.0, tuple(args.box))]
        context = dict(camera=camera, points_cam=pc, floor=floor, frame_index=frame, stamp=it)
        moved = object_motion(history, args, it, ct)
        if moved is not None and moved > args.max_object_motion:
            print(f"frame {frame}: the {args.label or 'object'} moved {moved * 100:.0f} cm (Vicon) between the "
                  f"image and the ToF frame, {abs(ct - it) / 1e6:.0f} ms apart: skipped")
            continue
        if not prompts and getattr(segmenter, "name", "") in ("sam", "efficientsam", "tof"):
            print(f"frame {frame}: no {args.label or 'matching'} detection within "
                  f"{args.max_detection_gap:.1f} s of this image, skipped")
            continue
        started = time.perf_counter()
        masks = segmenter.segment(rgb, prompts, context)
        seg_s = time.perf_counter() - started
        if not prompts and args.label:
            masks = [m for m in masks if m.label == args.label]
        if not masks:
            print(f"frame {frame}: no outline ({'no detections' if not prompts else 'segmenter found nothing'})")
        fits = []
        if args.align_tof != "off" and getattr(segmenter, "name", "") != "tof":
            # (not for the tof outline: it is made from the ToF, so it can't check it)
            fits = [fit_tof_alignment(m.mask, camera, pc @ fixed_rotation.T, floor) for m in masks]
        records.append(dict(frame=frame, ct=ct, it=it, rgb=rgb, camera=camera, pc=pc, masks=masks,
                            seg_s=seg_s, fits=fits, moved=moved))

    rotation, alignment = fixed_rotation, None
    fits = [f for r in records for f in r["fits"] if f is not None]
    if fits:
        angles = np.median([f["angles_deg"] for f in fits], axis=0)
        alignment = dict(frames=len(fits), angles_deg=[round(float(a), 2) for a in angles],
                         spread_deg=[round(float(a), 2) for a in np.ptp([f["angles_deg"] for f in fits], axis=0)],
                         f1_before=round(float(np.median([f["f1_before"] for f in fits])), 3),
                         f1_after=round(float(np.median([f["f1_after"] for f in fits])), 3),
                         shift_px=[round(float(a), 1) for a in np.median([f["shift_px"] for f in fits], axis=0)])
        total = small_rotation(*angles) @ fixed_rotation
        roll, pitch, yaw = (float(camera_config.get(f"detection_cam_{k}_deg", d))
                            for k, d in (("roll", 0.0), ("pitch", 90.0), ("yaw", 90.0)))
        alignment["detection_cam_rpy_deg"] = [round(a, 3) for a in mount_with_correction(roll, pitch, yaw, total)[:3]]
        alignment["total_correction"] = total.tolist()
        if args.align_tof == "apply":
            rotation = total

    # Pass 2: depth from each outline.
    rows, plotted = [], 0
    jsonl = (out / "results.jsonl").open("w")
    for rec in records:
        frame, ct, it, rgb, camera, pc_raw = (rec[k] for k in ("frame", "ct", "it", "rgb", "camera", "pc"))
        pc = pc_raw @ rotation.T
        floor = fit_floor_plane(pc, camera.expected_down())
        seg_s, masks = rec["seg_s"], rec["masks"]
        for k_mask, m in enumerate(masks):
            fit = rec["fits"][k_mask] if k_mask < len(rec["fits"]) else None
            started = time.perf_counter()
            res = estimate_from_outline(m.label, m.mask, camera, pc, floor,
                                        replace(params, outline_from_tof=m.source == "tof"))
            depth_s = time.perf_counter() - started
            if res is None:
                print(f"frame {frame}: {m.label}: too few ToF points inside the outline")
                continue
            base = tof_only_depth(pc_raw, camera, m.box, m.label, fit_floor_plane(pc_raw, camera.expected_down()))
            truth = vicon_truth(history, args, it, camera, res.level, res.box_level["z0"])
            row = dict(frame=frame, time_s=round((it - t0_bag) / 1e9, 3), pair_gap_ms=round(abs(ct - it) / 1e6, 1),
                       segmenter=m.source, confidence=round(m.score, 3), **result_summary(res),
                       front_z_m=round(res.box_level["z0"], 3), back_z_m=round(res.box_level["z1"], 3),
                       tof_only_m=None if base is None else round(base.depth_m, 3),
                       tof_only_method=None if base is None else base.method,
                       tof_align_deg=None if fit is None else fit["angles_deg"],
                       tof_align_f1=None if fit is None else [fit["f1_before"], fit["f1_after"]],
                       object_moved_m=None if rec["moved"] is None else round(rec["moved"], 3),
                       segment_ms=round(1000 * seg_s / max(1, len(masks)), 1),
                       depth_ms=round(1000 * depth_s, 1), **truth)
            if "true_back_m" in truth:
                row["back_gap_m"] = round(truth["true_back_m"] - row["back_z_m"], 3)
                row["depth_error_m"] = round(row["depth_m"] - truth["true_depth_m"], 3)
                if base is not None:
                    bz = res.level.to_level(base.corners_cam)[:, 2].max()
                    row["tof_only_back_gap_m"] = round(truth["true_back_m"] - float(bz), 3)
            rows.append(row)
            jsonl.write(json.dumps(row) + "\n")
            jsonl.flush()
            msg = (f"frame {frame:3d} t={row['time_s']:7.2f}s {m.label:8s} depth {res.depth_m:.3f} m "
                   f"({res.decided_by}{'' if res.depth_observed else ', back not visible'}); "
                   f"ToF-only {row['tof_only_m']}")
            if "true_depth_m" in truth:
                msg += f"; truth {truth['true_depth_m']:.3f}, back gap {row['back_gap_m']:+.3f}"
            elif "implied_depth_m" in truth:
                msg += f"; Vicon-implied {truth['implied_depth_m']:.3f}"
            print(msg + (f"  [{'; '.join(res.warnings)}]" if res.warnings else ""))
            if plotted < args.plots:
                title = (f"frame {frame}, {m.label}: outline depth {res.depth_m:.2f} m via {res.decided_by}"
                         + ("" if res.depth_observed else " (back not visible)")
                         + (f", ToF-only {base.depth_m:.2f} m" if base is not None else ""))
                plot_frame(out / f"frame_{frame:04d}_{m.label.replace(' ', '_')}.png", rgb, m.mask, res,
                           camera, m.box, base, truth, title)
                plotted += 1
    jsonl.close()
    if not rows:
        raise SystemExit("no frames produced a result (see the per-frame messages above)")

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (out / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})

    summary = {}
    for label in sorted({r["label"] for r in rows}):
        sub = [r for r in rows if r["label"] == label]
        entry = {"frames": len(sub)}
        for key in ["depth_m", "tof_only_m"] + [f"{m}_m" for m in METHODS] + [
                "true_depth_m", "implied_depth_m", "back_gap_m", "tof_only_back_gap_m", "camera_height_m",
                "vicon_camera_height_m", "alignment", "edge_sensitivity_cm_per_px", "segment_ms", "depth_ms"]:
            vals = np.array([r[key] for r in sub if r.get(key) is not None], dtype=float)
            if len(vals):
                entry[key] = {"median": round(float(np.median(vals)), 3),
                              "p10": round(float(np.percentile(vals, 10)), 3),
                              "p90": round(float(np.percentile(vals, 90)), 3)}
        entry["decided_by"] = {k: sum(r["decided_by"] == k for r in sub) for k in {r["decided_by"] for r in sub}}
        entry["depth_observed_frames"] = sum(bool(r["depth_observed"]) for r in sub)
        summary[label] = entry
    (out / "summary.json").write_text(json.dumps(
        {**summary, "tof_alignment": None if alignment is None else
         {**{k: v for k, v in alignment.items() if k != "total_correction"},
          "applied": bool(args.align_tof == "apply")}}, indent=2))
    plot_summary(out / "summary.png", rows)

    print("\nsummary (median, 10th-90th percentile over frames)")
    for label, entry in summary.items():
        print(f"  {label}: {entry['frames']} frames, decided by {entry['decided_by']}")
        print(f"    back visible (depth measured) in {entry['depth_observed_frames']} of {entry['frames']} frames"
              + ("" if entry['depth_observed_frames'] == entry['frames'] else
                 "; frames without it use the conservative class depth"))
        for key, name in (("depth_m", "outline depth"), ("tof_only_m", "ToF-only depth"),
                          ("true_depth_m", "Vicon truth"), ("implied_depth_m", "Vicon-implied depth"),
                          ("back_gap_m", "back gap (+ = box misses)"), ("tof_only_back_gap_m", "ToF-only back gap"),
                          ("camera_height_m", "camera height (ToF floor)"),
                          ("vicon_camera_height_m", "camera height (Vicon)"), ("alignment", "ToF/outline alignment"),
                          ("edge_sensitivity_cm_per_px", "ellipse depth change per px of outline [cm]"),
                          ("segment_ms", "outline (segmentation) time [ms]"), ("depth_ms", "depth math time [ms]")):
            if key in entry:
                e = entry[key]
                print(f"    {name:44s} {e['median']:7.3f}  ({e['p10']:.3f} .. {e['p90']:.3f})")
    if alignment:
        a = alignment
        print(f"\ncamera/ToF alignment, from {a['frames']} outline(s):")
        print(f"  outline/ToF agreement (F1) {a['f1_before']:.3f} -> {a['f1_after']:.3f} after rotating the ToF "
              f"rx {a['angles_deg'][0]:+.2f}, ry {a['angles_deg'][1]:+.2f}, rz {a['angles_deg'][2]:+.2f} deg "
              f"(camera axes), {a['shift_px'][0]:+.1f} px right / {a['shift_px'][1]:+.1f} px down at the object")
        if a["frames"] > 1:
            print(f"  spread between outlines: {a['spread_deg']} deg")
        print("  applied to the outline depths above" if args.align_tof == "apply" else
              "  not applied to the depths above: rerun with --align-tof apply to use it")
        if max(abs(v) for v in a["angles_deg"]) >= 0.5:
            r, pt, y = a["detection_cam_rpy_deg"]
            print("  if other bags give the same correction, make it permanent for perception too, in "
                  "config/llm_vision_planner.yaml:\n"
                  f"    detection_cam_roll_deg: {r}\n    detection_cam_pitch_deg: {pt}\n    detection_cam_yaw_deg: {y}\n"
                  "  (the same small correction; the angles look large because pitch 90 deg is a gimbal-lock "
                  "point of this angle convention. ToF positions in the body frame don't change.)")
    print(f"\nwrote {out}/results.csv, summary.json, summary.png and {plotted} frame plots")


if __name__ == "__main__":
    main()
