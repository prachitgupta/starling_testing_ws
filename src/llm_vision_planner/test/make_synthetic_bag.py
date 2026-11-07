#!/usr/bin/env python3
"""Write a small synthetic ROS 2 (Humble, sqlite3) bag of a simulated scene.

Topics mirror a real Starling recording: /hires_small_color (rgb8 image),
/tof_pc (PointCloud2 in the ToF optical frame, header frame 'world'),
/tflite_data (voxl_msgs/Aidetection, normalized box), /tf with Vicon frames
(vicon/vicon_world -> vicon/Starling2/Starling2, vicon/<object>/<object>).
The drone hovers in one place for every frame, as in a real snapshot. True
object outlines are also saved as PNGs for --segmenter maskdir.

  python3 test/make_synthetic_bag.py /tmp/synthetic_person --scene person
  python3 test/make_synthetic_bag.py /tmp/synthetic_closet --scene wardrobe
"""

import argparse
import json
import math
from pathlib import Path
import sqlite3
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "test"))

import bag_reader as br  # noqa: E402
import outline_scene as S  # noqa: E402

FIXTURE_MSGS = ROOT / "test" / "msg_fixtures"
TOPICS = {
    "/hires_small_color": "sensor_msgs/msg/Image",
    "/tof_pc": "sensor_msgs/msg/PointCloud2",
    "/tflite_data": "voxl_msgs/msg/Aidetection",
    "/tf": "tf2_msgs/msg/TFMessage",
}


def header(t_ns, frame):
    return {"stamp": {"sec": int(t_ns // 1_000_000_000), "nanosec": int(t_ns % 1_000_000_000)}, "frame_id": frame}


def scene_for(kind, drone_height):
    if kind == "person":
        prims, obj, label = S.person(drone_height=drone_height), 2, "person"
        # Vicon object at the footprint centre, toes (1.92) to back (2.24);
        # X = width (lateral), Y = depth.
        center_body, width, depth, yaw = (2.08, 0.0), 0.36, 0.32, 0.0
    elif kind == "chair":
        prims, obj, label = S.chair(yaw_deg=45, drone_height=drone_height), 3, "chair"
        center_body, width, depth, yaw = (2.3, 0.0), 0.44, 0.44, 45.0
    elif kind == "wardrobe":   # square-on closet: its back isn't visible
        prims, obj, label = S.wardrobe(drone_height=drone_height), 4, "closet"
        center_body, width, depth, yaw = (2.2 + 0.3, 0.0), 1.0, 0.6, 0.0
    else:
        raise ValueError(kind)
    return S.make_scene(prims, drone_height), obj, label, center_body, width, depth, yaw


def make_bag(path, kind="person", frames=3, drone_height=0.5, seed=0):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for old in path.glob("*.db3"):
        old.unlink()
    registry = br.TypeRegistry([FIXTURE_MSGS])
    scene, obj, label, center_body, width, depth, yaw = scene_for(kind, drone_height)
    camera, tof_r, tof_t, ids, rgb, _ = S.capture(scene, np.random.default_rng(seed))
    mask = ids == obj
    vs, us = np.nonzero(mask)
    box = (us.min() / camera.width, vs.min() / camera.height, us.max() / camera.width, vs.max() / camera.height)
    masks = path / "masks"
    masks.mkdir(exist_ok=True)
    from PIL import Image as PILImage
    for i in range(frames):
        PILImage.fromarray((mask * 255).astype(np.uint8)).save(masks / f"{i:06d}.png")

    db_name = f"{path.name}_0.db3"
    db = sqlite3.connect(path / db_name)
    db.executescript(
        "CREATE TABLE schema(schema_version INTEGER PRIMARY KEY,ros_distro TEXT NOT NULL);"
        "INSERT INTO schema VALUES(3,'humble');"
        "CREATE TABLE topics(id INTEGER PRIMARY KEY,name TEXT NOT NULL,type TEXT NOT NULL,"
        "serialization_format TEXT NOT NULL,offered_qos_profiles TEXT NOT NULL);"
        "CREATE TABLE messages(id INTEGER PRIMARY KEY,topic_id INTEGER NOT NULL,timestamp INTEGER NOT NULL,"
        "data BLOB NOT NULL);CREATE INDEX timestamp_idx ON messages (timestamp ASC);")
    for i, (name, typ) in enumerate(TOPICS.items(), start=1):
        db.execute("INSERT INTO topics VALUES(?,?,?,?,?)", (i, name, typ, "cdr", ""))
    ids_by_topic = {name: i for i, name in enumerate(TOPICS, start=1)}
    counts = {name: 0 for name in TOPICS}
    t0 = 1_791_000_000_000_000_000
    rng = np.random.default_rng(seed + 1)

    def write(topic, t_ns, obj_dict):
        db.execute("INSERT INTO messages(topic_id,timestamp,data) VALUES(?,?,?)",
                   (ids_by_topic[topic], int(t_ns), serialize(topic, obj_dict)))
        counts[topic] += 1

    def serialize(topic, obj_dict):
        return br.serialize(obj_dict, TOPICS[topic], registry)

    qz = math.sin(math.radians(yaw - 90) / 2), math.cos(math.radians(yaw - 90) / 2)
    for i in range(frames):
        t = t0 + i * 500_000_000
        tof = S.render_tof(scene, tof_r, tof_t, rng)
        cloud = tof.astype("<f4")
        write("/tof_pc", t, {
            "header": header(t, "world"), "height": 1, "width": len(cloud),
            "fields": [{"name": n, "offset": 4 * k, "datatype": 7, "count": 1} for k, n in enumerate("xyz")],
            "is_bigendian": False, "point_step": 12, "row_step": 12 * len(cloud),
            "data": np.frombuffer(cloud.tobytes(), dtype=np.uint8), "is_dense": True})
        write("/hires_small_color", t + 4_000_000, {
            "header": header(t + 4_000_000, "hires"), "height": camera.height, "width": camera.width,
            "encoding": "rgb8", "is_bigendian": 0, "step": camera.width * 3,
            "data": np.frombuffer(rgb.tobytes(), dtype=np.uint8)})
        write("/tflite_data", t + 6_000_000, {
            "timestamp_ns": t, "class_id": 0, "frame_id": i, "class_name": label, "cam": "hires_small_color",
            "class_confidence": 0.9, "detection_confidence": 0.9,
            "x_min": box[0], "y_min": box[1], "x_max": box[2], "y_max": box[3]})
        # Vicon: world z up, floor at 0; drone FLU marker over the world origin.
        tf = {"transforms": [
            {"header": header(t, "vicon/vicon_world"), "child_frame_id": "vicon/Starling2/Starling2",
             "transform": {"translation": {"x": 0.0, "y": 0.0, "z": drone_height},
                           "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}},
            {"header": header(t, "vicon/vicon_world"), "child_frame_id": f"vicon/{label}/{label}",
             "transform": {"translation": {"x": center_body[0], "y": -center_body[1], "z": 0.0},
                           "rotation": {"x": 0.0, "y": 0.0, "z": qz[0], "w": qz[1]}}}]}
        write("/tf", t + 1_000_000, tf)
    db.commit()
    db.close()

    total = sum(counts.values())
    end = t0 + (frames - 1) * 500_000_000 + 6_000_000
    lines = ["rosbag2_bagfile_information:", "  version: 5", "  storage_identifier: sqlite3",
             f"  duration:\n    nanoseconds: {end - t0}", f"  starting_time:\n    nanoseconds_since_epoch: {t0}",
             f"  message_count: {total}", "  topics_with_message_count:"]
    for name, typ in TOPICS.items():
        lines += [f"    - topic_metadata:\n        name: {name}\n        type: {typ}\n"
                  f"        serialization_format: cdr\n        offered_qos_profiles: \"\"\n"
                  f"      message_count: {counts[name]}"]
    lines += ["  compression_format: \"\"", "  compression_mode: \"\"", "  relative_file_paths:",
              f"    - {db_name}", "  files:", f"    - path: {db_name}",
              f"      starting_time:\n        nanoseconds_since_epoch: {t0}",
              f"      duration:\n        nanoseconds: {end - t0}", f"      message_count: {total}"]
    (path / "metadata.yaml").write_text("\n".join(lines) + "\n")
    truth = {"label": label, "width": width, "depth": depth, "camera_height": drone_height + 0.015}
    (path / "truth.json").write_text(json.dumps(truth))
    return truth


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--scene", default="person", choices=["person", "chair", "wardrobe"])
    ap.add_argument("--frames", type=int, default=3)
    a = ap.parse_args()
    print(make_bag(a.out, a.scene, a.frames))
