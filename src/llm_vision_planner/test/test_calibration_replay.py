#!/usr/bin/env python3
"""Record the residual calibration dataset from a replayed session.

Runs perception_detection.py, interactive_mission_gateway.py and
residual_calibration_recorder.py with the parameters that
launch/vision_error_calibration.launch.py gives them for `replay:=true`
(read from the launch file and config/llm_vision_planner.yaml), on a
simulated hover in front of a person and a stop sign that are moved between
captures. ROS is replaced by an in-process stand-in: messages are delivered in
time order and the ROS clock follows the recording, as `ros2 bag play --clock`
does with use_sim_time, while wall time keeps running as it would during a
replay. Two deviations keep the run deterministic: the outline worker thread
is replaced by inline computation (or by a stand-in that delivers each result
a set time after its frame, as a slow laptop would), and the segmenter returns
the renderer's exact outline (the EfficientSAM model is not part of the repo).

Checks: no takeoff node and no mission-state gate in replay; the CSV has the
same columns as fine_tuning/datasets/calibration_residual_raw_person.csv;
rows are synchronized with the Vicon stamps of the recording; perceived
footprints come from perception's measured depth (no GPT call), with the stop
sign 10 mm deep.
"""

from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time as real_time
import types
from types import SimpleNamespace

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
BAG_START = 1791737539.0          # recording time (2026-10-11), as in the person bag
VOXL_OFFSET_S = -1234.5           # the VOXL clock in sensor headers differs from the ground station's


# ----------------------------------------------------------------- stand-in ROS
class Bus:
    def __init__(self):
        self.subscribers = defaultdict(list)
        self.published = defaultdict(list)
        self.timers = []
        self.t = BAG_START

    def publish(self, topic, msg):
        self.published[topic].append(msg)
        for callback in list(self.subscribers[topic]):
            callback(msg)

    def run_timers_until(self, t):
        while True:
            due = [timer for timer in self.timers if timer[1] <= t]
            if not due:
                break
            timer = min(due, key=lambda item: item[1])
            self.t = timer[1]
            timer[1] += timer[0]
            timer[2]()
        self.t = t


BUS = Bus()
NODE_PARAMS = {}
LOG = []


class FakeTime:
    """Wall and monotonic clocks during a 1x replay: they advance with the bag
    but are not the bag's time."""

    wall_start = real_time.time()

    def monotonic(self):
        return 5000.0 + BUS.t - BAG_START

    def time(self):
        return self.wall_start + BUS.t - BAG_START

    perf_counter = staticmethod(real_time.perf_counter)
    sleep = staticmethod(lambda s: None)


class Logger:
    def __init__(self, name):
        self.name = name

    def _log(self, level, text):
        LOG.append((self.name, level, str(text)))

    def info(self, text):
        self._log("info", text)

    def warning(self, text):
        self._log("warning", text)

    def error(self, text):
        self._log("error", text)

    def debug(self, text):
        self._log("debug", text)


class FakeNode:
    def __init__(self, name, *args, **kwargs):
        self._name = name
        self._params = {}

    def declare_parameter(self, name, value=None, *args, **kwargs):
        self._params[name] = NODE_PARAMS.get(self._name, {}).get(name, value)
        return SimpleNamespace(value=self._params[name])

    def declare_parameters(self, namespace="", parameters=()):
        for item in parameters:
            self.declare_parameter(item[0], item[1] if len(item) > 1 else None)

    def get_parameter(self, name):
        return SimpleNamespace(value=self._params[name])

    def get_name(self):
        return self._name

    def create_subscription(self, msg_type, topic, callback, qos=None):
        BUS.subscribers[topic].append(callback)

    def create_publisher(self, msg_type, topic, qos=None):
        return SimpleNamespace(publish=lambda msg, topic=topic: BUS.publish(topic, msg))

    def create_timer(self, period, callback):
        BUS.timers.append([float(period), BUS.t + float(period), callback])

    def get_logger(self):
        return Logger(self._name)

    def get_clock(self):
        return SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=int(round(BUS.t * 1e9))))

    def destroy_node(self):
        return None


def message_class(name, **constants):
    def init(self, **fields):
        self.__dict__.update(fields)
    return type(name, (), {"__init__": init, **constants})


def install_fake_ros():
    def module(name, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        sys.modules[name] = mod
        return mod

    enum = SimpleNamespace(BEST_EFFORT=0, RELIABLE=1, KEEP_LAST=0, TRANSIENT_LOCAL=1, VOLATILE=0)
    module("rclpy", init=lambda *a, **k: None, shutdown=lambda *a, **k: None, ok=lambda: True,
           spin=lambda *a, **k: None)
    module("rclpy.node", Node=FakeNode)
    module("rclpy.qos", QoSProfile=lambda **k: SimpleNamespace(**k), QoSReliabilityPolicy=enum,
           QoSHistoryPolicy=enum, QoSDurabilityPolicy=enum)
    module("rclpy.executors", SingleThreadedExecutor=object, MultiThreadedExecutor=object)
    module("std_msgs")
    module("std_msgs.msg", String=message_class("String"))
    module("sensor_msgs")
    module("sensor_msgs.msg", Image=message_class("Image"), PointCloud2=message_class("PointCloud2"),
           PointField=message_class("PointField", FLOAT32=7))
    module("geometry_msgs")
    module("geometry_msgs.msg", TransformStamped=message_class("TransformStamped"))
    module("px4_msgs")
    module("px4_msgs.msg", VehicleOdometry=message_class("VehicleOdometry", POSE_FRAME_NED=1))
    module("voxl_msgs")
    module("voxl_msgs.msg", Aidetection=message_class("Aidetection"))
    module("openai", OpenAI=lambda *a, **k: (_ for _ in ()).throw(AssertionError("OpenAI must not be used")))


# ----------------------------------------------------------- launch file reader
class Subst:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs


def install_fake_launch():
    def module(name, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        sys.modules[name] = mod

    names = ("LaunchConfiguration", "PathJoinSubstitution", "PythonExpression", "FindPackageShare",
             "ParameterValue", "DeclareLaunchArgument", "TimerAction", "IfCondition", "UnlessCondition",
             "LaunchDescription", "Node")
    classes = {name: type(name, (Subst,), {}) for name in names}
    module("launch", LaunchDescription=classes["LaunchDescription"])
    module("launch.actions", DeclareLaunchArgument=classes["DeclareLaunchArgument"],
           TimerAction=classes["TimerAction"])
    module("launch.conditions", IfCondition=classes["IfCondition"], UnlessCondition=classes["UnlessCondition"])
    module("launch.substitutions", LaunchConfiguration=classes["LaunchConfiguration"],
           PathJoinSubstitution=classes["PathJoinSubstitution"], PythonExpression=classes["PythonExpression"])
    module("launch_ros")
    module("launch_ros.actions", Node=classes["Node"])
    module("launch_ros.parameter_descriptions", ParameterValue=classes["ParameterValue"])
    module("launch_ros.substitutions", FindPackageShare=classes["FindPackageShare"])


def read_launch(path, arguments):
    """{executable: parameters} for the nodes the launch file starts with `arguments`."""
    install_fake_launch()
    namespace = {}
    exec(compile(Path(path).read_text(), str(path), "exec"), namespace)
    description = namespace["generate_launch_description"]()
    values = {}

    def resolve(value):
        kind = type(value).__name__
        if isinstance(value, (list, tuple)):
            return "".join(str(resolve(item)) for item in value)
        if not isinstance(value, Subst):
            return value
        if kind == "LaunchConfiguration":
            return values[value.args[0]]
        if kind == "FindPackageShare":
            return str(ROOT)
        if kind == "PathJoinSubstitution":
            return os.path.join(*[str(resolve(item)) for item in value.args[0]])
        if kind == "PythonExpression":
            return str(eval("".join(str(resolve(item)) for item in value.args[0])))
        if kind == "ParameterValue":
            raw = str(resolve(value.args[0]))
            value_type = value.kwargs.get("value_type", str)
            return yaml.safe_load(raw) if value_type is bool else value_type(raw)
        raise TypeError(kind)

    def truth(condition):
        if condition is None:
            return True
        flag = str(resolve(condition.args[0])).strip().lower() in ("true", "1")
        return flag if type(condition).__name__ == "IfCondition" else not flag

    nodes = {}

    def visit(entity):
        kind = type(entity).__name__
        if kind == "DeclareLaunchArgument":
            name = entity.args[0]
            values[name] = str(arguments[name]) if name in arguments else str(resolve(entity.kwargs["default_value"]))
        elif kind == "TimerAction":
            for action in entity.kwargs["actions"]:
                visit(action)
        elif kind == "Node" and truth(entity.kwargs.get("condition")):
            params = {}
            for item in entity.kwargs.get("parameters", []):
                if isinstance(item, dict):
                    params.update({key: resolve(val) for key, val in item.items()})
                else:
                    data = yaml.safe_load(Path(resolve(item)).read_text()) or {}
                    params.update(data.get(entity.kwargs["name"], {}).get("ros__parameters", {}))
            nodes[entity.kwargs["executable"]] = params

    for entity in description.args[0]:
        visit(entity)
    return nodes


# ------------------------------------------------------------------ scenario
def import_nodes():
    install_fake_ros()
    for folder in (ROOT / "scripts", ROOT / "fine_tuning" / "scripts", ROOT / "test"):
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
    import interactive_mission_gateway as gateway
    import perception_detection as perception
    import residual_calibration_recorder as recorder
    for mod in (gateway, perception, recorder):
        mod.time = FakeTime()
    return perception, gateway, recorder


def stamp(t):
    return SimpleNamespace(sec=int(math.floor(t)), nanosec=int(round((t - math.floor(t)) * 1e9)) % 1_000_000_000)


VICON_FROM_NED = np.diag([1.0, -1.0, -1.0])     # Vicon world: x north, y west, z up
VICON_ORIGIN_NED = np.array([1.0, -0.5, 0.0])   # where the Vicon origin is in local NED


def vicon_pose(t, ned_position, yaw_ned):
    """TransformStamped of a level marker set (FLU body axes) at an NED pose."""
    position = VICON_FROM_NED @ (np.asarray(ned_position, float) - VICON_ORIGIN_NED)
    half = -0.5 * yaw_ned            # NED yaw (about down) is minus the yaw about Vicon up
    return SimpleNamespace(
        header=SimpleNamespace(stamp=stamp(t), frame_id="vicon/vicon_world"),
        transform=SimpleNamespace(
            translation=SimpleNamespace(x=position[0], y=position[1], z=position[2]),
            rotation=SimpleNamespace(x=0.0, y=0.0, z=math.sin(half), w=math.cos(half)),
        ),
    )


def cloud_message(t, points):
    points = np.asarray(points, dtype="<f4")
    return SimpleNamespace(
        header=SimpleNamespace(stamp=stamp(t + VOXL_OFFSET_S), frame_id="world"),
        height=1, width=len(points), is_bigendian=False, point_step=12, row_step=12 * len(points),
        fields=[SimpleNamespace(name=n, offset=4 * i, datatype=7, count=1) for i, n in enumerate("xyz")],
        data=points.tobytes(), is_dense=True,
    )


def image_message(t, rgb):
    return SimpleNamespace(header=SimpleNamespace(stamp=stamp(t + VOXL_OFFSET_S), frame_id="hires"),
                           height=rgb.shape[0], width=rgb.shape[1], encoding="rgb8", is_bigendian=0,
                           step=3 * rgb.shape[1], data=np.ascontiguousarray(rgb).tobytes())


def footprint(truth_body):
    """Centre and (width across, depth along) of a simulated object's floor footprint."""
    lo, hi = truth_body[:, :2].min(axis=0), truth_body[:, :2].max(axis=0)
    return 0.5 * (lo + hi), (float(hi[1] - lo[1]), float(hi[0] - lo[0]))


SCENES = [   # person (x_front, y), stop sign (x_front, y), held for 7 s each
    ((2.0, 0.0), (2.6, -0.75)),
    ((2.0, 0.35), (2.6, -0.75)),
    ((2.0, 0.65), (2.4, 0.0)),        # sign straight ahead, square-on
    ((1.9, 0.45), (2.8, -0.55)),
]
HOVER_NED = np.array([0.0, 0.0, -0.5])   # facing north, 0.5 m above the floor


def build_session(scene_lib, scenes=None):
    rng = np.random.default_rng(5)
    rendered, events = [], []
    t = BAG_START
    for vehicle_t in np.arange(t, t + 3.0, 0.02):       # still on the pad: frames align
        events.append((vehicle_t, "vehicle", None))
    t += 3.0
    for (person_x, person_y), (sign_x, sign_y) in scenes or SCENES:
        prims = scene_lib.person(x_front=person_x, y=person_y) + scene_lib.stop_sign(x_front=sign_x, y=sign_y)
        scene = scene_lib.make_scene(prims)
        camera, tof_r, tof_t, ids, rgb, tof = scene_lib.capture(scene, rng)
        objects = {}
        for object_id, obj, label in (("person-1", 2, "person"), ("stop-sign-1", 5, "stop sign")):
            center, dims = footprint(scene.truth[obj])
            vs, us = np.nonzero(ids == obj)
            objects[object_id] = dict(label=label, center=center, dims=dims, obj=obj,
                                      box=(us.min(), vs.min(), us.max() + 1, vs.max() + 1))
        rendered.append(dict(ids=ids, rgb=rgb, tof=tof, objects=objects))
        index = len(rendered) - 1
        for k, ts in enumerate(np.arange(t, t + 7.0, 0.02)):
            events.append((ts, "vehicle", index))
            if k % 50 == 25:
                events.append((ts + 0.004, "image", index))
            if k % 25 == 5:
                events.append((ts + 0.007, "cloud", index))
            if k % 10 == 3:
                events.append((ts + 0.011, "detections", index))
        t += 7.0
    events.sort(key=lambda item: item[0])
    return rendered, events


class SceneSegmenter:
    """Stands in for EfficientSAM: the renderer's outline of the prompted object."""

    name = "efficientsam"

    def __init__(self, rendered, outline_segmentation):
        self.rendered = rendered
        self.ObjectMask = outline_segmentation.ObjectMask
        self.box_iou = outline_segmentation.box_iou

    def segment(self, rgb, prompts, context=None):
        frame = next((item for item in self.rendered if np.array_equal(item["rgb"], rgb)), None)
        if frame is None:   # perception's start-up check on a blank image
            return []
        out = []
        for label, confidence, box in prompts:
            best = max(frame["objects"].values(), key=lambda item: self.box_iou(item["box"], box))
            out.append(self.ObjectMask(label, confidence, frame["ids"] == best["obj"], tuple(best["box"]),
                                       self.name))
        return out


DATA_FILES = {"vision_error_calibration_csv": str(ROOT / "fine_tuning" / "datasets" / "calibration_vision_error_dummy.csv")}


def replay(rendered, events, nodes, output_csv, worker_latency_s=0.0):
    BUS.__init__()
    LOG.clear()
    perception_mod, gateway_mod, recorder_mod = import_nodes()
    import outline_segmentation
    perception_mod.make_segmenter = lambda *a, **k: SceneSegmenter(rendered, outline_segmentation)
    NODE_PARAMS["semantic_obstacle_perception"] = dict(nodes["perception_detection.py"], outline_async=False)
    # (the installed share folder that the gateway also searches doesn't exist here)
    NODE_PARAMS["interactive_mission_gateway"] = dict(nodes["interactive_mission_gateway.py"], **DATA_FILES)
    NODE_PARAMS["residual_calibration_recorder"] = dict(nodes["residual_calibration_recorder.py"],
                                                        output_csv=str(output_csv))
    intent_calls = []
    recorder = recorder_mod.ResidualCalibrationRecorder()
    perception = perception_mod.SemanticObstaclePerception()   # no worker thread: outline_async is off
    gateway = gateway_mod.InteractiveMissionGateway()
    gateway.start_intent_request = lambda *a, **k: intent_calls.append(a)
    if worker_latency_s > 0.0:
        # Jobs queue as with the worker thread; one runs at a time and its
        # result appears worker_latency_s after it started.
        perception._params["outline_async"] = True
        busy = {"job": None, "until": None}

        def worker():
            if busy["job"] is not None and BUS.t >= busy["until"]:
                perception.run_outline_job(busy["job"])
                busy["job"] = None
            if busy["job"] is None and perception.outline_job is not None:
                busy["job"], perception.outline_job = perception.outline_job, None
                busy["until"] = BUS.t + worker_latency_s
        perception.create_timer(0.05, worker)
    frame_id = 0
    for t, kind, index in events:
        BUS.run_timers_until(t)
        if kind == "vehicle":
            BUS.publish("/vicon/Starling2/Starling2", vicon_pose(t, HOVER_NED, 0.0))
            BUS.publish("/fmu/out/vehicle_odometry", SimpleNamespace(
                pose_frame=1, position=HOVER_NED.tolist(), q=[1.0, 0.0, 0.0, 0.0],
                timestamp_sample=int((t + VOXL_OFFSET_S) * 1e6)))
            if index is not None:
                for object_id, item in rendered[index]["objects"].items():
                    ned = [item["center"][0], item["center"][1], -1.0]
                    BUS.publish(f"/vicon/{'person/person' if item['label'] == 'person' else 'stopsign/stopsign'}",
                                vicon_pose(t, ned, math.pi / 2))
        elif kind == "image":
            BUS.publish("/hires_small_color", image_message(t, rendered[index]["rgb"]))
        elif kind == "cloud":
            BUS.publish("/tof_pc", cloud_message(t, rendered[index]["tof"]))
        elif kind == "detections":
            frame_id += 1
            BUS.publish("/tflite", SimpleNamespace(header=SimpleNamespace(stamp=stamp(t + VOXL_OFFSET_S)),
                                                   width=1024, height=768))
            for item in rendered[index]["objects"].values():
                x1, y1, x2, y2 = item["box"]
                BUS.publish("/tflite_data", SimpleNamespace(
                    class_name=item["label"], class_confidence=0.9, detection_confidence=0.9,
                    x_min=float(x1), y_min=float(y1), x_max=float(x2), y_max=float(y2),
                    cam="hires_small_color", frame_id=frame_id, timestamp_ns=int(t * 1e9)))
    BUS.run_timers_until(events[-1][0] + 1.0)
    recorder.destroy_node()
    return perception, gateway, intent_calls


def outside_distance(points, corners):
    """Largest distance of `points` outside the convex quadrilateral `corners`."""
    hull = np.asarray(corners, dtype=float)
    center = hull.mean(axis=0)
    order = np.argsort(np.arctan2(hull[:, 1] - center[1], hull[:, 0] - center[0]))
    hull = hull[order]
    worst = 0.0
    for p in np.asarray(points, dtype=float):
        out = 0.0
        for a, b in zip(hull, np.roll(hull, -1, axis=0)):
            edge = b - a
            normal = np.array([edge[1], -edge[0]]) / np.linalg.norm(edge)
            if normal @ (center - a) > 0:
                normal = -normal
            out = max(out, float(normal @ (p - a)))
        worst = max(worst, out)
    return worst


# --------------------------------------------------------------------- tests
VICON_OBJECTS = None


def launch_arguments(output_csv, replay_flag="true"):
    return {
        "replay": replay_flag,
        "trial_id": "person-residual-replay-test",
        "output_csv": str(output_csv),
        "vicon_objects_json": json.dumps(VICON_OBJECTS),
        "params_file": str(ROOT / "config" / "llm_vision_planner.yaml"),
    }


def test_launch_file_for_replay_and_flight():
    replayed = read_launch(ROOT / "launch" / "vision_error_calibration.launch.py",
                           launch_arguments("/tmp/x.csv"))
    assert "control_law_executer.py" not in replayed            # nothing is sent to a vehicle
    assert replayed["interactive_mission_gateway.py"]["require_mission_state"] is False
    for executable in ("perception_detection.py", "interactive_mission_gateway.py",
                       "residual_calibration_recorder.py"):
        assert replayed[executable]["use_sim_time"] is True, executable
    gateway = replayed["interactive_mission_gateway.py"]
    assert gateway["nominal_depth_source"] == "perception" and gateway["intent_provider"] == "mock"
    assert replayed["perception_detection.py"]["depth_estimator"] == "outline"
    flight = read_launch(ROOT / "launch" / "vision_error_calibration.launch.py",
                         launch_arguments("/tmp/x.csv", "false"))
    assert "control_law_executer.py" in flight
    assert flight["interactive_mission_gateway.py"]["require_mission_state"] is True
    assert flight["perception_detection.py"]["use_sim_time"] is False
    gpt = read_launch(ROOT / "launch" / "vision_error_calibration.launch.py",
                      dict(launch_arguments("/tmp/x.csv"), nominal_depth_source="gpt"))
    assert gpt["interactive_mission_gateway.py"]["intent_provider"] == "openai"   # the mock parser has no depths


def run_session(scenes=None, worker_latency_s=0.0):
    """Record a simulated session through the launch's nodes; returns the CSV rows."""
    global VICON_OBJECTS
    import outline_scene as scene_lib
    rendered, events = build_session(scene_lib, scenes)
    first = rendered[0]["objects"]
    VICON_OBJECTS = [
        {"object_id": "person-1", "label": "person", "topic": "/vicon/person/person",
         "dimensions_m": list(first["person-1"]["dims"])},
        {"object_id": "stop-sign-1", "label": "stop sign", "topic": "/vicon/stopsign/stopsign",
         "dimensions_m": list(first["stop-sign-1"]["dims"])},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        output_csv = Path(tmp) / "calibration_residual_raw_replay.csv"
        nodes = read_launch(ROOT / "launch" / "vision_error_calibration.launch.py", launch_arguments(output_csv))
        _, _, intent_calls = replay(rendered, events, nodes, output_csv, worker_latency_s)
        with output_csv.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            return list(reader), reader.fieldnames, intent_calls, events


def test_replayed_session_records_the_dataset_with_measured_depths():
    rows, header, intent_calls, events = run_session()
    reference = ROOT / "fine_tuning" / "datasets" / "calibration_residual_raw_person.csv"
    with reference.open(newline="", encoding="utf-8") as stream:
        assert header == csv.DictReader(stream).fieldnames            # same dataset format
    assert not intent_calls, "calibration captures must not ask GPT for depths"
    snapshots = [json.loads(msg.data) for msg in BUS.published["/llm_vision/nominal_obstacles"]]
    assert snapshots and all(item["source"] == "perception_front_footprint" for item in snapshots)
    captures = {}
    for row in rows:
        captures.setdefault(row["capture_id"], {})[row["object_id"]] = row
    assert len(captures) >= len(SCENES) - 1, (len(captures), [m for m in LOG if m[1] != "info"][:5])
    for capture in captures.values():
        assert set(capture) == {"person-1", "stop-sign-1"}
        for object_id, row in capture.items():
            t, tv = float(row["timestamp_s"]), float(row["vicon_timestamp_s"])
            assert BAG_START <= t <= events[-1][0] + 1.0 and abs(t - tv) <= 0.1, (t, tv)   # bag time
            assert row["missed_detection"] == "false" and row["placeholder"] == "false", row
            depth = float(row["pred_chatgpt_depth_m"])
            pred = [(float(row[f"pred_corner_{i}_x"]), float(row[f"pred_corner_{i}_y"])) for i in range(4)]
            gt = [(float(row[f"gt_corner_{i}_x"]), float(row[f"gt_corner_{i}_y"])) for i in range(4)]
            missed_by = outside_distance(gt, pred)
            if object_id == "stop-sign-1":
                # 10 mm thick. Off to one side, the drone sees the plate at an
                # angle, so along the line of sight it spans width x sin(angle).
                bearing = math.atan2(float(row["gt_center_y"]) - HOVER_NED[1],
                                     float(row["gt_center_x"]) - HOVER_NED[0])
                if abs(bearing) < 0.03:
                    assert abs(depth - 0.01) < 1e-6, depth                      # square-on: exactly 10 mm
                else:
                    assert 0.01 < depth <= 0.6 * abs(math.sin(bearing)) + 0.05, (depth, bearing)
                assert missed_by < 0.03, missed_by
            else:
                assert 0.2 <= depth <= 0.6, depth
                assert missed_by < 0.08, missed_by
    sources = {obstacle["depth_estimate_source"] for item in snapshots for obstacle in item["obstacles"]}
    assert all(source.startswith("perception_outline_") for source in sources), sources


def test_a_slow_outline_never_records_the_previous_arrangement():
    """Moves just over the 0.15 m capture threshold, with each outline ready
    1.5 s after its frame (a slow laptop): a capture right after a move must
    not use the outline of where the object was."""
    scenes = [((2.0, 0.00), (2.6, -0.75)), ((2.0, 0.18), (2.6, -0.57)), ((2.0, 0.36), (2.6, -0.39)),
              ((2.0, 0.18), (2.6, -0.57)), ((2.0, 0.00), (2.6, -0.75))]
    rows, _, _, _ = run_session(scenes, worker_latency_s=1.5)
    assert len({row["capture_id"] for row in rows}) >= len(scenes) - 1, len(rows)
    for row in rows:
        pred = np.mean([[float(row[f"pred_corner_{i}_x"]), float(row[f"pred_corner_{i}_y"])] for i in range(4)], axis=0)
        error = pred[1] - float(row["gt_center_y"])
        assert abs(error) < 0.09, (row["capture_id"], row["object_id"], error)   # a stale one is off by 0.18
    # ... and the outlines are still used, not dropped for the ToF-only fit
    snapshots = [json.loads(msg.data) for msg in BUS.published["/llm_vision/nominal_obstacles"]]
    sources = {obstacle["depth_estimate_source"] for item in snapshots for obstacle in item["obstacles"]}
    assert sources and all(source.startswith("perception_outline_") for source in sources), sources


def outline_node(perception_mod):
    """A perception node with just the outline bookkeeping (no ROS)."""
    import threading
    node = object.__new__(perception_mod.SemanticObstaclePerception)
    params = {"outline_max_age_s": 10.0, "outline_wait_s": 4.0, "outline_settle_m": 0.08, "outline_max_image_gap_s": 0.5}
    node.get_parameter = lambda name: SimpleNamespace(value=params[name])
    node.outline_lock, node.outline_tracks, node.outline_used = threading.Lock(), [], set()
    node.outline_results, node.segmenter = [], object()
    return node


def test_each_object_waits_for_its_own_outline():
    """Two people: only A's outline ever arrives. B still becomes ready after
    outline_wait_s (with the ToF depth), and an object that left and came back
    waits for a fresh outline."""
    BUS.__init__()
    perception_mod, _, _ = import_nodes()
    node = outline_node(perception_mod)
    pose = {"position": np.zeros(3), "rotation_body_to_world": np.eye(3)}
    box_a, box_b = (0.10, 0.2, 0.25, 0.9), (0.60, 0.2, 0.75, 0.9)
    corners = np.array([[x, y, z] for x in (2.0, 2.3) for y in (-0.2, 0.2) for z in (-1.7, 0.0)])

    def tick(boxes):
        BUS.t += 0.5
        now = FakeTime().monotonic()
        node.outline_used = set()
        node.outline_results = [{"label": "person", "box": box_a, "source_s": now, "corners_world": corners,
                                 "front_world": np.array([2.0, -0.3, -0.85]), "view_axis_xy": np.array([1.0, 0.0]),
                                 "visible_width_m": 0.4, "depth_m": 0.3, "front_range_m": 2.0,
                                 "decided_by": "body_shape", "depth_observed": True}]
        out = []
        for box, y in boxes:
            payload = {"depth_source": "tof_projected_10of20pts_ellipse_fit", "depth_ready": True,
                       "front_surface_center": [2.0, y, -0.85]}
            node.apply_outline(payload, "person", box, pose, now)
            out.append(payload)
        node.outline_tracks = [t for t in node.outline_tracks if now - t["seen_s"] <= 1.0]   # as each publish does
        return out

    ready_b = []
    for _ in range(12):                                  # 6 s at 2 Hz
        a, b = tick([(box_a, -0.3), (box_b, 0.6)])
        assert a["depth_ready"] and a["depth_method"] == "outline_body_shape"
        ready_b.append(b["depth_ready"])
    assert ready_b[:7] == [False] * 7 and all(ready_b[8:]), ready_b    # 4 s wait, then the ToF depth
    for _ in range(120):                                 # B leaves for 60 s
        tick([(box_a, -0.3)])
    _, b = tick([(box_a, -0.3), (box_b, 0.6)])
    assert not b["depth_ready"] and b.get("outline_pending")


def test_gateway_waits_for_measured_depths_and_rejects_gpt_without_a_parser():
    BUS.__init__()
    _, gateway_mod, _ = import_nodes()
    NODE_PARAMS["interactive_mission_gateway"] = dict(
        DATA_FILES, calibration_only=True, nominal_depth_source="gpt", intent_provider="mock")
    try:
        gateway_mod.InteractiveMissionGateway()
    except ValueError as exc:
        assert "intent_provider" in str(exc)
    else:
        raise AssertionError("gpt depths with the mock parser would record nothing")
    NODE_PARAMS["interactive_mission_gateway"] = dict(DATA_FILES, intent_provider="mock")
    gateway = gateway_mod.InteractiveMissionGateway()
    gateway.latest_scene = {"healthy": True, "obstacles": [{"object_id": "obj-1", "depth_ready": False}]}
    gateway.latest_scene_received_s = FakeTime().time()
    assert "measuring object depths" in gateway.scene_context_error()
    gateway.latest_scene["obstacles"][0]["depth_ready"] = True
    assert gateway.scene_context_error() is None


if __name__ == "__main__":
    test_launch_file_for_replay_and_flight()
    print("ok  test_launch_file_for_replay_and_flight")
    test_replayed_session_records_the_dataset_with_measured_depths()
    print("ok  test_replayed_session_records_the_dataset_with_measured_depths")
    test_a_slow_outline_never_records_the_previous_arrangement()
    print("ok  test_a_slow_outline_never_records_the_previous_arrangement")
    test_each_object_waits_for_its_own_outline()
    print("ok  test_each_object_waits_for_its_own_outline")
    test_gateway_waits_for_measured_depths_and_rejects_gpt_without_a_parser()
    print("ok  test_gateway_waits_for_measured_depths_and_rejects_gpt_without_a_parser")
    print("calibration replay tests passed")
