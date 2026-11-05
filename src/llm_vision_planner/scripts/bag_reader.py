"""Read ROS 2 bags for offline analysis, with or without a ROS install.

Two readers give the same interface, (topic, log_time_ns, message) tuples
with attribute access (msg.header.stamp.sec, msg.data, ...):

  ros      rosbag2_py + rclpy deserialisation. Use inside the ROS container
           after sourcing install/setup.bash; handles .db3 and .mcap and any
           message type the workspace provides.
  builtin  sqlite3 + a small CDR decoder driven by .msg definitions. Reads
           .db3 bags anywhere Python runs. Humble bags don't store message
           definitions, so custom types (px4_msgs, voxl_msgs) are loaded from
           the workspace's msg folders, found automatically or via msg_dirs.

Also converts the message types this package uses into numpy.
"""

from pathlib import Path
import re
import sqlite3
import struct
from types import SimpleNamespace

import numpy as np


# ------------------------------------------------------------ definitions
STANDARD_DEFINITIONS = {
    "builtin_interfaces/msg/Time": "int32 sec\nuint32 nanosec",
    "builtin_interfaces/msg/Duration": "int32 sec\nuint32 nanosec",
    "std_msgs/msg/Header": "builtin_interfaces/Time stamp\nstring frame_id",
    "std_msgs/msg/String": "string data",
    "sensor_msgs/msg/Image": "std_msgs/Header header\nuint32 height\nuint32 width\nstring encoding\n"
                             "uint8 is_bigendian\nuint32 step\nuint8[] data",
    "sensor_msgs/msg/CompressedImage": "std_msgs/Header header\nstring format\nuint8[] data",
    "sensor_msgs/msg/PointField": "string name\nuint32 offset\nuint8 datatype\nuint32 count",
    "sensor_msgs/msg/PointCloud2": "std_msgs/Header header\nuint32 height\nuint32 width\n"
                                   "PointField[] fields\nbool is_bigendian\nuint32 point_step\n"
                                   "uint32 row_step\nuint8[] data\nbool is_dense",
    "sensor_msgs/msg/RegionOfInterest": "uint32 x_offset\nuint32 y_offset\nuint32 height\nuint32 width\n"
                                        "bool do_rectify",
    "sensor_msgs/msg/CameraInfo": "std_msgs/Header header\nuint32 height\nuint32 width\n"
                                  "string distortion_model\nfloat64[] d\nfloat64[9] k\nfloat64[9] r\n"
                                  "float64[12] p\nuint32 binning_x\nuint32 binning_y\nRegionOfInterest roi",
    "geometry_msgs/msg/Vector3": "float64 x\nfloat64 y\nfloat64 z",
    "geometry_msgs/msg/Point": "float64 x\nfloat64 y\nfloat64 z",
    "geometry_msgs/msg/Quaternion": "float64 x 0\nfloat64 y 0\nfloat64 z 0\nfloat64 w 1",
    "geometry_msgs/msg/Pose": "Point position\nQuaternion orientation",
    "geometry_msgs/msg/PoseStamped": "std_msgs/Header header\nPose pose",
    "geometry_msgs/msg/Transform": "Vector3 translation\nQuaternion rotation",
    "geometry_msgs/msg/TransformStamped": "std_msgs/Header header\nstring child_frame_id\n"
                                          "Transform transform",
    "tf2_msgs/msg/TFMessage": "geometry_msgs/TransformStamped[] transforms",
}

PRIMITIVES = {
    "bool": ("?", 1), "byte": ("B", 1), "char": ("B", 1), "uint8": ("B", 1), "int8": ("b", 1),
    "uint16": ("H", 2), "int16": ("h", 2), "uint32": ("I", 4), "int32": ("i", 4),
    "uint64": ("Q", 8), "int64": ("q", 8), "float32": ("f", 4), "float64": ("d", 8),
}
NUMPY_TYPES = {"bool": "u1", "byte": "u1", "char": "u1", "uint8": "u1", "int8": "i1",
               "uint16": "u2", "int16": "i2", "uint32": "u4", "int32": "i4",
               "uint64": "u8", "int64": "i8", "float32": "f4", "float64": "f8"}


def _normalize(type_name, package):
    t = type_name.strip()
    if t == "Header":
        return "std_msgs/msg/Header"
    if t in PRIMITIVES or t.startswith("string") or t.startswith("wstring"):
        return t
    parts = t.split("/")
    if len(parts) == 1:
        return f"{package}/msg/{parts[0]}"
    if len(parts) == 2:
        return f"{parts[0]}/msg/{parts[1]}"
    return t


class TypeRegistry:
    """Parsed .msg definitions: full name -> list of (field, base, array) where
    array is None (scalar), -1 (unbounded or bounded sequence) or N (fixed)."""

    def __init__(self, msg_dirs=()):
        self.texts = dict(STANDARD_DEFINITIONS)
        self.fields = {}
        for folder in msg_dirs:
            for path in Path(folder).expanduser().rglob("*.msg"):
                if path.parent.name == "msg":
                    self.texts[f"{path.parent.parent.name}/msg/{path.stem}"] = path.read_text(errors="replace")

    def knows(self, name):
        try:
            self.definition(name)
            return True
        except KeyError:
            return False

    def definition(self, name):
        if name in self.fields:
            return self.fields[name]
        if name not in self.texts:
            raise KeyError(name)
        package = name.split("/")[0]
        out = []
        for raw in self.texts[name].splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            tokens = line.split()
            if len(tokens) < 2 or "=" in tokens[1] or (len(tokens) > 2 and tokens[2].startswith("=")):
                continue   # constant
            type_token, field_name = tokens[0], tokens[1].split("=")[0]
            array = None
            m = re.match(r"^(.*?)\[(<=)?(\d*)\]$", type_token)
            if m:
                type_token = m.group(1)
                array = -1 if (m.group(2) or not m.group(3)) else int(m.group(3))
            base = re.sub(r"<=\d+$", "", type_token)
            base = _normalize(base, package)
            if base not in PRIMITIVES and not base.startswith(("string", "wstring")):
                self.definition(base)   # raises KeyError if unknown
            out.append((field_name, base, array))
        self.fields[name] = out
        return out


# -------------------------------------------------------------------- CDR
class _Cdr:
    def __init__(self, data):
        self.buf = memoryview(data)
        self.endian = "<" if data[1] in (1, 3) else ">"
        self.pos = 4

    def align(self, n):
        self.pos += (-(self.pos - 4)) % n

    def prim(self, base):
        fmt, size = PRIMITIVES[base]
        self.align(size)
        value = struct.unpack_from(self.endian + fmt, self.buf, self.pos)[0]
        self.pos += size
        return value

    def string(self):
        self.align(4)
        n = struct.unpack_from(self.endian + "I", self.buf, self.pos)[0]
        self.pos += 4
        s = bytes(self.buf[self.pos:self.pos + max(n - 1, 0)]).decode("utf-8", "replace")
        self.pos += n
        return s

    def prim_array(self, base, count):
        size = PRIMITIVES[base][1]
        if count:
            self.align(size)
        arr = np.frombuffer(self.buf, dtype=np.dtype(NUMPY_TYPES[base]).newbyteorder(self.endian),
                            count=count, offset=self.pos)
        self.pos += size * count
        return arr


def _decode(cdr, registry, name):
    obj = SimpleNamespace()
    for field_name, base, array in registry.definition(name):
        if array is None:
            setattr(obj, field_name, _value(cdr, registry, base))
            continue
        if array == -1:
            cdr.align(4)
            count = struct.unpack_from(cdr.endian + "I", cdr.buf, cdr.pos)[0]
            cdr.pos += 4
        else:
            count = array
        if base in PRIMITIVES:
            setattr(obj, field_name, cdr.prim_array(base, count))
        else:
            setattr(obj, field_name, [_value(cdr, registry, base) for _ in range(count)])
    return obj


def _value(cdr, registry, base):
    if base in PRIMITIVES:
        return cdr.prim(base)
    if base.startswith("string"):
        return cdr.string()
    if base.startswith("wstring"):
        raise ValueError("wstring fields are not supported by the builtin reader")
    return _decode(cdr, registry, base)


def deserialize(data, type_name, registry):
    return _decode(_Cdr(data), registry, type_name)


class _CdrWriter:
    def __init__(self):
        self.out = bytearray(b"\x00\x01\x00\x00")

    def align(self, n):
        self.out += b"\x00" * ((-(len(self.out) - 4)) % n)

    def prim(self, base, value):
        fmt, size = PRIMITIVES[base]
        self.align(size)
        self.out += struct.pack("<" + fmt, value)

    def string(self, value):
        raw = str(value).encode("utf-8") + b"\x00"
        self.prim("uint32", len(raw))
        self.out += raw


def serialize(obj, type_name, registry):
    """Inverse of deserialize (little-endian CDR). obj: dict or namespace tree."""
    w = _CdrWriter()

    def get(o, k):
        return o[k] if isinstance(o, dict) else getattr(o, k)

    def put(o, name):
        for field_name, base, array in registry.definition(name):
            v = get(o, field_name)
            if array is None:
                put_value(v, base)
                continue
            if array == -1:
                w.prim("uint32", len(v))
            if base in PRIMITIVES:
                a = np.asarray(v, dtype=np.dtype(NUMPY_TYPES[base]).newbyteorder("<"))
                if len(a):
                    w.align(PRIMITIVES[base][1])
                w.out += a.tobytes()
            else:
                for item in v:
                    put_value(item, base)

    def put_value(v, base):
        if base in PRIMITIVES:
            w.prim(base, v)
        elif base.startswith("string"):
            w.string(v)
        else:
            put(v, base)

    put(obj, type_name)
    return bytes(w.out)


# ---------------------------------------------------------------- readers
def _metadata(bag):
    meta = Path(bag) / "metadata.yaml"
    if not meta.is_file():
        return {}
    text = meta.read_text()
    try:
        import yaml
        return yaml.safe_load(text).get("rosbag2_bagfile_information", {})
    except Exception:   # minimal fallback without PyYAML
        storage = re.search(r"storage_identifier:\s*(\S+)", text)
        files = re.findall(r"^\s*-\s*(\S+\.(?:db3|mcap))\s*$", text, re.M)
        return {"storage_identifier": storage.group(1) if storage else "",
                "relative_file_paths": sorted(set(files))}


def bag_files(bag):
    bag = Path(bag)
    if bag.is_file():
        return [bag]
    meta = _metadata(bag)
    files = [bag / f for f in meta.get("relative_file_paths", []) if (bag / f).is_file()]
    return files or sorted(bag.glob("*.db3")) or sorted(bag.glob("*.mcap"))


class BuiltinReader:
    name = "builtin"

    def __init__(self, bag, msg_dirs=()):
        self.bag = Path(bag)
        self.files = [f for f in bag_files(bag) if f.suffix == ".db3"]
        if not self.files:
            raise RuntimeError(f"no .db3 files in {bag} (the builtin reader can't read .mcap; "
                               "run inside the ROS container to use the ros reader)")
        meta = _metadata(bag)
        if meta.get("compression_format"):
            raise RuntimeError("compressed bag: run inside the ROS container to use the ros reader")
        self.registry = TypeRegistry(msg_dirs)
        self._types = {}
        for f in self.files:
            with sqlite3.connect(f"file:{f}?mode=ro", uri=True) as db:
                for name, typ in db.execute("SELECT name, type FROM topics"):
                    self._types[name] = typ

    def topics(self):
        return dict(self._types)

    def can_decode(self, topic):
        return self.registry.knows(self._types[topic])

    def messages(self, topics, decode=None):
        """Yield (topic, log_time_ns, msg). Topics not in `decode` (default:
        all) yield the raw bytes instead, which is fast for timing passes."""
        topics = [t for t in topics if t in self._types]
        decode = set(topics if decode is None else decode)
        for f in self.files:
            db = sqlite3.connect(f"file:{f}?mode=ro", uri=True)
            ids = {row[0]: row[1] for row in db.execute("SELECT id, name FROM topics")}
            wanted = [i for i, n in ids.items() if n in topics]
            if not wanted:
                db.close()
                continue
            q = f"SELECT topic_id, timestamp, data FROM messages WHERE topic_id IN ({','.join('?' * len(wanted))}) ORDER BY timestamp"
            for topic_id, stamp, data in db.execute(q, wanted):
                topic = ids[topic_id]
                yield topic, int(stamp), (deserialize(data, self._types[topic], self.registry)
                                          if topic in decode else data)
            db.close()


class RosReader:
    name = "ros"

    def __init__(self, bag, msg_dirs=()):
        import rosbag2_py  # noqa: F401  (raises ImportError outside ROS)
        from rclpy.serialization import deserialize_message  # noqa: F401
        self.bag = Path(bag)
        meta = _metadata(bag)
        storage = meta.get("storage_identifier") or (
            "mcap" if any(f.suffix == ".mcap" for f in bag_files(bag)) else "sqlite3")
        self.storage = storage
        reader = self._open()
        self._types = {t.name: t.type for t in reader.get_all_topics_and_types()}
        del reader

    def _open(self):
        import rosbag2_py
        reader = rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=str(self.bag), storage_id=self.storage),
                    rosbag2_py.ConverterOptions(input_serialization_format="cdr",
                                                output_serialization_format="cdr"))
        return reader

    def topics(self):
        return dict(self._types)

    def can_decode(self, topic):
        try:
            from rosidl_runtime_py.utilities import get_message
            get_message(self._types[topic])
            return True
        except Exception:
            return False

    def messages(self, topics, decode=None):
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
        topics = [t for t in topics if t in self._types]
        decode = set(topics if decode is None else decode)
        classes = {t: get_message(self._types[t]) for t in decode}
        reader = self._open()
        reader.set_filter(rosbag2_py.StorageFilter(topics=topics))
        while reader.has_next():
            topic, data, stamp = reader.read_next()
            yield topic, int(stamp), (deserialize_message(data, classes[topic]) if topic in decode else data)


def find_msg_dirs(start):
    """Workspace msg folders (px4_msgs, voxl_msgs, ...) above `start`."""
    for parent in [Path(start).resolve()] + list(Path(start).resolve().parents):
        src = parent / "src" if (parent / "src").is_dir() else (parent if parent.name == "src" else None)
        if src is not None and any(src.glob("*/msg")):
            return sorted(str(p) for p in src.glob("*/msg"))
    return []


def open_bag(bag, reader="auto", msg_dirs=None):
    # Folders given explicitly are loaded last, so they override the workspace's.
    msg_dirs = find_msg_dirs(Path(__file__).parent) + list(msg_dirs or [])
    if reader in ("auto", "ros"):
        try:
            return RosReader(bag, msg_dirs)
        except ImportError:
            if reader == "ros":
                raise RuntimeError("the ros reader needs ROS 2: run inside the container after "
                                   "sourcing /opt/ros/humble/setup.bash and install/setup.bash")
    return BuiltinReader(bag, msg_dirs)


# ------------------------------------------------------------- conversions
def stamp_ns(header):
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def _bytes(data):
    return np.frombuffer(data, dtype=np.uint8) if not isinstance(data, np.ndarray) else data.view(np.uint8)


def _yuv_to_rgb(y, u, v):
    y, u, v = (np.asarray(c, dtype=np.float32) for c in (y, u, v))
    r = y + 1.402 * (v - 128)
    g = y - 0.344136 * (u - 128) - 0.714136 * (v - 128)
    b = y + 1.772 * (u - 128)
    return np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)


def image_to_rgb(msg):
    """sensor_msgs/Image or CompressedImage -> HxWx3 uint8 RGB."""
    if hasattr(msg, "format") and not hasattr(msg, "encoding"):
        import io
        from PIL import Image as PILImage
        return np.asarray(PILImage.open(io.BytesIO(bytes(_bytes(msg.data)))).convert("RGB"))
    enc = str(msg.encoding).lower()
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    raw = _bytes(msg.data)
    if enc in ("nv12", "nv21", "yuv420", "i420"):
        y = raw[:h * w].reshape(h, w)
        if enc == "i420" or enc == "yuv420":
            u = raw[h * w:h * w + h * w // 4].reshape(h // 2, w // 2)
            v = raw[h * w + h * w // 4:h * w + h * w // 2].reshape(h // 2, w // 2)
        else:
            uv = raw[h * w:h * w + h * w // 2].reshape(h // 2, w // 2, 2)
            u, v = (uv[..., 0], uv[..., 1]) if enc == "nv12" else (uv[..., 1], uv[..., 0])
        u, v = u.repeat(2, 0).repeat(2, 1), v.repeat(2, 0).repeat(2, 1)
        return _yuv_to_rgb(y, u, v)
    rows = raw[:h * step].reshape(h, step)
    if enc in ("rgb8", "bgr8", "8uc3"):
        img = rows[:, :w * 3].reshape(h, w, 3)
        return img[..., ::-1].copy() if enc == "bgr8" else img.copy()
    if enc in ("rgba8", "bgra8", "8uc4"):
        img = rows[:, :w * 4].reshape(h, w, 4)[..., :3]
        return img[..., ::-1].copy() if enc == "bgra8" else img.copy()
    if enc in ("mono8", "8uc1"):
        g = rows[:, :w]
        return np.repeat(g[..., None], 3, axis=2)
    if enc in ("mono16", "16uc1"):
        g = rows[:, :w * 2].copy().view("<u2" if not msg.is_bigendian else ">u2")
        g = (g >> 8).astype(np.uint8)
        return np.repeat(g[..., None], 3, axis=2)
    if enc in ("yuv422", "uyvy", "yuv422_yuy2", "yuyv"):
        px = rows[:, :w * 2].reshape(h, w // 2, 4).astype(np.float32)
        if enc in ("yuv422", "uyvy"):
            u, y0, v, y1 = (px[..., i] for i in range(4))
        else:
            y0, u, y1, v = (px[..., i] for i in range(4))
        y = np.stack([y0, y1], axis=-1).reshape(h, w)
        return _yuv_to_rgb(y, u.repeat(2, 1), v.repeat(2, 1))
    raise ValueError(f"unsupported image encoding '{msg.encoding}'")


def cloud_xyz(msg):
    """sensor_msgs/PointCloud2 -> (N, 3) float x, y, z (finite points only)."""
    fields = {f.name: f for f in msg.fields}
    if any(n not in fields for n in "xyz") or any(int(fields[n].datatype) != 7 for n in "xyz"):
        return np.empty((0, 3))
    endian = ">" if msg.is_bigendian else "<"
    dtype = np.dtype({"names": list("xyz"), "formats": [endian + "f4"] * 3,
                      "offsets": [int(fields[n].offset) for n in "xyz"], "itemsize": int(msg.point_step)})
    raw = _bytes(msg.data)
    rows = [np.frombuffer(raw, dtype=dtype, count=int(msg.width), offset=r * int(msg.row_step))
            for r in range(int(msg.height))]
    if not rows:
        return np.empty((0, 3))
    data = np.concatenate(rows)
    pts = np.column_stack([data["x"], data["y"], data["z"]]).astype(float)
    return pts[np.all(np.isfinite(pts), axis=1)]


def transforms(msg, type_name):
    """(parent, child, stamp_ns, translation[3], quaternion xyzw[4]) from TF-like messages."""
    if type_name.endswith("TFMessage"):
        items = msg.transforms
    elif type_name.endswith("TransformStamped"):
        items = [msg]
    else:
        return []
    out = []
    for t in items:
        tr, q = t.transform.translation, t.transform.rotation
        out.append((str(t.header.frame_id).lstrip("/"), str(t.child_frame_id).lstrip("/"), stamp_ns(t.header),
                    np.array([tr.x, tr.y, tr.z], float), np.array([q.x, q.y, q.z, q.w], float)))
    return out


def quaternion_matrix(q):
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def detection_fields(msg):
    """(label, confidence, (x_min, y_min, x_max, y_max) as published, camera) from a voxl Aidetection-like message."""
    label = str(getattr(msg, "class_name", "")).strip().lower()
    conf = -1.0
    for name in ("class_confidence", "detection_confidence"):
        try:
            conf = float(getattr(msg, name))
            break
        except (AttributeError, TypeError, ValueError):
            continue
    vals = tuple(float(getattr(msg, k)) for k in ("x_min", "y_min", "x_max", "y_max"))
    return label, conf, vals, str(getattr(msg, "cam", ""))


def box_to_pixels(vals, width, height):
    """Detector box (normalized or pixel, like perception_detection.py) -> pixels."""
    vals = list(vals)
    if all(-0.01 <= v <= 1.01 for v in vals):
        vals = [vals[0] * width, vals[1] * height, vals[2] * width, vals[3] * height]
    x1, x2 = sorted(vals[0::2])
    y1, y2 = sorted(vals[1::2])
    return (max(0.0, x1), max(0.0, y1), min(width - 1.0, x2), min(height - 1.0, y2))


def detection_box(msg, width, height):
    """(label, confidence, (x1, y1, x2, y2) pixels, camera)."""
    label, conf, vals, cam = detection_fields(msg)
    return label, conf, box_to_pixels(vals, width, height), cam
