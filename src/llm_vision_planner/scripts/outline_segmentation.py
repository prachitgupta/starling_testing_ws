"""Object outlines (pixel masks) from the drone's camera image.

Backends:
  sam      Segment Anything prompted with the onboard detector's box. Works
           for whatever label the detector reports, because SAM is
           class-agnostic: the box says where, SAM finds the outline.
           Default model: MobileSAM (fast on CPU). sam2.1_b.pt / sam_b.pt
           give sharper edges but are slower.
  yolo     YOLO instance segmentation (COCO classes: person, chair, couch,
           bottle, ...). Needs no detector boxes, so it also works on bags
           without /tflite_data.
  efficientsam  EfficientSAM-Ti as ONNX, prompted with the detector box. Runs
           with OpenCV's DNN module (or onnxruntime), no PyTorch needed.
  tof      No neural network: projects the ToF points that fall in the
           detector box into the image. Its edges come from the ToF itself, so
           it adds nothing the ToF doesn't show: depths fall back to the
           visible extent and class depths. It only exercises the pipeline.
  maskdir  Hand-drawn or saved masks: <dir>/<frame:06d>.png (any non-zero
           pixel is the object), for checking against a known-good outline.

sam and yolo use the ultralytics package (pip install ultralytics); model
weights download automatically on first use.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


EFFICIENTSAM_URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
                    "image_segmentation_efficientsam/image_segmentation_efficientsam_ti_2025april.onnx")
EFFICIENTSAM_SHA256 = "4eb496e0a7259d435b49b66faf1754aa45a5c382a34558ddda9a8c6fe5915d77"
EFFICIENTSAM_FILE = "efficientsam_ti_2025april.onnx"


@dataclass
class ObjectMask:
    label: str
    score: float
    mask: np.ndarray        # HxW bool
    box: tuple              # x1, y1, x2, y2 pixels
    source: str


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def mask_box(mask):
    vs, us = np.nonzero(mask)
    return (float(us.min()), float(vs.min()), float(us.max()), float(vs.max())) if len(us) else None


def _fit_to_image(masks, height, width):
    """Masks at another resolution (no letterbox) -> image size, nearest neighbour."""
    if masks.shape[1:] == (height, width):
        return masks
    rows = np.minimum((np.arange(height) + 0.5) * masks.shape[1] / height, masks.shape[1] - 1).astype(int)
    cols = np.minimum((np.arange(width) + 0.5) * masks.shape[2] / width, masks.shape[2] - 1).astype(int)
    return masks[:, rows][:, :, cols]


def _clip_to_box(mask, box, grow=0.08):
    x1, y1, x2, y2 = box
    gx, gy = grow * (x2 - x1), grow * (y2 - y1)
    clipped = np.zeros_like(mask)
    r0, r1 = int(max(0, y1 - gy)), int(min(mask.shape[0], y2 + gy + 1))
    c0, c1 = int(max(0, x1 - gx)), int(min(mask.shape[1], x2 + gx + 1))
    clipped[r0:r1, c0:c1] = mask[r0:r1, c0:c1]
    return clipped


class YoloSegmenter:
    name = "yolo"

    def __init__(self, model="yolo11n-seg.pt", conf=0.25, device=None):
        from ultralytics import YOLO
        self.model = YOLO(model)
        self.conf = conf
        self.device = device

    def instances(self, image_rgb):
        kwargs = dict(conf=self.conf, retina_masks=True, verbose=False)
        if self.device:
            kwargs["device"] = self.device
        res = self.model.predict(np.ascontiguousarray(image_rgb[..., ::-1]), **kwargs)[0]  # ultralytics expects BGR
        if res.masks is None:
            return []
        h, w = image_rgb.shape[:2]
        data = _fit_to_image(res.masks.data.cpu().numpy() > 0.5, h, w)
        names = res.names
        out = []
        for i in range(len(data)):
            label = str(names[int(res.boxes.cls[i])]).lower()
            box = tuple(float(v) for v in res.boxes.xyxy[i].cpu().numpy())
            out.append(ObjectMask(label, float(res.boxes.conf[i]), data[i], box, self.name))
        return out

    def segment(self, image_rgb, prompts, context=None):
        found = self.instances(image_rgb)
        if not prompts:
            return found
        out = []
        for label, conf, box in prompts:
            scored = [(box_iou(box, m.box) + (0.5 if m.label == label else 0.0), m) for m in found]
            scored = [s for s in scored if s[0] >= 0.3 + (0.0 if s[1].label == label else 0.5)]
            if scored:
                best = max(scored, key=lambda s: s[0])[1]
                out.append(ObjectMask(label, conf, best.mask, box, self.name))
        return out


class SamSegmenter:
    name = "sam"

    def __init__(self, model="mobile_sam.pt", device=None):
        from ultralytics import SAM
        self.model = SAM(model)
        self.device = device

    def segment(self, image_rgb, prompts, context=None):
        if not prompts:
            raise RuntimeError("the sam segmenter needs detector boxes; use --segmenter yolo for bags without them")
        h, w = image_rgb.shape[:2]
        bgr = np.ascontiguousarray(image_rgb[..., ::-1])
        out = []
        for label, conf, box in prompts:
            kwargs = dict(bboxes=[list(box)], verbose=False)
            if self.device:
                kwargs["device"] = self.device
            res = self.model.predict(bgr, **kwargs)[0]
            if res.masks is None or len(res.masks.data) == 0:
                continue
            masks = _fit_to_image(res.masks.data.cpu().numpy() > 0.5, h, w)
            best = max(masks, key=lambda m: m.sum())
            out.append(ObjectMask(label, conf, _clip_to_box(best, box), box, self.name))
        return out


class TofSegmenter:
    """Outline from ToF points projected into the image (no neural network)."""
    name = "tof"

    def __init__(self, window_m=0.9, floor_clearance_m=0.04):
        self.window_m = window_m
        self.floor_clearance_m = floor_clearance_m

    def segment(self, image_rgb, prompts, context):
        camera, pts, floor = context["camera"], context["points_cam"], context.get("floor")
        h, w = image_rgb.shape[:2]
        camera = camera.scaled_to(w, h)
        uv, ok = camera.project(pts)
        if floor is not None:
            ok &= floor.distance_above(pts) > self.floor_clearance_m
        ranges = np.linalg.norm(pts, axis=1)
        out = []
        for label, conf, box in prompts:
            x1, y1, x2, y2 = box
            inb = ok & (uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)
            if inb.sum() < 10:
                continue
            front = np.percentile(ranges[inb], 5)
            sel = inb & (ranges <= front + self.window_m)
            p = uv[sel]
            # Close gaps between neighbouring ToF samples: radius from their
            # median nearest-neighbour spacing in the image.
            sample = p[np.random.default_rng(0).choice(len(p), min(len(p), 1500), replace=False)]
            d = np.linalg.norm(sample[:, None] - sample[None], axis=2)
            np.fill_diagonal(d, np.inf)
            radius = int(np.clip(np.ceil(0.75 * np.median(d.min(axis=1))), 2, 12))
            grid = np.zeros((h, w), dtype=bool)
            ui = np.clip(np.round(p[:, 0]).astype(int), 0, w - 1)
            vi = np.clip(np.round(p[:, 1]).astype(int), 0, h - 1)
            grid[vi, ui] = True
            from outline_depth import shift_mask
            mask = shift_mask(shift_mask(grid, radius), -(radius - 1))
            out.append(ObjectMask(label, conf, _clip_to_box(mask, box, 0.0), box, self.name))
        return out


class MaskDirSegmenter:
    name = "maskdir"

    def __init__(self, folder, label="object"):
        self.folder = Path(folder)
        self.label = label or "object"

    def segment(self, image_rgb, prompts, context):
        from PIL import Image as PILImage
        path = self.folder / f"{int(context['frame_index']):06d}.png"
        if not path.is_file():
            return []
        mask = np.asarray(PILImage.open(path).convert("L")) > 0
        if mask.shape != image_rgb.shape[:2]:
            mask = _fit_to_image(mask[None], *image_rgb.shape[:2])[0]
        label, conf, box = prompts[0] if prompts else (self.label, 1.0, mask_box(mask))
        return [ObjectMask(label, conf, mask, box, self.name)]


class EfficientSamSegmenter:
    """EfficientSAM-Ti (ONNX, OpenCV model zoo) prompted with the detector box.

    Needs only OpenCV's DNN module, or onnxruntime if it is installed: no
    PyTorch. About 2.6 s per box on a slow 2-core CPU, so it belongs on the
    ground-station laptop, not the drone.
    """
    name = "efficientsam"

    def __init__(self, model_path=None):
        path = Path(model_path or EFFICIENTSAM_FILE)
        if not path.is_file():
            raise SystemExit(f"EfficientSAM model not found: {path}\nDownload it (48 MB) with:\n"
                             f"  curl -L -o {path} {EFFICIENTSAM_URL}\n  sha256sum {path}   # {EFFICIENTSAM_SHA256}")
        self.session = self.net = None
        try:
            import onnxruntime
            options = onnxruntime.SessionOptions()
            options.log_severity_level = 3   # hide a harmless shape-merge warning from this model
            self.session = onnxruntime.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        except ImportError:
            import cv2
            self.net = cv2.dnn.readNet(str(path))

    def _forward(self, image, points, labels):
        if self.session is not None:
            return self.session.run(["output_masks", "iou_predictions"], {
                "batched_images": image, "batched_point_coords": points, "batched_point_labels": labels})
        self.net.setInput(image, "batched_images")
        self.net.setInput(points, "batched_point_coords")
        self.net.setInput(labels, "batched_point_labels")
        return self.net.forward(["output_masks", "iou_predictions"])

    def segment(self, image_rgb, prompts, context=None):
        import cv2
        if not prompts:
            return []   # needs a detector box (or --box) to outline
        h, w = image_rgb.shape[:2]
        size = 1024
        image = cv2.resize(image_rgb, (size, size)).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        image = np.ascontiguousarray(image)
        out = []
        for label, conf, box in prompts:
            x1, y1, x2, y2 = box
            points = np.zeros((1, 1, 6, 2), np.float32)
            labels = np.full((1, 1, 6), -1.0, np.float32)
            points[0, 0, :2] = [[x1 * size / w, y1 * size / h], [x2 * size / w, y2 * size / h]]
            labels[0, 0, :2] = [2.0, 3.0]          # box corners: top-left, bottom-right
            masks, ious = self._forward(image, points, labels)
            logits = np.asarray(masks)[0, 0, int(np.argmax(np.asarray(ious)[0, 0]))].astype(np.float32)
            mask = cv2.resize(logits, (w, h), interpolation=cv2.INTER_LINEAR) >= 0
            n, comp, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
            if n > 2:   # keep the largest piece
                mask = comp == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            if mask.any():
                out.append(ObjectMask(label, conf, _clip_to_box(mask, box), box, self.name))
        return out


def make_segmenter(kind, prompts_available=True, sam_model="mobile_sam.pt", yolo_model="yolo11n-seg.pt",
                   mask_dir=None, device=None, label=None, efficientsam_model=None):
    if kind == "auto":
        try:
            import ultralytics  # noqa: F401
            kind = "sam" if prompts_available else "yolo"
        except ImportError:
            if prompts_available and Path(efficientsam_model or EFFICIENTSAM_FILE).is_file():
                kind = "efficientsam"
            else:
                print("no segmentation model available (pip install ultralytics, or download the "
                      "EfficientSAM ONNX model): falling back to the ToF outline, which only gives "
                      "the visible extent and class depths")
                kind = "tof"
    if kind == "sam":
        return SamSegmenter(sam_model, device)
    if kind == "efficientsam":
        return EfficientSamSegmenter(efficientsam_model)
    if kind == "yolo":
        return YoloSegmenter(yolo_model, device=device)
    if kind == "tof":
        return TofSegmenter()
    if kind == "maskdir":
        if not mask_dir:
            raise ValueError("--mask-dir is required with --segmenter maskdir")
        return MaskDirSegmenter(mask_dir, label)
    raise ValueError(f"unknown segmenter '{kind}'")
