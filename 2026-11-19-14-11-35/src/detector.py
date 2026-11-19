"""推理模块（T010–T013）。

职责：加载 YOLOv8n、只保留 person(classes=[0])、单图/批量推理、预热+中位数计时。
所有可调参数来自 configs/exp.yaml；计时口径见 docs/timing_note.md。
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from config import load_config, resolve

# 计时口径（T004，见 docs/timing_note.md）：不依赖 exp.yaml
WARMUP_RUNS = 10
TIMED_RUNS = 30


@dataclass
class Detection:
    """单个检测框。bbox 为像素坐标 (x1, y1, x2, y2)。"""

    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str = "person"


@dataclass
class DetectResult:
    """单张图的推理结果。"""

    image_id: str
    image_path: Path
    detections: list[Detection] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def person_count(self) -> int:
        """预测人数 = 保留下来的 person 框数量（数量口径，见 T014 / metrics_note）。"""
        return len(self.detections)


def load_model(config: dict | None = None):
    """加载权重，设备与 classes 从配置读取（T011/T012）。"""
    from ultralytics import YOLO

    cfg = config if config is not None else load_config()
    weights = resolve(cfg["model"]["weights"])
    if not weights.exists():
        raise FileNotFoundError(
            f"权重文件不存在: {weights}。请确认 weights/yolov8n.pt 已随仓库分发。"
        )
    model = YOLO(str(weights))
    return model


def _predict(model, image, cfg: dict, conf: float | None = None) -> list[Detection]:
    """执行一次推理，返回 person 检测框列表。imgsz/iou/classes/device 全部读配置。"""
    m = cfg["model"]
    inf = cfg["inference"]
    results = model.predict(
        source=image,
        imgsz=int(inf["imgsz"]),
        conf=float(conf if conf is not None else inf["conf"]),
        iou=float(inf["iou"]),
        classes=list(m["classes"]),       # [0] = person
        device=str(m["device"]),          # cpu
        verbose=False,
    )
    detections: list[Detection] = []
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return detections
    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    names = result.names
    for (x1, y1, x2, y2), c, cls_id in zip(xyxy, confs, cls_ids):
        detections.append(
            Detection(
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                confidence=float(c),
                class_id=int(cls_id),
                class_name=str(names.get(int(cls_id), "person")),
            )
        )
    return detections


def detect_image(model, image, image_id: str | None = None, cfg: dict | None = None,
                 conf: float | None = None) -> DetectResult:
    """单图推理（T011）。image 可为路径、numpy 数组；conf 覆盖用于阈值扫描。"""
    cfg = cfg if cfg is not None else load_config()
    path = Path(image) if isinstance(image, (str, Path)) else None
    if image_id is None:
        image_id = path.stem if path else "array"
    detections = _predict(model, image, cfg, conf=conf)
    return DetectResult(
        image_id=image_id,
        image_path=path.resolve() if path else Path("."),
        detections=detections,
    )


def detect_batch(model, images: list, image_ids: list[str] | None = None,
                 cfg: dict | None = None, conf: float | None = None) -> list[DetectResult]:
    """批量推理（T012）。images 为路径或数组列表，逐张调用 detect_image。"""
    cfg = cfg if cfg is not None else load_config()
    if image_ids is None:
        image_ids = [
            Path(im).stem if isinstance(im, (str, Path)) else f"arr_{i}"
            for i, im in enumerate(images)
        ]
    return [
        detect_image(model, im, image_id=iid, cfg=cfg, conf=conf)
        for im, iid in zip(images, image_ids)
    ]


def measure_latency(model, image, cfg: dict | None = None) -> dict:
    """预热 + 中位数计时（T004/T013）。返回统计字典，波动超 10% 时标记 warning。"""
    cfg = cfg if cfg is not None else load_config()
    for _ in range(WARMUP_RUNS):
        _predict(model, image, cfg)
    samples: list[float] = []
    for _ in range(TIMED_RUNS):
        t0 = time.perf_counter()
        _predict(model, image, cfg)
        samples.append((time.perf_counter() - t0) * 1000.0)
    median_ms = statistics.median(samples)
    spread = (max(samples) / min(samples) - 1.0) if min(samples) > 0 else float("inf")
    return {
        "median_ms": median_ms,
        "min_ms": min(samples),
        "max_ms": max(samples),
        "spread_ratio": spread,
        "stable": spread < 0.10,
        "samples": samples,
    }


def iter_image_paths(config: dict | None = None) -> list[tuple[str, Path]]:
    """按配置收集五场景图片，返回 [(scene, path)]，全英文文件名、按名排序。"""
    cfg = config if config is not None else load_config()
    data = cfg["data"]
    base = resolve(data["samples_dir"])
    exts = {str(e).lower() for e in data["image_ext"]}
    pairs: list[tuple[str, Path]] = []
    for scene in data["scenes"]:
        scene_dir = base / scene
        if not scene_dir.is_dir():
            continue
        for p in sorted(scene_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                pairs.append((scene, p))
    return pairs


if __name__ == "__main__":
    # T003 冒烟验证：临时图跑通即可，不写正式代码之外的东西
    import sys

    cfg = load_config()
    model = load_model(cfg)
    image = sys.argv[1] if len(sys.argv) > 1 else None
    if image is None or not Path(image).exists():
        print("用法: python src/detector.py <图片路径>")
        raise SystemExit(1)
    result = detect_image(model, image, cfg=cfg)
    print(f"检出 person 框: {result.person_count}")
    for det in result.detections:
        print(f"  {det.class_name} conf={det.confidence:.3f} bbox={tuple(round(v, 1) for v in det.bbox)}")
