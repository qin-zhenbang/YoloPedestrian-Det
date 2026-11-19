"""行人检测推理：加载 exp.yaml 中的模型参数，对单张图片输出 person 检测框。

参数唯一来源是 configs/exp.yaml，本文件不写死阈值、尺寸或权重路径。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "exp.yaml"


@dataclass
class Detection:
    """一个 person 检测框，坐标为原图像素 xyxy。"""

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls_id: int = 0


def load_config(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """读取 exp.yaml。"""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(cfg: dict[str, Any]) -> YOLO:
    """按配置加载 YOLOv8 权重。"""
    weights = Path(cfg["model"]["weights"])
    if not weights.is_absolute():
        weights = PROJECT_ROOT / weights
    return YOLO(str(weights))


def detect_image(
    model: YOLO, image_path: str | Path, cfg: dict[str, Any]
) -> list[Detection]:
    """对单张图片推理，只保留配置中的类别（person=0），返回检测框列表。"""
    m = cfg["model"]
    result = model.predict(
        source=str(image_path),
        imgsz=int(m["imgsz"]),
        conf=float(m["conf"]),
        iou=float(m["iou"]),
        classes=list(m["classes"]),
        device=m["device"],
        verbose=False,
    )[0]

    detections: list[Detection] = []
    for box in result.boxes:
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        detections.append(
            Detection(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                conf=float(box.conf[0]),
                cls_id=int(box.cls[0]),
            )
        )
    return detections
