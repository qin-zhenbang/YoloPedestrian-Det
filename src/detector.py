"""行人检测推理：加载 exp.yaml 中的模型参数，提供单图/批量推理与 CPU 耗时测量。

参数唯一来源是 configs/exp.yaml，本文件不写死阈值、尺寸或权重路径。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
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


@dataclass
class SpeedStats:
    """单张平均耗时（毫秒），口径见 exp.yaml 的 timing 段。"""

    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    total_ms: float
    n_images: int
    warmup: int


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


def _predict_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """把 exp.yaml 的 model 段翻译成 ultralytics 的推理参数。"""
    m = cfg["model"]
    return {
        "imgsz": int(m["imgsz"]),
        "conf": float(m["conf"]),
        "iou": float(m["iou"]),
        "classes": list(m["classes"]),
        "device": m["device"],
        "verbose": False,
    }


def _predict(model: YOLO, source: Any, cfg: dict[str, Any]) -> list[Any]:
    return model.predict(source=source, **_predict_kwargs(cfg))


def _to_detections(result: Any) -> list[Detection]:
    """把 ultralytics 的结果对象转成 Detection 列表。"""
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


def detect_image(
    model: YOLO, image_path: str | Path, cfg: dict[str, Any]
) -> list[Detection]:
    """对单张图片推理，只保留配置中的类别（person=0），返回检测框列表。"""
    return _to_detections(_predict(model, str(image_path), cfg)[0])


def detect_batch(
    model: YOLO, image_paths: Sequence[str | Path], cfg: dict[str, Any]
) -> dict[str, list[Detection]]:
    """批量推理，返回 {图片路径: 检测框列表}。

    逐张调用单图推理而不是把多张图塞进一个 tensor：ultralytics 批量推理会强制
    rect=False（方形补边），检测结果与单图 rect=True 不一致，会污染后续统计。
    """
    return {str(path): detect_image(model, path, cfg) for path in image_paths}


def benchmark_latency(
    model: YOLO, image_paths: Sequence[str | Path], cfg: dict[str, Any]
) -> SpeedStats:
    """按 exp.yaml 的 timing 口径测 CPU 单张耗时：预热若干张后取指定张数的均值。

    只统计 results[0].speed 中的预处理/推理/后处理，不含读图、画框和写 CSV。
    """
    paths = [str(p) for p in image_paths]
    t = cfg["timing"]
    warmup = int(t["warmup"])
    repeats = int(t["repeats"])
    if len(paths) < warmup + repeats:
        raise ValueError(f"计时需要至少 {warmup + repeats} 张图，当前只有 {len(paths)} 张")

    prev_threads = torch.get_num_threads()
    torch.set_num_threads(int(t["threads"]))
    try:
        for path in paths[:warmup]:
            _predict(model, path, cfg)
        rows = [_predict(model, path, cfg)[0].speed for path in paths[warmup:warmup + repeats]]
    finally:
        torch.set_num_threads(prev_threads)

    def mean_ms(key: str) -> float:
        return sum(float(r[key]) for r in rows) / len(rows)

    components = {key: mean_ms(key) for key in t["include"]}
    return SpeedStats(
        preprocess_ms=components["preprocess"],
        inference_ms=components["inference"],
        postprocess_ms=components["postprocess"],
        total_ms=sum(components.values()),
        n_images=repeats,
        warmup=warmup,
    )


if __name__ == "__main__":
    cfg = load_config()
    model = load_model(cfg)
    samples = sorted((PROJECT_ROOT / cfg["data"]["samples_dir"]).glob("*/*.jpg"))

    dets = detect_image(model, samples[0], cfg)
    print(f"[单图] {samples[0].name}: {len(dets)} 个 person 框")

    batch = detect_batch(model, samples[:5], cfg)
    summary = ", ".join(f"{Path(p).name}={len(d)}" for p, d in batch.items())
    print(f"[批量] {len(batch)} 张: {summary}")

    stats = benchmark_latency(model, samples, cfg)
    print(
        f"[计时] 预热 {stats.warmup} 张取 {stats.n_images} 张均值: "
        f"预处理 {stats.preprocess_ms:.1f} ms + 推理 {stats.inference_ms:.1f} ms + "
        f"后处理 {stats.postprocess_ms:.1f} ms = {stats.total_ms:.1f} ms/张"
    )
