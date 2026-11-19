"""行人检测推理：加载 exp.yaml 中的模型参数，提供单图/批量推理与 CPU 耗时测量。

参数唯一来源是 configs/exp.yaml，本文件不写死阈值、尺寸或权重路径。
"""

# 文件从上到下分四块：
# 1. 定义 Detection（一个检测框）和 SpeedStats（一组耗时）两个数据结构；
# 2. load_config / load_model 负责读参数、加载权重；
# 3. detect_image / detect_batch 负责推理，把结果转成 Detection；
# 4. benchmark_latency 按统一口径测 CPU 单张耗时。
# 直接运行本文件时，__main__ 会依次做单图、批量、计时三项自测。

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

    # 坐标单位是“原图像素”，不是模型输入的 640×640；
    # ultralytics 已经帮我们换算回原图尺寸，这里直接存。
    # (x1, y1) 是框左上角，(x2, y2) 是框右下角。
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float          # 置信度，0~1，越大越可信
    cls_id: int = 0      # 类别编号；本实验只保留 person(0)，所以通常都是 0


@dataclass
class SpeedStats:
    """单张平均耗时（毫秒），口径见 exp.yaml 的 timing 段。"""

    preprocess_ms: float   # 预处理：缩放、补边、转张量等
    inference_ms: float    # 模型前向推理
    postprocess_ms: float  # 后处理：主要是 NMS 和结果整理
    total_ms: float        # 上面三项相加，单位毫秒/张
    n_images: int          # 正式计时用了多少张（不含预热）
    warmup: int            # 预热了多少张（不计入统计）


def load_config(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """读取 exp.yaml。"""
    # 只负责把 yaml 变成字典，不在代码里补默认值；
    # 这样 exp.yaml 就是唯一的参数来源，改参数不用改代码。
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(cfg: dict[str, Any]) -> YOLO:
    """按配置加载 YOLOv8 权重。"""
    # exp.yaml 里写的是相对路径 weights/yolov8n.pt，
    # 这里拼成“仓库根目录下的绝对路径”，保证从任意目录运行都能找到权重。
    weights = Path(cfg["model"]["weights"])
    if not weights.is_absolute():
        weights = PROJECT_ROOT / weights
    return YOLO(str(weights))


def _predict_kwargs(cfg: dict[str, Any], conf: float | None = None) -> dict[str, Any]:
    """把 exp.yaml 的 model 段翻译成 ultralytics 的推理参数。

    conf 传 None 时用配置里的基线阈值；阈值扫描时由调用方覆盖。
    """
    m = cfg["model"]
    return {
        # 推理前把图等比缩放到长边 640 左右，短边补边凑成正方形
        "imgsz": int(m["imgsz"]),
        # 置信度低于该值的框直接丢掉；扫描实验会临时覆盖这个值
        "conf": float(m["conf"] if conf is None else conf),
        # NMS 的 IoU 阈值：两个框重叠超过 0.70 就只保留置信度高的那个
        "iou": float(m["iou"]),
        # 只要 COCO 第 0 类 person，其他类别在推理阶段就被过滤掉
        "classes": list(m["classes"]),
        "device": m["device"],
        # 关闭 ultralytics 每张图都打印的日志，避免刷屏
        "verbose": False,
    }


def _predict(
    model: YOLO, source: Any, cfg: dict[str, Any], conf: float | None = None
) -> list[Any]:
    # 统一入口：所有推理都经过这里，保证单图、批量、计时用的是同一套参数。
    return model.predict(source=source, **_predict_kwargs(cfg, conf))


def _to_detections(result: Any) -> list[Detection]:
    """把 ultralytics 的结果对象转成 Detection 列表。"""
    # result.boxes 是 ultralytics 的 Boxes 对象，可能包含 0 到多个框。
    # box.xyxy[0] 是 4 个数 [x1, y1, x2, y2]，是 tensor，要转成普通 float。
    # 转成自定义 Detection 后，后面的 evaluate / visualize 就不依赖 ultralytics 了。
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
    model: YOLO, image_path: str | Path, cfg: dict[str, Any], conf: float | None = None
) -> list[Detection]:
    """对单张图片推理，只保留配置中的类别（person=0），返回检测框列表。"""
    # model.predict 返回的是“每张图一个结果对象”的列表；
    # 这里只传了一张图，所以取 [0]，再转成 Detection。
    return _to_detections(_predict(model, str(image_path), cfg, conf)[0])


def detect_batch(
    model: YOLO,
    image_paths: Sequence[str | Path],
    cfg: dict[str, Any],
    conf: float | None = None,
) -> dict[str, list[Detection]]:
    """批量推理，返回 {图片路径: 检测框列表}。

    逐张调用单图推理而不是把多张图塞进一个 tensor：ultralytics 批量推理会强制
    rect=False（方形补边），检测结果与单图 rect=True 不一致，会污染后续统计。
    """
    # 这里用字典推导式逐张调用 detect_image；
    # 键是图片路径字符串，值是该图的 Detection 列表。
    return {str(path): detect_image(model, path, cfg, conf) for path in image_paths}


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
    # 预热和正式计时都要有图可用；不够就直接报错，避免悄悄少测几张。
    if len(paths) < warmup + repeats:
        raise ValueError(f"计时需要至少 {warmup + repeats} 张图，当前只有 {len(paths)} 张")

    # 先记下当前线程数，测完在 finally 里恢复，避免影响调用方后续代码。
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(int(t["threads"]))
    try:
        # 预热：前几张包含初始化、缓存冷启动，耗时偏高，所以只跑不计。
        for path in paths[:warmup]:
            _predict(model, path, cfg)
        # 正式计时：从预热之后取 repeats 张，每张跑一次。
        # results[0].speed 是 ultralytics 记录的字典，单位毫秒，
        # 包含 preprocess / inference / postprocess 三个字段。
        rows = [_predict(model, path, cfg)[0].speed for path in paths[warmup:warmup + repeats]]
    finally:
        torch.set_num_threads(prev_threads)

    def mean_ms(key: str) -> float:
        # 对指定耗时字段求平均；rows 里每项都是一张图的 speed 字典。
        return sum(float(r[key]) for r in rows) / len(rows)

    # 只统计 exp.yaml 的 include 里列出的字段，目前是三项之和。
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
    # 自测入口：不依赖 evaluate/visualize，单独验证“模型能加载、能出框、能计时”。
    cfg = load_config()
    model = load_model(cfg)
    # 把所有场景目录下的 jpg 排成一列，后面单图/批量/计时都从这里取。
    samples = sorted((PROJECT_ROOT / cfg["data"]["samples_dir"]).glob("*/*.jpg"))

    # 1) 单图：确认至少能画出一个 person 框。
    dets = detect_image(model, samples[0], cfg)
    print(f"[单图] {samples[0].name}: {len(dets)} 个 person 框")

    # 2) 批量：只取前 5 张，确认逐张结果与单图一致。
    batch = detect_batch(model, samples[:5], cfg)
    summary = ", ".join(f"{Path(p).name}={len(d)}" for p, d in batch.items())
    print(f"[批量] {len(batch)} 张: {summary}")

    # 3) 计时：按 exp.yaml 的预热/重复次数测均值。
    stats = benchmark_latency(model, samples, cfg)
    print(
        f"[计时] 预热 {stats.warmup} 张取 {stats.n_images} 张均值: "
        f"预处理 {stats.preprocess_ms:.1f} ms + 推理 {stats.inference_ms:.1f} ms + "
        f"后处理 {stats.postprocess_ms:.1f} ms = {stats.total_ms:.1f} ms/张"
    )
