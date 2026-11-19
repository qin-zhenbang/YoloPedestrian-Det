"""图像级统计：把检测框数与 labels.csv 的人工真值比较，算检出 / 误检 / 漏检。

口径（真值是每张图的人工人数，不是逐人框，因此按计数比较）：
- 检出 = min(检测数, 真值)
- 误检 = max(检测数 - 真值, 0)
- 漏检 = max(真值 - 检测数, 0)
- 检出率 = Σ检出 / Σ真值，漏检率 = Σ漏检 / Σ真值
- 精确率 = Σ检出 / Σ检测数，误检率 = Σ误检 / Σ检测数
密集场景的真值本身是人工计数（部分为估算），误差记在 labels.csv 的 note 列。
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from detector import Detection, detect_batch, load_config, load_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Label:
    image_id: str
    scene: str
    person_count: int


@dataclass(frozen=True)
class ImageStats:
    image_id: str
    scene: str
    gt: int
    det: int
    tp: int
    fp: int
    fn: int


@dataclass(frozen=True)
class Metrics:
    n_images: int
    gt: int
    det: int
    tp: int
    fp: int
    fn: int
    recall: float
    precision: float
    miss_rate: float
    fp_rate: float


def load_labels(cfg: dict) -> dict[str, Label]:
    """读取 labels.csv，按图片文件名索引。"""
    path = PROJECT_ROOT / cfg["data"]["labels_csv"]
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = csv.DictReader(f)
        return {
            row["image_id"]: Label(
                image_id=row["image_id"],
                scene=row["scene"],
                person_count=int(row["person_count"]),
            )
            for row in rows
        }


def list_images(cfg: dict) -> list[Path]:
    """按 exp.yaml 的场景顺序列出全部测试图。"""
    root = PROJECT_ROOT / cfg["data"]["samples_dir"]
    paths: list[Path] = []
    for scene in cfg["data"]["scenes"]:
        paths.extend(sorted((root / scene).glob("*.jpg")))
    return paths


def run_detections(
    model,
    cfg: dict,
    image_paths: Sequence[str | Path] | None = None,
    conf: float | None = None,
) -> dict[str, list[Detection]]:
    """对指定图片（默认全部）跑推理，返回 {路径: 检测框列表}。

    conf 传 None 用 exp.yaml 基线阈值；阈值扫描时传低阈值一次跑完再逐档过滤。
    """
    paths = list(image_paths) if image_paths is not None else list_images(cfg)
    return detect_batch(model, paths, cfg, conf=conf)


def _make_row(label: Label, det: int) -> ImageStats:
    return ImageStats(
        image_id=label.image_id,
        scene=label.scene,
        gt=label.person_count,
        det=det,
        tp=min(det, label.person_count),
        fp=max(det - label.person_count, 0),
        fn=max(label.person_count - det, 0),
    )


def rows_from_detections(
    detections: dict[str, list[Detection]],
    labels: dict[str, Label],
    conf_threshold: float | None = None,
) -> list[ImageStats]:
    """把检测结果与真值逐图对比。conf_threshold 非空时先按置信度过滤检测框。"""
    rows: list[ImageStats] = []
    for path, boxes in detections.items():
        image_id = Path(path).name
        if image_id not in labels:
            raise KeyError(f"labels.csv 缺少图片 {image_id} 的真值")
        if conf_threshold is None:
            det = len(boxes)
        else:
            det = sum(1 for box in boxes if box.conf >= conf_threshold)
        rows.append(_make_row(labels[image_id], det))
    return rows


def summarize(rows: Iterable[ImageStats]) -> Metrics:
    """把逐图统计汇总成一组指标，分母为 0 时比率记 0。"""
    rows = list(rows)
    gt = sum(r.gt for r in rows)
    det = sum(r.det for r in rows)
    tp = sum(r.tp for r in rows)
    fp = sum(r.fp for r in rows)
    fn = sum(r.fn for r in rows)
    return Metrics(
        n_images=len(rows),
        gt=gt,
        det=det,
        tp=tp,
        fp=fp,
        fn=fn,
        recall=tp / gt if gt else 0.0,
        precision=tp / det if det else 0.0,
        miss_rate=fn / gt if gt else 0.0,
        fp_rate=fp / det if det else 0.0,
    )


def summarize_by_scene(rows: Iterable[ImageStats]) -> dict[str, Metrics]:
    """按场景分组汇总，键顺序与传入行的场景出现顺序一致。"""
    grouped: dict[str, list[ImageStats]] = {}
    for row in rows:
        grouped.setdefault(row.scene, []).append(row)
    return {scene: summarize(items) for scene, items in grouped.items()}


def evaluate(
    model,
    cfg: dict,
    image_paths: Sequence[str | Path] | None = None,
    conf: float | None = None,
) -> list[ImageStats]:
    """一次完整评估：推理 + 与真值逐图对比，返回逐图统计。"""
    labels = load_labels(cfg)
    detections = run_detections(model, cfg, image_paths, conf)
    return rows_from_detections(detections, labels)


if __name__ == "__main__":
    cfg = load_config()
    model = load_model(cfg)
    rows = evaluate(model, cfg)
    overall = summarize(rows)

    print(
        f"[基线 conf={cfg['model']['conf']}] {overall.n_images} 张图："
        f"真值 {overall.gt}，检测 {overall.det}，检出 {overall.tp}，"
        f"误检 {overall.fp}，漏检 {overall.fn}"
    )
    print(
        f"检出率 {overall.recall:.1%}，精确率 {overall.precision:.1%}，"
        f"漏检率 {overall.miss_rate:.1%}，误检率 {overall.fp_rate:.1%}"
    )
    print("场景        图数   真值   检测   检出   误检   漏检   检出率   漏检率")
    for scene, m in summarize_by_scene(rows).items():
        print(
            f"{scene:<10} {m.n_images:>4} {m.gt:>6} {m.det:>6} {m.tp:>6} "
            f"{m.fp:>6} {m.fn:>6}  {m.recall:>6.1%}  {m.miss_rate:>6.1%}"
        )
