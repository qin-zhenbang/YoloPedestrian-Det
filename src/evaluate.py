"""图像级统计：把检测框数与 labels.csv 的人工真值比较，算检出 / 误检 / 漏检。

口径（真值是每张图的人工人数，不是逐人框，因此按计数比较）：
- 检出 = min(检测数, 真值)
- 误检 = max(检测数 - 真值, 0)
- 漏检 = max(真值 - 检测数, 0)
- 检出率 = Σ检出 / Σ真值，漏检率 = Σ漏检 / Σ真值
- 精确率 = Σ检出 / Σ检测数，误检率 = Σ误检 / Σ检测数
密集场景的真值本身是人工计数（部分为估算），误差记在 labels.csv 的 note 列。
"""

# 文件从上到下分四块：
# 1. 三个数据结构：Label（真值）、ImageStats（单图对比结果）、Metrics（汇总指标）；
# 2. 读真值、列图片、跑推理；
# 3. 核心公式：检测数 vs 真值 → 检出/误检/漏检，再汇总成率和分场景统计；
# 4. 写 CSV（阈值扫描、耗时、场景统计）以及 __main__ 的一键运行流程。

import csv
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from detector import (
    Detection,
    SpeedStats,
    benchmark_latency,
    detect_batch,
    load_config,
    load_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "results" / "tables"


@dataclass(frozen=True)
class Label:
    image_id: str        # 图片文件名，例如 day_005.jpg
    scene: str           # 所属场景：day / night / occlusion / crowd / small
    person_count: int    # 人工数的真实人数（这张图只有一个总数，没有逐人框）


@dataclass(frozen=True)
class ImageStats:
    """一张图的对比结果，所有字段都是“人数/框数”，不是比率。"""

    image_id: str
    scene: str
    gt: int              # ground truth：真值人数
    det: int             # detections：模型输出的框数
    tp: int              # 检出：min(det, gt)
    fp: int              # 误检：max(det - gt, 0)
    fn: int              # 漏检：max(gt - det, 0)


@dataclass(frozen=True)
class Metrics:
    """把多张 ImageStats 相加后的汇总，比率都是 0~1 的小数。"""

    n_images: int
    gt: int
    det: int
    tp: int
    fp: int
    fn: int
    recall: float        # 检出率 = tp / gt
    precision: float     # 精确率 = tp / det
    miss_rate: float     # 漏检率 = fn / gt
    fp_rate: float       # 误检率 = fp / det


def load_labels(cfg: dict) -> dict[str, Label]:
    """读取 labels.csv，按图片文件名索引。"""
    path = PROJECT_ROOT / cfg["data"]["labels_csv"]
    # utf-8-sig 用来去掉 Excel 可能写入的 BOM，否则第一列列名会带隐藏字符。
    # 结果字典的键是文件名（如 day_005.jpg），后面用图片路径的 name 来匹配。
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
    # 顺序很重要：CSV 行序、图表横轴顺序都跟着它走。
    # day → night → occlusion → crowd → small，每个目录内按文件名排序。
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
    # 默认跑全部图；阈值扫描会显式传入 0.05，只推理一次。
    paths = list(image_paths) if image_paths is not None else list_images(cfg)
    return detect_batch(model, paths, cfg, conf=conf)


def _make_row(label: Label, det: int) -> ImageStats:
    # 这里是整套指标的核心：真值只有一个总人数，没法做逐人 IoU 匹配，
    # 所以只能按“框数”和“人数”比较：
    # - 检测数比真值少：少掉的部分算漏检；
    # - 检测数比真值多：多出的部分算误检；
    # - 两者取较小值，就是“对上了”的检出数。
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
        # detections 的键是完整路径，labels 的键是文件名，所以要先取 Path(...).name。
        image_id = Path(path).name
        if image_id not in labels:
            raise KeyError(f"labels.csv 缺少图片 {image_id} 的真值")
        if conf_threshold is None:
            # 不再过滤：推理时已经按 exp.yaml 的 conf 丢过低置信框。
            det = len(boxes)
        else:
            # 阈值扫描：boxes 是在 0.05 低阈值下跑的，这里按当前档位重新数一遍。
            # 用 >= 是为了让 0.25 档把 conf 恰好等于 0.25 的框也算进来。
            det = sum(1 for box in boxes if box.conf >= conf_threshold)
        rows.append(_make_row(labels[image_id], det))
    return rows


def summarize(rows: Iterable[ImageStats]) -> Metrics:
    """把逐图统计汇总成一组指标，分母为 0 时比率记 0。"""
    # 先把可迭代对象转成列表：后面要多次遍历，生成器只能用一次。
    rows = list(rows)
    # 所有计数直接相加：先加人数和框数，再算率，避免“先算每张图的率再平均”
    # 导致大图和小图权重相同。
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
        # 三元表达式是防止分母为 0（例如空图片列表或没有检测框）。
        recall=tp / gt if gt else 0.0,
        precision=tp / det if det else 0.0,
        miss_rate=fn / gt if gt else 0.0,
        fp_rate=fp / det if det else 0.0,
    )


def summarize_by_scene(rows: Iterable[ImageStats]) -> dict[str, Metrics]:
    """按场景分组汇总，键顺序与传入行的场景出现顺序一致。"""
    # 用普通 dict 分组：Python 3.7+ 会保留插入顺序，
    # 而传入顺序来自 list_images（day → night → …），所以 CSV 行序稳定。
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
    # 三行对应三步：读真值 → 跑检测 → 逐图比较。
    # 返回的是“逐图明细”，需要总体指标时再传给 summarize()。
    labels = load_labels(cfg)
    detections = run_detections(model, cfg, image_paths, conf)
    return rows_from_detections(detections, labels)


def threshold_grid(cfg: dict) -> list[float]:
    """按 exp.yaml 的 threshold_sweep 生成含起止值的阈值序列。"""
    t = cfg["threshold_sweep"]
    start, stop, step = float(t["start"]), float(t["stop"]), float(t["step"])
    # 用 round 避免 0.05 反复相加出现 0.30000000000000004 这种浮点误差。
    # n 是档数：0.05~0.90 步进 0.05 正好 18 档（含两端）。
    n = int(round((stop - start) / step)) + 1
    return [round(start + i * step, 2) for i in range(n)]


def sweep_thresholds(
    model,
    cfg: dict,
    image_paths: Sequence[str | Path] | None = None,
) -> list[tuple[float, Metrics]]:
    """在最低阈值推理一次，再逐档过滤检测框并汇总，返回 [(阈值, 指标)]。"""
    # 为什么只推理一次：模型推理最慢，阈值过滤只是数框，几乎不耗时。
    # 先在 0.05 把所有可能的框都拿出来，后面 18 档都在这批框上过滤。
    # 注意：NMS 是在 0.05 下做的，所以结果与“每档单独推理”会有极小差异；
    # 已验证 0.25 档与直接按 0.25 推理的基线完全一致。
    thresholds = threshold_grid(cfg)
    detections = run_detections(model, cfg, image_paths, conf=min(thresholds))
    labels = load_labels(cfg)
    return [
        (thr, summarize(rows_from_detections(detections, labels, thr)))
        for thr in thresholds
    ]


def write_threshold_sweep(
    sweep: Iterable[tuple[float, Metrics]],
    path: str | Path = TABLES_DIR / "threshold_sweep.csv",
) -> Path:
    """写 threshold_sweep.csv，比率保留 4 位小数。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig 会带 BOM，Excel 双击打开时中文表头不会乱码。
    # 一行对应一个阈值档位，比率统一保留 4 位小数，方便版本对比。
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["threshold", "n_images", "gt", "det", "tp", "fp", "fn",
             "recall", "precision", "miss_rate", "fp_rate"]
        )
        for thr, m in sweep:
            writer.writerow(
                [f"{thr:.2f}", m.n_images, m.gt, m.det, m.tp, m.fp, m.fn,
                 f"{m.recall:.4f}", f"{m.precision:.4f}",
                 f"{m.miss_rate:.4f}", f"{m.fp_rate:.4f}"]
            )
    return path


def write_model_benchmark(
    cfg: dict,
    stats: SpeedStats,
    path: str | Path = TABLES_DIR / "model_benchmark.csv",
) -> Path:
    """写 model_benchmark.csv：本机 yolov8n 自测 + 官方 yolov8s 参照。"""
    m = cfg["model"]
    official = cfg["official"]
    model_name = Path(m["weights"]).stem
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 表里固定两行：
    # - self：本次在本机 CPU 上实测的 yolov8n，包含完整三段耗时；
    # - official：ultralytics 文档里的 yolov8s 数字，只有总耗时，没有分段。
    # 两行协议不同（PyTorch vs ONNX），note 列必须写清楚，避免被当成公平对比。
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["model", "source", "device", "imgsz", "n_images", "warmup",
             "preprocess_ms", "inference_ms", "postprocess_ms", "total_ms", "note"]
        )
        writer.writerow(
            [model_name, "self", "CPU", m["imgsz"], stats.n_images, stats.warmup,
             f"{stats.preprocess_ms:.1f}", f"{stats.inference_ms:.1f}",
             f"{stats.postprocess_ms:.1f}", f"{stats.total_ms:.1f}",
             f"本机 PyTorch CPU 单线程、{stats.n_images} 张均值；"
             f"CPU: {platform.processor()}"]
        )
        writer.writerow(
            ["yolov8s", "official", "CPU", 640, "", "", "", "", "",
             f"{official['yolov8s_cpu_ms']:.1f}",
             f"官方 {official['protocol']}，非本机自测；"
             f"来源 {official['source']}；口径不同仅作对照"]
        )
    return path


def write_scene_breakdown(
    rows: Iterable[ImageStats],
    path: str | Path = TABLES_DIR / "scene_breakdown.csv",
) -> Path:
    """写 scene_breakdown.csv：按场景汇总检出/误检/漏检，比率保留 4 位小数。"""
    # 先按场景分组求和；字典顺序就是 list_images 的场景顺序。
    by_scene = summarize_by_scene(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["scene", "n_images", "gt", "det", "tp", "fp", "fn",
             "recall", "precision", "miss_rate", "fp_rate"]
        )
        for scene, m in by_scene.items():
            writer.writerow(
                [scene, m.n_images, m.gt, m.det, m.tp, m.fp, m.fn,
                 f"{m.recall:.4f}", f"{m.precision:.4f}",
                 f"{m.miss_rate:.4f}", f"{m.fp_rate:.4f}"]
            )
    return path


if __name__ == "__main__":
    # 一键跑完三组实验：基线统计 → 阈值扫描 → CPU 计时，并重写三份 CSV。
    cfg = load_config()
    model = load_model(cfg)
    # 1) 基线：按 exp.yaml 的 conf=0.25 跑全部 40 张，得到逐图明细。
    rows = evaluate(model, cfg)
    overall = summarize(rows)

    # 先打印总体，再打印分场景，数字应与 CSV 完全一致。
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

    scene_path = write_scene_breakdown(rows)
    print(f"\n[场景统计] 已写入 {scene_path.relative_to(PROJECT_ROOT)}")

    # 2) 阈值扫描：内部只在 0.05 推理一次，再逐档过滤。
    sweep = sweep_thresholds(model, cfg)
    csv_path = write_threshold_sweep(sweep)
    print(f"\n[阈值扫描] 已写入 {csv_path.relative_to(PROJECT_ROOT)}（{len(sweep)} 档）")
    print("阈值   检测   检出   误检   漏检   检出率   精确率")
    for thr, m in sweep:
        print(
            f"{thr:>4.2f} {m.det:>6} {m.tp:>6} {m.fp:>6} {m.fn:>6}  "
            f"{m.recall:>6.1%}  {m.precision:>6.1%}"
        )

    stats = benchmark_latency(model, list_images(cfg), cfg)
    # 3) 计时：按 exp.yaml 的预热 3 张 + 正式 20 张测均值，写模型耗时表。
    bench_path = write_model_benchmark(cfg, stats)
    print(
        f"\n[耗时] {Path(cfg['model']['weights']).stem} {stats.total_ms:.1f} ms/张"
        f"（预处理 {stats.preprocess_ms:.1f} + 推理 {stats.inference_ms:.1f} + "
        f"后处理 {stats.postprocess_ms:.1f}），已写入 {bench_path.relative_to(PROJECT_ROOT)}"
    )
