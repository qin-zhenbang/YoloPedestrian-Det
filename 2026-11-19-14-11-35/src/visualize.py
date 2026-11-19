"""出图模块（T020–T027）。

风格常量集中定义（T020）；画框单图、四宫格、阈值双线图、耗时对比图、场景漏检柱状图。
字体自动探测，保证中文无乱码；输出 PNG 直接可拖进 PPT（T027）。
所有数据来自 evaluate.py 生成的 CSV 与 detector.py 的结果，参数来自 exp.yaml。
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from config import load_config, resolve
from detector import Detection, DetectResult, detect_image, iter_image_paths, load_model

# ---- T020 图表风格常量 ----
DPI = 200
FIGSIZE = (8, 4.5)
FONT_CANDIDATES = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans CN",
    "PingFang SC", "WenQuanYi Micro Hei", "Arial Unicode MS",
]
COLORS = {"main": "#1f77b4", "alt": "#d62728", "grid": "#cccccc"}
BOX_COLOR = (255, 0, 0)   # BGR，OpenCV 画框用红色


def setup_style() -> str:
    """探测可用中文字体，设置 matplotlib 风格，返回实际使用的字体名。"""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    font = next((name for name in FONT_CANDIDATES if name in installed), "DejaVu Sans")
    plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = DPI
    plt.rcParams["savefig.dpi"] = DPI
    plt.rcParams["font.size"] = 11
    return font


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    return path


def draw_boxes(image, result: DetectResult, out_path: Path, cfg: dict | None = None) -> Path:
    """T021：在单图画框并保存 det_<image_id>.png。image 为 BGR numpy 数组或图片路径。"""
    import cv2

    cfg = cfg if cfg is not None else load_config()
    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image))
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image}")
    for det in result.detections:
        x1, y1, x2, y2 = (int(round(v)) for v in det.bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, 2)
        label = f"person {det.confidence:.2f}"
        cv2.putText(image, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 1, cv2.LINE_AA)
    return _save_image(out_path, image)


def _save_image(path: Path, image_bgr) -> Path:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image_bgr)
    return path


def grid_overview(samples: list[tuple], out_path: Path, title: str = "四场景检测概览") -> Path:
    """T022：2×2 四宫格。samples 为 [(标题, BGR图像)]，取前四张。"""
    import cv2

    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (label, img) in zip(axes.ravel(), samples[:4]):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(label, fontsize=11)
        ax.axis("off")
    for ax in axes.ravel()[len(samples[:4]):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return _save(fig, out_path)


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def threshold_pr_curve(table_csv: Path, out_path: Path) -> Path:
    """T023：阈值双线图，precision / recall 随 conf 变化。"""
    setup_style()
    rows = _read_csv(table_csv)
    confs = [float(r["conf"]) for r in rows]
    precisions = [float(r["precision"]) for r in rows]
    recalls = [float(r["recall"]) for r in rows]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(confs, recalls, marker="o", color=COLORS["main"], label="召回率 recall")
    ax.plot(confs, precisions, marker="s", color=COLORS["alt"], label="精确率 precision")
    ax.set_xlabel("置信度阈值 conf")
    ax.set_ylabel("指标值")
    ax.set_title("置信度阈值对检出效果的影响")
    ax.set_ylim(0, 1.05)
    ax.grid(True, color=COLORS["grid"], alpha=0.6)
    ax.legend()
    fig.tight_layout()
    return _save(fig, out_path)


def latency_compare(table_csv: Path, out_path: Path) -> Path:
    """T024：n 本地实测与 s 官方数据柱状对比；s 未填时只画 n 并标注。"""
    setup_style()
    rows = _read_csv(table_csv)
    names, values = [], []
    for r in rows:
        raw = r.get("median_ms", "")
        if raw in (None, ""):
            continue
        names.append(r["model"])
        values.append(float(raw))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(names, values, color=[COLORS["main"], COLORS["alt"]][:len(values)])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("单图耗时 (ms)")
    ax.set_title("模型 CPU 推理耗时对比（n 实测 / s 官方）")
    ax.grid(axis="y", color=COLORS["grid"], alpha=0.6)
    if len(values) < 2:
        ax.text(0.5, 0.5, "yolov8s 官方数据待填（T028）", transform=ax.transAxes,
                ha="center", va="center", color=COLORS["alt"])
    fig.tight_layout()
    return _save(fig, out_path)


def scene_miss_rate(table_csv: Path, out_path: Path) -> Path:
    """T025：五场景漏检率柱状图。"""
    setup_style()
    rows = _read_csv(table_csv)
    scenes = [r["scene"] for r in rows]
    rates = [float(r["miss_rate"]) for r in rows]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(scenes, rates, color=COLORS["main"])
    ax.set_ylabel("漏检率")
    ax.set_title("五场景漏检率对比")
    ax.grid(axis="y", color=COLORS["grid"], alpha=0.6)
    for i, rate in enumerate(rates):
        ax.text(i, rate, f"{rate:.2f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    return _save(fig, out_path)


def run_all(detection_pairs: int = 6) -> dict[str, Path]:
    """从三份 CSV 出图，并抽若干张图画框。detection_pairs 为画框样本数。"""
    cfg = load_config()
    setup_style()
    tables = resolve(cfg["output"]["tables_dir"])
    figures = resolve(cfg["output"]["figures_dir"])
    det_dir = resolve(cfg["output"]["detection_samples_dir"])
    outputs: dict[str, Path] = {
        "threshold_pr_curve": threshold_pr_curve(
            tables / "threshold_sweep.csv", figures / "threshold_pr_curve.png"),
        "latency_compare": latency_compare(
            tables / "model_benchmark.csv", figures / "latency_compare.png"),
        "scene_miss_rate": scene_miss_rate(
            tables / "scene_breakdown.csv", figures / "scene_miss_rate.png"),
    }
    # T021：抽图画框；同时收集用于四宫格
    import cv2

    model = load_model(cfg)
    pairs = iter_image_paths(cfg)
    grid_samples: list[tuple[str, object]] = []
    for scene, path in pairs[:detection_pairs]:
        result = detect_image(model, path, image_id=path.stem, cfg=cfg)
        out = det_dir / f"det_{path.stem}.png"
        draw_boxes(path, result, out, cfg)
        image_bgr = cv2.imread(str(out))
        if image_bgr is not None:
            grid_samples.append((f"{scene} / {path.stem}", image_bgr))
    outputs["detection_samples"] = det_dir
    if grid_samples:
        outputs["grid_overview"] = grid_overview(
            grid_samples, figures / "grid_overview.png")
    return outputs


if __name__ == "__main__":
    outputs = run_all()
    print("已生成：")
    for key, path in outputs.items():
        print(f"  {key}: {path}")
