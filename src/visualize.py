"""可视化：统一的图表风格与检测框绘制。

统一图表风格，并产出 results/detection_samples/ 的画框图；
四宫格、阈值曲线、耗时图、场景柱状图都复用这里的 PALETTE 与 apply_style()。
"""

# 文件分三块：
# 1. 风格与字体：PALETTE 配色、apply_style（matplotlib）、_load_font（PIL 画框用）；
# 2. 画框图：draw_boxes，以及 5 张样图 / 7 张失败案例；
# 3. 统计图：四宫格、阈值双线图、耗时对比图、场景漏检柱状图。
# 统计图都直接读 results/tables 下的 CSV，保证图里的数字和表完全一致。

from pathlib import Path

import csv

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from detector import Detection, detect_image, load_config, load_model
from evaluate import TABLES_DIR, load_labels

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = PROJECT_ROOT / "results" / "detection_samples"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

# 全套图共用的一套颜色：
# - primary 深青：主要数据序列（检出率、柱状图、实测耗时）；
# - secondary / box 珊瑚：对比序列（精确率）和检测框；
# - neutral 浅灰：官方参照、辅助线；subtle 深灰：脚注文字；
# - grid 更浅的灰：网格线；text 接近黑：标题和正文。
PALETTE = {
    "primary": "#0F766E",
    "secondary": "#F87454",
    "neutral": "#94A3B8",
    "subtle": "#64748B",
    "text": "#1F2937",
    "grid": "#E2E8F0",
    "box": "#F87454",
}

# 每场景挑一张代表性图片出框，路径与真值见 data/samples 与 labels.csv。
SAMPLE_IMAGES = [
    "day_005.jpg",
    "night_004.jpg",
    "occlusion_001.jpg",
    "crowd_008.jpg",
    "small_002.jpg",
]

# 2×2 四宫格选图：day 作基线，night / crowd / small 分别对应低光、密集、远小。
GRID_IMAGES = [
    "day_005.jpg",
    "night_004.jpg",
    "crowd_008.jpg",
    "small_002.jpg",
]

SCENE_CN = {
    "day": "白天",
    "night": "夜间",
    "occlusion": "遮挡",
    "crowd": "密集",
    "small": "远小",
}

# 失败案例：前 6 张以漏检为主，最后 1 张以误检为主。
FAILURE_IMAGES = [
    "crowd_005.jpg",
    "night_007.jpg",
    "crowd_006.jpg",
    "crowd_004.jpg",
    "small_008.jpg",
    "small_007.jpg",
    "occlusion_008.jpg",
]

_FONT_CANDIDATES = [
    (r"C:\Windows\Fonts\msyhbd.ttc", True),
    (r"C:\Windows\Fonts\msyh.ttc", True),
    (r"C:\Windows\Fonts\simhei.ttf", True),
    ("arialbd.ttf", False),
]


def _available_font() -> str | None:
    """从常见中文字体里挑一个已安装的，供 matplotlib 使用。"""
    # matplotlib 只认字体名，不认文件路径，所以先列出系统已注册的字体名再匹配。
    # 找不到中文字体时返回 None，调用方会保持 matplotlib 默认字体。
    names = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "PingFang SC"):
        if name in names:
            return name
    return None


def apply_style() -> dict[str, str]:
    """冻结全套图表的 matplotlib 风格，返回配色表。"""
    # 注意：这里只影响 matplotlib 画的统计图；
    # 画框图和四宫格用的是 PIL，字体由下面的 _load_font 单独控制。
    font = _available_font()
    if font:
        plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "axes.unicode_minus": False,
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "axes.edgecolor": PALETTE["grid"],
            "axes.labelcolor": PALETTE["text"],
            "axes.titlecolor": PALETTE["text"],
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.color": PALETTE["text"],
            "ytick.color": PALETTE["text"],
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 10,
        }
    )
    return PALETTE


def _load_font(size: int) -> tuple[ImageFont.FreeTypeFont, bool]:
    """加载中文字体，返回 (字体, 是否支持中文)。找不到中文字体时退回英文字体。"""
    # 按优先级尝试：微软雅黑粗体 → 微软雅黑 → 黑体 → Arial。
    # 前三个 supports_cjk=True，最后一个 False，调用方据此决定标签写“行人”还是 person。
    for path, supports_cjk in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size), supports_cjk
        except OSError:
            continue
    return ImageFont.load_default(), False


def draw_boxes(
    image: str | Path | Image.Image,
    detections: list[Detection],
    color: str = PALETTE["box"],
    show_conf: bool = True,
) -> Image.Image:
    """在图片上画 person 检测框，返回新的 RGB 图像（不改原图）。

    标签优先放在框上方，放不下依次尝试下方、框内；与已放标签冲突时跳过，
    避免密集场景标签互相压字。
    """
    img = Image.open(image).convert("RGB") if isinstance(image, (str, Path)) else image.convert("RGB")
    # 统一转 RGB：部分测试图是灰度图（如 crowd_005），PIL 画彩色框前必须先转换。
    draw = ImageDraw.Draw(img)
    w, h = img.size
    # 线宽和字号按图片短边缩放，保证大图小图上框和字看起来粗细一致。
    line_width = max(2, round(min(w, h) / 250))
    font_size = max(13, round(min(w, h) / 28))
    font, supports_cjk = _load_font(font_size)
    prefix = "行人" if supports_cjk else "person"
    # placed 记录已经画过的标签矩形，用来判断新标签会不会和旧标签重叠。
    placed: list[tuple[int, int, int, int]] = []

    def overlaps(rect: tuple[int, int, int, int]) -> bool:
        return any(
            not (rect[2] <= p[0] or p[2] <= rect[0] or rect[3] <= p[1] or p[3] <= rect[1])
            for p in placed
        )

    for det in detections:
        x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
        # 先画一圈更粗的白框当描边，再画珊瑚色细框；
        # 这样无论背景是白天亮色还是夜间暗色，框都能看清。
        draw.rectangle([x1 - 1, y1 - 1, x2 + 1, y2 + 1], outline="white", width=line_width + 2)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        label = f"{prefix} {det.conf:.2f}" if show_conf else prefix
        # textbbox 量出文字实际占多大，标签底色框按这个尺寸画。
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        tw, th = right - left, bottom - top
        # 横坐标夹在图片内，避免框靠右时标签被裁掉。
        tx = min(max(0, x1), max(0, w - tw - 8))
        # 标签候选位置：框上方 → 框下方 → 框内顶部。
        # 上方放不下（超出图片）或会和已有标签重叠时，自动换下一个位置。
        candidates = (y1 - th - 8, y2 + 2, y1 + 1)
        for ty in candidates:
            rect = (tx, ty, tx + tw + 8, ty + th + 6)
            if ty < 0 or rect[3] > h or overlaps(rect):
                continue
            draw.rectangle(rect, fill="white", outline=color, width=1)
            draw.text((tx + 4, ty + 2), label, fill=PALETTE["text"], font=font)
            placed.append(rect)
            break
        # 三个位置都冲突就跳过这个标签，但框本身已经画上了。

    return img


def render_samples(
    model,
    cfg: dict,
    sample_ids: list[str] = SAMPLE_IMAGES,
    out_dir: str | Path = SAMPLES_DIR,
) -> dict[str, int]:
    """按 SAMPLE_IMAGES 生成画框图，返回 {图片名: 框数}。"""
    root = PROJECT_ROOT / cfg["data"]["samples_dir"]
    index = {p.name: p for p in root.glob("*/*.jpg")}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for image_id in sample_ids:
        # 每张图单独推理 + 画框，文件名固定为 {图片名}_pred.png，方便报告/README 引用。
        detections = detect_image(model, index[image_id], cfg)
        image = draw_boxes(index[image_id], detections)
        save_path = out_dir / f"{Path(image_id).stem}_pred.png"
        image.save(save_path)
        counts[image_id] = len(detections)
        print(f"{image_id}: {len(detections)} 框 -> {save_path.relative_to(PROJECT_ROOT)}")
    return counts


def render_failure_cases(
    model,
    cfg: dict,
    image_ids: list[str] = FAILURE_IMAGES,
    out_dir: str | Path = SAMPLES_DIR,
) -> dict[str, int]:
    """生成失败案例图（漏检/误检），返回 {图片名: 框数}。"""
    root = PROJECT_ROOT / cfg["data"]["samples_dir"]
    index = {p.name: p for p in root.glob("*/*.jpg")}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for image_id in image_ids:
        # 与样图相同的画法，只是文件名加 _fail 后缀，便于区分用途。
        detections = detect_image(model, index[image_id], cfg)
        image = draw_boxes(index[image_id], detections)
        save_path = out_dir / f"{Path(image_id).stem}_fail.png"
        image.save(save_path)
        counts[image_id] = len(detections)
        print(f"{image_id}: {len(detections)} 框 -> {save_path.relative_to(PROJECT_ROOT)}")
    return counts


def _letterbox(image: Image.Image, size: tuple[int, int] = (960, 720)) -> Image.Image:
    """把不同长宽比的图等比缩放后贴到统一白底画布，保证四宫格对齐且不拉伸。"""
    canvas = Image.new("RGB", size, "white")
    # 取宽、高两个缩放比例中较小的一个：保证整张图都能塞进画布，不裁切。
    # 例如 640×480 的图放进 960×720，缩放比取 1.5；
    # 而 640×361 的图宽度先到 960，高度只到 541，上下会留白。
    scale = min(size[0] / image.width, size[1] / image.height)
    thumb = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.LANCZOS,
    )
    # 把缩放后的图放到画布正中间，左右/上下留白的部分就是白底。
    canvas.paste(thumb, ((size[0] - thumb.width) // 2, (size[1] - thumb.height) // 2))
    return canvas


def make_grid_overview(
    model,
    cfg: dict,
    image_ids: list[str] = GRID_IMAGES,
    out_path: str | Path = FIGURES_DIR / "grid_overview.png",
) -> Path:
    """生成 2×2 四宫格总览图，每格标注场景、真值和检测数。

    四张图长宽比不同，先用白底 letterbox 统一成 960×720，再用 PIL 拼版；
    不用 matplotlib 子图，避免不同长宽比在子图里留下大片空白。
    """
    labels = load_labels(cfg)
    root = PROJECT_ROOT / cfg["data"]["samples_dir"]
    index = {p.name: p for p in root.glob("*/*.jpg")}

    cell_w, cell_h = 960, 720
    # 每个格子上方留 56px 写图注，格子之间和四周留 24px 空隙，最上方 76px 放总标题。
    caption_h, gutter, title_h = 56, 24, 76
    cols = 2
    rows = (len(image_ids) + cols - 1) // cols
    # 画布尺寸按“格子数 × 格子大小 + 间隙”算出来，保证所有元素都落在画布内。
    width = cols * cell_w + (cols + 1) * gutter
    height = title_h + rows * (caption_h + cell_h) + (rows + 1) * gutter
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    title_font, _ = _load_font(34)
    caption_font, _ = _load_font(24)
    # anchor="mm" 表示以文字中心点为基准，所以坐标直接给画布/格子的中心。
    draw.text(
        (width // 2, title_h // 2),
        f"YOLOv8n 行人检测示例（conf={cfg['model']['conf']}，CPU）",
        font=title_font,
        fill=PALETTE["text"],
        anchor="mm",
    )

    for i, image_id in enumerate(image_ids):
        detections = detect_image(model, index[image_id], cfg)
        image = draw_boxes(index[image_id], detections)
        label = labels[image_id]
        caption = (
            f"{SCENE_CN[label.scene]} · {Path(image_id).stem}　"
            f"真值 {label.person_count} / 检测 {len(detections)}"
        )
        # i % cols 决定第几列，i // cols 决定第几行；
        # 图注占格子上方 caption_h，图片从图注下面开始贴。
        x = gutter + (i % cols) * (cell_w + gutter)
        y = title_h + gutter + (i // cols) * (caption_h + cell_h + gutter)
        draw.text(
            (x + cell_w // 2, y + caption_h // 2),
            caption,
            font=caption_font,
            fill=PALETTE["text"],
            anchor="mm",
        )
        canvas.paste(_letterbox(image, (cell_w, cell_h)), (x, y + caption_h))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"四宫格 -> {out_path.relative_to(PROJECT_ROOT)}")
    return out_path


def plot_threshold_curve(
    cfg: dict,
    table_path: str | Path = TABLES_DIR / "threshold_sweep.csv",
    out_path: str | Path = FIGURES_DIR / "threshold_pr_curve.png",
) -> Path:
    """画阈值双线图：检出率与精确率随 conf 的变化，并标出基线阈值。"""
    apply_style()
    # 直接读 evaluate.py 生成的 CSV，不重新推理；
    # 这样图上的每个点和表格里的数字一定一致。
    with open(table_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    # CSV 里的比率是 0~1 小数，画图时乘 100 换成百分比，纵轴更好读。
    thresholds = [float(r["threshold"]) for r in rows]
    recall = [100 * float(r["recall"]) for r in rows]
    precision = [100 * float(r["precision"]) for r in rows]
    baseline = float(cfg["model"]["conf"])

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(thresholds, recall, color=PALETTE["primary"], linewidth=2,
            marker="o", markersize=4, label="检出率")
    ax.plot(thresholds, precision, color=PALETTE["secondary"], linewidth=2,
            marker="s", markersize=4, label="精确率")
    # 竖虚线标出 exp.yaml 的基线阈值，让读者一眼看到 0.25 在曲线上的位置。
    ax.axvline(baseline, color=PALETTE["neutral"], linestyle="--", linewidth=1)
    ax.text(baseline + 0.012, 3, f"基线 conf={baseline}", color=PALETTE["text"], fontsize=10)

    ax.set_xlim(0.03, 0.92)
    ax.set_ylim(0, 105)
    ax.set_xticks([0.05] + [round(0.1 * i, 1) for i in range(1, 10)])
    ax.set_xlabel("置信度阈值")
    ax.set_ylabel("百分比（%）")
    ax.set_title("阈值对检出率与精确率的影响（40 张测试图）")
    ax.legend(loc="lower left")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    print(f"阈值双线图 -> {out_path.relative_to(PROJECT_ROOT)}")
    return out_path


def plot_latency_compare(
    table_path: str | Path = TABLES_DIR / "model_benchmark.csv",
    out_path: str | Path = FIGURES_DIR / "latency_compare.png",
) -> Path:
    """画 CPU 单张耗时对比：本机 yolov8n 实测 vs 官方 yolov8s 参照。"""
    apply_style()
    # 按 source 列建索引（self / official），这样就不依赖 CSV 的行顺序。
    with open(table_path, encoding="utf-8-sig", newline="") as f:
        rows = {r["source"]: r for r in csv.DictReader(f)}

    self_row = rows["self"]
    official_row = rows["official"]
    # barh 的 y 轴从下往上画：列表第一个在最下面。
    # 所以把官方参照放第一个（下方），本机实测放第二个（上方）。
    values = [float(official_row["total_ms"]), float(self_row["total_ms"])]
    labels = ["YOLOv8s（官方参照）", "YOLOv8n（本机实测）"]
    colors = [PALETTE["neutral"], PALETTE["primary"]]

    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars = ax.barh(labels, values, color=colors, height=0.55)
    # 在每根柱子末端写具体毫秒数，读者不用对着刻度估。
    for bar, value in zip(bars, values):
        ax.text(value + 2, bar.get_y() + bar.get_height() / 2,
                f"{value:.1f} ms", va="center", fontsize=11, color=PALETTE["text"])

    ax.set_xlim(0, max(values) * 1.25)
    ax.set_xlabel("单张耗时（毫秒）")
    ax.set_title("CPU 单张推理耗时对比（imgsz=640）")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)

    # 脚注把两种口径写清楚，防止读者把两根柱子当成公平对比。
    fig.text(
        0.01, 0.02,
        "注：n 为本机 PyTorch CPU 单线程、20 张均值；s 为 ultralytics 官方 COCO val / CPU ONNX 数据，口径不同仅作量级参照。",
        fontsize=9,
        color=PALETTE["subtle"],
    )
    fig.subplots_adjust(bottom=0.22)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    print(f"耗时对比图 -> {out_path.relative_to(PROJECT_ROOT)}")
    return out_path


def plot_scene_miss_rate(
    cfg: dict,
    table_path: str | Path = TABLES_DIR / "scene_breakdown.csv",
    out_path: str | Path = FIGURES_DIR / "scene_miss_rate.png",
) -> Path:
    """画五场景漏检率柱状图，柱顶标漏检率与漏检数/真值数。"""
    apply_style()
    with open(table_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    # CSV 可能不是按 exp.yaml 的顺序写的，这里按配置里的场景顺序重排，
    # 保证每天出图时横轴顺序都一样。
    order = list(cfg["data"]["scenes"])
    rows.sort(key=lambda r: order.index(r["scene"]))

    scenes = [r["scene"] for r in rows]
    miss = [100 * float(r["miss_rate"]) for r in rows]
    fn = [int(r["fn"]) for r in rows]
    gt = [int(r["gt"]) for r in rows]
    # 总体漏检率用“总漏检 / 总真值”算，不是五个场景漏检率的平均；
    # 否则人数少的场景会和人数多的场景权重相同。
    overall = 100 * sum(fn) / sum(gt)
    labels = [f"{SCENE_CN[s]}\n{s}" for s in scenes]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(labels, miss, color=PALETTE["primary"], width=0.6)
    # 柱顶写“漏检率 + 漏检数/真值数”，避免只看到百分比却不知道基数。
    for bar, m, f, g in zip(bars, miss, fn, gt):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 2, f"{m:.1f}%\n{f}/{g}",
                ha="center", va="bottom", fontsize=9, color=PALETTE["text"])

    # 灰色虚线是全部 40 张的总体漏检率，用来看各场景高于还是低于总体。
    ax.axhline(overall, color=PALETTE["subtle"], linestyle="--", linewidth=1)
    ax.text(0.3, overall + 1.5, f"总体 {overall:.1f}%",
            ha="left", va="bottom", fontsize=10, color=PALETTE["subtle"])

    ax.set_ylim(0, 100)
    ax.set_ylabel("漏检率（%）")
    ax.set_title("各场景漏检率（conf=0.25，40 张测试图）")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    fig.text(
        0.01, 0.02,
        "柱顶数字：漏检率与 漏检数/真值数；虚线为全部 40 张的总体漏检率。",
        fontsize=9,
        color=PALETTE["subtle"],
    )
    fig.subplots_adjust(bottom=0.2)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    print(f"场景漏检柱状图 -> {out_path.relative_to(PROJECT_ROOT)}")
    return out_path


if __name__ == "__main__":
    # 一条命令重画全部图：
    # 先加载一次模型，再依次生成样图、失败案例、四宫格和三张统计图。
    # 统计图读取 results/tables 的 CSV，所以要先跑 evaluate.py。
    apply_style()
    cfg = load_config()
    model = load_model(cfg)
    render_samples(model, cfg)          # 5 张场景样图
    render_failure_cases(model, cfg)    # 7 张失败案例
    make_grid_overview(model, cfg)      # 2×2 四宫格
    plot_threshold_curve(cfg)           # 阈值双线图
    plot_latency_compare()              # CPU 耗时对比
    plot_scene_miss_rate(cfg)           # 五场景漏检柱状图
