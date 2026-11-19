"""评估模块（T014–T019）。

三组实验：阈值扫描、模型耗时对比、五场景漏检，各出一份 CSV。
指标口径为人数口径，见 docs/metrics_note.md；所有参数来自 exp.yaml。
"""
from __future__ import annotations

import csv
from decimal import Decimal
from pathlib import Path

from config import load_config, resolve
from detector import (
    detect_image,
    detect_batch,
    iter_image_paths,
    load_model,
    measure_latency,
)


def load_labels(config: dict | None = None) -> dict[str, dict]:
    """读取 labels.csv，返回 {image_id: {'scene':..., 'person_count':int, 'note':...}}。"""
    cfg = config if config is not None else load_config()
    path = resolve(cfg["data"]["labels_csv"])
    labels: dict[str, dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            image_id = (row.get("image_id") or "").strip()
            if not image_id:
                continue
            try:
                count = int(row.get("person_count", "0"))
            except (TypeError, ValueError):
                count = 0
            labels[image_id] = {
                "scene": (row.get("scene") or "").strip(),
                "person_count": count,
                "note": (row.get("note") or "").strip(),
            }
    return labels


def count_stats(pred: int, gt: int) -> dict:
    """人数口径的检出/误检/漏检（T014）。"""
    tp = min(pred, gt)
    fp = max(pred - gt, 0)
    fn = max(gt - pred, 0)
    return {"pred": pred, "gt": gt, "tp": tp, "fp": fp, "fn": fn}


def aggregate(results: list, labels: dict[str, dict]) -> dict:
    """对一批 DetectResult 按人数口径汇总。"""
    totals = {"pred": 0, "gt": 0, "tp": 0, "fp": 0, "fn": 0, "images": 0, "matched": 0}
    for r in results:
        label = labels.get(r.image_id)
        if label is None:
            continue
        s = count_stats(r.person_count, int(label["person_count"]))
        for key in ("pred", "gt", "tp", "fp", "fn"):
            totals[key] += s[key]
        totals["images"] += 1
        if r.image_id in labels:
            totals["matched"] += 1
    totals["recall"] = totals["tp"] / totals["gt"] if totals["gt"] else 0.0
    totals["precision"] = totals["tp"] / totals["pred"] if totals["pred"] else 0.0
    totals["miss_rate"] = totals["fn"] / totals["gt"] if totals["gt"] else 0.0
    return totals


def _thresholds(config: dict) -> list[float]:
    """生成 0.05–0.90 步进 0.05 的 18 档阈值，用 Decimal 避免浮点累积误差。"""
    ts = config["threshold_sweep"]
    start, stop, step = Decimal(str(ts["start"])), Decimal(str(ts["stop"])), Decimal(str(ts["step"]))
    values = []
    cur = start
    while cur <= stop:
        values.append(float(cur))
        cur += step
    return values


def threshold_sweep(model, config: dict | None = None, images: list | None = None,
                    labels: dict | None = None) -> Path:
    """T015：18 档阈值全图集推理，输出 threshold_sweep.csv。"""
    cfg = config if config is not None else load_config()
    labels = labels if labels is not None else load_labels(cfg)
    pairs = images if images is not None else iter_image_paths(cfg)
    if not pairs:
        raise RuntimeError("没有找到测试图片，请按 T006 先把图片放入 data/samples/ 五场景目录。")
    rows = []
    for conf in _thresholds(cfg):
        results = []
        for scene, path in pairs:
            r = detect_image(model, path, image_id=path.stem, cfg=cfg, conf=conf)
            r.scene = scene
            results.append(r)
        stats = aggregate(results, labels)
        stats["conf"] = round(conf, 2)
        stats["precision"] = round(stats["precision"], 6)
        stats["recall"] = round(stats["recall"], 6)
        rows.append(stats)
    out = resolve(cfg["output"]["tables_dir"]) / "threshold_sweep.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["conf", "images", "matched", "gt", "pred", "tp", "fp", "fn", "precision", "recall"]
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out


def model_benchmark(model, config: dict | None = None, image=None) -> Path:
    """T016：n 本地 CPU 实测耗时 + s 官方数据（引用填写），输出 model_benchmark.csv。"""
    cfg = config if config is not None else load_config()
    pairs = iter_image_paths(cfg)
    if image is None:
        if not pairs:
            raise RuntimeError("没有测试图，无法实测耗时。请按 T006 准备图片。")
        image = pairs[0][1]
    t = measure_latency(model, image, cfg)
    ref = cfg.get("reference", {})
    s_ms = ref.get("official_yolov8s_cpu_ms")
    source = ref.get("official_source_url") or ""
    condition = ref.get("official_condition") or ""
    rows = [
        {
            "model": "yolov8n",
            "source": "本地实测",
            "median_ms": round(t["median_ms"], 3),
            "spread_ratio": round(t["spread_ratio"], 4),
            "stable": t["stable"],
            "source_url": "",
            "condition": "CPU, imgsz=%s, 预热%d+中位数%d次" % (
                cfg["inference"]["imgsz"], 10, 30),
        },
        {
            "model": "yolov8s",
            "source": "官方数据" if s_ms is not None else "待填（T028）",
            "median_ms": s_ms if s_ms is not None else "",
            "spread_ratio": "",
            "stable": "",
            "source_url": source,
            "condition": condition,
        },
    ]
    out = resolve(cfg["output"]["tables_dir"]) / "model_benchmark.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "source", "median_ms", "spread_ratio", "stable", "source_url", "condition"]
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return out


def scene_breakdown(model, config: dict | None = None, images: list | None = None,
                    labels: dict | None = None) -> Path:
    """T017：五场景漏检统计，输出 scene_breakdown.csv。"""
    cfg = config if config is not None else load_config()
    labels = labels if labels is not None else load_labels(cfg)
    pairs = images if images is not None else iter_image_paths(cfg)
    per_scene: dict[str, list] = {s: [] for s in cfg["data"]["scenes"]}
    for scene, path in pairs:
        r = detect_image(model, path, image_id=path.stem, cfg=cfg)
        r.scene = scene
        per_scene.setdefault(scene, []).append(r)
    out = resolve(cfg["output"]["tables_dir"]) / "scene_breakdown.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["scene", "images", "matched", "gt", "pred", "tp", "fp", "fn",
              "miss_rate", "recall", "precision"]
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for scene, results in per_scene.items():
            stats = aggregate(results, labels)
            stats["scene"] = scene
            stats["miss_rate"] = round(stats["miss_rate"], 6)
            stats["recall"] = round(stats["recall"], 6)
            stats["precision"] = round(stats["precision"], 6)
            writer.writerow(stats)
    return out


def run_all() -> list[Path]:
    """按顺序跑完三组实验。"""
    cfg = load_config()
    labels = load_labels(cfg)
    pairs = iter_image_paths(cfg)
    model = load_model(cfg)
    paths = [
        threshold_sweep(model, cfg, pairs, labels),
        model_benchmark(model, cfg),
        scene_breakdown(model, cfg, pairs, labels),
    ]
    return paths


if __name__ == "__main__":
    outputs = run_all()
    print("已生成：")
    for p in outputs:
        print(f"  {p}")
