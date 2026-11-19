# 引用标注清单

核对日期：2026-11-19。目的：报告和答辩中出现的官方数据、数据集、软件，都能说清出处、口径和硬件条件。

## 1. 官方性能数据（已核对）

- 来源：Ultralytics YOLOv8 文档，Performance Metrics 表
- URL：<https://docs.ultralytics.com/models/yolov8/#performance-metrics>
- 访问日期：2026-11-19
- 表口径：Detection (COCO)，imgsz=640；Speed 列为 CPU ONNX（ms），另有 A100 TensorRT 列

| 模型 | imgsz | mAP<sup>val 50-95</sup> | CPU ONNX (ms) | params (M) | FLOPs (B) |
|---|---|---|---|---|---|
| YOLOv8n | 640 | 37.3 | 80.4 | 3.2 | 8.7 |
| YOLOv8s | 640 | 44.9 | 128.4 | 11.2 | 28.6 |

核对结论：

- 128.4 ms 确实是 YOLOv8s / 640 / CPU ONNX 的值，与 `configs/exp.yaml` 的 `official` 段一致。
- 官方页面**没有注明 CPU 型号、核心数、ONNX Runtime 版本**；只有 GPU 列注明了 A100 TensorRT。因此只能写“官方 CPU ONNX 口径”，不能写成“在某型号 CPU 上 128.4 ms”。
- 官方同时给出 YOLOv8n 的 CPU ONNX 80.4 ms。当前图表按任务要求只放 s 作对照；报告若要讨论 n 的速度，建议同时给出 80.4，避免拿 n 的 PyTorch 实测去对 s 的 ONNX 值。

## 2. 本机实测口径（已核对）

- CPU：Intel Core Ultra 5 125H（Win32_Processor）；`platform.processor()` 返回 `Intel64 Family 6 Model 170 Stepping 4, GenuineIntel`；本环境报告 6 物理核 / 6 逻辑处理器。
- 软件：Python 3.13.14，torch 2.15.0+cpu，ultralytics 8.4.211。
- 推理：imgsz=640，batch=1，device=cpu，classes=[0]，conf=0.25，iou=0.70。
- 计时：torch 单线程（测前设为 1，默认 6，测完恢复）；预热 3 张、正式 20 张取均值；只计 preprocess + inference + postprocess。
- 结果：117.5 ms/张（`results/tables/model_benchmark.csv`）；机器负载会造成 110.8–146.9 ms 波动。

## 3. 对比注意

- 本机 117.5 ms（PyTorch CPU）与官方 128.4 ms（ONNX CPU）**不是同口径**：运行时、导出格式、CPU、线程设置都不同，只能作量级参照。
- 不能从这两个数得出“n 只比 s 快约 9%”的结论。
- 同口径比较有两种做法：
  - 官方对官方：n 80.4 ms vs s 128.4 ms（同为 CPU ONNX）；
  - 本机对本机：用同一套脚本分别测 n 和 s（当前只测了 n）。
- 官方 n 80.4 ms 低于本机 n 117.5 ms，差异主要来自 ONNX Runtime 与 PyTorch、CPU 和线程设置，不是模型本身差异。

## 4. 可直接放进报告的标注

- 官方：Ultralytics YOLOv8 Docs，“Performance Metrics”，COCO val / imgsz 640 / CPU ONNX；YOLOv8s 128.4 ms；官方未注明 CPU 型号。访问于 2026-11-19。
- 本机：Intel Core Ultra 5 125H，PyTorch 2.15.0+cpu 单线程，预热 3 张 + 正式 20 张均值，117.5 ms/张。
- 数据：COCO val2017（Lin et al., ECCV 2014），选取 40 张，5 场景各 8 张。

## 5. 软件与模型引用

YOLOv8 官方 BibTeX：

```bibtex
@software{yolov8_ultralytics,
  author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
  title = {Ultralytics YOLOv8},
  version = {8.0.0},
  year = {2023},
  url = {https://github.com/ultralytics/ultralytics},
  license = {AGPL-3.0}
}
```

- 实际安装版本为 8.4.211；官方引用模板给的是 8.0.0，两者不冲突。报告可写“Ultralytics YOLOv8（AGPL-3.0）”。
- 权重 `yolov8n.pt` 来自官方 assets release；YOLOv8 模型与代码采用 AGPL-3.0 / Enterprise 双许可。

COCO 论文：

> Lin, T.-Y., Maire, M., Belongie, S., Bourdev, L., Girshick, R., Hays, J., Perona, P., Ramanan, D., Zitnick, C. L., & Dollár, P. (2014). Microsoft COCO: Common Objects in Context. ECCV 2014. arXiv:1405.0312. DOI: 10.1007/978-3-319-10602-1_48.

- 数据集：<https://cocodataset.org/>，使用 val2017 split。
- 标注许可为 CC BY 4.0；图片来自 Flickr，单张图片的版权/许可见原页面，不能笼统写成“图片本身是 CC BY 4.0”。
- 本仓库 40 张图的原始 URL 与逐图来源见 `data/SOURCES.md`。

## 6. 风险与待办

- 报告中引用官方 128.4 ms 时，脚注必须带“官方未注明 CPU 型号，仅作量级参照”。
- 如果老师追问“为什么 n 和 s 差不多快”，用第 3 节解释；必要时补测官方 n 的 80.4 ms 作对照。
- `exp.yaml` 的 `source` 仍是文档首页，精确位置见本文第 1 节；若要 CSV 备注也带“CPU 未注明”，需改 `protocol` 并重写 `model_benchmark.csv`，本次未改。

## 7. 核对结论

PASS（带条件）：数值与协议和官方页面一致；官方硬件条件无法获得，已明确标注为“未注明”，不当作同口径对比。
