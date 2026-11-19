# 行人检测课程设计 · 开工文档

只交 PPT（5–7 分钟），代码只是用来跑出图和表。Python 3.13 + ultralytics YOLOv8n（CPU），不训练、不做界面。

仓库：https://github.com/qin-zhenbang/YoloPedestrian-Det（main 分支，骨架已推送）。

## 项目结构

```
YoloPedestrian-Det/
├── .gitignore               排除 venv/、*.pt、IDE 配置
├── docs/KICKOFF.md          本文档
├── configs/exp.yaml         所有参数：模型、conf、iou、imgsz、场景分组
├── weights/                 yolov8n.pt 已随仓库分发
├── data/
│   ├── samples/             测试图，按 day/night/occlusion/crowd/small 分目录
│   └── labels.csv           每张图的人工人数真值
├── src/
│   ├── detector.py          推理：加载模型、只留 person(classes=[0])、计时
│   ├── evaluate.py          三组实验统计，出 3 份 CSV
│   └── visualize.py         画框、四宫格、柱状图和折线图
├── results/
│   ├── figures/             直接拖进 PPT 的图
│   ├── tables/              直接粘进 PPT 的表
│   └── detection_samples/   画框后的单图
├── venv/                    本地环境，不进仓库；用 requirements.txt 重建
└── requirements.txt         版本锁定（含 CPU 版 torch 源），换机器重建用
```

三个脚本够了：detector 出结果，evaluate 出数字，visualize 出图。参数只从 `exp.yaml` 读，别在代码里写死。

## 任务单元

一行一件事，编号固定，按顺序做，不指定谁做。每个任务都能单独做完、单独验收。

### S0 环境与基线

| ID | 事 | 做完拿到什么 |
| --- | --- | --- |
| T001 | 建 venv、装依赖、锁版本 | `venv/`、`requirements.txt` |
| T002 | 预置 yolov8n.pt 权重 | `weights/yolov8n.pt`（已随仓库分发） |
| T003 | 一次性命令跑通推理（临时图，不写正式代码） | 终端确认能出 person 框 |
| T004 | 写并冻结 `exp.yaml`（含计时口径） | `configs/exp.yaml`，之后不再改 |

T003 用官方示例图或随手一张临时图；正式图集是 T005，画框图是 T014。参数尽早冻结，后面所有实验只读它。

### S1 数据与真值

| ID | 事 | 做完拿到什么 |
| --- | --- | --- |
| T005 | 准备测试图集：30–50 张、5 场景分组、统一命名、登记来源 | `data/samples/`、`data/SOURCES.md` |
| T006 | 人工数人数 | `data/labels.csv` |

### S2 推理

| ID | 事 | 做完拿到什么 |
| --- | --- | --- |
| T007 | 写 `detector.py`：数据结构 + 单图推理 | `load_model()`、`detect_image()` |
| T008 | 补批量推理、`classes=[0]`、计时与自测 | `detector.py` 完整 + 自测记录 |

### S3 三组实验

| ID | 事 | 做完拿到什么 |
| --- | --- | --- |
| T009 | 写 `evaluate.py` 统计函数（检出 / 误检 / 漏检口径） | `evaluate.py` |
| T010 | 阈值扫描 0.05–0.90 步进 0.05 | `threshold_sweep.csv`（18 档） |
| T011 | 自测 n 的 CPU 耗时，官方 s 数据作对照 | `model_benchmark.csv` |
| T012 | 五场景漏检统计 | `scene_breakdown.csv` |
| T013 | 三份 CSV 抽样复核 + 写指标口径说明 | 复核记录、`docs/metrics_note.md` |

### S4 出图

| ID | 事 | 做完拿到什么 |
| --- | --- | --- |
| T014 | 定图表风格 + `draw_boxes()` | `visualize.py` 基础、`detection_samples/` |
| T015 | 2×2 四宫格 | `grid_overview.png` |
| T016 | 阈值双线图 | `threshold_pr_curve.png` |
| T017 | n 的 CPU 耗时图（附官方 s 对照） | `latency_compare.png` |
| T018 | 五场景漏检柱状图 | `scene_miss_rate.png` |
| T019 | 挑失败案例图 + 全图乱码裁剪检查 | 6–8 张案例图、检查记录 |

### S5 装订

| ID | 事 | 做完拿到什么 |
| --- | --- | --- |
| T020 | 核对官方数据引用（出处、COCO val / 640、硬件条件） | 引用标注清单 |
| T021 | 组装 8 页 PPT | `presentation.pptx` |
| T022 | 写 6 分 30 秒讲稿 + 静态兜底页 | `speaker_notes.md`、备用 PNG |
| T023 | 计时彩排 | 实测 ≤ 7 分钟 |
| T024 | 逐页核对图 / 表 / 数字齐全且说得出来源 | 复核记录 |

S2 / S3 卡着后面所有出图和装订，优先做。老师若明确要求训练，再插三件事：整数据、CPU 微调 10–20 epoch、把曲线放进 PPT。

进度：T001、T002、T004、T005、T006 已完成（venv + requirements.txt + yolov8n.pt + exp.yaml 已冻结 + 40 张五场景图集与 SOURCES.md + labels.csv 40 行真值）；T003 冒烟待补，T007 起未开始。
