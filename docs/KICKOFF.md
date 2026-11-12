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

一行一件事，编号固定，按顺序做，不指定谁做。

### S0 环境与基线

| ID | 事 | 做完拿到什么 |
| --- | --- | --- |
| T001 | 建 venv、装依赖、锁版本 | `venv/`、`requirements.txt` |
| T002 | 下载并预置 yolov8n.pt | `weights/yolov8n.pt`（已随仓库分发） |
| T003 | 一次性命令跑通推理（临时图，不写正式代码） | 终端确认能出 person 框 |
| T004 | 定计时口径（预热 + 中位数） | 一段口径说明（先记文档，不依赖 exp.yaml） |

T003 用官方示例图或随手一张临时图即可，正式图集由 T006 提供；画框图是 T021 的事。

### S1 数据与真值

| ID | 事 | 做完拿到什么 |
| --- | --- | --- |
| T005 | 登记图片来源和授权 | `data/SOURCES.md` |
| T006 | 收 30–50 张图，按 5 场景分组 | `samples/` 五个目录，每组 ≥ 6 张 |
| T007 | 统一命名、检查路径 | 全英文无空格的图集 |
| T008 | 人工数人数 | `labels.csv` |
| T009 | 写并冻结 `exp.yaml` | 参数定了就不再改 |

### S2 推理

| ID | 事 | 做完拿到什么 |
| --- | --- | --- |
| T010 | 定 Detection / DetectResult | `detector.py` 数据类 |
| T011 | `load_model()` + `detect_image()` | 单图能出 person 框 |
| T012 | `detect_batch()` + `classes=[0]` | 批量 50 张不报错 |
| T013 | 计时逻辑 + 自测 | 耗时波动 < 10% |

### S3 评估

| ID | 事 | 做完拿到什么 |
| --- | --- | --- |
| T014 | 检出 / 误检 / 漏检统计函数 | `evaluate.py` |
| T015 | 阈值扫描 0.05–0.90 步进 0.05 | `threshold_sweep.csv`（18 档） |
| T016 | 自测 n 的 CPU 耗时，s 引官方数据 | `model_benchmark.csv` |
| T017 | 五场景漏检统计 | `scene_breakdown.csv` |
| T018 | 三份 CSV 抽样人眼复核 | 复核记录 |
| T019 | 写指标口径说明（PPT 脚注） | `docs/metrics_note.md` |

### S4 出图

| ID | 事 | 做完拿到什么 |
| --- | --- | --- |
| T020 | 定图表风格（字体 / dpi / 配色） | `visualize.py` 风格常量 |
| T021 | `draw_boxes()` + `det_` 单图 | `detection_samples/` |
| T022 | 2×2 四宫格 | `grid_overview.png` |
| T023 | 阈值双线图 | `threshold_pr_curve.png` |
| T024 | n 的 CPU 耗时图（附官方 s 对照） | `latency_compare.png` |
| T025 | 五场景漏检柱状图 | `scene_miss_rate.png` |
| T026 | 挑失败案例图 | 6–8 张 good / mid / bad |
| T027 | 查乱码和裁剪 | 所有图能直拖 PPT |

### S5 装订

| ID | 事 | 做完拿到什么 |
| --- | --- | --- |
| T028 | 核对官方数据引用 | 出处和条件标齐 |
| T029 | 产物登记表打勾 | 6 项都有来源 |
| T030 | 组装 8 页 PPT | `presentation.pptx` |
| T031 | 写 6 分 30 秒讲稿 | `speaker_notes.md` |
| T032 | 做静态兜底页 | 备用 PNG |
| T033 | 计时彩排 | 实测 ≤ 7 分钟 |
| T034 | 最终数字复核 | 每个数字说得出来源 |

S2 / S3 卡着后面所有出图和装订，优先做。老师若明确要求训练，再插三件事：整数据、CPU 微调 10–20 epoch、把曲线放进 PPT。

进度：T001、T002 已完成（venv + requirements.txt + yolov8n.pt）；T003 起未开始。
