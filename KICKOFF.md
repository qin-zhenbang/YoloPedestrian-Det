# 行人检测课程设计 · 开工文档

只交 PPT（5–7 分钟），代码只是用来跑出图和表。Python 3.13 + ultralytics YOLOv8n（CPU），不训练、不做界面。

## 项目结构

```
YoloPedestrian-Det/
├── docs/KICKOFF.md          本文档
├── configs/exp.yaml         所有参数：模型、conf、iou、imgsz、场景分组
├── weights/                 yolov8n.pt / yolov8s.pt（预置，免得现场联网）
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
└── requirements.txt         版本锁定，自己重跑用
```

三个脚本够了：detector 出结果，evaluate 出数字，visualize 出图。参数只从 `exp.yaml` 读，别在代码里写死。

## 任务单元

一行一件事，编号固定，谁认领就把名字填进「谁」。

### S0 环境与基线

| ID | 事 | 谁 | 做完拿到什么 |
| --- | --- | --- | --- |
| T001 | 建 venv、装依赖、锁版本 | A | `venv/`、`requirements.txt` |
| T002 | 下载并预置两个权重 | A | `weights/yolov8n.pt`、`yolov8s.pt` |
| T003 | 跑通单图推理 | B | 一张画框图 + 一次耗时 |
| T004 | 定计时口径（预热 + 中位数） | B | `exp.yaml` 里的 warmup / repeat |

### S1 数据与真值

| ID | 事 | 谁 | 做完拿到什么 |
| --- | --- | --- | --- |
| T005 | 登记图片来源和授权 | A | `data/SOURCES.md` |
| T006 | 收 30–50 张图，按 5 场景分组 | A | `samples/` 五个目录，每组 ≥ 6 张 |
| T007 | 统一命名、检查路径 | A | 全英文无空格的图集 |
| T008 | 人工数人数 | A（全员帮） | `labels.csv` |
| T009 | 写并冻结 `exp.yaml` | A | 参数定了就不再改 |

### S2 推理

| ID | 事 | 谁 | 做完拿到什么 |
| --- | --- | --- | --- |
| T010 | 定 Detection / DetectResult | B | `detector.py` 数据类 |
| T011 | `load_model()` + `detect_image()` | B | 单图能出 person 框 |
| T012 | `detect_batch()` + `classes=[0]` | B | 批量 50 张不报错 |
| T013 | 计时逻辑 + 自测 | B | 耗时波动 < 10% |

### S3 评估

| ID | 事 | 谁 | 做完拿到什么 |
| --- | --- | --- | --- |
| T014 | 检出 / 误检 / 漏检统计函数 | B | `evaluate.py` |
| T015 | 阈值扫描 0.05–0.90 步进 0.05 | B | `threshold_sweep.csv`（18 档） |
| T016 | n vs s 延迟对比 | B | `model_benchmark.csv` |
| T017 | 五场景漏检统计 | B | `scene_breakdown.csv` |
| T018 | 三份 CSV 抽样人眼复核 | A + B | 复核记录 |
| T019 | 写指标口径说明（PPT 脚注） | B | `docs/metrics_note.md` |

### S4 出图

| ID | 事 | 谁 | 做完拿到什么 |
| --- | --- | --- | --- |
| T020 | 定图表风格（字体 / dpi / 配色） | C | `visualize.py` 风格常量 |
| T021 | `draw_boxes()` + `det_` 单图 | C | `detection_samples/` |
| T022 | 2×2 四宫格 | C | `grid_overview.png` |
| T023 | 阈值双线图 | C | `threshold_pr_curve.png` |
| T024 | n / s 延迟柱状图 | C | `latency_compare.png` |
| T025 | 五场景漏检柱状图 | C | `scene_miss_rate.png` |
| T026 | 挑失败案例图 | C | 6–8 张 good / mid / bad |
| T027 | 查乱码和裁剪 | C | 所有图能直拖 PPT |

### S5 装订

| ID | 事 | 谁 | 做完拿到什么 |
| --- | --- | --- | --- |
| T028 | 核对官方数据引用 | C | 出处和条件标齐 |
| T029 | 产物登记表打勾 | 负责人 | 6 项都有来源 |
| T030 | 组装 8 页 PPT | C | `presentation.pptx` |
| T031 | 写 6 分 30 秒讲稿 | C | `speaker_notes.md` |
| T032 | 做静态兜底页 | C | 备用 PNG |
| T033 | 计时彩排 | 全员 | 实测 ≤ 7 分钟 |
| T034 | 最终数字复核 | 负责人 | 每个数字说得出来源 |

S2 / S3（B 线）卡着后面所有出图和装订，优先给人。老师若明确要求训练，再插三件事：整数据、CPU 微调 10–20 epoch、把曲线放进 PPT。
