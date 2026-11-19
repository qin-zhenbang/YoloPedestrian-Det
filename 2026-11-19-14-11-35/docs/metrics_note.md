# 指标口径说明（T019，PPT 脚注用）

本项目的真值 `data/labels.csv` 只记录每张图的**人工人数**（`person_count`），没有标注框坐标。
因此所有指标按**人数口径**统计，不是基于框 IoU 匹配的目标检测 mAP。

## 数量口径定义

对每张图：

- `gt` = 人工数出的人数
- `pred` = 模型保留的 person 框数量（即 `detections` 长度）
- **检出（近似 TP）** = `min(pred, gt)`：预测与真值能对上的最大人数
- **误检（FP）** = `max(pred - gt, 0)`：多报的人数
- **漏检（FN）** = `max(gt - pred, 0)`：少报的人数
- **召回率 recall** = 检出 / gt（gt=0 时记为 1 并标注）
- **精确率 precision** = 检出 / pred（pred=0 时记为 0 并标注）

汇总时先对每张图按上述公式计算，再按场景或阈值分组求和：`总检出 = Σmin(pred,gt)`，`总误检 = Σmax(pred-gt,0)`，`总漏检 = Σmax(gt-pred,0)`。

## 阈值扫描（T015）

conf 从 0.05 到 0.90 步进 0.05，共 18 档；每一档在全图集上统计检出/误检/漏检，输出 `results/tables/threshold_sweep.csv`。
`threshold_pr_curve.png` 的双线即这一档序列上的 recall 与 precision。

## 模型耗时（T016）

n 的耗时为本地 CPU 实测（预热 10 次 + 30 次中位数，口径见 `docs/timing_note.md`）；
s 的耗时引用 ultralytics 官方文档，配置里的 `reference.*` 必须填出处 URL、测试条件（设备/imgsz/是否含前处理与 NMS），
在 PPT 脚注注明"n 为本地实测，s 为官方数据，条件：……"，不可混为一谈。

## 场景漏检（T017）

按 day / night / occlusion / crowd / small 五场景分组，输出 `results/tables/scene_breakdown.csv`。
`漏检率 = 场景总漏检 / 场景总人数`。
