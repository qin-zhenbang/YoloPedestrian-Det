# 图件检查记录（T019）

检查日期：2026-11-19　环境：Python 3.13 / matplotlib Agg / Pillow / Microsoft YaHei

## 一、图件清单

| 文件 | 尺寸（px） | 用途 |
|---|---|---|
| results/figures/grid_overview.png | 1992×1700 | 2×2 四宫格总览 |
| results/figures/threshold_pr_curve.png | 1547×983 | 阈值双线图 |
| results/figures/latency_compare.png | 1610×746 | 耗时对比 |
| results/figures/scene_miss_rate.png | 1642×918 | 场景漏检柱状图 |
| results/detection_samples/*_pred.png | 原图尺寸 | 五张单图样图 |
| results/detection_samples/*_fail.png | 原图尺寸 | 八张失败案例 |

失败案例（conf=0.25，逐图统计与 CSV 一致）：

| 图片 | 场景 | 真值 | 检测 | 检出 | 误检 | 漏检 | 主要问题 |
|---|---|---|---|---|---|---|---|
| crowd_005 | crowd | 165 | 8 | 8 | 0 | 157 | 百人合影，远处人脸过小 |
| night_007 | night | 160 | 13 | 13 | 0 | 147 | 夜间看台，极小目标 |
| crowd_006 | crowd | 46 | 15 | 15 | 0 | 31 | 室内密集人群 |
| crowd_004 | crowd | 29 | 12 | 12 | 0 | 17 | 人群遮挡与尺度变化 |
| small_008 | small | 6 | 0 | 0 | 0 | 6 | 海边极小人像，全部漏检 |
| small_007 | small | 11 | 5 | 5 | 0 | 6 | 雪坡远景小目标 |
| occlusion_008 | occlusion | 3 | 6 | 3 | 3 | 0 | 救生圈等被误判为行人 |
| day_005 | day | 5 | 7 | 5 | 2 | 0 | 右缘低置信框误检 |

## 二、检查项与结果

1. 中文渲染：matplotlib 与 PIL 均使用 Microsoft YaHei；生成时未出现 Glyph missing / findfont 警告，图中标题、坐标轴、图例、标签均正常。
2. 文本裁剪：四宫格标题与图注经 PIL textbbox 排版在画布内；失败案例标签增加了边界判断与冲突跳过，未出现超出画面或整块被裁的文字。
3. 图片变形：四宫格使用等比缩放后 letterbox 到 960×720，未拉伸；单图与案例图保持原图尺寸。
4. 数据一致：四张图的数值均直接读取 results/tables 下的 CSV，与 T010–T012 表格逐项一致。
5. 文件完整性：全部 PNG 可正常打开，尺寸与上表一致，无 0 字节文件。
6. 目检结果：四宫格、双线图、耗时图、柱状图逐张查看，无乱码、无遮挡关键信息；失败案例中漏检区域（未画框的人）与误检框清晰可辨。

## 三、T019 中的修改

- draw_boxes() 增加标签防重叠：优先框上方，其次下方、框内；空间不足或与已有标签冲突时跳过该标签。密集场景（如 crowd_005、crowd_008）不再出现标签压字。
- 新增 render_failure_cases()，统一输出八张 `_fail.png`。

## 四、已知瑕疵与说明

- crowd_005、crowd_008 等图标签较密，防重叠策略会主动跳过部分标签，框本身完整保留。
- small_008 检测数为 0，图中没有框，属模型能力问题，不是绘图错误。
- night_007 与 crowd_005 的真值为估算值（见 data/labels.csv 备注），对应漏检率存在计数误差。
- day_005 的两个误检框置信度 0.29/0.33，位于画面右缘，可作为“基线精确率并非 100%”的例子。

结论：图件检查通过，可用于 PPT。
