# 测试图集来源登记

共 40 张，5 场景各 8 张。统一命名 `{scene}_{NNN}.jpg`。本表登记每张图的原始出处与下载 URL，供核对引用。

## 数据集级出处

- **COCO val2017**（全部 40 张，含 night）：Lin et al., *Microsoft COCO: Common Objects in Context*, ECCV 2014。官网 <https://cocodataset.org/>。图像为 Flickr 用户照片（各自许可），标注 CC BY 4.0。本仓库经 HF 镜像 `LibreYOLO/coco-val2017-mini500`（val2017 的 500 张子集）下载，原始直链 `http://images.cocodataset.org/val2017/{id}.jpg`。**选 val2017 而非 train2017：预训练 YOLOv8 见过 train2017，用验证集才不构成数据泄漏。**

- **night 场景选型说明**：初版曾用 LLVIP 可见光（极低光，平均像素 1–11，人几乎不可见），按反馈弃用；也试过 BDD100K 夜驾（够亮但几乎无行人）与 ExDark（镜像标注损坏）。最终 night = COCO val2017 中**夜间/傍晚户外、人工照明（路灯/泛光/活动灯）下行人清晰可见**的 8 张，平均亮度 40–78，兼顾「是夜景」与「人看得清、可人工计数」。

## 场景定义

- day：白天户外、光照充足、行人中等尺寸（基线）。
- night：夜间/傍晚户外，人工照明下行人清晰可见。
- occlusion：行人被物体或其他人部分遮挡。
- crowd：多人密集同框。
- small：行人在画面中占比极小（远距）。

## 逐图登记

| 新文件名 | 场景 | 原始标识 | 下载 URL | 许可 | 内容备注 |
| --- | --- | --- | --- | --- | --- |
| day_001.jpg | day | COCO val2017 000000002473.jpg | http://images.cocodataset.org/val2017/000000002473.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 晴日滑雪腾空，蓝天背景 |
| day_002.jpg | day | COCO val2017 000000006954.jpg | http://images.cocodataset.org/val2017/000000006954.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 草地儿童持飞盘 |
| day_003.jpg | day | COCO val2017 000000008690.jpg | http://images.cocodataset.org/val2017/000000008690.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 围栏边女孩与山羊 |
| day_004.jpg | day | COCO val2017 000000013291.jpg | http://images.cocodataset.org/val2017/000000013291.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 黄昏草地飞盘人群 |
| day_005.jpg | day | COCO val2017 000000016249.jpg | http://images.cocodataset.org/val2017/000000016249.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 户外长椅坐着的人 |
| day_006.jpg | day | COCO val2017 000000031817.jpg | http://images.cocodataset.org/val2017/000000031817.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 戴头盔骑踏板车的人 |
| day_007.jpg | day | COCO val2017 000000042628.jpg | http://images.cocodataset.org/val2017/000000042628.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 雪堆后远处行走的人 |
| day_008.jpg | day | COCO val2017 000000047585.jpg | http://images.cocodataset.org/val2017/000000047585.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 撑伞婚礼新人 |
| night_001.jpg | night | COCO val2017 000000001268.jpg | http://images.cocodataset.org/val2017/000000001268.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 黄昏/夜水边桥下，行人可见 |
| night_002.jpg | night | COCO val2017 000000011511.jpg | http://images.cocodataset.org/val2017/000000011511.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 夜广场雕像旁行人 |
| night_003.jpg | night | COCO val2017 000000018193.jpg | http://images.cocodataset.org/val2017/000000018193.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 夜户外坐吃，灯光照亮人物 |
| night_004.jpg | night | COCO val2017 000000025393.jpg | http://images.cocodataset.org/val2017/000000025393.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 夜街路灯下两名行人 |
| night_005.jpg | night | COCO val2017 000000026690.jpg | http://images.cocodataset.org/val2017/000000026690.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 夜滑板场泛光，选手与观众 |
| night_006.jpg | night | COCO val2017 000000033005.jpg | http://images.cocodataset.org/val2017/000000033005.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 夜网球场泛光下单人 |
| night_007.jpg | night | COCO val2017 000000039405.jpg | http://images.cocodataset.org/val2017/000000039405.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 夜网球馆泛光球场与看台人群 |
| night_008.jpg | night | COCO val2017 000000047828.jpg | http://images.cocodataset.org/val2017/000000047828.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 夜滨水蓝灯桥，岸边人群 |
| occlusion_001.jpg | occlusion | COCO val2017 000000000872.jpg | http://images.cocodataset.org/val2017/000000000872.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 棒球跑垒者前后重叠 |
| occlusion_002.jpg | occlusion | COCO val2017 000000001000.jpg | http://images.cocodataset.org/val2017/000000001000.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 网球场儿童相互遮挡 |
| occlusion_003.jpg | occlusion | COCO val2017 000000002685.jpg | http://images.cocodataset.org/val2017/000000002685.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 柜台后人被遮 |
| occlusion_004.jpg | occlusion | COCO val2017 000000006771.jpg | http://images.cocodataset.org/val2017/000000006771.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 前排女子挡住后排人群 |
| occlusion_005.jpg | occlusion | COCO val2017 000000012748.jpg | http://images.cocodataset.org/val2017/000000012748.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 男子抱婴儿被马头部分遮 |
| occlusion_006.jpg | occlusion | COCO val2017 000000034139.jpg | http://images.cocodataset.org/val2017/000000034139.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 两人下半身被行李箱雕塑遮 |
| occlusion_007.jpg | occlusion | COCO val2017 000000043314.jpg | http://images.cocodataset.org/val2017/000000043314.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 滑雪者被旗门杆遮 |
| occlusion_008.jpg | occlusion | COCO val2017 000000046497.jpg | http://images.cocodataset.org/val2017/000000046497.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 船上儿童被索具与彼此遮 |
| crowd_001.jpg | crowd | COCO val2017 000000002299.jpg | http://images.cocodataset.org/val2017/000000002299.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 黑白学校合影多排 |
| crowd_002.jpg | crowd | COCO val2017 000000005001.jpg | http://images.cocodataset.org/val2017/000000005001.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 剪彩仪式密集人群 |
| crowd_003.jpg | crowd | COCO val2017 000000014439.jpg | http://images.cocodataset.org/val2017/000000014439.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 公园放风筝散布人群 |
| crowd_004.jpg | crowd | COCO val2017 000000016228.jpg | http://images.cocodataset.org/val2017/000000016228.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 马车游行路边人群 |
| crowd_005.jpg | crowd | COCO val2017 000000024021.jpg | http://images.cocodataset.org/val2017/000000024021.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 黑白学校密集合影 |
| crowd_006.jpg | crowd | COCO val2017 000000036494.jpg | http://images.cocodataset.org/val2017/000000036494.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 食堂密集就座人群 |
| crowd_007.jpg | crowd | COCO val2017 000000037689.jpg | http://images.cocodataset.org/val2017/000000037689.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 雪场半管旁观者列队 |
| crowd_008.jpg | crowd | COCO val2017 000000049759.jpg | http://images.cocodataset.org/val2017/000000049759.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 室内排球多人 |
| small_001.jpg | small | COCO val2017 000000001761.jpg | http://images.cocodataset.org/val2017/000000001761.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 桥与飞机，极小行人 |
| small_002.jpg | small | COCO val2017 000000007281.jpg | http://images.cocodataset.org/val2017/000000007281.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 海滩骑者，背景极小行人 |
| small_003.jpg | small | COCO val2017 000000013348.jpg | http://images.cocodataset.org/val2017/000000013348.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 机场飞机旁极小地勤 |
| small_004.jpg | small | COCO val2017 000000020571.jpg | http://images.cocodataset.org/val2017/000000020571.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 港口船边极小码头人 |
| small_005.jpg | small | COCO val2017 000000028993.jpg | http://images.cocodataset.org/val2017/000000028993.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 广场护柱后远处极小人 |
| small_006.jpg | small | COCO val2017 000000032941.jpg | http://images.cocodataset.org/val2017/000000032941.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 街道杆后远处极小人 |
| small_007.jpg | small | COCO val2017 000000051938.jpg | http://images.cocodataset.org/val2017/000000051938.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 雪坡远处极小滑雪者 |
| small_008.jpg | small | COCO val2017 000000052891.jpg | http://images.cocodataset.org/val2017/000000052891.jpg | COCO val2017 (Lin et al., ECCV 2014); 图源自 Flickr, 标注 CC BY 4.0 | 海滩狗前景，背景极小行人 |
