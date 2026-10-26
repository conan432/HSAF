> **HSAF release note.** This document is adapted from UniRank and describes all five UniRank datasets for provenance. The HSAF paper and this repository use four of them: Taobao, KuaiRand, MerRec, and TAAC-25. QK-Video is not used here.

# UniRank 五个公开数据集：获取、规模、处理与计算资源

本文档依据论文 `../papers/Unifying Sequence Modeling and Feature Interaction/UniRank.pdf`、仓库 `README.md`、`data/*/preprocess_*.py`、`config/*.yaml` 和 `benchmark/*/*.log` 整理。它回答四个问题：数据能否公开获取、论文使用的规模、UniRank 如何处理、复现大概需要多少 GPU。

## 结论先行

五个数据集均有公开来源，UniRank 也为五者提供了可直接训练的处理后版本。**优先下载 UniRank 的预处理 Parquet 数据**，不要从原始日志重建，除非你需要修改标签或切分规则。主 benchmark 统一使用：`max_len=100`、embedding dim `16`、token dim `256`、3 层、每 GPU batch size `8192`、训练 1 epoch、bf16、DDP。

论文正式实验使用 NVIDIA H20；仓库 `run_all.sh` 的默认配置是 **4 卡 DDP**。论文没有给出“最低显卡数”的硬限制。若只复现一个模型的一个数据集，单张 80 GB 级卡通常是可行的，但需要降低 per-GPU batch 并用梯度累积；要复现全部 15 模型或缩短总耗时，应使用 4 张 80 GB 级卡或 H20。

## 1. 论文处理后的规模

下表的实例数是论文 Table 2 报告的**预处理后完整样本数**，不是原始压缩文件大小。`AvgL`/`MaxL` 是用户可用行为历史长度；主表训练时仍统一截断到 100。

| 数据集 | UniRank dataset ID | 实例数 | 用户数 | 物品数 | 字段数 | 任务数 | 平均历史长度 | 最大历史长度 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| QK-Video | `QK_Video_Action` | 493,306,303 | 4,996,176 | 3,752,235 | 10 | 4 | 99 | 6,013 |
| KuaiRand | `KuaiRand_Video_Action` | 323,464,444 | 27,285 | 32,038,725 | 40 | 6 | 11,855 | 228,030 |
| TAAC-25 | `TencentGR_10M_Action` | 757,207,146 | 7,706,778 | 15,707,425 | 30 | 2 | 98 | 100 |
| Taobao | `Taobao_Action` | 23,601,301 | 470,570 | 831,643 | 23 | 4 | 50 | 3,756 |
| MerRec | `MerRec_Action` | 172,304,959 | 1,697,072 | 42,577,610 | 20 | 5 | 102 | 26,576 |

规模最大的是 TAAC-25，item vocabulary 最大的是 MerRec，其次是 KuaiRand。虽然 KuaiRand 用户仅约 2.7 万，但用户历史极长，原始最大长度超过 22 万；因此它是长序列实验的主要压力测试集。

## 2. 统一处理协议

五个数据集在进入模型前被统一为 point-wise autoregressive 样本：每个可保留事件是一个目标，模型只能使用该目标之前的 item、action、timestamp 历史。一个事件可以同时激活多个二分类标签，多标签组合被编码成一个 history action token。

共同规则：

- 默认删除有效事件数小于 10 的用户，见各预处理脚本的 `--min_user_interactions 10`。
- 类别值 `0` 留给 padding/OOV；默认 `min_feat_count=2`，低频非 ID 类别值映射为 OOV。资源紧张时论文建议提高到 10，减少 vocab 和 embedding 状态。
- 全局 `user_id`、`item_id` 不做低频裁剪，保证 ID 表示一致。
- 连续值不直接输入，价格、时长等被离散成 bucket；时间戳派生 weekday/weekend/hour。
- 输出为 `train`、`valid`、`test` 三个 split，每个 split 有匹配的 `data/`、`user_info/`、`item_info/` Parquet block；`block_manifest.json` 描述 block。
- block 内的 `user_index`、`item_index` 是局部连续索引，用于有界的 side-information cache；`user_id`、`item_id` 仍是全局特征 ID。
- `user_info` 保存完整 `full_item_seq`、`full_action_seq` 和可用的 `full_timestamp_seq`；dataloader 根据当前 target 截断/补齐因果前缀。

仓库默认分块数量通常是 train 32 块、valid 8 块、test 8 块，且 `config/dataset_config.yaml` 设为 `blocked: True`、`block_cache_size: 2`。这是避免把数亿样本一次性装入 CPU 内存的关键。

## 3. 获取方式与逐数据集处理

### 3.1 QK-Video

**来源与下载**

- 原始数据：TenRec 的 QK-Video 页面：<https://static.qblv.qq.com/qblv/h5/algo-frontend/tenrec_dataset.html>
- UniRank 处理后数据：<https://huggingface.co/datasets/salmon1802/QK-Video>
- 处理脚本：`data/QK_Video/preprocess_QK_seq_action.py`

**原始含义与标签**

QQ 看点推荐流短视频曝光，时间范围为 2021-09-17 至 2021-12-07。每条记录是曝光的 user-video 对，四个任务为 `click`、`follow`、`like`、`share`；四者均为 0 时就是仅曝光。

**UniRank 处理**

- 保留每条 impression；四个 binary 标签打包成 4-bit multi-hot action，理论上有 `2^4=16` 种 action pattern。
- 数据无公开 timestamp，因此按每个用户的发布行顺序排序，用 `_row_id` 保留该顺序；无法做按日期切分。
- 每个用户按事件位置切为约 80%/10%/10% 的 train/valid/test。
- 过滤交互数小于 10 的用户；`video_category`、`watching_times`、`gender`、`age` 做频次 OOV 过滤。
- `video_category` 写入 item_info；用户和视频 ID 维持独立全局 vocab。

**脚本默认参数**

```bash
python data/QK_Video/preprocess_QK_seq_action.py \
  --input_file /path/to/QK-video.csv \
  --output_dir /path/to/QK_Video_Action
```

默认分块：32/8/8，CSV 分块读取大小为 4,000,000 行。

### 3.2 KuaiRand-27K

**来源与下载**

- 原始数据主页：<https://kuairand.com/>
- 原始下载脚本：`data/KuaiRand/download.sh`，下载 <https://zenodo.org/records/10439422/files/KuaiRand-27K.tar.gz>。
- UniRank 处理后数据：<https://huggingface.co/datasets/salmon1802/KuaiRand>
- 处理脚本：`data/KuaiRand/preprocess_Kuairand_seq_action.py`

**原始含义与标签**

快手短视频曝光数据，覆盖 2022-04-08 至 2022-05-08，合并普通推荐流量和随机干预流量。任务为 `is_click`、`is_follow`、`is_like`、`is_comment`、`is_forward`、`long_view`。`long_view` 定义为：视频不长于 18 秒时看完，或更长视频观看至少 18 秒。

**UniRank 处理**

- 合并 standard-policy 与 random-intervention 的五个日志文件，按毫秒级 `time_ms` 排序。
- 连接 26 个用户画像字段，加入 `video_type`、`primary_tag`、`music_type`、`duration_bucket` 等视频元数据。
- 将时间转为 Asia/Shanghai 的日期、星期、是否周末、小时；播放时长和视频时长分别 bucket 化。
- 六个标签打包成 6-bit action，理论上 `2^6=64` 种历史 action。
- 按日期切分：26 天 train、3 天 valid、3 天 test。
- 用户静态字段和部分 item 字段做低频 OOV；代码注释说明当前实现主要对用户特征做 OOV，ID 和 action 不裁剪。

**脚本默认参数**

```bash
bash data/KuaiRand/download.sh
python data/KuaiRand/preprocess_Kuairand_seq_action.py \
  --data_dir /path/to/KuaiRand-27K/data \
  --output_dir /path/to/KuaiRand_Video_Action
```

这是长序列最重的数据集。即使训练时 `max_len=100`，预处理仍需保留完整用户时间序列和 block-local side information。

### 3.3 TAAC-25 / TencentGR-10M

**来源与下载**

- 原始公开数据：<https://huggingface.co/datasets/TAAC2025/TencentGR-10M>
- UniRank 处理后数据：<https://huggingface.co/datasets/salmon1802/TAAC-25>
- 导出原始 HF 数据到 Parquet：`data/TAAC2025/export_parquet.py`
- 处理脚本：`data/TAAC2025/preprocess_TAAC2025_seq_action.py`

导出命令：

```bash
python data/TAAC2025/export_parquet.py
python data/TAAC2025/preprocess_TAAC2025_seq_action.py \
  --data_dir data/TAAC2025 \
  --output_dir /path/to/TencentGR_10M_Action
```

`export_parquet.py` 会下载并导出 `item_feat`、`seq`、`user_feat` 三个 HF config。

**原始含义与标签**

腾讯 2025 广告算法比赛的匿名工业流量。论文中处理后的时间跨度为 2024-10-15 至 2025-08-05。原始 action type：0 为曝光，1 为点击，2 为下游转化。UniRank 令 type 1/2 都使 `is_click=1`，只有 type 2 使 `is_conversion=1`，强制 conversion 是 click 的子集。

**UniRank 处理**

- 用户没有 click 或 conversion 的完整序列直接删除。
- 4 个 scalar user fields、4 个 list user fields、13 个 item fields；list field 截断/补齐为最多 5 个值。
- 非 ID 的 user/item/list 元素做低频 OOV 过滤。
- 按用户 hash 分区读取 `seq.parquet`，避免把嵌套序列一次性展开到内存。
- 切分不是简单 8:1:1：先找含 click 和 conversion 正例的最短尾部 test 段，再找紧邻的最短正例 valid 段，剩余给 train。
- 论文报告的实际结果是 226 天 train、1 天 valid、68 天 test。

TAAC-25 是实例数最大的主 benchmark 数据集，完整复现时最需要 DDP 和高速本地/共享 NVMe 存储。

### 3.4 Taobao Display Ad Click / User Behavior

**来源与下载**

- 原始数据：<https://tianchi.aliyun.com/dataset/56>
- UniRank 处理后数据：<https://huggingface.co/datasets/salmon1802/Taobao>
- 处理脚本：`data/Taobao/preprocess_Taobao_seq_action.py`

原始文件包括 `raw_sample`、`ad_feature`、`user_profile`、`behavior_log`，支持 `.csv`、`.csv.gz`、`.csv.tar.gz`。

**原始含义与标签**

广告曝光位于 2017-05-06 至 2017-05-13；购物行为日志覆盖 2017-04-22 至 2017-05-13。任务为 `is_click`、`cart`、`fav`、`buy`。

**UniRank 处理**

- 每条 ad impression 是一个 anchor sample；`raw_sample.clk` 产生 click 标签。
- 连接 8 个 user profile 字段和 5 个广告属性：类目、campaign、customer、brand、价格 bucket。
- 通过同一 user、`cate_id`、`brand` 匹配曝光后 24 小时内的 `behavior_log`，将 cart/fav/buy 回填到该 impression。这里使用的是“任一匹配行为可命中”的实现，不要求两套行为日志同时出现。
- 将 click/cart/fav/buy 打包为 4-bit action；价格切成 10 个区间，时间派生北京时区的 weekend/hour。
- 按曝光日期切分：官方 8 天样本对应 6 天 train、1 天 valid、1 天 test。
- 过滤交互数小于 10 的用户，user 和 item 的非 ID 类别特征做 OOV 过滤。

**脚本默认命令**

```bash
python data/Taobao/preprocess_Taobao_seq_action.py \
  --data_dir /path/to/Taobao \
  --output_dir /path/to/Taobao_Action
```

这是五者中实例量最小，适合先做正确性验证和单卡开发。

### 3.5 Mercari MerRec

**来源与下载**

- 原始数据：<https://huggingface.co/datasets/mercari-us/merrec>
- UniRank 处理后数据：<https://huggingface.co/datasets/salmon1802/MerRec>
- 专用下载脚本：`data/MerRec/download_merrec_20231001.py`
- 处理脚本：`data/MerRec/preprocess_MerRec_seq_action.py`

下载 October 分区：

```bash
python data/MerRec/download_merrec_20231001.py \
  --local-dir /path/to/MerRec

python data/MerRec/preprocess_MerRec_seq_action.py \
  --data_dir /path/to/MerRec \
  --output_dir /path/to/MerRec_Action
```

脚本默认下载 `mercari-us/merrec` 下的 `20231001/`。若 Hugging Face 访问受限，脚本支持 `HF_ENDPOINT=https://hf-mirror.com`；私有/受限访问时可提供 `HF_TOKEN`。

**原始含义与标签**

Mercari C2C 市场事件，完整 release 覆盖 2023-05 至 2023-10，UniRank 使用最大的 October partition。任务为 `Like`、`Cart`、`Offer`、`Checkout`、`Purchase`。

**UniRank 处理**

- 先按 `user_id`、timestamp、source-row 排序。
- 每个 `(user_id, item_id)` 仅保留最早的 `item_view` 作为曝光 anchor；随后发生的 like/cart/offer/checkout/purchase 回填成该 view 的 multi-hot 标签。
- 非 view 事件只用于回填标签，不单独构成样本，也不进入历史 token。
- 保留 session、日历上下文和 10 个 item 字段，价格 bucket 化，5 标签组合编码为 action。
- 默认优先按日期 8:1:1 切分；若可用日期不足 3 个，`--split_strategy auto` 回退到每用户位置比例切分。
- 论文实际使用 25 天 train、3 天 valid、3 天 test。

MerRec 的 item 数超过 4250 万，虽然样本数低于 QK/TAAC，但 embedding table、Adagrad 累积状态和 item_info I/O 都很重。

## 4. 如何下载处理后数据并启动

推荐直接下载对应的 `salmon1802/*` dataset repository，例如：

```bash
hf download salmon1802/KuaiRand \
  --repo-type dataset \
  --local-dir /data/UniRank-data/KuaiRand_Video_Action
```

之后修改 `config/dataset_config.yaml` 中该 dataset ID 的：

```yaml
data_root: /data/UniRank-data
train_data: /data/UniRank-data/KuaiRand_Video_Action/train/data
valid_data: /data/UniRank-data/KuaiRand_Video_Action/valid/data
test_data: /data/UniRank-data/KuaiRand_Video_Action/test/data
```

同时更新三组 `*_user_info` 和 `*_item_info` 路径。不要只改 `data_root` 而遗留仓库作者的 `/mnt/ceph-...` 绝对路径。

## 5. GPU、CPU、内存和磁盘建议

### 5.1 作者的正式设置

论文 Section 3.2.6 明确给出的主 benchmark 设置：

- GPU：NVIDIA H20。
- 默认每 GPU batch size：8192。
- OOM 时：per-GPU batch 按比例降低，同时按相同比例提高 `accumulation_steps`，保持有效 batch size。
- precision：bf16 autocast；BCE 在 FP32 计算。
- 模型：embedding dim 16、token dim 256、3 层、1 epoch。
- `run_all.sh`：默认 `CUDA_VISIBLE_DEVICES=0,1,2,3`、`NPROC=4`，即 4 卡 DDP。

论文 Figure 4 还使用 OneTrans、sequence length 1000 说明工程优化效果：相对单卡基础实现，启用全部优化的 4 卡模式达到 14.24 倍吞吐，并将单卡峰值显存降至基础实现的 31%。这不是主表 `max_len=100` 的直接显存数值，但说明长序列实验需要 DDP、bf16、compile、attention kernel 和 checkpointing 一起使用。

### 5.2 推荐配置

| 目标 | 建议 GPU | 建议配置 | 说明 |
| --- | --- | --- | --- |
| 代码正确性、Taobao 单模型 | 1 x 24-48 GB | batch 512-2048，`accumulation_steps` 补足 | 先确认下载、路径、dataloader、日志和指标。|
| Taobao 完整基线 | 1 x 40-80 GB | batch 4096-8192 | 数据最小；默认 `max_len=100` 时单卡相对友好。|
| QK / KuaiRand / MerRec 单模型复现 | 1 x 80 GB 或 H20 | 从 batch 1024/2048 开始，逐步升高；保留 bf16 | 单卡能做，但完整 epoch 时间长，且 MerRec/KuaiRand 的 embedding 与 I/O 压力更大。|
| TAAC-25 单模型复现 | 1 x 80 GB 仅作调试；推荐 4 x 80 GB/H20 | 4 卡，8192 per GPU | 7.57 亿样本，单卡可跑但耗时不现实。|
| 论文主 benchmark 的单模型可比复现 | 4 x H20 或 4 x 80 GB A100/H100 | 8192 per GPU，bf16，DDP | 与 `run_all.sh` 的默认并行度一致。|
| 15 模型 x 5 数据集完整 benchmark | 至少 4 x 80 GB/H20；建议多节点或任务并发 | 每项 1 epoch，任务级并行 | 75 次主实验，不应串行单卡执行。|
| `max_len>=1000` 的 OneTrans/TokenFormer/UltraHSTU scaling | 4 x 80 GB/H20 起步 | 降 batch、启用 checkpoint 和 Flex/SDPA | 长序列 attention 活跃 token 更多，不能照搬 batch 8192。|

这张表是工程建议，不是论文承诺的最小硬件。最稳妥的启动方式是先用 `batch_size=1024`，观察峰值显存，再按 2 倍逐级增加；每次减小 batch 时同步增加 `accumulation_steps`。

### 5.3 CPU RAM 与磁盘

没有一个官方“压缩后数据集占用多少 GB”的固定数值，因为 Parquet 压缩、PyArrow 版本、数据集 revision、block 数和是否保留临时分区都会改变占用。不能把“实例数”直接换算成精确磁盘容量。

但按当前脚本的处理方式，建议：

- **开发/仅下载一个较小数据集**：至少 200 GB 可用 SSD、64 GB RAM。
- **QK、KuaiRand、TAAC-25 或 MerRec 的预处理**：至少 0.5-1 TB NVMe/高速共享存储、128 GB RAM；最好预留原始文件、临时 user-hash Parquet 分区、最终 Parquet blocks 三份空间，峰值可能接近原始加处理后数据之和。
- **全五数据集都保留**：建议多 TB 高吞吐存储和 256 GB RAM。blocked loader 的意义是控制训练期间 CPU 内存，而不是让预处理阶段完全不需要磁盘空间。

对超长历史的 KuaiRand，训练主表虽截断到 100，但 `user_info` 仍保存完整序列；因此磁盘与读取吞吐比模型激活显存更早成为瓶颈。

## 6. 运行时间的实测参照

仓库 benchmark 日志与 `run_all.sh` 默认 4 卡 DDP 配置相匹配：日志里的 `local batches/epoch` 乘 4 后约等于对应数据的 80% train 实例。以下是相同基准下、15 个模型日志中“开始训练到 epoch-1 验证”的范围，包含不同模型的速度差异：

| 数据集 | 每 rank local batches/epoch | 一轮训练时间范围 | 中位数 |
| --- | ---: | ---: | ---: |
| Taobao | 540 | 1.2-5.8 分钟 | 2.0 分钟 |
| MerRec | 4,405 | 9.8-57.4 分钟 | 18.9 分钟 |
| KuaiRand | 7,814 | 18.4-77.6 分钟 | 28.2 分钟 |
| QK-Video | 12,038 | 10.2-91.6 分钟 | 23.8 分钟 |
| TAAC-25 | 17,884/17,885 | 49.8-159.8 分钟 | 76.8 分钟 |

这不是不同硬件之间可直接套用的 SLA：作者使用 H20，具体模型、存储带宽、`torch.compile` 首次编译、FlexAttention 内核、GPU 代际都会改变时间。它的用途是排期：在接近作者的 4 卡 H20/80 GB 环境上，TAAC-25 的完整 15 模型比较是数十 GPU 小时级任务，不能按 Taobao 的分钟级时间估计。

## 7. 实际复现顺序

1. 先下载 `salmon1802/Taobao`，用 `RankMixer_Taobao_Action` 在单卡跑通。
2. 下载 `salmon1802/KuaiRand`，先保持 `max_len=100`，验证 blocked loader 和 action mask。
3. 再做 QK、MerRec；确认磁盘读取没有成为 GPU 空转瓶颈。
4. 最后运行 TAAC-25，并使用 4 卡 DDP。
5. 只有在主表 `max_len=100` 的结果稳定后，再跑 KuaiRand 的 1000+ 长度 scaling；此时需专门调整 batch、accumulation、checkpointing 和 attention window。

