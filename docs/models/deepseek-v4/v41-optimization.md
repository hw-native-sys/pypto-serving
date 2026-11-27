# DeepSeek V4.1-Flash 优化记录

本页集中记录 V4.1 的优化方向、采用决定和实测结果。后续实验按日期追加，注明所测版本、工作负载、正确性检查以及采用或放弃的依据。早期 CPU Engram 与常驻 decode 缓冲区的设计见[开发者记录](../../developer-guide/v41-engram-decode-optimization.md)。

## 2026-11-23 · 四卡 DSpark K5

本轮保留方向 1、4、5。Serving 负责 prefill 元数据与 RoPE 绑定。配套使用 Lib #1916 时，TP1 直接使用动态行数验证头。[Serving #355](https://github.com/hw-native-sys/pypto-serving/pull/355) · [Lib #1916](https://github.com/hw-native-sys/pypto-lib/pull/1916)

### 当前实现与 chunk 512 验证

Serving 通过 Lib 的标准入口 `l3_decode_fwd_dspark` 注册一张 recurrent decode 图。在 Lib #1916 中，TP1 默认且仅使用动态行数，无需环境开关。每个 rank 读取自己的 `attention_num_tokens`，按 16 行向上对齐后投影，再按有效行数采样。空 rank 保留一个占位采样行。K5 下，一个请求对应 6 个位置；同一 rank 上 1、2、3 个请求分别投影 16、16、32 行。Lib 独立入口在 TP>1 时取同一 TP 组内最大的有效行数，再按 16 行向上对齐，使各个词表分片使用相同的通信步长。采样仍使用各 rank 自己的有效行数。TP2 双卡验证中，两个 rank 的有效行数为 0、18 时，投影从固定 80 行缩至 32 行，有效输出与参考结果逐元素一致。Serving 的服务入口当前仅支持 TP1。下表使用改为唯一入口前的动态实现测得。

以下均为四卡 TP1/DP4/EP4、固定 K5、chunk 512 的请求中位数。方向 1、4 在各组均开启，所有对照使用相同卡组和权重／采样 ABI 适配器。串行比较复用了同配置已有标杆，各版本完整预热 16 题后再测这 16 题。并发比较各启动一次，将相同的 9 个请求同时提交，包含批次内等待时间。

| 负载 | 验证头 | TTFT (ms) | TPOT (ms/token) |
|---|---|---:|---:|
| 单请求 | 完整头 | 753.78 | 19.46 |
| 单请求 | 早期固定小头 | 748.41 | 19.12 |
| 单请求 | 动态头 | 751.11 | 19.57 |
| 9 请求并发 | 完整头 | 8283.98 | 59.43 |
| 9 请求并发 | 动态头 | 8356.03 | 56.93 |

动态头的单请求 TPOT 比早期固定小头高 0.45 ms/token；与完整头的逐题配对差值区间跨零。9 请求并发时比完整头低 2.50 ms/token（4.2%），TTFT 高 72.05 ms。单请求 16/16、并发 9/9 的输入、输出 token IDs、停止原因和逐轮草稿计数分别与同负载的完整头一致。并发实测覆盖每 rank 最多 3 个请求。另一次 12 请求实验在 prefill 回放起点校验处失败，因此只报告完成的 9 请求对照。

### 五个方向与取舍（早期固定头实验）

下表是 **chunk 128** 的独立实验。每行给出该实验基线与候选的请求中位数，单位依次为 ms 和 ms/token。第 1～4 项的 Serving 共同基线是 `37578bb`，第 5 项在同一版本上切换完整头与小头。

| 方向 | 实现与决定 | TTFT 前 → 后 | TPOT 前 → 后 | 依据 |
|---|---|---:|---:|---|
| 1 · Prefill 滑动窗口地址 | **保留**。一次生成 cache 地址序列，让相邻 query 复用重叠的因果窗口。 | 970.05 → 957.38 | 26.27 → 26.11 | 地址准备累计耗时从 46.20 降至 9.86 ms。逐题配对的 TTFT 中位数快 31.42 ms，16 题中 14 题更快。288 组 CPU 元数据逐元素一致。 |
| 2 · Prefill 压缩 KV 页地址 | **舍弃**。实验将 C1A/C2A 写地址和 RoPE 位置改为张量批量生成。 | 1001.26 → 999.02 | 26.13 → 26.61 | 地址准备从 8.95 降至 3.14 ms，576 组元数据一致，但端到端 TTFT、TPOT 的配对区间均跨零。局部节省不足以支持合入。 |
| 3 · Prefill 复用 packed embedding | **舍弃**。实验让相同拓扑的元数据准备和输入上传共享一次打包。 | 996.39 → 994.05 | 26.05 → 26.88 | begin_step 的逐题配对中位数快 13.25 ms，但 TTFT、TPOT 的配对区间均跨零，未观察到稳定的整网收益。 |
| 4 · Decode 复用 drafter RoPE 表 | **保留**。当 cos、sin 都是启动时上传的原始张量，直接绑定常驻设备表。自定义表沿用原有路径。 | 1026.73 → 1020.91 | 27.13 → 26.48 | 避免重复传入相同表，保留自定义输入语义。配对 TPOT 差值为 −0.15 ms/token，95% 区间为 −1.31 至 +1.05，独立收益仍有不确定性。 |
| 5 · 单请求 K5 小验证头 | **早期方案，已改用动态头**。当时由 Serving 在完整图和固定小图之间选择。 | 1038.18 → 1085.35 | 31.40 → 28.20 | 合并样本的 TPOT 较低，但 ABBA 两对比较方向相反，TTFT 中位数升高。额外图增加启动和常驻开销，因此默认关闭。 |

早期固定头实验使用 `PYPTO_V41_DSPARK_HEAD_BUCKET=1` 和当时的 Lib。TP1、K5 下，所有 DP rank 的第 1 行及以后均无活跃请求时可选小图，包含空 rank；第 0 行可引用任意有效 KV slot。任一 rank 的后续行活跃时，所有 rank 使用完整图。本次容量下，每 rank 的小图投影 16 个填充后的行、采样 6 行，完整图分别为 80 行和 72 行。

另一次 **1+2+3** 的 prefill 组合实验测得 TTFT 1027.56 → 967.90 ms。它使用完整 decode 验证头，不能用于估计保留组合 **1+4+5** 的收益。

### 早期固定小头组合的 chunk 512 整网结果

四张 Ascend 950DT，TP1/DP4/EP4，单并发，固定 DSpark K5，greedy。两轮使用同一卡组，每轮独立启动，预热第一题，再运行相同的 16 道 GSM8K 题。输入 603～684 tokens，输出上限 256，关闭 prefix caching，Torch intra-op 为 1。TTFT 从请求开始计到首个非空流式输出，每题 TPOT =（末 token 到达时间 − 首 token 到达时间）/（输出 token 数 − 1）。下表均为正式请求中位数。

| 版本与轮次 | 请求数 | TTFT (ms) | TPOT (ms/token) | 接受草稿 / 提出草稿 |
|---|---:|---:|---:|---:|
| 1+4+5 第一轮 | 16 | 717.77 | 19.28 | 1050 / 2235 |
| 1+4+5 第二轮 | 16 | 737.02 | 19.35 | 1050 / 2235 |
| 1+4+5 合并 | 32 | 730.84 | 19.31 | 2100 / 4470（46.98%） |
| 官方历史基线，Serving `37578bb`、Lib `eac6fb1` | 32 | 760.46 | 19.61 | 2080 / 4510（46.12%） |

组合实测取自压缩提交前的 Serving `c75ce52` 和 Lib `00deb150` 的验证头图。Prefill 与 bootstrap 使用 Serving 锁定的 Lib 图，测试适配器补齐验证头所需的 FP8 权重、scale 和采样输入。两轮均确认选中小头，逐题输出 token IDs、停止原因和逐轮草稿接受序列一致，每轮输出 1493 tokens。

官方基线使用同节点的另一组四卡、完整预热 16 题，直接使用锁定的 Lib。表中两组测量的卡组、预热、ABI 适配和草稿接受率均有差异；要量化 1+4+5 的净收益，仍需在相同卡组、预热和适配器下做配对对照。


## 2026-11-24 · 四卡串行 prefill 与 decode

先合入 [Serving #355](https://github.com/hw-native-sys/pypto-serving/pull/355)，再审阅 [Serving #364](https://github.com/hw-native-sys/pypto-serving/pull/364) 与 [Lib #1951](https://github.com/hw-native-sys/pypto-lib/pull/1951)。下表用 PR 提交定位移植代码，用实验版本定位测量。两个 PR 独立跟踪；Serving 的 Lib gitlink 未随实验切换。

四张 Ascend 950DT，同一卡组，TP1/DP4/EP4，DSpark K5，chunk 512，并发 1。每个正式进程预热同一批 16 题，再测 16 题。TTFT 为请求至首个非空输出的墙钟时间，TPOT 为每题首末输出的时间差除以输出 token 数减一，再取请求中位数。以下数据来自移植前的实验。各行有不同的直接父版本，不能相加；当前主线移植及八卡满载、P/D 分离需要分别验证。

| 方向 | PR 提交 | 实验记录与取舍 |
|---|---|---|
| Prefill MoE 分块 | [Serving 5b4d1d4](https://github.com/hashiqiqixian/pypto-serving/commit/5b4d1d464079a97940fcd614ad366e2d15f61103) | 请求 chunk 固定 512，内部从 80 增至 256 行时，TTFT 714.35 → 430.30 ms；小 decode 图下，256 → 512 行为 431.75 → 380.99 ms。图内循环分别为 7、2、1 次。 |
| 每 DP 分区的 decode 容量 | [Serving a39b997](https://github.com/hashiqiqixian/pypto-serving/commit/a39b9978adf3e597ee9ea1d893b366ab965f5417) | 实验 83934ee。每分区最大请求数 12 → 1，对齐后的 80 行缩至 16 行。相同 prefill 下 TPOT 18.00 → 16.23 ms/token，16/16 题变快。设置 PYPTO_V41_DECODE_BATCH_PER_DP=1 复现单请求配置；默认 12，满载时按目标并发选容量。 |
| 跳过完整 prefill 不使用的分段输入上传 | [Serving 6fff69d](https://github.com/hashiqiqixian/pypto-serving/commit/6fff69dec1040b4da3e9500d4148c76e3da563ec) | 实验 0e3eb6b，TTFT 323.45 ms，TPOT 16.160 ms/token。完整图直接使用已绑定的 embedding，仍保留输出行映射。 |
| 大图处理满块，192 行图处理短尾 | [Serving 2a9ecda](https://github.com/hashiqiqixian/pypto-serving/commit/2a9ecda70e3ae3ccc8b1fcac64ad2968edf2ed61) | 选图实验 7ed3e7e，组合 7408a80 使用 512/192 图。两次独立进程 TTFT 297.09 / 296.15 ms，TPOT 16.163 / 16.166 ms/token。组合还含其他主机侧改动，数字不能单独归因于短尾图。新增一张图及对应资源，按各分区实际行数的最大值选图。 |
| 空闲 rank 跳过本地注意力 | [Lib c00891fd](https://github.com/hashiqiqixian/pypto-lib/commit/c00891fd8c8ad08f6fbf49891c3e924bc7518602) | 实验 a8f6654。保留每层专家通信。相同快速 prefill 下逐题 TPOT 配对差中位数为 −0.392 / −0.331 ms/token，两轮均 16/16 更快；组合 TPOT 为 13.357 / 13.423。满载各 rank 都有工作时没有空闲分支收益。 |
| 16 行专家输出融合 HC 残差 | [已从 PR 撤回 · 历史 C++ 实验 eca5aff5](https://github.com/hashiqiqixian/pypto-lib/commit/eca5aff536a2eca50d62038e9190d2ffd4c88f5e) | 实验 f4c4ce6。接在空闲 rank 优化之后，两轮 TTFT 301.17 / 297.02 ms，TPOT 13.0215 / 12.7914 ms/token。该方案需要修改专家 kernel，已撤回。 |
| 16 行目标 decode 的逆 RoPE 量化融合 O-A | [已从 PR 撤回 · 历史 C++ 实验 e21e25a3](https://github.com/hashiqiqixian/pypto-lib/commit/e21e25a3b3e6cf1e64bf8e0cb293c125c029fa43) | 实验 9322d542。叠加 HC 残差后单轮 TTFT 296.46 ms，TPOT 12.4218 ms/token。16/16 题的输出、停止原因和逐轮草稿计数均与固定对照一致。该最佳轮尚无独立复测。此 C++ 实现已撤回；纯 DSL 实验另行验证。 |

Lib 主线已有专家计数的缓存行读取优化和融合专家后端。原实验 60f02e51 的同卡 ABBA 中，TPOT 19.78 → 18.36 ms/token，TTFT 754.54 → 773.34 ms。固定 BF16 投影的拆分/融合专家对照为 17.127 → 13.934 ms/token。两个历史结果分别记录来源，后续 PR 复用主线实现。[计数读取所在提交 #1928](https://github.com/hw-native-sys/pypto-lib/pull/1928) · [融合专家后端 #1819](https://github.com/hw-native-sys/pypto-lib/pull/1819)

同卡 DeepInfer 历史中位数为 298.95 ms TTFT、4.5629 ms/token TPOT。PyPTO 的 12.4218 是单请求、16 行图的最佳轮，不能用作八卡 AgentX 满载的性能预测。
