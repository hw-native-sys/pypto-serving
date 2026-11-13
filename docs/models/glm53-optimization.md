# GLM-5.3-Flash 优化计划与进展

## 优化分支提审（2026-11-05）

- Lib 提交为 [PR #1710](https://github.com/hw-native-sys/pypto-lib/pull/1710)，
  分支 `feat/next_glm53_optimize`，提交 `a6f6e3a`，基于主线 `eb76b3b`，整合为一笔提交。
  包含 Indexer 有效范围和短历史快速路径、BF16 Linear K128、MLA 列 tile128、
  权重头信息预检，以及原 FP32 KDA decode 的 packed 请求与缓存状态适配。
- Serving 继续使用 [PR #286](https://github.com/hw-native-sys/pypto-serving/pull/286)。
  优化分支的运行代码与该 PR 更新前的 `b1138f4` 完全相同，额外提交优化报告、
  文档入口和上述 Lib 依赖，不另开重复的 Serving PR。
- 保留最新主线的 KDA 逐行投影和稀疏注意力空块跳过，未覆盖回旧实现。
  Lib #1710 包含隔离 [PR #1702](https://github.com/hw-native-sys/pypto-lib/pull/1702)
  的 FP32 decode 适配，两者存在重叠；合入时应协调，不能视作两项独立收益。
- 同步后的源码已通过 TP8／TP16 完整 prefill／decode 共4项代码生成、
  28项 KDA NPU 用例、8项 MLA 投影／吸收／输出投影 NPU 用例、
  177项 golden runner 单测和41项 Serving GLM CPU 单测。
  NPU 组件验证使用单卡上的对应 head 分片几何，不代表多卡整网验收。
- 本次主线同步后尚未重新采集组合版本的整网吞吐；下文性能数字仍对应各自冻结版本。
  隔离 #1702 在197的基线与候选8卡运行均已结束，prefix 恢复和交错请求的一致性检查通过，
  **独立 CPU 整网精度对照尚未通过**：新组合的 CPU 参考与 NPU 基线在 prefill logits 上
  相对 L2 差约65.78%，需先核对整网参考组合与输入契约。该差异在尚未切换 decode 的
  基线已经存在，当前不能归因于本次 decode 替换，也不能据此标记整网精度验收完成。

## 接入原有 FP32 KDA decode（2026-11-05，验收中）

代码已单独提交为 [Lib PR #1702](https://github.com/hw-native-sys/pypto-lib/pull/1702)（待评审），
基于主线 `fbdf6de`，隔离 PR 当前提交 `d9a7c00`。下文历史优化分支测量对应 `d2658fe`，
改名前的整网实测版本为 `af0d053`。共享函数已统一命名为 `dense_layer`／`kda_attention`，
同步修改全部导入与inline调用别名；归一化符号后的AST完全一致，改名后4项完整入口代码生成通过。
PR分支重新通过18个NPU核心用例、4个请求数／初态／多token兼容用例和TP8／TP16的4项完整入口代码生成，
完整pre-commit通过。下文整网吞吐测于隔离PR之前的优化分支，尚未在PR分支复测；
相同历史的完整logits及整网prefix／混合请求验收仍待完成。

沿用已有 `decode_kda`，仅适配 packed 请求与 seed／working／checkpoint 状态元数据，
不新增另一套递推函数。完整 decode 显式选择该 FP32 递推；prefill 保留原 chunk 路径，
单 token 的 prompt 尾块也继续走 prefill。输入和输出仍为 BF16，递推状态仍为 FP32；
中间舍入按各自的 golden 验证，两条路径不宣称逐元素一致。Serving 的计算代码和 ABI9 不变。

已完成：单 NPU 上 TP8／TP16 head 分片几何的18个边界用例、20步连续状态验证、
TP8／TP16 完整 prefill 和 decode 的4项代码生成，以及108项 Serving 相关 CPU 测试。
连续状态验证覆盖请求暂停恢复、共享初态、checkpoint 发布与恢复，NPU 与独立历史 golden
分别连续携带状态；数学参考与冻结的原 FP32 golden 完全相同。逐点阈值保持 rtol=atol=0.02，
只读／未用状态与输出填充另外要求完全不变。

下表为本机 A3 的单 NPU 核心对照，不包含投影、卷积、通信或整网。相同物理16行 query、
51行 state，原地 working 且不发布 checkpoint，每轮恢复相同初态和输出；恢复开销包含在两侧。
各5次预热／100次正式，取 `[RUN] effective_us` 的中位数。两侧分别通过对应精度契约的 golden。

| Head 分片几何 | 活跃请求 | chunk 基线 μs | 原有 FP32 decode 适配后 μs | 耗时下降 |
|---|---:|---:|---:|---:|
| TP8 | 1 | 877.6 | 453.2 | 48.36% |
| TP8 | 4 | 903.3 | 483.5 | 46.47% |
| TP8 | 16 | 962.1 | 555.6 | 42.25% |
| TP16 | 1 | 461.1 | 238.4 | 48.30% |
| TP16 | 4 | 475.5 | 255.2 | 46.33% |
| TP16 | 16 | 501.8 | 281.6 | 43.88% |

工具链为 PyPTO `ff2e0a4`、runtime `31f9e72`、ISA `f83d4b7`、PTOAS0.72、CANN9.0.0。
证据为 `build_output/glm53-next-optimize/fp32-decode-integration-20261105/`。
旧的紧凑有效行测试与当前适配后的固定16请求槽有不同调度开销，不能直接沿用其222.780μs等数字。
本机8卡整网任务已完成：基线与候选依次使用同一组设备，TP8／EP8、context512、
chunk32、预算128，固定160输入／32输出，每档1组预热与3组正式，关闭 prefix 和 MTP。
下面是直接调用 `Glm53ModelRunner` 的端到端输出吞吐，包含输入准备、下发、读回与采样，
排除初始化、请求重置和落盘；它不是 HTTP 指标，不能直接对照历史 HTTP 表。

| 并发 | chunk 基线 token/s | FP32 decode 候选 token/s | 吞吐提升 | 单步 decode 中位耗时 |
|---|---:|---:|---:|---:|
| 1 | 5.440 | 5.899 | 8.45% | 142.14 → 126.73 ms |
| 4 | 19.270 | 20.967 | 8.81% | 152.53 → 136.20 ms |

两侧简单问答均得到“5”和“巴黎”，运行与清理成功。固定4条长输出样本中，首次 token
分歧分别出现在第32、4、17个输出 token；第4条32个 token 完全一致。两侧输出均为连贯续写，
这不替代精度验收。下步先在相同 token 历史下对照完整 logits 与对应 FP32 golden，
再完成组合 KDA、prefix 恢复与混合请求回归；**该切换仍未完成最终验收**。
按已授权的逐项优化流程，验收后独立提交，再以新时间线选择后续热点；
优先重测16行 decode 的 KDA 投影合并，并对每项单独采集整网前后性能。

## 撤回 KDA 混合精度专用 decode（2026-11-05）

**该候选不再保留在当前优化代码中，独立 [PR #1694](https://github.com/hw-native-sys/pypto-lib/pull/1694) 撤回。**
移除 `decode_kda_mixed_precision` 及其生产分支和专属验证入口，生产 KDA 恢复使用既有
`prefill_kda` chunk 路径。原有 FP32 `decode_kda`、其 golden 和其他优化不变。
没有将不同舍入规则的 FP32 路径直接替换进整网。

补测显示此前只比较 chunk 基线不够完整。相同 A3 单个逻辑 NPU、16 query 行、51 state 行，
使用双方都支持的原地状态更新并关闭 checkpoint；旧函数只处理有效 token 的借用视图，
新函数按16请求槽处理相同有效 token。每轮恢复相同初态及输出填充，设备耗时包含相同恢复开销。
每项5次预热、100次正式，取设备耗时中位数；无投影、卷积、TP通信、Serving或整网。

| Head 分片几何 | 活跃请求 | 原 FP32 decode μs | 新 mixed decode μs | 新函数耗时变化 |
|---|---:|---:|---:|---:|
| TP8 | 1 | 222.780 | 471.110 | +111.47% |
| TP8 | 4 | 284.840 | 485.690 | +70.51% |
| TP8 | 16 | 559.220 | 539.570 | -3.51% |
| TP16 | 1 | 136.410 | 257.020 | +88.42% |
| TP16 | 4 | 167.320 | 266.360 | +59.19% |
| TP16 | 16 | 303.219 | 281.430 | -7.19% |

两条路径分别通过自己的未修改 golden，但输出及状态不逐元素一致，不能将性能表当作相同精度契约的
替换验收。工具链为 PyPTO `ff2e0a4`、runtime `31f9e72`、PTOAS0.72、CANN9.0.0。
证据为 `build_output/glm53-next-optimize/kda-decode-withdrawal-20261105/DECODE_COMPARISON.json`。
下文11月3日、11月4日的专用 decode 实现、验收和测量均作为历史记录保留，不再表示当前代码启用它。

撤回后补测：原有decode的single／batch及分组chunk的paged共3项NPU回归通过；
TP8／TP16完整prefill、decode共4项代码生成通过，两仓pre-commit通过。
原有KDA计算／golden函数及投影tile的AST核对一致。本轮没有整网吞吐复测。



## 主线同步（2026-11-04）

本轮同步到 Lib `d02a6bb`、Serving `61b350e`，并合入现有 GLM 接入分支的 ABI9 适配。
主线将 prefill、decode 各自改为一个完整45层入口，decode 包含设备 Embedding，
两个入口均包含 NPU LM-head。原分阶段 HOST backbone、独立阶段入口和额外的 generation
封装已被完整入口替代。KDA 状态使用分组缓存的 seed／working／checkpoint 三类行；
专用单 token decode 已迁移到该契约，保留 BF16 中间边界。

继续保留 Linear K128、attention 列 tile128／无效块跳过、Indexer pool 事件范围收缩、
短窗口免 query／score 计算，以及权重头信息和 dtype 预检。
原手动 slab 上传和去重 logits 回读依赖同步执行，不能在有未完成 forward 时调用设备控制接口；
此次采用新接入分支的双缓冲 HOST_SHARED 参数和 runtime 有序拷贝，避免异步下发冲突。
prefix cache 使用公共 grouped cache manager，移除旧双 bank snapshot 协议及其配置。
默认开启前缀缓存；命中必须短于 prompt，按新接入分支的方式计算剩余尾部。
旧版精确整段命中的 logits 直接复用不再启用，对应历史零 query 结果不代表当前行为。

**下面的性能数字均为对应冻结版本的历史结果，不能当作 ABI9 合并后版本的吞吐。**
本轮验证完成：

- TP8／TP16 的完整 prefill 与 decode 共4个 L3 入口代码生成通过，ABI 均为9。
- 同一197 NPU上采用 TP8／TP16 head 分片几何，各8个新旧 KDA 核心对照通过：
  覆盖槽15、空洞请求、全16请求、全空、零 q／k、beta 边界、原地状态和共享 seed；
  完整输出与51行状态逐元素相等，未用行、只读 seed 和输出填充未改变。
  这是单 NPU 上的两种分片几何验证，不是 TP8／TP16 多卡通信实测。
- 模拟器通过相同16个核心对照，以及分组缓存的 paged prefill、paged convolution、
  公共 decode CLI 的 packed／packed-empty，共20例。
- 两种拓扑的 CPU 权重加载自检通过，golden runner 单测177项通过。
  Serving 单测首轮1064项通过；另外4项因嵌套 Lib 未初始化而找不到 ABI 源文件，
  临时链接现有 Lib worktree 后4项全部通过，链接随后删除，未复制仓库。
- 两仓完整 pre-commit 通过。本轮未重新运行整网吐字、prefix cache HTTP 验收或吞吐测试。

证据位于 `build_output/glm53-next-optimize/mainline-sync-20261104T080417Z/`；
保留源码 hash、编译元数据、NPU／SIM 对照与原始检查日志。MTP 不属于本次同步改动。

本文件统一记录 GLM-5.3-Flash 在 PTO3 和 pypto-serving 中的优化计划、验证进度、实测性能和最终取舍。后续每完成一项，更新这里的状态、数据和结论；运行命令见[模型使用说明](glm53.md)。

更新日期：**2026-11-05，11月5日记录使用北京时间；更早记录以 UTC 为准，各章节另有注明的除外**。文中 `<artifact-root>` 表示测试机器上的实验产物父目录，具体本地路径保存在未提交的运行记录中。

**提交历史整理：**Lib 原 `993be25` 之后的13笔、Serving 原 `cdfc912` 之后的26笔，以及本轮工作区改动，在两个仓各整合为一笔后续提交。Lib 随后在最新主干上重放，保留上游 Indexer 独立维度修复，并补回原 PR 的公开计时字段。代码已推送 private 仓；[Lib #1553](https://github.com/hw-native-sys/pypto-lib/pull/1553) 已以 `eacc677` 合入 upstream/main，Serving 接入对应 [PR #286](https://github.com/hw-native-sys/pypto-serving/pull/286)。下文保留测量时的原始提交编号，对应历史快照和验证产物；旧历史及补丁已本地备份，已采纳数量、数值门槛和既有性能结论不变。

同步主干后的补测：本机任务 `task_20261026_061431_18436767974` 的3个 Indexer score NPU 用例通过，覆盖普通、部分尾块及稀疏请求段；8项产物重载单测和6项受控时钟检查通过。原 TP8／TP16 整网缓存验收仍对应其冻结源码；该次补测未重跑整网或重新采集性能。计时修复仅影响主机时间戳及报告字段，29个 kernel AST 保持不变。

**2026-10-30 主线同步说明：**优化 worktree 的 Lib 已同步到公开主线 `5e7430d`；Serving 以 `7e53f3f` 合并当前主线 `fbfa125` 和已有 GLM 接入 `6c7df9a`。上游已整合 KDA／MLA 文件并将运行时放回 Serving，主线新增的 MTP／视觉实现一并保留。原 K128／列 tile 修改迁到合并后的算子，权重头信息预检迁到 CPU 加载代码，输入缓冲合并迁到 Serving runner；原未提交内容已备份，仍作为优化分支的工作区改动保留。本轮同步后已完成组件NPU对照、TP8／TP16自然EOS吐字检查及TP8短输出基线；缓存生命周期与设备时间线继续验收。**下文历史吞吐、指纹和21项试验计数仍对应各自冻结版本，不能作为这次新代码组合的性能或精度结果。**

## KDA 专用 decode：保留 BF16 中间边界（2026-11-03）

**实现、TP8 正确性及 TP16 完整缓存回归已完成，核心设备计时显示明显下降，整网性能尚未验收。**修改位于 Lib 的 `feat/next_glm53_optimize` worktree，计算代码仅涉及本轮的 `kda.py`、`prefill_layer.py`。Serving 没有为此新增计算实现或配置开关。下方11月2日的组件性能及更早的优化表均保留原冻结版本口径，不能作为本项修改后的测量。

### 做了什么

新增 `decode_kda_mixed_precision`：当所有非空请求各有一个 query token 时，`kda_attention_sublayer` 选择该核心；多 token prefill、混合 prefill/decode 及全空批次保留原 chunk 核心。按 `query_start_loc` 找打包 token，按请求索引读取 `kda_state_rows`，支持空洞请求、高编号槽及置换后的状态行。

单 token 的严格下三角 Gram 矩阵为零，因此原逆矩阵为单位阵，可以省去初始化、四轮求逆及单位阵乘法。核心由原 chunk 的9组 SPMD 阶段缩减到准备、状态更新2组。**这里仅指递推核心；投影、卷积、输出投影、TP归约及外层87个L2主干调用没有因此合并。**

新核心保留 chunk 在归一化、q／k 衰减、beta、状态矩阵乘法、残差和最终输出处的 BF16 边界，以及 FP32 递推状态。Cube 的填充及归约尺寸也保留原语义。既有独立 `decode_kda` 仍为 FP32 递推参考路径，未替换或改变；所有9个既有 `golden_*` 函数 AST 不变，没有放宽比较门槛。当前仓中的 DeepSeek V4 DSpark、MTP、V4.1 实现均没有 KDA，不能直接从其 attention 搬来 KDA 递推。

### 已完成验证

- TP8／TP16 的 Dense、KDA＋MoE、HOST backbone 短查询及 KDA＋MoE 原容量入口共8组 codegen 通过；CPU Serving 单测39项通过。
- 递推核心 TP8／TP16 分片几何各6个 NPU 用例，共12项通过。覆盖槽15、空洞、全16请求、全空、零q／k及beta边界；使用非零初始状态、置换状态行和未使用区域哨兵。与原 mixed-precision golden 比较通过，且新旧 NPU 核心的完整输出和状态**逐元素相等**。该项是单NPU上的两种分片几何，不冒充多卡通信验证。
- 完整投影→卷积→递推→输出投影组合 TP8／TP16 几何各5例，共10项 NPU 对照通过，包含17＋1的混合prefill和全空。仍使用原golden及 `rtol=atol=0.02`，未用状态及填充区不变。
- TP8／TP16各1例NPU分支证明通过：将仅chunk路径使用的三角矩阵／单位阵输入填为NaN，新单token路径仍通过独立golden，证明实际选中了专用核心；该诊断不修改模型常量或比较器。
- TP16／EP16完整缓存回归16个HTTP用例及断开连接检查全部通过；精确128命中首步零query、部分160命中恢复128后计算32，两对8步完整FP32向量历史字节hash一致；4次真实native restore、实际多请求混批、驱逐、取消后复用、中英文自然EOS及退出清理通过，保留179行采样记录。
- TP8／EP8 完整 Serving 新旧各跑同一组160输入／128输出，C1、C4各1组预热和3组正式。20条响应的文本、用量及结束原因均一致；保存的前16次真实forward完整FP32词表logits逐字节一致。后续forward未全部保存logits，不能外推为整段逐步logits均逐位相同。两版中英文自然EOS检查均通过。

### 首次整网性能：包含回退，不作提速结论

同一197 A3、TP8／EP8、DP1、BF16主KV、NPU完整FP32 LM-head、chunk32／预算128、ctx2048／16槽，关闭prefix cache和MTP；无设备Profiler，启动和编译不计入吞吐。两次独立服务依次运行，保留全部样本：

| 并发 | 本轮原 chunk 控制 token/s | 专用 decode token/s | 变化 |
|---:|---:|---:|---:|
| 1 | 4.296742 | 4.323752 | +0.63% |
| 4 | 16.501807 | 15.040388 | **-8.86%** |

C4纯decode的三组forward中位耗时，控制版为214.019／212.466／213.494 ms，候选为214.939／217.034／259.040 ms。候选第三组同时出现backbone提交、seed／head提交和logits读回变慢；这是进一步排查的线索，**尚不足以认定为机器噪声或排除核心回退**。不删除该轮，不与历史4.253／16.111的基线混算收益。随后尝试在同一进程、同一权重和持久Worker中按旧／新／新／旧交替测C4；每个真实token仅执行一条路径。该诊断在首次切到旧版control backbone的forward13发生 `SCHEDULER_TIMEOUT S1:running-stalled`，未完成任何正式组。已终止本实验的measurement子进程，由controller关闭服务；源码不变、服务退出码0、全部NPU恢复空闲。保留 `AB_ABORT.json`、`ab-http-tp8/server.log` 及失败report；根因未确认，不能把这一诊断失败当作新核心的吞吐，也不能据此将首轮回退判为机器噪声。

### 补充：递推核心设备耗时

同一197、单NPU分别采用TP8／TP16的head分片几何，随机固定输入、非零初态、置换状态行；C1使用槽15，C4使用槽0／3／7／15。每次dispatch先从只读输入恢复全部状态及输出填充，再执行原chunk或新核心，避免重复推进状态。每项5次预热、100次正式，使用原生STRACE的L2 device_wall中位数，**包含相同的状态／输出恢复开销**；不含投影、卷积、输出投影和TP通信，不代表完整KDA或整网。新旧输出和状态逐元素一致，原golden及0.02门槛不变。

| 分片几何 | 活跃请求 | 原chunk入口 μs | 专用decode入口 μs | 耗时下降 |
|---|---:|---:|---:|---:|
| TP8 | 1 | 797.370 | 318.920 | 60.00% |
| TP8 | 4 | 819.730 | 332.390 | 59.45% |
| TP16 | 1 | 435.560 | 186.120 | 57.27% |
| TP16 | 4 | 445.880 | 190.540 | 57.27% |

上述结果证明含恢复的核心入口更快；**不能替代前面的整网性能验收，当前不记录新增整网提速百分比**。另做完全相同输入的分支开销隔离：TP8几何C1直接核心319.210 μs、带生产query长度选择逻辑322.130 μs；C4为329.220／334.000 μs，分别增加约2.92／4.78 μs，输出及状态仍逐元素一致。该数据不支持将整网回退归因于query长度检查，未据此扩改生产代码。后续需在完整生产KDA L2与HOST调度的关键路径上继续归因，并用已验证的独立服务方式复测C4。证据为 `core-bench-tp8/report.json`、`core-bench-tp16/report.json`、`core-bench-controller-report.json`，均保留100次设备样本及host样本。初版模拟诊断误用了legacy platform参数、在本机设备初始化失败；改为显式backend／execution_mode后4例模拟检查通过，未将失败尝试计入设备样本。

本轮冻结源码、原始数据及脚本统一保存在 `build_output/glm53-next-optimize/kda-decode-20261103T112000Z/`。197同名实验目录为 `<artifact-root>/glm53-kda-decode-20261103T112000Z`，主要结果为 `NPU_COMPONENT_GATES.json`、`MATCHED_HTTP_AUDIT.json`、`baseline-http-tp8/`、`candidate-http-tp8/` 及核心／selector计时报告。本地 `evidence-197/` 解出结构化结果，设备采样原始日志另留在 `performance-evidence.tar.gz`。临时诊断脚本不加入受版本管理的测试文件。

## 真实整网组件性能对照：PTO3 与 vLLM Ascend（2026-11-02）

**本轮完成了同机TP16／EP16的MoE、KDA、DSA设备计时。C4时，PTO3最快rank的单层耗时分别约为vLLM的4.70、10.76、4.33倍；此外，rank0的主干组件间空隙中位数约212 ms。差距同时包含组件实现及调度，不能全部归因于MatMul。**本轮为带采样的诊断，原TP8生产吞吐基线保持原口径。

### 实际负载与统计口径

- 同一197 A3：16个逻辑NPU，即8张双芯板卡；TP16、EP16、DP1。
- 同一套官网下载的W8A8权重、同一组160-token文本，每请求生成16个token，温度0、忽略EOS；C1、C4各1组预热、1组正式。
- 两边上下文上限2048、最多16请求、chunk32、token预算128；MTP、prefix cache关闭，主KV为BF16。vLLM保留FULL_DECODE_ONLY入图及shared expert侧流重叠。
- 取正式组中全部请求完成prefill后的前5个真实decode步。PTO3按query／context元数据筛选；vLLM按静态图、实际shape及每请求16个输出的上界核对全批次。没有为计时额外推进同一token的模型状态。
- 覆盖42个MoE层及其31个KDA、11个DSA attention。前三个Dense层的KDA＋MLP单列，不混入KDA或MoE单项。
- 每步先对同类层取平均，再对5步取中位，主表取16个rank中的最小值，同时保留最慢rank。**单位ms/层，不是token/s或整网单步时间。**
- hidden、KDA状态、KV和专家路由由各自整网自然产生；同prompt不代表两边中间张量和专家分配逐位相同。

### Decode单层设备耗时

| 并发 | 组件 | PTO3 ms/层 | vLLM ms/层 | PTO3／vLLM |
|---:|---|---:|---:|---:|
| 1 | KDA | 1.911 | 0.143 | 13.41倍 |
| 1 | DSA | 1.451 | 0.301 | 4.82倍 |
| 1 | MoE | 2.192 | 0.319 | 6.87倍 |
| 4 | KDA | 1.968 | 0.183 | 10.76倍 |
| 4 | DSA | 1.477 | 0.341 | 4.33倍 |
| 4 | MoE | 1.659 | 0.353 | 4.70倍 |

范围为mHC前处理、RMSNorm、完整attention／MoE及其mHC后处理。KDA包含投影、卷积／状态计算及输出TP归约；DSA包含MLA、cache、Indexer、稀疏attention及输出归约；MoE包含Gate、shared／routed expert、EP dispatch／combine及输出恢复。

PTO3使用单次执行的原生STRACE device_wall，包含每个L2调用的设备端进入、编排和退出；vLLM使用真实图执行中HcPre开始至对应HcPost完成的设备时间，整网图统一启动不分摊到各组件。**计算范围对应，入口组织开销仍不同，不能将表中倍数理解为同一个底层kernel的倍数。**

C4的rank间范围如下，各值均为该rank的5步中位，单位ms/层：

| 组件 | PTO3最快～最慢rank | vLLM最快～最慢rank |
|---|---:|---:|
| KDA | 1.968～5.628 | 0.183～0.185 |
| DSA | 1.477～3.797 | 0.341～0.344 |
| MoE | 1.659～2.552 | 0.353～0.355 |

PTO3有明显rank差异和步间波动。早启动rank可能在TP通信中等待其他rank，需要继续区分启动偏斜、等待及实际计算；本轮不能直接认定为专家负载不均衡。orch／sched窗口并集也已单独导出，该effective窗口仍含编排和等待，不是纯AICore忙碌时间。

### 主干组件间空隙

下表取rank0，在同一设备时钟上量取第一层开始至最后一层完成。空隙为主干窗口扣除87个组件设备窗口的并集，没有跨rank相加。各列分别对5步取中位，不能直接相加。

| 并发 | PTO3主干窗口 ms | vLLM主干窗口 ms | PTO3组件间空隙 ms | vLLM组件间空隙 ms |
|---:|---:|---:|---:|---:|
| 1 | 566.99 | 22.06 | 238.15 | 0.10 |
| 4 | 573.60 | 25.34 | 211.83 | 0.10 |

窗口只包含45层主干，不含Embedding、最终LM-head及HTTP。HOST入口虽已减少上层L3调用，**每rank每步仍有87个L2主干调用，16个rank共1392次**。参数准备、下发、等待及其运行时路径需要继续细分；时间线尚不能证明某个具体函数解释了全部空隙。

### 实现差异与结论边界

1. PTO3纯decode仍按CAP16组织工作；vLLM图的query行数为C1=1、C4=4，真实请求相同、填充工作量不同。
2. 主KV均为BF16；Indexer KPool仍为PTO3的INT8＋FP32 scale、vLLM的BF16。当前上下文约160～165，PTO3走已有短窗口免评分分支，不能据此推断超过512个pool时的长上下文DSA性能。
3. PTO3 shared保持不做TP的既定方案；vLLM本轮shared按TP16分片。实际task显示shared局部中间维度128（2048／16），routed为每rank18个完整expert。MoE比较包含并行组织差异。
4. PTO3生产KDA走chunk，vLLM走FlaNPU recurrent。已有单步候选曾出现生成序列差异，后续应保持BF16边界和原有验收门槛，不直接替换。
5. PTO3沿用197已验证的PyPTO／runtime／ISA组合，vLLM使用已验证的0.30.0rc2 A3镜像；CANN分别为9.0.0和9.2.0。本轮是实际部署实现比较，工具链也有差异。
6. Prefill保留了独立原始样本，但包含动态图下发、混合批次和Profiler影响。本节主表限定为上述decode步；prefill样本及本轮HTTP吞吐不作为正式无采样TTFT或生产性能目标。样本量有限，不能宣称稳定跨负载收益。

### 后续方向与证据

先细分L2之间的参数／下发／等待和rank启动偏斜，再研究C1／C4查询容量收缩、保持数值契约的KDA decode、MoE grouped matmul／量化epilog及通信重叠。当前HOST入口继续保留；此前设备常驻循环的依赖超时仍未解决。本轮没有实施新的模型优化，也没有改变shared并行契约。

本地证据统一在 build_output/glm53-next-optimize/operator-comparison-20261102T025500Z/：

- comparison.json、component-comparison.csv：协议、最快／最慢rank及比较结果。
- pto-tp16-r3/ 下的 operator-report.json、operator-samples.csv、forward-timing-audit.json、backbone-device-samples.json：每次调用、每层、每rank及窗口明细。
- vllm-r2/ 下同名统计及 representative-device-tasks.json：32份profile的导出及真实任务样例；197保留原始采样和Native日志。
- PTO3的17份日志共3,465,386条STRACE，记录头与解析数量一致，无截断及丢失告警；43个正式组forward均完整匹配16个rank、每rank87个主干调用。HTTP用量、源码身份及退出清理检查通过，PTO3中英文自然EOS检查通过。本轮不新增整网CPU-golden结论。
- 只纳入PTO3第三轮及vLLM正式profile的设备数据，排障早期轮次不混入。普通NPU Event在图重放中时间戳不更新的方法已拒绝；vLLM使用标准任务时间线和[官方离线解析API](https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/torch_npu-profiler/torch_npu-profiler-profiler-analyse.md)。

## 当前整体优化路线

本节按2026-10-30同步后的Lib `5e7430d`、Serving `7e53f3f`及保留的工作区改动制定。已取得组件、复制路径、双拓扑吐字及TP8基线；分块单变量试验完成，裁行候选的C1改善而C4基本持平，尚未采纳。**已采纳KDA＋MoE融合、Indexer pool写事件收缩、attention空块裁剪、短窗口免评分、DSA＋MoE融合、decode执行容量分离、HOST整网调度及完整logits读回去重。当前TP8、160输入／128输出为C1 4.253083、C4 16.110647、C8 28.400740、C16 42.725300 token/s；最近HOST调度单项+39.60%／+29.78%，读回去重单项+3.79%／+16.24%。C4首字中位数分别增加19.42%和10.66%，当前为3.822139秒；读回对照的控制版本有一轮偏慢，保留全部样本，不把聚合增幅当作稳定跨负载收益。**空块裁剪和免评分分别测得C1／C4 +1.75%／+2.60%和+2.22%／+1.23%，属于本次样本中的小幅改善，尚未证明跨负载稳定收益。KDA融合另有匹配短负载C1 +8.08%、C4 +7.34%；5G、容量分离、HOST调度及读回去重均已通过TP16完整缓存及自然EOS回归。早期FP32单步因生成序列变化未采纳；11月3日保留BF16边界的新核心已通过正确性及TP16缓存回归，核心计时下降57～60%、整网性能尚未验收，见本文首节。后续改动继续集中在两个仓的 `feat/next_glm53_optimize` worktree，每项独立验证和记录。本节为当前实施顺序；下方旧候选及性能表保留为历史，不直接沿用其实现状态或收益。

### 已有能力与本轮判断

同一197、TP8／EP8、160输入／128输出、chunk32／预算128，每档1组预热及3组正式的逐项结果如下。每一步均与直接父版本比较，C4为四请求合计吞吐；KDA融合本身的收益另有32输出对照，不能与本表混算。

| 保留版本 | C1 token/s | C4 token/s | 本项C1／C4变化 |
|---|---:|---:|---:|
| KDA＋MoE融合，后续各项的长输出起点 | 1.936322 | 7.496042 | 本表起点 |
| 再收缩pool写事件 | 2.103769 | 8.096075 | +8.65%／+8.00% |
| 再裁剪attention空块 | 2.140627 | 8.306750 | +1.75%／+2.60% |
| 再跳过短窗口无用评分 | 2.188225 | 8.408665 | +2.22%／+1.23% |
| 再融合DSA＋MoE | 2.322731 | 8.959438 | +6.15%／+6.55% |
| 再将纯decode查询容量降至16 | 2.935380 | 10.679046 | +26.38%／+19.19% |
| 再用HOST入口组织45层纯decode | 4.097659 | 13.859591 | +39.60%／+29.78% |
| 再去重完整logits读回，**11月2日保留版本** | **4.253083** | **16.110647** | **+3.79%／+16.24%** |

每项性能对应冻结源码和同口径HTTP对照。16行decode的TP16功能验收也已通过；尚无本轮TP16吞吐。早期FP32 KDA单步、5F及投影原型未混入上述版本；11月3日新核心另列于本文首节，不沿用本表吞吐。

- 主线的Serving权重加载已选择原生W8A8 MLA，并已有对应的INT8／scale程序参数；本轮TP8／TP16头信息预检、projection／output组件golden及双拓扑整网吐字检查已通过，TP8短输出性能已测。尚未取得本轮完整45层CPU golden的逐步logits对照。因此原opt48改为现有路径验收与归因，不再安排从零实现。
- 主线已合入Indexer TopK、gather和cache-write优化。先纳入新基线，再按实际时间线判断旧opt26等候选是否还有独立价值，不能重复开发或重复计算收益。
- KDA混合精度专用decode已于11月5日撤回；随后对原有FP32 `decode_kda` 适配分组元数据并接入完整decode。TP8整网初测吞吐C1／C4提升8.45%／8.81%，相同历史下的logits及缓存回归仍待验收，详见本文首节。前述历史计时不代表当前版本。
- MTP已有草稿头、缓存及独立生成入口；当前 `generation.py` 仍每生成一个token执行一次target forward，再逐token统计draft是否命中，尚未减少target调用次数。HTTP CLI的投机配置目前只对DeepSeek V4开放。GLM的MTP提速工作是批量target验证、接受点状态处理及Serving接入，不是新增一个开关或再写一套draft。
- 当前单实例最多16个请求槽。总32并发可先用两个独立TP8实例承接；单TP16扩到32需要同步验证缓存槽位、元数据、程序ABI、输出和内存，不能只放宽CLI上限。

### 实施顺序与收益判断

| 顺序 | 项目 | 实际要做什么 | 收益判断／当前状态 | 主要仓库 |
|---|---|---|---|---|
| 0，先验收 | **新主线NPU正确性、基线和时间线** | 固定工具链及权重，验收原生W8A8 MLA、迁移后的输入缓冲、TP8／TP16整网、缓存和多请求；采一次真实执行的Host／设备／通信时间线。 | CPU1198项、权重自测和头预检已过；197已完成6组组件NPU对照及两卡缓冲复制回归，TP8整网与短输出基线、TP16吐字检查已通过；TP16完整缓存回归及单次真实执行PMU／时间线已完成，后续结构改变后需重新剖析。原1.09秒/token及9.73～12.75倍差距只是旧版本参照。 | Lib＋Serving |
| 1，已完成分块试验 | **prefill分块与总预算分别扫描** | 固定编译容量和其他条件比较32／64／128分块，再单独研究更大总预算；检查对已有decode请求的影响。 | chunk64／128的C1短输出改善，C4分别退化9.06%／11.05%，保留32。总预算扫描待容量解耦，不与分块混测。 | Serving |
| 2，已采纳 | **prefill与decode执行容量解耦** | prefill／混批保留128行，纯单步批次复用同一算子的16行程序；持久KV／KDA及权重共用，工作区借用前部视图，两组通信epoch独立。 | 匹配长输出C1 +26.38%、C4 +19.19%，20条响应完全一致；TP16完整缓存及自然EOS通过，169次forward实际覆盖16／128切换。C4首字中位数增加4.95%，单列保留。更大预算仍待独立扫描。 | Lib＋Serving |
| 3，已撤回（2026-11-05） | **KDA专用decode** | 早期FP32递推因生成变化未采纳；新核心保留chunk的BF16边界，只消除单token时的逆矩阵及单位阵工作，卷积保持。 | 12组核心新旧逐元素一致、10组完整KDA组合对照、TP8的20条响应与16份logits字节一致，TP16缓存回归通过；首轮C1 +0.63%、C4 -8.86%，不作提速结论，交替诊断在旧版control首步阻塞、无有效性能组；核心入口耗时下降57～60%，不替代整网收益。 | Lib；Serving仅诊断和报告 |
| 4，独立合并投影 | **KDA六路输入投影合并** | 一次打包q／k／v／b／f_a／g_a，另保留f_b／g_b，把8路GEMM尝试减少到3路；TP复制／切分及数值边界保持。 | 修正后10组TP8／TP16组件原golden及新旧逐字节一致通过。实际TP8 CAP128耗时增加2.99%，当时暂不接入；已有16行组件wall为TP8 324.65→281.82 μs、TP16 314.12→173.69 μs，四组输出逐字节一致。容量分离后可重新做独立整网试验，未将组件降幅当成整网收益。旧原型因错误及Out别名遮蔽而作废，不混用其数据。 | Lib |
| 第一阶段已采纳，继续优先推进 | **减少主干提交与稳定任务图** | KDA＋MoE已合并至58次，DSA整层合并至47次；下一阶段参考DeepSeek，将45层循环交给一个常驻程序。 | HOST整网调度已采纳，纯decode为3次L3提交，匹配C1／C4 +39.60%／+29.78%，TP16完整缓存通过，C4首字+19.42%保留。跨层L2设备循环曾在首次decode依赖读取超时，未采纳、根因未确认；后续须先缩小复现。 | Lib＋Serving；编译器问题单独跟踪 |
| 5，按实际热点推进 | **无效访存与计算、MoE及mHC融合** | pool写事件、5E空块裁剪和5D短窗口免评分均已逐项采纳；5F短上下文host读取裁剪已测但暂不采纳，5C gather收缩留给更长上下文，随后按新profile定序MoE／mHC。 | 5E／5D各20条HTTP响应与父版本完全一致；5D的512／513边界、长短混合及MTP复用共6组NPU检查通过。5F吞吐只增约1%，C4首字中位数增加19.21%，保留5D。保留现有EP和shared完整权重契约。 | Lib＋必要的Serving适配 |
| 已采纳，继续按新热点推进 | **完整logits读回去重** | 读取相同head索引的代表行，再恢复完整16行FP32输出；密集／碎片选择回退完整读回。head_hidden读回保持。 | 匹配C1／C4 +3.79%／+16.24%，控制C4有一轮偏慢且候选首字+10.66%，全部保留；TP16的168次真实forward与独立完整读回逐字节相同，缓存及自然EOS通过。剩余head_hidden消费者及成本需单独研究。 | Serving |
| 后续优先，尚未实施 | **复用现有设备greedy采样通路** | 参考DeepSeek现有`DecodeResult.sampled_token_ids`与`supports_device_sampling`，为满足采样条件的GLM请求只读回token／小候选；沿用当前FP32 head的结果及argmax同分规则。需要完整logits的采样与prefix捕获保持完整通路，有限值检查保持原覆盖。 | 已确认公共Serving接口及DeepSeek消费者存在，GLM当前尚未接入。先验收逐token与现有完整logits采样一致，再单独测吞吐和TTFT；不预填收益。 | Lib＋Serving模型适配，复用公共执行器协议 |
| 6，独立评估部署效率 | **提高batch利用率和比较TP8／TP16** | 先测单实例C1／4／8／16，再比较固定16逻辑NPU下单TP16与双TP8；双实例可做总C32，单实例32另行设计。 | 当前phase8的TP8单实例C8／C16已实测28.400740／42.725300 token/s，96条含预热响应按原始SSE与相同输入C4参考核对一致，首字中位数4.637038／9.221955秒。保持原算子与精度，是并发配置测量；双TP8／TP16及多副本路由仍待另验。旧短负载双TP8约33.92%的改善仅作线索。 | Serving |
| 7，结构稳定后实施 | **真正减少target调用的MTP** | 从MTP1验收起步，逐步做多个draft的一次target验证、各验证位置的logits、接受点KDA／卷积／KV提交或恢复，再接HTTP调度。 | 先验证每个接受token的正确性，再看每轮推进token数和总成本；不能仅以draft接受率或MTP已启用宣称加速。 | Lib＋Serving |
| 8，容量驱动的后续项 | **C8、长上下文和进一步异步重叠** | KV内存或带宽实际受限时再引入C8；主干耗时降低后重新衡量上传、读回与CPU回收的占比。 | C8、prefix及FlashComm类似改法均不保证普通短请求吞吐受益。已有两任务在途的异步能力不再记作新完成项。 | Lib＋Serving |
| 单列冷启动，未实施 | **权重拼接、重排和上传** | 对启动的权重处理阶段单独计时，再决定预打包、复用及上传并发；保留源权重校验与布局契约。 | 本次TP8只读栈采样位于`iter_rank_uploads`／`fuse_gate_up`及上传线程；代码使用每rank一个线程，并调用Torch拼接，是否存在嵌套并行或重复工作需再测。它不属于已获推理吞吐收益，未改本轮A/B环境。 | Lib＋Serving |

原生MLA、Indexer主线改动的性能归因属于新基线分析；文件重构、备份、CPU检查和本次路线整理不增加原21项性能试验计数。当前不预填收益百分比或把候选收益相乘。上述顺序以实际主干时间线为准：高占比且可独立落地的项目优先，编译器阻塞项保持并行跟踪。

### 对照与每项验收

1. **区分两种对照。** 单项优化用同一版本、同一负载的基线／候选A/B，除本项外固定参数。跨引擎追赶则统一硬件、权重、有效输入、实际输出长度、采样和主要cache dtype，让双方分别选择合适的调度预算；另外保留共同小块配置的控制实验，不能把它当成vLLM最佳性能。
2. **先固定输出长度再比较。** 主负载建议160输入／128输出，160／32另列；输入采用官方tokenizer／完整模板并保存IDs，避免截断模板或把提示复述当成回答质量通过。先测共同支持的并发数，总C32单列部署拓扑。基线关闭MTP及prefix、主KV保留BF16，后续功能开关逐项测、组合再测。
3. **同时报告TTFT、有效TPOT和TPS。** warm-up不计时，记录均值、中位数、p99、全部原始轮次和API实际accepted output token；TPS以相同HTTP窗口计算，不把draft token或额外drain观察混入。不同长度、并发及主机的数据不相除，也不混称纯kernel时间。
4. **正确性和状态先过。** 先算子及新主线现有golden，再整层／整网、TP8／TP16、缓存、并发、取消和槽位复用；保留现有误差门槛。跨版本量化路径变化要按其真实运算参考检查，不能直接套旧BF16输出指纹，也不能通过降低golden标准换速度。
5. **逐项落地和记录。** 先做一个候选，过门槛后采匹配性能；有效才保留，无收益则回退并记录。每个阶段重新排序剩余项目，不把普通功能可运行、CPU编译或预测收益算成已完成的整网提速。

## 2026-10-30 本轮执行记录

固定Lib `5e7430d`、Serving `7e53f3f`及本轮开始时的工作区快照，使用官网发布W8A8权重。197上的独立工具链为PyPTO `69dbfe2`、runtime `9c27de9`、ISA `1b54ac7`、PTOAS 0.70、CPU Torch 2.14；PyPTO和runtime在197从源码构建，PTOAS使用SHA256核验的官方发布包。runtime及ISA沿用此前明确选定的更新版本，保留PyPTO的默认pin差异提示和版本检查。

| 项目 | 本轮已取得的证据 | 状态与限定 |
|---|---|---|
| 组件NPU正确性 | Hello；官网L43的原生W8A8 MLA projection／output（17 tokens）；KDA prefill ragged、decode batch及decode卷积，6组均通过各自既有golden及门槛。 | TP8分片的单卡算子验证，不代替8／16卡通信或45层整网验收。 |
| 减少主干提交的编译复验 | 按当前生产签名分别组合KDA attention＋MoE、DSA attention＋MoE；两种单独attention控制均通过CPU codegen及原生binary构建，组合分别触发78／12个E104形状等价校验错误。 | 继续跟踪PyPTO #3203；未创建NPU Worker或提交设备任务，不算性能试验或提速成果。 |
| 输入缓冲接口修正 | 请求重置、prefix恢复、MTP上传统一使用分配基址＋显式偏移。29项GLM CPU回归、仓库静态检查通过；两卡NPU验证新值、共享owner的两个子视图及相邻数据均精确正确。 | 原始失败请求日志保留；这是主线迁移遗漏的复制路径修正，不记作吞吐优化。 |
| TP8完整Serving | EN回答`5`、ZH回答`巴黎`，均自然EOS、各3个输出token；160输入／32输出的C1／C4 HTTP控制通过。 | 新短输出基线见下表；尚无该版本160／128主负载结果。 |
| TP16吐字 | EN回答`5`、ZH回答`巴黎`，均自然EOS；源码哈希不变，服务正常退出、设备空闲。 | 功能smoke通过；尚无本轮TP16吞吐。 |
| 缓存、多请求状态 | 融合R2的TP8实际HTTP验收通过：完整128前缀命中、160前缀部分命中、完整FP32 logits一致、四并发、驱逐、断连、取消后复用及自然EOS。 | 新接口观察器适配后使用原验收标准；不作为性能数据。TP16缓存验收尚待本轮补测。 |
| 设备诊断 | 单次真实执行的PMU及依赖图取得32份rank级记录，共206,883条物理核记录；逐任务数量审计通过。 | 分项见下文；尚无任务级设备／通信绝对时间线。 |

**四项优化叠加后的功能回归：**最终保留5D R4，未混入KDA单步、5F或投影原型。TP8缓存HTTP回归全部通过，明确观察到128完整前缀命中首步零query、160部分命中恢复128后只计算32；两对8步完整FP32向量历史逐字节hash一致，实际四请求混批、驱逐、断连／取消后复用及自然EOS通过，保留179行采样记录。TP16 EN回答`5`、ZH回答`巴黎`，均3 tokens自然EOS；两拓扑源码前后不变，服务正常关闭，设备恢复空闲。这是功能验收，没有TP16吞吐数据。证据为`FINAL_VALIDATION.json`、`evidence-197/final-optimized-tp8-cache/report.json`及`final-optimized-tp16-smoke/report.json`。本次TP16初始化较慢，启动期间做过只读栈诊断；正常完成后未将其归为死锁，也不使用启动耗时计算TPS。

**TP8新短输出基线：**197，TP8／EP8／DP1，CAP128／context2048／chunk32／预算128，NPU LM-head，BF16主KV，prefix及MTP关闭。每档1组预热、3组正式；压力prompt使用完整官网模板、固定160个输入token，强制32个输出token。它们与上述自然EOS回答分开记录。

| 并发 | 合计输出token/s | TTFT中位数 | 有效TPOT中位数 | 正式输出token |
|---:|---:|---:|---:|---:|
| 1 | **1.661999** | 2.975808 s | 0.525893 s | 96 |
| 4 | **6.414770** | 3.391239 s | 0.532457 s | 384 |

本表属于修正后的新基线，不能与旧TP16、不同上下文或不同输出长度直接相除并称作单项优化收益。统计窗口为首个正式请求组提交至最后HTTP DONE；源码前后哈希一致，服务正常关闭、16逻辑NPU恢复空闲。分块单变量对照已完成，结果见下表。

本轮执行产物统一保存在Serving的 `build_output/glm53-next-optimize/execution-20261030T065321Z/`；197任务目录为同名时间戳的独立package目录。环境准备及测试启动夹具的失败日志保留，尚无有效数据的尝试不计入原21项性能试验。本轮新增采纳结果独立于原21项历史计数，下列各项分别记录父版本及其实际取舍。

### 第1项：分块大小单变量对照——保留默认32

上述TP8条件、源码、预算128、CAP128、输入160／输出32及1组预热／3组正式均不变，只修改chunk。各档C1正式96 tokens、C4正式384 tokens；包含预热在内的20条响应，文本、usage及结束原因逐项与chunk32一致，EN／ZH自然EOS检查也均通过。

| chunk | C1 token/s | 相对32 | C1 TTFT中位数 | C4 token/s | 相对32 | C4 TPOT中位数 |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1.661999 | — | 2.975808 s | 6.414770 | — | 0.532457 s |
| 64 | 1.708246 | +2.78% | 1.890419 s | 5.833368 | −9.06% | 0.608439 s |
| 128 | 1.800404 | +8.33% | 1.315857 s | 5.705961 | −11.05% | 0.617014 s |

较大分块改善本次单请求首字，四并发却退化，暂不改默认值，也不把它记作通用吞吐优化。它是短输出负载上的配置取舍；160／128主负载、更大预算及混合prefill／decode需另验，当前未宣称完成。原始JSON／SSE、源码前后哈希及资源清理证据已取回，审计为`phase1-audit.json`。

### 第2A项：KDA投影按有效token行计算——正确性通过，吞吐基本持平

持久缓冲仍为CAP128，按`query_start_loc[BATCH]`对输入及四族输出创建前部视图，复用同一`kda_projection`。不改变权重、matmul累加、BF16／FP32边界、缓存布局或MoE的EP契约。两个受影响的生产入口均通过CPU codegen及原生binary构建；NPU的有效1／4／17／67行均通过原golden的`rtol=0.02, atol=0.02`，未使用尾部哨兵逐点不变。

匹配上述chunk32基线的完整HTTP结果：C1为**1.663649 token/s（+0.10%）**，C4为**6.501153 token/s（+1.35%）**，每档1组预热、3组正式；20条响应的文本、usage和结束原因与基线一致，EN／ZH自然EOS、源码哈希及关闭清理均通过。幅度较小，判为吞吐基本持平，不记作明确提速。快照作为第2B项前置保留；切换融合候选时已撤出生产工作区，融合收益不含裁行改动。

首个组件脚本漏填运行平台，误入默认模拟平台并在设备执行前的编译阶段失败，已保留原记录；显式`platform=a2a3`后的四组才是设备验证结果。该脚本修正不算模型优化或候选数值失败。原始HTTP证据及逐项文本审计位于本轮执行目录的`evidence-197/phase2-active-projection-tp8-c1-c4-out32`和`phase2-active-projection/AUDIT.json`。

### 第2B项：KDA输出按有效token行计算——C1改善，C4基本持平，暂不采纳

复用同一个`kda_output`，输入、gate及FP32交换输出只给有效前部视图；将它收回现有attention组合函数，由两个生产调用者共用。输出仍先舍入BF16再转FP32参加原TP归约。有效1／4／17／67行均通过原`golden_kda_output`、原`compare_kda_output`和尾部哨兵检查；两个受影响生产入口的CPU codegen及原生构建通过。独立候选的匹配短输出HTTP实测已完成：C1为**1.759857 token/s**，相对原基线**+5.89%**、相对仅投影候选**+5.78%**；C4为**6.385830 token/s**，相对原基线**−0.45%**、相对仅投影候选**−1.77%**。20条压力响应的文本、usage、结束原因与原基线一致，两条自然EOS回答也通过；源码及退出清理通过。判为单请求改善、四并发基本持平，仍保留为候选，尚未替换工作区生产代码；不能把C1的百分比套到多请求。160／128主负载及与后续融合的组合效果未验收。

### 主干融合提前验收：KDA＋MoE 89→58次——已采纳，C1 +8.08%、C4 +7.34%

该项的潜在收益高于当前裁行小改，提前进行设备验收。原始合并图的78个E104，先通过显式的编译期工作区行数降至2个，再明确本来就固定的请求槽数后消除；对应问题继续跟踪[PyPTO #3203](https://github.com/hw-native-sys/pypto/issues/3203)，没有修改SDK或关闭校验。

候选仍调用相同的KDA attention和MoE子算子，只将31个KDA＋MoE层的两次提交合为一次。45层的主干计划由89次变为58次，两个通信窗口及EP窗口各自保留、各推进一次，shared／routed的EP权重与计算契约不变；DSA的11个attention＋FFN仍分开，47次目标尚未达成。TP8／TP16的合并入口与受固定容量参数影响的Dense入口，共4个CPU codegen＋原生binary构建均通过；计划、缓冲绑定及epoch也已检查。该结果不算NPU执行或性能收益。

显式容量KDA在TP8／TP16分片下的ragged、single共4组NPU对照通过，复用原golden及`rtol=0.02, atol=0.02`，同时比较输出与递推状态。融合R1在设备计算前被参数排列校验拒绝；R2将Tensor集中在前、标量ordinal移至最后，4个原生构建和编译后参数排列检查均通过，未跳过Serving校验。失败快照和日志保留。

| 并发 | 原基线token/s | 融合R2 token/s | 提升 | 融合TTFT中位数 | 融合有效TPOT中位数 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.661999 | **1.796290** | **+8.08%** | 2.699984 s | 0.486562 s |
| 4 | 6.414770 | **6.885462** | **+7.34%** | 3.145273 s | 0.496515 s |

沿用TP8、160输入／32输出、chunk32／预算128、每档1组预热／3组正式。20条压力响应的文本、usage、结束原因与原基线完全一致；TP8和TP16的EN／ZH自然EOS检查均通过，源码前后不变、服务正常关闭且设备恢复空闲。TP8额外缓存验收16条响应通过，包括完整／部分命中、FP32 logits一致、四并发、驱逐及断连恢复。已将冻结R2的3个Lib源文件原样应用到优化worktree，29项Serving单测、相关Ruff及diff检查通过。收益对应完整候选（含必要固定编译容量适配），不能全部归因于减少31次提交；未包含裁行候选。尚无该候选160／128和TP16吞吐。

原始证据：`phase5a-fused-kda-moe-r2/AUDIT.json`、`WORKTREE_ADOPTION.json`、`evidence-197/phase5a-r2-fused-kda-moe-tp8-c1-c4-out32`、197的`baseline/fused-cache-tp8-r2`。

### 第5B项：Indexer仅处理可能有效的pool写事件——已采纳，长输出C1 +8.65%、C4 +8.00%

在融合R2上只收缩pool写事件的执行范围，复用原`indexer_pool_write`及量化公式。CAP128／16请求时有效事件最多44个，保守容量48再按现有量化tile64对齐，实际执行64个事件，而非物理表的10240个。剩余`pool_valid`批量清零，缓存布局、运行时输入和其他计算不变。TP8／TP16的DSA入口原生构建通过；0／1／17／44个有效事件均通过原cache／scale golden及原误差门槛，全部10240行valid标记精确一致，未触及的缓存与scale保持逐点一致。

160输入／32输出的匹配HTTP A/B：C1 **1.889941 token/s**（相对融合R2 **+5.21%**），C4 **6.963629 token/s**（**+1.14%**，短负载先判基本持平）。TTFT中位数分别2.646415／3.010189秒，TPOT中位数0.456242／0.466482秒。

160输入／128输出的独立匹配对照已完成，均为同一TP8／CAP128／chunk32／预算128、NPU head、BF16主KV，prefix／MTP关闭，每档1组预热／3组正式：

| 并发 | 融合版本token/s | 收缩事件后token/s | 本项提升 | 候选TTFT中位数 | 候选TPOT中位数 | 正式输出token |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.936322 | **2.103769** | **+8.65%** | 2.556592 s | 0.458311 s | 384 |
| 4 | 7.496042 | **8.096075** | **+8.00%** | 3.013399 s | 0.466721 s | 1536 |

两个长度系列各20条响应的文本、usage、结束原因与各自融合控制完全一致，自然EOS、源码及退出清理均通过；128系列还逐项核对输入IDs、请求payload和计时定义。已将冻结候选的`prefill_layer.py`原样应用到优化worktree，29项Serving单测、相关Ruff及diff检查通过。收益按实际输出长度分别记录，不将两个百分比混用或与短输出融合收益相乘。TP16仍有本项DSA原生构建证明，尚无收缩事件后的TP16整网吞吐。

证据：`phase5b-pool-events/AUDIT.json`、`AUDIT_STRICT32.json`、`AUDIT_LONG128.json`、`WORKTREE_ADOPTION.json`，以及`evidence-197/`下对应32／128输出原始HTTP报告。

### 第3项：复用KDA单步递推——试验完成，生成序列有差异，未采纳

仅当所有活跃请求各有1个query token时，复用既有`kda_conv_decode`与`decode_kda`；prefill／混合批次继续使用原chunk路径。依据`query_start_loc`将打包token映射至`kda_state_rows`，验证槽位15、空洞槽位、16个活跃请求及混合prefill，并使用非恒等状态行映射、非零初始状态和未使用状态哨兵。投影及输出仍计算原CAP128，不混入第2项裁行候选。TP8／TP16的KDA＋MoE与Dense共4个生产入口codegen及原生构建通过；8组NPU组合对照均通过原投影、卷积、decode／prefill及输出golden，沿用`rtol=0.02, atol=0.02`，未用状态行和输出填充区逐点不变。单步算子保持其既有FP32递推，chunk保持既有BF16中间边界，不能声称两条算法逐位相同。

后续R2将分支条件明确为最大query长度**恰好等于1**，空批次保留原chunk路径，避免误更新状态行0；以已采纳pool事件候选为父版本。TP8／TP16共10组原golden对照通过，新增的两组空批次中所有状态行均逐点不变。

匹配160／128 HTTP试验完成：C1 **2.279776 token/s（+8.37%）**，C4 **8.581910 token/s（+6.00%）**，自然EOS及源码／退出清理通过。**但C1的4条、C4的16条生成文本均与控制版本不同，不能作为等价优化采纳。**此处“组件golden通过”和“整网生成一致”是两个不同结论；现有FP32单步与BF16 chunk中间边界不同，尚未做逐步teacher-forced logits归因，不能据此声称原单步算子错误或新模型精度更差。

生产worktree保留原chunk路径；后续5E／5D已重新基于已验收的pool事件版本，完全不叠加本项。快照、10组组件结果、原始长文本及性能保留在`phase3-single-step-r2`、`evidence-197/phase3-r2-single-step-tp8-c1-c4-out128`，取舍见`AUDIT_LONG128.json`和`DECISION.json`。如继续接单步，应单独研究与chunk边界一致的实现，或先完成真实权重的逐步logits精度验收。

### 第5C项候选：Indexer按可见窗口收缩pool gather——CPU通过，尚未NPU验证

PMU显示生产传入10240项page表，原prologue按每项4个pooled rows准备41024行scratch，而score逻辑轴只有10240个pooled rows。候选保持外部输入及缓存ABI，按所有query的`seg_start＋pool_count`最大有效末端，推导所需page前缀，再调用同一个`indexer_score`；高编号请求槽及稀疏窗口均按真实末端处理，不假定活跃请求位于前几个槽。TP8／TP16的DSA生产入口codegen和原生构建通过，原golden／窗口比较器的独立夹具也已过codegen。空窗口、部分尾页、高槽位、跨sort leaf和满逻辑轴夹具已准备；尚无NPU或性能结论。新发现的第5D项能在当前短上下文直接省掉评分，优先于本项；本项先保留给需要评分的更长上下文。

### 第5D项：短窗口省掉无用Indexer query与评分——已采纳，小幅改善

现有TopK在可见池数不超过512时直接返回全部池的紧凑索引，不读取score。context2048下每个query最多512个池，原路径仍计算query投影、head weights、Hadamard／INT8 query量化、pool gather和score，结果没有消费者。候选仅在整批窗口均不超过512时，复用原`skip_query`分支省掉query／head-weights投影，并跳过score；仍调用原TopK、原expand，K／gate投影及所有缓存写入保持。任何query超过512时整批保留原评分路径；原MTP selection复用开关仍保留。

以单步R2候选为父版本，TP8／TP16的DSA生产入口codegen及原生构建通过。R1直接别名外部scalar时，生成的C++引用了未声明的`ext_skip_topk`，在CPU原生构建被拒绝；R2以等价的常量及条件表达谓词，未改SDK、未跳过检查。原失败快照保留。NPU夹具复用原score／TopK golden和原窗口／tie-aware比较器，检查空窗口、高槽位、512／513边界及长短混合；短窗口要求整数选择逐点一致，未使用score的NaN哨兵保持，另检查query投影决策标记。新增MTP复用夹具要求超过512时也保留旧选择，普通／复用夹具的codegen均通过。

按新发现的实际无效计算，将第5E项attention填充块裁剪提前。后续发现单步候选生成序列有变化，故本项R4重新基于已验收的pool事件版本及5E，保留原chunk KDA；重过TP8／TP16的DSA原生构建。6组NPU检查全部通过：空窗口、高槽位、512／513边界、长短混合，以及超过512时的MTP selection复用；原score／TopK golden和比较器不变。初次复用夹具在父进程仍持有Worker时启动子进程，设备打开报权限错误；改为进程退出后串行运行，保留失败记录，没有重置设备或改动模型门槛。

匹配160／128、每档1组预热／3组正式，相对5E：C1 **2.140627→2.188225 token/s（+2.22%）**，C4 **8.306750→8.408665 token/s（+1.23%）**；TTFT中位数分别为2.480782／2.952092秒，TPOT中位数0.440976／0.453022秒。20条响应的文本、usage和结束原因完全一致，EN／ZH自然EOS、源码哈希、关闭及设备空闲检查通过。属于本次样本的小幅改善，已保留修改，不外推到需要评分的长上下文。证据为`phase5d-short-selection-r4/AUDIT_LONG128.json`、`WORKTREE_ADOPTION.json`及对应原始HTTP报告。

### 第5E项：Sparse attention只遍历非空前缀块——已采纳，小幅改善

原flash循环每行固定遍历`ceil(2051／128)=17`个块，末尾空块虽被mask，仍会gather并执行QK、softmax和PV。候选利用现有“索引前紧凑、后缀全−1”布局，先检查每块首项，循环只遍历非空前缀块；无效query得到0次迭代，沿用原空行写零逻辑。非空块的公式、BF16边界、缓存与value expansion不变，不修改TopK或截短有效上下文。

首版使用DSL不支持的`+=`，在CPU解析即被拒绝；R2改为普通赋值后通过，未创建该失败版本的设备任务。由于单步候选的生成序列有变化，R3重新基于已验收的pool事件版本，保留原chunk KDA；TP8／TP16的DSA生产入口codegen及原生构建再次通过。

双拓扑旧／新算子各4组，共16次原golden对照全部通过，覆盖原ragged、128行仅1行有效、全空行、127／128／129／2048／2051索引边界；对应8组旧／新NPU输出**逐字节一致，最大差异0**，所有空行严格为0。未放宽原`rtol=1／32, atol=0.002`或原比较器。

匹配160／128、每档1组预热／3组正式，相对5B：C1 **2.103769→2.140627 token/s（+1.75%）**，C4 **8.096075→8.306750 token/s（+2.60%）**；TTFT中位数分别为2.565114／2.988057秒，TPOT中位数0.449001／0.458034秒。20条响应的文本、usage和结束原因完全一致，自然EOS、源码及关闭清理通过，已采纳。收益较小，未宣称统计稳定或所有上下文均受益。原始张量与报告见`phase5e-live-sparse-blocks-r3/golden-parent`、`golden-candidate`、`NPU_EXACT_COMPARISON.json`及`AUDIT_LONG128.json`。

### 第5F项：短上下文免去host窗口扫描——试验完成，暂不采纳

第5D项在host orchestration中逐行读取128个pool_count，再决定是否需要评分。Serving已保证每个请求不超过编译时context上限；context≤2048时，按4个token一个pool的上界即可证明不会超过512，无需再次扫描。候选在现有共享函数内使用这一编译期上界；更长上下文保留实际窗口扫描，MTP复用及缓存写入不变，不增加入口或配置开关。不将这些读取未经测量地称为128次D2H复制。

TP8／TP16分别在context2048和8192通过codegen及原生构建，共4组。9组NPU选择器检查通过：短上下文的空窗口／高槽位／512，以及8192上下文的空窗口、高槽位、512／513边界、长短混合和MTP复用，沿用原golden与比较器。首次CPU探针误选16384，超出当前API的8192支持上限，在编译前即被拒绝；未修改支持范围。

匹配160／128对照相对5D：C1 **2.188225→2.219048 token/s（+1.41%）**，C4 **8.408665→8.494626 token/s（+1.02%）**，20条响应完全一致，源码和关闭清理通过。但C4 TTFT中位数**2.952092→3.519231秒（+19.21%）**，TPOT仅由0.453022降至0.446499秒。吞吐变化较小且首字指标退化，不将其作为通用收益采纳；最终组合回归仍选择5D。候选和完整数据保留于`phase5f-short-context-specialization/AUDIT_LONG128.json`、`DECISION.json`及对应原始HTTP报告。

### 第4项与验证框架修正：投影合并仍是原型

六路宽投影原型首次使用跨kernel的带stride视图，多行结果实际不正确。诊断还发现`golden.runner`给pure Out使用Tensor初始化器时，参考输出与实际输出可能共用存储，从而掩盖错误；返回同一Tensor的factory也有此问题。修正为独立clone后，在现有`tests/golden/test_runner.py`补充两种初始化器的错误输出回归，先确认旧实现两例失败，再确认修正后该文件177项全部通过，随后`tests/golden`全量324项通过。没有新增独立校验文件，原比较器和数值门槛不变。

首次投影原型的精度通过和性能数字全部作废，不计提速。R2将中间瓶颈结果显式写入紧凑缓冲，每次actual／golden独立初始化，重新完成TP8／TP16和1／4／16／67／128行共10组NPU验证；原golden及`0.02／0.02`门槛通过，四组输出全部与原实现逐字节一致。首次R2编译探针仍指向旧原型的记录也已作废，在任何R2 NPU执行前更正并重过两拓扑codegen。

197 a2a3、单逻辑设备0，原／新各3次预热、30次正式，指标为有效设备wall中位数（微秒），并保留全部样本；没有fallback或不稳定dispatch slot。以下只列1行与当前生产CAP128，其余4／16／67行见原始报告。

| TP分片形状 | 行数 | 原投影设备耗时 | 合并投影设备耗时 | 耗时变化 |
|---:|---:|---:|---:|---:|
| 8 | 1 | 318.600 μs | 273.490 μs | −14.16% |
| 8 | 128 | 1118.680 μs | 1152.180 μs | **+2.99%** |
| 16 | 1 | 306.600 μs | 168.910 μs | −44.91% |
| 16 | 128 | 683.090 μs | 667.320 μs | −2.31% |

生产decode仍以CAP128执行，不能使用1行的收益推算整网提速。TP8当前尺寸略慢，暂不修改生产权重ABI或接入Serving，留待容量解耦后重新评估。证据为`phase4-packed-projection-poc/R2_AUDIT.json`及`npu-tp8-r2`／`npu-tp16-r2/report.json`。上述框架缺陷不影响已采纳项目的独立HTTP文本及缓存logits对照；各组件夹具已核对Out初始化和InOut克隆路径。

### 第5G项：DSA＋MoE进一步融合——已采纳，C1 +6.15%、C4 +6.55%

原DSA融合的12个E104集中在attention的动态token行数穿过循环及条件合流。现有`absorb_query`与`prefill_sparse_attn`增加可选编译期token行数，生产入口传入本来固定的CAPACITY，独立动态组件仍沿用原调用方式；同一计算、同一BF16边界，没有修改SDK或关闭校验。固定维度后，11个DSA＋MoE层可各合为一次，主干从58降至47次。attention TP窗口、FFN TP窗口和EP窗口仍相互独立，各推进一次epoch。原DSA、MoE独立程序保留给真实MTP消费者，未将两个子程序误删。

TP8／TP16的融合及独立DSA入口在context2048通过原生构建，融合入口另在context8192通过，共6组；每次都核对47步计划、每层参数绑定、tensor在前／scalar在后的ABI及通信epoch。接着完成双拓扑各4组attention NPU原golden，覆盖ragged、仅末行有效、全空行及块边界；与已保存的旧NPU输出逐字节比较，8组全部一致、最大差异0，原比较器与门槛不变。匹配160／128、每档1组预热／3组正式，相对5D：C1 **2.188225→2.322731 token/s（+6.15%）**，C4 **8.408665→8.959438 token/s（+6.55%）**。TTFT中位数分别2.351339／2.755979秒，TPOT中位数0.415050／0.426045秒；20条响应文本、usage和结束原因完全一致，自然EOS及源码／关闭清理通过。

TP16随后完成完整缓存HTTP验收：128完整命中首步零query、160部分命中恢复128后计算32、两对8步完整FP32向量历史逐字节hash一致、四并发、驱逐、断连／取消复用及EN／ZH自然EOS均通过；保留179行采样记录，源码不变、runner／registry正常关闭，16逻辑NPU恢复空闲。首轮旧观察器将常驻程序数写死为6，在新7程序版本启动末尾断言失败，未进入HTTP；保留失败报告。v5仅改为核对模型声明的完整程序清单和唯一编译对象，原`run_acceptance.py`逐字节不变，数值门槛、客户端协议及时间上限均不变，R2重测通过。

三个模型文件已按冻结候选精确同步到优化worktree，其余模型源码不变。证据为`phase5g-dsa-fusion/{CPU_GATES.json,NPU_EXACT_COMPARISON.json,AUDIT_LONG128.json,FINAL_VALIDATION.json,WORKTREE_ADOPTION.json}`及`evidence-197/phase5g-final-tp16-cache-r2/report.json`。TP8性能与TP16缓存属于各自负载，不据此声称已有TP16吞吐。

### 第2项：decode执行容量分离（执行阶段phase6）——已采纳，C1 +26.38%、C4 +19.19%

现有Indexer的`PROJ_T_TILE`已经是16，但整网入口仍残留64倍数的旧限制；首次直接16行探针在导入时被拒绝，没有编译或NPU执行。简单只传constexpr也不会改变签名尺寸。后续通过同一入口的constexpr容量和匹配的metadata-only tensor形状构建16行变体；TP8／TP16的Dense、KDA＋MoE、DSA＋MoE共6组原生构建、参数和完整计划检查通过，未创建NPU Worker。

接入采用PyPTO现有JIT校验及缓存，复用权重与持久KV／KDA分配；只借用查询工作区前16行。两组程序保持独立通信epoch，所有活跃请求各一个query且总数不超过16时才选小容量，prefill／混批保留128行。128行与16行共12组目标程序原生构建通过，经共享KernelCompiler的真实路径复验通过；31项Serving单测覆盖原功能、容量交替时epoch独立，以及带byte_offset的嵌套视图、常量物理布局保留。最终采用后在真实worktree重跑31项单测及Ruff、diff检查通过。

第一次设备启动在编译阶段发现旧独立attention入口漏传新增constexpr，未创建设备Worker或执行NPU计算。R2为全部调用点显式绑定容量；与首版相比仅两个旧HOST函数增加参数，目标计算函数及Serving源码逐字节／AST核对不变。随后按真实`prepare_kernels`完成TP8／TP16各15个程序的CPU预检，包含常驻主干、三个16行变体、MTP及缓存辅助程序，改变的旧DSA入口也完成原生构建。

匹配160／128整网实测相对5G：C1 **2.322731→2.935380 token/s（+26.38%）**，C4 **8.959438→10.679046 token/s（+19.19%）**。TPOT中位数分别从0.415050／0.426045降至0.325312／0.349838秒；TTFT由2.351339／2.755979变为2.377803／2.892375秒，C4首字增加约0.136秒（+4.95%），单列保留，不声称首字改善。20条响应的文本、usage和结束原因完全一致，EN／ZH自然EOS、源码及关闭清理通过。

TP16随后通过原完整缓存HTTP验收：128完整命中首步零query、160部分命中恢复128后计算32、两对8步完整FP32向量历史逐字节hash一致、四并发、驱逐、断连／取消复用及EN／ZH自然EOS均通过；169次真实forward明确覆盖16与128容量，选择规则全部符合query长度，两组epoch均实际推进。另将同一首个串行请求的8步完整FP32 logits与父版本5G比较，全部SHA256及token ID相同；此跨版本结论仅覆盖这8步。源码前后不变，runner／registry正常关闭，设备恢复空闲；没有将该功能负载用于TP16吞吐声明。

另补齐离线manifest的16行程序保存／重载，使用实际编译元数据完成CPU往返，确认拒绝缺项变体，并兼容只有完整容量的manifest。该补丁仅处理离线产物元数据，不在HTTP A/B调用路径中。模型文件及Serving运行代码已精确采用冻结R2，离线补丁一并纳入；未新建校验文件，仅在已有Serving测试文件补两项回归。候选与失败记录位于`phase6-query-capacity`、`phase6-query-capacity-r2`和`phase6-capacity-feasibility`；验收证据为R2目录的`AUDIT_LONG128.json`、`FINAL_VALIDATION.json`、`ACTUAL_QUERY_CAPACITY.json`、`CROSS_PARENT_FIRST8_SHA.json`、`WORKTREE_ADOPTION.json`及`offline-manifest/AUDIT.json`。

### 整网调度第二阶段（phase7）——HOST方案已采纳，C1 +39.60%、C4 +29.78%

两个独立候选都只替换纯decode的上层计划：seed、一次主干调度、head，共3次L3提交；prefill／混批保留47次。Dense、KDA＋MoE及DSA＋MoE各自持有通信窗口，epoch按本程序forward次数与层序号计算。KV、KDA持久状态、权重及数值边界保持，逐层输出调试保留原路径。新程序作为已有auxiliary产物保存，沿用共享编译／运行接口，没有SDK补丁或新增低层计算实现。

第一种候选把45层内联到同一L2设备循环，通过三个CPU原生预查、TP8／TP16各16程序的真实编译、32项Serving单测，另通过8192上下文CPU原生构建。它在TP8首次完整prefill后、首次decode中失败：原始设备日志首先报告`get_tensor_data`等待生产任务超时（code 8），位于MoE的`expert_offsets`读取；当时`prefill_dispatch_count_exchange`运行中，还有KDA任务缺少依赖。随后才出现507018／507901和失败后的`get_ref`断言，不能把这些次生错误当作根因，也未认定是数学精度或参数数量问题。退出未完成后仅清理本实验进程组，设备恢复空闲，没有手工reset。日志、生成源码及校验信息保存在`phase7-decode-backbone/failure-evidence/`，不含跨主机传输的原生二进制；该候选未采纳、没有性能结果，根因仍需缩小复现。

第二种候选保留已验证的L2程序边界，只在HOST入口循环组织原有逐rank调用；它减少上层提交，未将整个网络合成一个设备L2程序。主干内仍有每rank 87次原生子调用，TP8／TP16分别为696／1392次。对照确认原有40个函数／类AST完全相同，再通过TP8／TP16各16程序的编译／原生构建及32项单测，ABI、形状、计划绑定和epoch溢出检查均通过。实际HOST调用的设备、张量、窗口、标量与epoch，在TP8／TP16及forward序号0／1／7均与旧45层计划逐项一致；该CPU证明不能代替设备回归。

TP8匹配160／128实测相对容量分离版：C1 **2.935380→4.097659 token/s（+39.60%）**，C4 **10.679046→13.859591 token/s（+29.78%）**。TPOT中位数分别为0.227074／0.244906秒；TTFT分别为2.394398／3.453983秒，相对父版本增加0.70%／**19.42%**，C4多约0.562秒，保留该首字取舍。20条响应的文本、usage和结束原因完全相同，EN／ZH自然EOS、源码不变、正常关闭及设备空闲均通过。

TP16完整缓存回归通过全部16项及179次采样，覆盖完整／部分命中、零query恢复、四并发、驱逐、取消复用和EN／ZH自然EOS。168次真实forward覆盖16／128容量且选择规则全符，纯decode确实使用整网HOST计划及独立epoch；另外与父版本比较同一首个串行请求的8步完整FP32词表向量，全部SHA256及token ID一致。该跨版本字节结论只覆盖这8步；没有TP16吞吐声明。已精确采用冻结源码，25个模型Python文件全部匹配，实际worktree的32项Serving单测、Ruff及diff检查通过。

源码与证据为`phase7-host-backbone/{DESIGN.json,CPU_INTEGRATION_GATES.json,HOST_ORDER_TP8.json,HOST_ORDER_TP16.json,AUDIT_LONG128.json,FINAL_VALIDATION.json,ACTUAL_QUERY_CAPACITY.json,CROSS_PARENT_FIRST8_SHA.json,WORKTREE_ADOPTION.json}`及`full-set-tp{8,16}/RESULT.json`。缓存观察器适配3／47次主干调用及新程序计数，客户端验收脚本逐字节不变，原数值、状态和时间门槛不变。另只读记录相同head行索引对应的全部FP32分片字节，168次forward均一致，供下一项读回去重判断，不增加设备执行或更改原验收标准。

### 输出读回去重（phase8）——已采纳，C1 +3.79%、C4 +16.24%

候选仅改Serving的logits搬运，保持Lib计算、NPU LM-head和完整head_hidden读回不变。对于相同`logit_row_indices`，先读代表行，再恢复原16行完整FP32结果；代表位置合并为连续区间，超过半数行唯一或超过两个区间时回退原完整读回。仍保留全部行、padding、有限值检查和既有输出接口，未缩减数值精度。

已有测试文件中39项单测通过。197两逻辑NPU的14项真实搬运对照随后全部通过，覆盖全重复、prefill索引、高槽位、全唯一、碎片回退和元数据缺失，另核对worker逆序、嵌套byte_offset、负零、NaN payload及第二次设备数据更新；每项都与独立完整读回及原上传字节比较，所有16行、padding及父缓冲canary完全一致。这项只执行内存搬运，没有模型计算或吞吐声明。上一项真实权重的168次forward中，相同head索引对应的全部FP32分片字节也全部一致，覆盖prefill、decode及并发。

匹配160／128实测相对HOST调度版：C1 **4.097659→4.253083 token/s（+3.79%）**，C4 **13.859591→16.110647 token/s（+16.24%）**；20条响应的文本、usage和结束原因完全相同，自然EOS、源码不变、正常关闭及设备空闲通过。本地从取回的原始SSE再次核对，结果一致。TPOT中位数为0.217245／0.220316秒，相对父版本降低4.33%／10.04%；TTFT为2.353768／3.822139秒，分别降低1.70%／增加**10.66%**，保留C4首字代价。

控制版本C4三轮完整窗口为34.450／42.895／33.481秒，候选为31.837／31.841／31.663秒。16.24%使用全部三轮聚合计算，未删除控制慢轮，不能当作稳定的跨负载增幅。CPU恢复完整输出的成本也计入端到端窗口。**TP16完整缓存回归已通过，168次真实forward的完整FP32分片（包括padding）与独立完整NPU读回全部逐字节一致，覆盖16／128查询容量。**全部16项HTTP验收及179次采样、并发、取消复用、自然EOS与正常关闭通过，设备恢复空闲。与HOST父版本比较首个串行请求的8步完整词表向量，SHA256和token ID也相同；此跨版本结论限于这8步。已精确采用冻结Serving代码，25个Lib模型Python文件保持一致；实际worktree的39项单测、Ruff及diff检查通过，没有新增跟踪的测试文件。证据为`phase8-compact-logits/{CPU_GATES.json,device-copy/RESULT.json,REAL_DUPLICATE_ROWS.json,AUDIT_LONG128.json,AUDIT_LONG128_LOCAL.json,PER_ROUND_STABILITY.json,FINAL_VALIDATION.json,CROSS_PARENT_FIRST8_SHA.json,WORKTREE_ADOPTION.json}`。

### 当前代码的并发扫描（phase9）——C8／C16已实测

197 A3、TP8／EP8／DP1，共8个逻辑NPU，完整官方W8A8权重；当前已采纳phase8，BF16主KV、NPU FP32 head、prefix／MTP关闭、chunk32／预算128／context2048。每请求160输入／128输出，每档1轮预热、3轮正式。下表为整个实例的聚合输出吞吐；C1／C4沿用phase8匹配实测，C8／C16在另一个相同配置进程中依次测得。

| 并发 | 聚合输出 token/s | TTFT中位数（秒） | TPOT中位数（秒） |
|---:|---:|---:|---:|
| 1 | 4.253083 | 2.353768 | 0.217245 |
| 4 | 16.110647 | 3.822139 | 0.220316 |
| 8 | 28.400740 | 4.637038 | 0.246053 |
| 16 | 42.725300 | 9.221955 | 0.293236 |

C8／C16正式输出分别为3072／6144个token。96条含预热响应重新解析原始SSE，8种相同输入的文本、usage及结束原因与已验收C4参考全部一致；输入IDs、模板、采样及源码哈希相同，源码前后未变。吞吐按首轮正式dispatch到最后HTTP DONE的完整窗口重算，包含轮间客户端开销，全部样本保留。服务退出0、设备恢复空闲。

提高并发增加聚合吞吐，同时增加单请求的首字与逐token等待；这是当前代码的配置测量，不作为新的算子优化A/B增幅，也没有将C16并发写成TP16设备拓扑。尚未在该版本实测双TP8总C32或单TP16，也未据此计算与旧vLLM不同条件数据的追赶倍数。证据为`evidence-197/phase9-batch-scan-tp8-c8-c16-out128/`及`phase9-batch-scan/AUDIT.json`。

### 2026-10-31 容量分离后重新剖析：继续优先减少主干提交

197，当前容量分离R2，TP8／EP8、CAP128／decode16／context2048／chunk32，1请求、160输入／16输出的独立诊断。48次真实forward成功；排除一次开启PMU的decode和另外5次预热后，统计24个普通decode窗口。没有重放状态，也不将该诊断任务的TPS用作匹配性能结果。

| 主机API时间窗 | 当前中位数 |
|---|---:|
| 一轮完整forward | **320.249 ms** |
| 16次输入上传，共10,406,400 bytes | 2.809 ms |
| 47次主干submit | 278.015 ms |
| 47次主干wait | 5.784 ms |
| NPU LM-head调用 | 1.214 ms |
| 完整logits读回，9,928,704 bytes | 17.028 ms |
| head_hidden读回，2,097,152 bytes | 8.682 ms |

约284 ms仍在提交／等待窗口内，其中31个KDA整层约186 ms、11个DSA整层约75 ms；这些窗口包含设备执行、通信及反压，不能全部视为Python开销。输入及两种读回大小未缩小。新PMU覆盖4个逻辑程序、48份rank级子调度；84,702条物理核记录逐任务与`block_num×活跃kernel槽`核对一致。KDA投影、mHC系数、TP等待和EP count exchange仍显眼；PMU累计周期包含并行核及等待，未换算为整网延迟占比。相比早期剖析，各种优化已叠加，不能把诊断耗时的总下降归给容量分离单项。

下一项优先检验整网常驻循环，先保留prefill路径，只尝试把纯decode的45次层调用合为一次，seed／head分开。参考DeepSeek V4.1的层循环，复用现有Dense、KDA、DSA和MoE函数，保留三类独立通信窗口与epoch；每层加作用域约束临时资源生命周期。实验已通过TP8行16／128及TP16行16的代码生成、ABI和CPU原生构建；尚未证明设备正确性、资源上限或性能收益。初版未按JIT规则复用orchestration，之后发现新边界未声明已有NZ布局；各失败和修正快照均保留，SDK未改。

剖析证据：`evidence-197/profile-phase6-tp8-single-pass/{report.json,PROFILE_SUMMARY.json}`、`phase6-query-capacity-r2/profiling/{PHYSICAL_COUNT_AUDIT.json,ALL_PMU_SUMMARY.json}`；整网循环预查为`phase7-backbone-feasibility-r5/CPU_GATES.json`。以上是phase6剖析时的排序，其后的HOST调度及logits读回去重均已完成验收，见上节；剩余head_hidden用途、小容量投影合并及更少设备边界仍待独立评估。

### 原始新基线分项：主干调用仍为首要优化对象

197原始新基线，TP8／CAP128／chunk32，1请求、160输入／16输出，独立诊断运行。48次真实forward均成功；仅一次实际decode启用PMU＋依赖图，swimlane为0、没有状态重放。统计24个非PMU的decode窗口，另外排除5步预热，不使用该诊断任务的TPS作性能对照。

| 主机API时间窗 | 中位数 |
|---|---:|
| 一步forward | 532.345 ms |
| 16次输入复制（共10,406,400 B） | 2.877 ms |
| 89次主干submit | 484.447 ms |
| 89次主干wait | 7.920 ms |
| NPU LM-head调用 | 1.250 ms |
| logits读回（9,928,704 B） | 17.356 ms |
| head_hidden读回（2,097,152 B） | 8.536 ms |

各项为独立中位数，不要求严格相加。主干submit＋wait占整步的逐样本比例中位数为**92.49%**；其中包括设备推进、运行时及通信等待，不能称为92.49%纯CPU开销，也不能预言融合能消掉全部时间。

seed、含KDA的Dense、DSA及MoE共32份rank级PMU记录取得**206,883条物理核记录**，逐任务与依赖图的`block_num×有效kernel槽`完整匹配，并用各自program的name map标注。KDA QKV投影、mHC系数、Indexer pool gather／写事件值得继续研究；PMU核周期总和不是设备wall time，不用它直接排出整网收益百分比。**任务级设备／通信绝对时间线仍未取得**。

后续小项补记：研究按有效请求槽读回logits，以及NPU head下head_hidden读回的实际消费者；需保持相同的有限性检查、缓存及MTP语义，先不预填收益。当前优先完成融合、KDA单步和容量解耦。

原始观测器、完整FP32 logits、32份PMU／依赖图、name map及物理记录审计均保留在本轮`evidence-197/`。profiling首轮因客户端不接受0组预热而停止，正常关闭且设备空闲；修正为1组预热后上述R2通过，不改变模型、数值门槛或性能基线。

## 当前进展

以下为此前记录的分项状态；主线同步后的实施顺序和状态解释以上一节为准。

| 项目 | 当前状态 |
|---|---|
| 整网功能 | 已集成官网 W8A8 权重加载、45 层文本主干、prefill、增量 decode 和 pypto-serving。 |
| 并行配置 | TP8/EP8/DP1、TP16/EP16/DP1 均有真实 NPU 验证。Attention 和 Dense FFN 使用 TP；MoE 使用 EP，共享专家保留完整权重。 |
| 请求调度 | 已支持对齐的 chunked prefill、多请求 decode、等待队列、取消和请求槽位复用。 |
| 已完成试验 | **21 项：12 项采纳、8 项否决或回退、1 项配置研究。** 测试夹具失败和编译准备不计为优化完成。另有独立的权重 BYPASS 试验被否决。 |
| 已提交基线的关键实现 | shared INT8 NZ、routed INT8 ND、最多两个设备任务在途；Embedding、最终归一化和 LM-head 在 CPU，LM-head 每次前向计算 16 行，并保留设备端完整 32 行、所有 rank 的读回及有限值检查。本地 NPU-head 候选另见 opt38。 |
| Prefix cache | opt15 已采纳，默认关闭；opt31 完整 prompt 命中功能已接入本地 Serving，**历史冻结版本的 TP8、TP16 功能验收均通过**。当前R3也已在两拓扑通过128完整命中／160部分命中的HTTP复验；当前统一通过 `--enable-prefix-caching` 开启部分及完整命中，性能仍待单独对照。 |
| 异步下发 | Engine 调度和前向内部有界异步下发已支持，opt13 已采纳。设备执行与 CPU 输出处理的进一步重叠属于 opt14B，尚未完成 NPU 和性能验收。 |
| 当前代码与验收基线 | 历史已采纳基线为 Lib `2e075a2`、Serving 运行时代码 `486c4b1`，布局版本 3；本次整合提交包含后续 opt38 和 opt31，opt38 的数值参考仍待确认。 |
| 最近结论 | opt19 静态 CAP64 attention、opt16B BLAS 线程调整均无有效收益，已否决；默认 prefill chunk 仍为 32。 |
| 历史功能验收 | **对应冻结版本的 prefix cache 功能验收通过。** 同一 NPU LM-head 后端下，完整命中、部分命中、16 请求并发、取消重启及关闭检查均通过；两种拓扑各保留299个完整FP32向量，不混用旧 CPU-head 输出或吞吐控制。 |
| opt38 状态 | PTO3/NPU 完整 LM-head 已实现并完成设备、整网与 Serving 测量。本地默认 NPU；FP64 独立对照通过，旧 CPU 逐点微差保留，正式采纳仍等待参考标准确认。prefix 验收不改变这一状态。 |
| 当前源码复验 | **TP8／TP16 R3 的当前源码 HTTP功能验收均通过，并完成第二独立审计。** 每拓扑16个必要响应案例及断连检查全过，覆盖raw64、缓存、4请求native混批、英文／中文自然EOS及清理。K128仍待正式匹配性能对照；旧R2约18.5秒／forward及失败报告保留。 |
| 新性能分支 | `feat/next_glm53_optimize` 的共享输入缓冲及元数据合并已通过TP8／TP16整网及独立审计：每拓扑58行完整FP32指纹一致、4请求混批通过。TP8 decode主机窗口627.91→572.51ms（耗时降低8.82%）；TP16当前1062.94ms，无匹配基线，不计算提升比例。均非正式Serving TPS。 |
| 固定16卡部署实测 | 同C2、11输入／8输出，两个独立TP8实例总吞吐2.580460 token/s，单TP16为1.926905，提高33.92%；两侧400行完整FP32指纹通过。尚非单入口DP2，下一步补单TP8 C2／C4卡效率。 |
| vLLM Ascend 对照 | **同机同客户端重测完成：**16逻辑NPU、同发布版W8A8／BF16主KV、160输入／32输出、无MTP／prefix，C1／C2／C4的vLLM为10.1916／17.1636／30.3009 TPS，PTO3为0.7995／1.5931／3.1141，约9.73～12.75倍。优化组合另测160输入／128输出，混合文本C32最高503.24 TPS（C8＋MTP5，FlashComm1关闭）。协议、延迟、限制及原始证据见后文两节。 |
| 最新依赖候选 | 已在独立环境构建PyPTO69dbfe2／runtime9c27de9／ISA1b54ac7／PTOAS0.70；65单测、TP8／TP16共14个普通生产程序native构建通过。4个融合probe仍报Error104；新栈尚未NPU回归。 |
| 当前接入审查补测 | 官方新 stack 的最小构建复现、focused42、Serving182、weights inline 及两仓重跑 pre-commit 通过；本机 Hello／Qwen accuracy／JIT cold-hot 均过，197 CPU42／Hello-only 过。附加 replay_types 仍保留17项 fixture 构造失败和覆盖缺口；不报 CPU 全绿，也不把HTTP功能验收当作独立CPU-golden整网数值验收。 |

“采纳”表示实现已保留且通过对应数值和性能验收；“验证中”表示只完成部分关卡；“否决”也是已完成的试验，其候选实现不保留。不能把组件提速、编译通过或 CPU 模拟通过计入整网收益。

### 当前吐字验证到什么程度

已验收版本能够连续生成，1、2、4 并发的短输出回归均完成。历史已验收性能样本使用英文补全，在填充文本之后输入 `The capital of France is`，实测续写为 ` Paris. The capital of France is Paris`。测试忽略 EOS 并生成 8 个 token。

这说明生成链路及优化前后结果一致；首个城市正确，后续有重复续写。**目前不能据此认定长回答质量已经验证通过。** 更长输出、自然 EOS、中文与英文回答的连贯性和重复情况，纳入 opt17B 补测。

本轮R3已另行完成两拓扑的64-token raw SSE，并以两道受控题检查完整回答及自然EOS：英文算术答案为 `5`、中文法国首都为 `巴黎`。两条HTTP原文仍含 `</think>`（分别为 `</think>5`、`</think>巴黎`）；64-token续写按长度结束，两个简短问答只作为回答质量smoke，不等于广泛长文或领域准确性验收。

数值回归方面，已保留原 TP8 的 25 份和 TP16 的 29 份输出，后续优化与这些已验收 PTO3 结果逐字节比较；另有前缀、缓存和 Serving 回归。它们属于 PTO3 实现之间的回归对照，**不等同于独立 CPU golden 对完整 45 层模型的验收**。独立组件仍保留各自的 golden 门槛。

### 接入审查补测：权重预检已完成，官方升级已通过编译复现

按 `adapt-new-model` 的完成标准补齐本轮接入证据。权重预检修复最初在 `03d5190` 基础上仅修改 `weights.py`：读取 tensor payload 前，验证驻留计划需要的全部源 tensor、各 EP owner 的专家、实际存储 dtype、量化声明和 scale／offset header。该文件冻结源码 SHA256 为 `d21603309b308acd2307c81c66c7742c1db48abbd1f097491b0d0e8076bbbc19`；该权重预检修复没有更改 kernel、量化数学或误差门槛。随后集成的三文件 K128 修复及其独立验收见下文。

| 检查 | 当前结果 |
|---|---|
| 原有 inline CPU 自检 | 全部通过，含微型夹具的 57 个驻留栈逐字节对照，以及 19 类坏 header 在零 payload 读取时拒绝；不冒充真实完整权重的逐值比较。 |
| 真实权重预检 | 官网发布版、自制 W8A8 各在 TP8／TP16 完成完整计划预检，均为零 payload 读取。真实 Serving `load_checkpoint` 路径也在 TP8／TP16 通过，检查中禁止 `Checkpoint.read` 读取 payload。 |
| 双向 index 覆盖 | 两套权重均有 113,449 项：计划消费者对应 113,098 项；其余 351 项有明确排除理由，即 347 项视觉权重、3 项未使用的 MTP host-table 副本和 1 项离线 `rot.weight`。这不表示已支持视觉或 MTP 推理。 |
| 测试与静态检查 | 182 项相关 Serving 单测及两仓全量 pre-commit 检查通过。 |
| 旧 stack 本机 TP8 启动 | 任务 `task_20261026_194024_8502562437` 已结束，外层 exit1、server exit3；Gate 编译失败，报告 `passed=false`，Ascend 日志文件数为 0，未进入推理。 |
| 旧 stack 197 TP16 启动 | `glm53-skill-acceptance-20261027-197/native-tp16-r1` 同样 exit1、server exit3，出现相同 Gate 编译错误，未进入推理。源码前后哈希相同；结束后只读复查确认 16 个逻辑设备健康、无 NPU 进程。 |
| 新 stack 的本机 TP8 HTTP R2 | 已启动真实模型并完成 raw8 JSON／SSE；约 18.5 秒的稳定 forward 耗时使 20 分钟客户端预算耗尽。raw64 未完成，缓存、并发及自然 EOS 未验收；断连后 30 秒内释放请求槽的观察失败。最终关闭后资源清空，源码未变；不是新的 Gate 编译失败。 |

两次均在 Gate 生成代码的 `TMOV` 编译中出现 `(0, 0)` valid shape 及 `GetValidRow`／`GetValidCol` 模板错误；后续 `invalid KV cache block count: 0` 是启动失败后的级联错误。**根因已定位到 PyPTO 的 `tile.assemble` 发射：局部 scratch 复制把 zero-valid replay 哨兵错误固化为静态 `(0, 0)` subview。** Lib `03d5190`（#1555）在 Gate 中引入 `bf16_linear`／scratch 复制后触发了该问题，并非本轮 `weights.py` 元数据修复引入。

旧 stack 下的61行独立复现仅含合法的 16×16 BF16 matmul、FP32 add 和局部 scratch 复制：CPU 侧 codegen 成功，但实际 C++ binary build 失败；去掉 scratch 复制的对照构建通过。`skip_ptoas` 保存的原始 IR 已出现零 valid，错误发生在 PTOAS 之前。两机旧版本链已核对一致：PyPTO `6a7c862`、runtime gitlink／源码／已安装二进制 `6e383fc`、ISA pin／实际 checkout `c0d7148`、PTOAS pin／实际版本 `0.65`。复现未创建 Worker 或提交 NPU；源码和证据保存在 Lib 的 `build_output/glm53-gate-compile-repro-20261027/{repro.py,EVIDENCE.json}`。

原报告的 `actual_NPU_execution: true` 只表示发起了 `--execute` 尝试；它不证明 kernel 编译通过、模型 forward 已执行或 HTTP 已生成响应。旧 `native-tp8-r1`／`native-tp16-r1` 失败报告保持不变；其 zero-valid 编译问题通过官方 stack 升级处理，没有添加本地 compiler patch 或模型绕过。后续 K128及必要列 tile适配另按组件／调用者和整网阶段验证。

**官方升级里程碑：**旧 stack 为 PyPTO `6a7c862`（10月15日）、simpler `6e383fc`、PTOAS `0.65`。现已构建安装官方 PyPTO `ce0eb7c07fad364b6b169ee99a55950283e60dae`（10月27日 UTC）、simpler `d8f997f`、ISA `b1d460ee8c001585ede36b68ba3678dc5a226bbf`、PTOAS `0.67`，本机两个实际 venv 的 pin 链检查均通过。latest 的 PR #3078 对浮点 Vec assemble 使用 `TINSERT`；原61行源码保持 SHA256 `60147d961bfc921e90d2ed2c668e20b1915ec16a8f3d7b61fd181d54360f4936`，repro 和 control 均已实际通过 codegen 及 C++ binary build，无本地 patch。旧 Gate 的 zero-valid 编译阻塞已由官方升级解决；这仍是 CPU 构建验证，不是模型 NPU 或 HTTP 验收。升级后的 focused compiler CPU 回归42项通过；Serving 相关182项重新通过（1.16秒），weights inline 自检也通过。

**保留的 CPU 失败与覆盖缺口：**附加 `replay_types` 共25项，8 PASS／17 FAIL。17项均在旧 fixture 将 `Expr`／`int` 混用于手工构造 `TensorType` 时失败，未进入目标编译 pass，不能算作这些 pass 已获验证。独立审计已排除旧 core 或双 core 混用，正常 `pl.Tensor` 前端 specialization 在本机两个实际 venv 均通过，旧 `6a7c862` 与新 `ce0` 的 constructor binding 相同，GLM 不手工调用该 API。失败原样保留，未修改官方测试或阈值，**不宣称 CPU 全绿**；审计见 Lib 的 `build_output/glm53-pypto-refresh-20261027/replay-constructor-audit/AUDIT_SUMMARY.json`。

**设备 sanity 已完成：**本机单卡任务 `task_20261026_210327_1411884178` 的 Hello、Qwen accuracy、JIT cold／hot 均已通过。197 也已完成官方新 stack 升级、focused CPU42 和 Hello-only 真实设备检查；197 缺少 Qwen 权重，未在该机执行 Qwen accuracy／JIT 检查，不能冒称通过。单卡 sanity 不替代 GLM 整网、HTTP 或长回答验收。

构建过程保留原始记录：本机初始保守的2 jobs推进较慢，在进度264/326后仅将任务的 ignored 构建 profile 调为8 jobs，保留已有 `.o` 继续构建。这是构建配置记录，不计模型性能优化试验。升级及本轮补测不增加21项性能试验计数，也不改变 opt38 状态。

旧 stack 的两次真实 CLI 启动均尝试当前源码 JIT 编译，未用旧编译产物覆盖当前实现。升级后的 R2 已进入 GLM forward，raw8 JSON／SSE 通过；raw64 因整网过慢未完成。这些R2未完成项随后已由当前R3在TP8／TP16完成：128-token完整命中、160-token部分命中、4请求并发及官方low配置的英文算术／中文首都chat。R3两条chat实际设置上限256，均在3个token内由官方EOS自然结束。自然 EOS 必须保留实际终止证据，不能仅凭启用 EOS 策略或到达输出长度上限判定通过。旧版本的缓存和短输出证据继续保留，不替代当前源码的结果。

**R2 运行与客户端协议收尾：**本机 TP8 的 `native-tp8-r2/report.json` 保留 `passed=false`。原始 1200 秒客户端预算和 30 秒断连释放窗口均未放宽；断连观察未通过，最终关闭及缓存／请求资源清理完成，源码哈希不变。当前记录是稳定的 forward 过慢，不能与旧 stack 的 Gate 编译错误混为一谈。harness-v2 修正了受控 graceful stop 的原始 `-15` 退出码分类，以及未触达用例必须记录 `NOT_RUN` 的口径，15 项 CPU 协议检查通过；原 strict EOS、完整源码哈希检查及两个时间上限保持不变，harness-v2现已在197的TP8／TP16 R3完成16个必要响应案例及断连检查，并经独立审计通过；R2原始失败不改写。证据位于 Serving 的 `build_output/glm53-skill-acceptance-20261027/{native-tp8-r2,harness-v2}/`。

### 共享 BF16 linear K128：已集成，双拓扑HTTP功能通过，正式性能待测

`03d5190` 的共享 `bf16_linear` 以 K1 逐元素推进；K4096 每个输出 tile 循环4096次。现已正式集成 K128 分块，循环改为32次，末块按有效 K 长度零填充，保留块间 TwoSum 补偿、全部 BF16／FP32 边界、原 golden 和比较器。

较大的 K tile 使两个原有256列 tile 超出 Vec 预算：KDA projection 和 MLA epilog 在相关组合下出现 **233,472 > 188,416 字节**，原失败保留。必要适配仅将 KDA `N_TILE` 上限和 MLA `O_N_TILE` 改为128；它们控制输出列分块、grid 和 offset，不改变模型维度、权重形状或外部 ABI。没有修改内存 reserve、硬件限制或工具链验证。MLA 的 prefill 和独立 decode 两个 owner 均受此常量影响，因此分别验证。

| 验证项 | 实际覆盖 | 结果与边界 |
|---|---|---|
| 初始共享 linear 组件 | 同卡、同 seed533，`[16,4096] × [128,4096]`；K1／K128各一次运行。 | 两侧2048个 FP32点均通过原 `atol=1e-5, rtol=1e-6` 零超差门槛，2048个 BF16点逐点一致；5次预热、100次正式轮次的设备中位数 **16.0748 ms → 0.0988 ms**。编译、输入生成／上传及启动不计入该设备指标，全部原始轮次保留；**仅组件数据，不是整网收益**。 |
| 额外 linear：10个 NPU进程 | K129尾块两种权重方向、seed534、BF16 midpoint、有限大／小项抵消；K1与K128各运行。 | 原严格 FP32／BF16双输出门槛全部通过，不改 golden；不把这些独立矩阵输入当作整网 hidden。 |
| Gate：12个 NPU进程 | 官网第3／23／44层、全部288路由权重行、tokens1／17；既有 embedding／mHC 派生输入。 | 原专家集合精确匹配、route weights 的3e-3／1%及 INT8 ±1／0.1%预算通过；全部权重行参加路由打分不等于每个专家均被选中执行，更不等于完整 MoE／45层验收。 |
| mHC：40个 NPU 用例 | 随机 mixes／pre／post／head；官网第3／23／44层 attention／FFN两侧的 mixes；tokens1／17。 | 原 owner／golden／门槛通过；是 mHC调用者验证，不是 attention／FFN或完整层组合。 |
| KDA／MLA projection：12个 NPU进程 | TP8 rank0，官网L44 KDA projection、L43 MLA prolog及 query absorption，tokens1／17。 | 原门槛通过；未据此宣称 TP通信、完整注意力或整层已验收。 |
| MLA epilog：6个 NPU进程 | TP8 rank0，官网L43的 `w_o`／`w_o_absorbed`；原 prefill owner 的 tokens1／17，以及独立 decode owner 的 tokens1。 | 原 `atol=1e-4, rtol=1e-3, max_error_ratio=0.005` 不变；155,648个保存的 FP32值按原容差复核零超差。输入仍是原 seeded owner 夹具，不是整网 attention 捕获。 |
| 最终三文件完整生产图 | TP8/EP8、TP16/EP16，capacity128／context512；每种拓扑的6个 production program。 | 全12个 CPU codegen／PTOAS、实际 native binary build／assembly及新 full6 manifest公共重载通过；未创建 Worker、提交 NPU或加载权重，不代替设备整网精度或性能。正常复用源匹配的编译缓存，没有旧 manifest覆盖。 |

Gate／额外 linear 的初次 R1 将 ScalarSpec 错存为 Python int，12个 Gate进程在编译和 NPU前被拒绝，**不是候选精度失败**；该失败保留。R2仅将6份标量编码改成0-dim INT32 tensor，数值仍为1／17，其余输入和全部 golden输出文件字节不变，再完成上述 Gate验证。各调用者均核对实际 alias identity 和生成 PTO 的 Kstep128，未因提前 import误测原K1。

**正式集成状态：**仅更新必要注释／docstring后，去除 docstring 的 AST与实测候选一致；golden、spec、CLI和接口保持不变。正式文件 SHA256前缀为 `linear.py: 277ad87fc7af`、`kda_projection.py: 3accbf036ac2`、`mla_epilog.py: 54902c8c6a65`，完整哈希及 AST证明在 Lib 的 `build_output/glm53-pypto-refresh-20261027/final-integration/{FINAL_SHA256.json,NORMALIZED_AST_PROOF.json}`。两仓 pre-commit 已重新通过。

本轮组件／调用者证据分别位于同一 refresh目录下的 `perf-linear-candidates/`、`perf-mhc-callers/`、`perf-projection-callers/r3-remote/` 和 `full-production-candidate-compile/`。这些是原 owner数值门槛及生产图构建证据，**不是独立 CPU golden对45层整网的验收**，也不增加历史已采纳计数。

**R3当前源码验收：TP8／TP16均通过。** 两次运行分别完成16个必要HTTP响应案例及一次断连检查，第二独立审计均通过。部署用250个Python文件与正式树一致，实际导入源码前后哈希未变；实际编译产物核实48处Kloop均为step128，KDA和MLA epilog列tile均为128。每种拓扑执行168次完整45层forward、14,952次主干程序调用及168次NPU LM-head辅助调用；没有旧manifest覆盖或CPU前向替代。

| 验收项 | TP8／TP16结果与限定 |
|---|---|
| HTTP／SSE | raw8 JSON及SSE、raw64完整输出、usage、结束原因和单个`[DONE]`等协议检查通过。raw64按指定长度结束，不冒称自然EOS。 |
| 完整FP32 logits | 每拓扑重载64个154,880维向量，共9,912,320个FP32值，原始字节hash及argmax与采样时记录一致；128／160两对cold-hit各8步完整向量和生成IDs一致。这是各拓扑内部回归，不是跨TP逐字节对照或独立CPU golden。 |
| 完整／部分前缀 | 精确128-token命中query=0，不增加主干／LM-head调用，仍执行设备reset／六族状态restore；160-token命中128后只补32个query token。 |
| 真实并发 | 有同一次native forward处理4个活动请求的记录，并非仅同时发起4个HTTP连接。 |
| 断连与后续请求 | 原30秒窗口通过；从更早首个Engine交付token到native release的保守上界，TP8为1.252438秒、TP16为2.601491秒。没有记录精确client-close时刻，不能称精确取消延迟；随后请求正常完成。 |
| 自然EOS及答案 | EN答案`5`、ZH答案`巴黎`，均以官方EOS154827结束，实际3个token，非长度上限截断。HTTP保留`</think>`；完整token串末尾EOS解码为`<|user|>`。仅两道受控smoke，不是广泛语言质量验收。 |
| 关闭与资源 | 受控graceful退出、Runner／registry各关闭一次，源码哈希一致；最终全部16卡健康空闲、无物理卡FD、无wrapper／controller残留，全局锁释放。设备日志中`Init: wrote`的ERROR级记录已按成功初始化分类，未发现其他ERROR／FATAL。 |

**保留的生命周期观察：**两拓扑的每条自然EOS chat之后，都还有一次已在途decode及采样未交付；公开输出IDs／usage仍为原3个token。该尾部工作原样保留，不能写成EOS之后零计算或吞吐收益。

原1200秒客户端预算、30秒断连释放窗口、strict EOS及源码检查不变。TP8／TP16的独立审计分别见 Lib 的 `build_output/glm53-pypto-refresh-20261027/TP8_R3_INDEPENDENT_AUDIT.json`、`TP16_R3_INDEPENDENT_AUDIT.json`，原始证据在同目录 `tp8-r3-http-audit/`、`tp16-r3-http-audit/`。旧R2的约18.5秒／forward及全部失败报告原样保留；当前功能通过不改变opt38数值参考待确认状态，也不增加21项历史性能试验／12项采纳。

CPU 证据位于 Lib 的 `build_output/glm53-checkpoint-preflight-{inline.log,real.json,coverage.json,precommit.log}`；最后全量静态检查另存 `build_output/glm53-checkpoint-preflight-final-precommit.log`。本机失败报告位于 Serving 的 `build_output/glm53-skill-acceptance-20261027/native-tp8-r1/`；197 的原始报告、源码前后哈希和退出码经 SHA256 校验取回同层 `evidence-197-r1/`，另存少量编译定位摘录与退出后设备查询，记录见 `EVIDENCE_AUDIT.json`。本轮最终 Serving 静态检查保存在 case 内 `final-precommit.log`。这些是正确性维护和接入验收，**不增加 21 项历史性能试验计数、不声明新整网提速，也不改变 opt38 的数值参考待确认状态**；历史 TP8／TP16 缓存验收仍只对应原冻结版本。

## 当前性能

**修复前 R2 的回退记录（`03d5190` 基础、K1）：**官方新 stack 下，`model.forward` 主机调用稳定约 **18.5秒／次**，采样约 **2ms**；20分钟客户端预算内只完成 raw8 JSON／SSE，raw64 未完成。该区间包含设备执行、通信和运行时等待，不能直接写成纯 NPU 算力耗时，也不是标准 Serving TPS 对照。K128及两处必要tile适配现已集成，TP8／TP16 R3的HTTP功能验收通过；**正式匹配性能对照尚未完成**。不能用组件16.0748→0.0988 ms，或跨主机／不同边界的R2与R3窗口，推算整网增益。**下列已采纳及opt38性能仍属于各自冻结历史快照。**

R3保留的raw64连续63个单token诊断窗口，中位数为TP8 **0.620699秒**、TP16 **1.291511秒**；全部query=1窗口的中位数分别为0.619779／1.302165秒。窗口含Runner／native到Session返回，包含相应设备、通信和主机等待，不是纯NPU计算或完整客户端TPOT／TPS；本轮TP16窗口更慢，全部观察保留。它们不是同硬件优化前后配对benchmark，不据此声明约30倍整网提速。原始口径见同refresh目录的 `TP8_TP16_R3_OBSERVER_DIAGNOSTICS.json`。

表中吞吐均为所有请求合计的输出 token/s。1、2、4 表示并发请求数，不是卡数。

| 指标 | 含义 |
|---|---|
| 首字延迟 TTFT | 客户端提交请求到收到第一个 token 的时间。 |
| token 间隔 | 同一请求相邻输出 token 的时间间隔；包含 Serving、CPU 输出处理等开销。 |
| 总吞吐 | 所有请求合计输出的 token 数除以请求组耗时。 |

### 新优化分支：TP8 主机时间分项

`feat/next_glm53_optimize` 首次基线采样已完成：同一台197、TP8／EP8／DP1、CAP128／context512／chunk32、NPU LM-head、prefix关闭，使用已通过R3验收的Lib源码和官方 `ce0eb7c` stack。一次真实CLI请求生成26个token：1次prefill、25次decode，前5次decode预热，保留后20次；`PYPTO_BENCH=0`，没有状态重放或swimlane双执行。下表是加入低侵入观测器后的主机API时间窗，不是纯设备时间或正式Serving吞吐。

| 分项 | 20步中位数 | 逐步占比中位数 |
|---|---:|---:|
| 整步forward | 627.913 ms | 100% |
| 输入上传，112次／9,078,336B | 83.047 ms | 13.27% |
| 89次主干提交与等待 | 505.914 ms | 80.83% |
| 其中：submit入队 | 235.115 ms | 37.56% |
| 其中：result等待 | 272.559 ms | 42.94% |
| 独立NPU LM-head | 1.380 ms | 0.22% |
| logits读回 | 16.425 ms | 2.60% |
| head_hidden读回 | 8.154 ms | 1.31% |
| 输入准备 | 2.009 ms | 0.32% |
| 有限值检查调用／拼接／其余CPU及观测器残余 | 1.129／0.235／7.461 ms | 0.18%／0.04%／1.16% |

父子API窗口已排除重复计数；各项中位数独立计算，不能要求它们精确相加。`submit`包含参数校验、调度和入队，`result`包含设备、IPC及结果发布等待，不能将任一项直接写成可全部消除的纯固定成本。主干按句柄归属的窗口以MoE最大，其次KDA，但这也不是独立算子device wall。

**优先级据此调整：**先研究减少主干提交及其相关等待；输入合并仍有价值，但TP8即使完全消除上传，其整步收益空间也只有约13%，不能再将它作为主要慢因或承诺数倍加速。独立NPU LM-head当前约0.2%，不列本轮高收益首项。TP16输入合并后的相同分项也已完成，数据列于下一节；不能直接用TP8占比代替TP16归因。

26个生成ID与R3相同prompt的前26个逐个相同，源码前后不变；Runner清空、退出、全16卡空闲、无设备FD和全局锁释放均独立核验。原始数据及离线分析器见本仓 `build_output/glm53-next-optimize/host-profile/tp8-r1-raw/`；独立结论为同级 `TP8_R1_HOST_ATTRIBUTION_INDEPENDENT.json`（SHA256 `b00dff54d3d49a54152f2d41630a4e9fc1445ffecfddcf3bf222925f168e88ad`）。本次不改变历史21项试验或12项采纳计数。

### 输入上传优化：TP8／TP16 整网验证通过

仅修改Lib `prefill_fwd.py` 的主机端：prepare前注册两块持久共享CPU缓冲，BF16 embedding一块、13个INT32元数据合为一块；同dtype的设备view保持原参数shape／dtype。每步填入新输入后，每rank同步复制两块owner，避免逐字段创建和释放暂存buffer。INT32按64B对齐，每rank仅增加120B。其余249个Lib Python文件与R3逐字节相同，51个原顶层计算／Golden函数及global／import AST完全相同。

同197、TP8／EP8／DP1、同raw26及预热5／观察20窗口、prefix OFF和NPU head，对照上述基线。Serving采用同步后的 `5d200bc`；相较基线唯一runtime文件差异是CLI统一prefix参数，本次实际配置仍为关闭，前向实现未变。

| 同口径主机窗口 | 基线 | 输入合并后 |
|---|---:|---:|
| 整步forward中位数 | 627.913 ms | **572.506 ms，降低8.82%** |
| 上传中位数 | 83.047 ms | **2.504 ms** |
| 每步H2D调用数／字节数 | 112／9,078,336B | 16／9,079,296B |
| BF16上传，均为8次复制 | 15.602 ms | 1.795 ms |
| INT32上传 | 67.402 ms，104次 | 0.707 ms，8次 |
| 89次主干提交与等待中位数 | 505.914 ms | 531.188 ms |
| 其中submit入队中位数 | 235.115 ms | 259.462 ms |
| 其中result等待中位数 | 272.559 ms | 275.035 ms |

上传节省约80.54ms，但主干等窗口有所增加，**整步实际只减少55.41ms**；没有将上传节省冒充整网收益。每项中位数独立计算，不能直接相加。候选20步范围544.290–602.509ms；这是一次带观测器的host窗口对照，不是正式Serving吞吐、客户端TPOT或纯设备时间。submit增加的原因尚待独立归因，不能仅据此认定是view校验。

同一次真实模型加载还完成四请求回归：共58个完整FP32向量／8,983,040元素逐一回读、finite／argmax／全byte SHA对照R3全部通过；raw26的旧参考只保留指纹，不能称两份原向量literal memcmp。40次实际forward均为16次上传、89次主干提交、1次独立head及2次完整D2H；出现5次真实四请求native cohort，slots为互异的 `[0,1,2,3]`，不等于验证16请求。额外并发回归在raw26计时之后，不混入20个性能窗口。

433源码／25个harness／10个控制文件前后hash保持；正常关闭、12个观察进程全部消失、全16卡idle／无设备FD／全局锁释放均通过独立复核。实际执行公开`alloc_tensor/view_tensor`路径和两个owner目标有证据，但没有逐地址dump native owner registry。冻结源为Lib `c13033c`，harness READY `61233277`、SOURCE READY `bd4db6e6`、CONTROL READY `328d1ccd`。记录在本仓 `build_output/glm53-next-optimize/input-slabs-profile/input-slabs-tp8-r1-audit/`，主审 `FINAL_AUDIT.json` SHA256 `e071ad3287187478067c499787aaf003098662892dc0ffadbb741db4d16ec75c`，独立审计 `INDEPENDENT_AUDIT.json` SHA256 `6006a926924f2c559ecb3946f7a5334a8c5228fcb5b969c07f421d5322cce758`。

TP16同源码回归也已完成。20步decode当前主机窗口中位数为1062.945ms，上传5.257ms、主干961.762ms（submit592.106ms、result等待368.487ms），独立head2.085ms、logits／hidden读回33.789／22.577ms。没有匹配的新TP16基线，**不使用R3旧1.29秒诊断值计算提升百分比**；同一实现的TP8和TP16数字也不是优化前后A/B。

TP16的58个完整FP32向量均对其自己的R3参考逐行核验通过，未混用TP8参考；40次实际forward每步32次上传、18,158,592B，主干89次／head1次／完整D2H2次，5个四请求cohort的slots为 `[0,1,2,3]`。433源码／18个harness／12个控制文件前后一致；19个观察进程全部结束、无强杀、正常关闭及全16卡idle／无FD／锁释放均独立通过。记录见本仓 `build_output/glm53-next-optimize/input-slabs-profile-tp16/input-slabs-tp16-r1-audit/INDEPENDENT_AUDIT.json`（SHA256 `68fdb52316da26172867964a1e1dab5c3d3bf3577c0bb6442d76ebd81fb00d1c`）。

当前保留已验证的输入上传实现；本轮尚未提交或推送新性能分支，也不改变历史21项试验／12项采纳计数。

### 固定16卡、C2的当前部署对照

197、当前输入合并实现、同一官方stack和发布权重；prefix OFF，NPU head，每请求11个输入token／8个输出token，总并发2。每种配置5组预热＋20组计时；正式320个accepted token除以从第一组提交到最后一组两请求释放／排空的完整client wall，含组间等待。下表包含HTTP／调度／前向／采样及排空，不含启动、预热和关闭；不是单步host窗口。

| 部署 | 正式完整wall | 总吞吐 | TTFT中位数 | 请求HTTP延迟中位数 |
|---|---:|---:|---:|---:|
| 单TP16／EP16，native C2 | 166.069441 s | 1.926905 token/s | 1.058365 s | 8.054267 s |
| 两个独立TP8／EP8，各C1 | 124.008891 s | **2.580460 token/s** | 0.806077 s | 6.105944 s |

双TP8总吞吐提高**33.92%**；两端口、两份模型，并非单入口DP2。每实例host线程配置16，双实例合计32；这是固定16卡的整体部署对照，不能将差异全归因TP通信，也不证明它优于单TP8 native C2。下一优先项是补单TP8的C2／C4卡效率对照；不跨输入长度或拓扑与历史表计算增益。

两配置均完成50请求／400个完整154880维FP32指纹（双实例各200行），逐一对照各自拓扑R3参考；三个backend各首请求8个完整向量另行读回通过。全部25组的实际native cohort符合配置，无未交付的额外sample；不把指纹回归称为独立CPU-golden整网验收。r2整体exit0，主审和第二独立审计均通过。433源码／13个harness／23个控制文件及stack前后一致；三服务均受控退出、无强杀，912个观察进程全部结束，16卡空闲、无设备FD、锁释放；HBM对本次初始值最大仅+1MiB。128份Ascend日志中，32条ERROR级均为已知初始化成功行，其余ERROR为0，原warnings保留。证据在本仓 `build_output/glm53-next-optimize/two-tp8-throughput/paired-c2-r2-audit/`，主审 `FINAL_AUDIT.json` SHA256 `8822299aff1e02d2b3918d07db8aaef3e1344db3916076dd438d4eaf06f7573d`，第二审计 `INDEPENDENT_AUDIT.json` SHA256 `ccf0a50d20b1b256ab1c767f0b43681e75a1109ee3648df916aa16d35d8521b3`。r1采集错误及原始失败完整保留，未改写成通过。

### 历史已采纳基线：TP8、160输入／8输出

同一 A3 主机、TP8/EP8、160 个输入 token、8 个输出 token、上下文上限 512、每步 token 预算 128、chunk 32、prefix cache 关闭、owner 的 PyTorch 线程设置为 16。

线程数表示 PyTorch 的配置值，不能据此认定后端每次计算都实际使用了相同数量的物理核。

| 并发请求数 | 首字延迟 | 平均 token 间隔的中位数 | 总吞吐 |
|---|---:|---:|---:|
| 1 | 2.571 秒 | 0.504 秒/token | **1.306 token/s** |
| 2 | 2.636 秒 | 0.502 秒/token | **2.605 token/s** |
| 4 | 2.774 秒 | 0.511 秒/token | **5.057 token/s** |

每档并发预热 1 组、计时 3 组，总计 12 组，其中 3 组预热、9 组计时；表中只汇总计时组。

对应 opt33：CPU LM-head 从计算 32 行改为计算调用方返回的 16 行。历史同负载对照的吞吐为 0.760 / 1.498 / 3.500 token/s，观察到的增幅为 71.79% / 73.85% / 44.49%。主要耗时变化出现在 CPU head；历史对照未与本次交错采样，CPU 波动也有影响，因此不将全部差值归因为该改动，不声称统计显著性。

这是最新已采纳的 **160 输入 / 8 输出、prefix 关闭**基线。opt31 使用新的 128 输入精确命中负载，尚无完整开关性能结论，不替换本表。

### 已采纳的异步下发和前缀缓存收益

| 优化 | 独立对照条件 | 4 并发总吞吐 | 观察到的变化 |
|---|---|---:|---:|
| opt13 有界异步设备下发 | 同一主机，128 输入 / 8 输出，基于 shared NZ | 4.077 → 4.910 token/s | +20.43% |
| opt15 前缀缓存 | 同一主机，160 输入 / 8 输出，共享 128-token 前缀，单独的开关对照 | 3.500 → 4.682 token/s | +33.77% |

opt15 的收益主要体现在重复前缀请求的首字时间；2、4 并发的 token 间隔反而增加，因此不将它描述为 decode 提速。这两项与 opt33 的测量条件、实现阶段不同，收益不能相乘或直接相加。

原机器、128 输入 / 8 输出的早期累计系列，从最初 0.135 / 0.277 / 0.523 token/s 提升到 opt09 的 1.054 / 2.094 / 4.081 token/s，约为 7.55～7.80 倍。该系列与另一主机的 160-input 系列分别保留。

### opt38：PTO3/NPU LM-head 的当前验证进展

本项将最终 RMSNorm 和完整词表投影移到 PTO3/NPU，保持归一化中间值、矩阵乘输入与累加、最终 logits 为 FP32，没有增加 BF16 中间舍入。各 rank 按词表切分；每卡权重在 TP8/TP16 下分别增加 303/152 MiB。词表分片补零到 64 行边界，读回后逐分片去掉 padding，再向 Serving 返回完整 154,880 维 FP32 logits。Embedding 仍在 CPU，通用采样仍由 Serving 负责；原完整 32 行、所有 rank 的读回和有限值检查保留。

CPU 参考可用 `GLM53_LM_HEAD_BACKEND=cpu`，或构造 Session 时传入 `lm_head_backend="cpu"`。NPU 后端在同一驻留 worker 上注册辅助程序，原六个主干程序和布局版本 3 不变。

| 验证项 | 当前结果 |
|---|---|
| TP8、TP16 辅助程序 | 当前 padding 实现的 host/PTOAS 编译均通过。 |
| TP8 官网权重组件 | 16 行 × 154,880，共 **2,478,080 个 logits**，原逐点 `rtol=atol=1e-5` 全部通过，16 行 top-1 一致；覆盖普通、零、大幅值和重复输入。 |
| 词表尾部 | 初版 3,072 个尾部值错误，非尾部零失败；改用完整 64 列 tile 和补零的分片权重后，上述严格对照零失败。保留失败记录，不将其当作普通 FP32 舍入。 |
| 组件耗时 | 197 A3、TP8、两次预热／十次计时、CPU/NPU AB/BA 交错；NPU 辅助调用＋完整 logits 读回＋拼接中位数 **20.34 ms**。同轮 CPU RMSNorm＋GEMM 为 606.29 ms；CPU 波动较大，组件数据不换算成整网或 Serving 收益。 |
| TP8 独立整网对照 | **56 次完整 forward 已完成**，只加载一次权重，CPU/NPU 各自独立 reset 后执行同输入；全部 rank 的头部输入相同，连续 8 个 token 一致。32 行／4,956,160 个 logits 中 3 个未过旧 CPU 逐点门槛，原报告保留 `passed=false`。 |
| 独立 FP64 对照 | 对上述 32 行重新以 FP64 计算最终归一化和点积，NPU 对同样的逐点 `1e-5` 门槛 **零失败**，CPU 有 1 点失败；固定 CPU FP32 归一化后仅以 FP64 做点积，也得到相同计数。原 CPU 门槛未自动替换，已请用户确认采用哪种参考。 |
| 旧冻结随机夹具 | 本机 TP8 任务 `task_20261026_021816_93030626708` 已重放原 32 行／4,956,160 值：与旧 CPU 对照 7 点未过，与独立 FP64 对照零失败、32 个 top-1 一致；旧 CPU 自身对 FP64 有 4 点失败。原冻结文件未修改。 |
| TP16 设备组件 | **197 的 16 个逻辑设备实测完成。** 新 16 行组件对原 CPU 全点通过；旧冻结 32 行与 CPU 对照 7 点未过，与两种 FP64 参考均零失败。原本机排队任务已取消，未留后台排队任务。 |
| Serving | **真实 Engine 12 组全部完成**，80 个完整 FP32 向量／12,390,400 个 logits 与旧 CPU 比较有 11 点超差，全部 top-1、采样和生成历史一致，取消与槽复用协议通过；保留 `precision_accepted=false`。没有声称这 80 行已独立重算 FP64。 |

完整 forward 的同进程交错对照如下。每种负载、每种后端各一次预热，随后三组固定 AB/BA 顺序，共六次计时；重置请求、首次 CPU FP32 缓存转换、保存和比较均在计时外。指标包含实际主干、头部执行、完整 logits 读回及返回，不是客户端 token/s。原 CPU 逐点门槛未全部通过，因此暂记为诊断数据，尚未采纳。

| 同轮负载 | CPU head 完整 forward | NPU head 完整 forward | 耗时变化 |
|---|---:|---:|---:|
| 1 请求，各 32 个 query token | 511.60 ms | 341.56 ms | −33.24% |
| 4 请求，各 32 个 query token | 558.19 ms | 384.78 ms | −31.07% |

本机 TP8 的另一份组件对照为 CPU 166.41 ms、NPU 21.53 ms，同样使用两次预热、十次 AB/BA 计时并保留全部样本。它与 197 的 606.29／20.34 ms 分开保存，进一步表明 CPU 头部存在主机和运行条件波动，不能把组件比值直接当作 Serving 提速。

**Serving 实测：**与 opt33 的同机历史控制比较，保持 TP8/EP8、160 输入／8 输出、context512、budget128、chunk32、prefix off、owner Torch16；每档一次预热、三次计时。CPU 控制未交错重测，因此这里记录观察值，不作统计显著性或独立因果百分比声明。

| 并发 | 原 CPU-head 吞吐 | NPU-head 吞吐 | 观察到的变化 | NPU 首字延迟 | NPU token 间隔 |
|---|---:|---:|---:|---:|---:|
| 1 | 1.306 token/s | **1.813 token/s** | +38.86% | 1.866 s | 0.338 s |
| 2 | 2.605 token/s | **3.689 token/s** | +41.65% | 1.841 s | 0.340 s |
| 4 | 5.057 token/s | **5.199 token/s** | +2.81% | 2.366 s | 0.343 s |

4 并发三轮分别为 **7.352／3.417／6.740 token/s**，没有剔除慢轮或重测替换。第二轮比第一轮多 5.012 s，5.011 s 位于 Session 内：head 前区间多 3.575 s、NPU 最终归一化和投影调用多 0.165 s、head 后到返回多 1.270 s。Session 外调度／IPC／采样／交付余量仅多约 1.1 ms，四个请求同步变慢，队列等待不足 2.5 ms。现有记录不能继续区分具体主干 kernel、系统调度、copy、拼接或有限值检查，不能把它直接归因于某个算子。

实际续写仍为 ` Paris. The capital of France is Paris`，8 个 token 与控制一致；随机采样、并发、取消后槽复用的既有历史也相同。本项实测结束时，模型源码仅更正文档和注释，除文档字符串外的 AST 与实测版本相同，证明保存在 `final-source-equivalence.json`；后续主干同步及主机计时字段恢复见本文开头的补测记录。代码已随整合提交推送至 Lib #1553 的来源分支。

本项与历史 `trial05_gpu_head` 分开记录。尚未把本地候选计入已采纳项目，也未以 top-1 相同代替完整 logits 的逐点检查。实验入口、失败及成功数据保存在 Lib 的 `build_output/glm53-perf-iterations/npu-head-opt38/`；197 执行目录为同层的 `npu-head-opt38-197/`。

### opt39：历史性能记录的进一步分析

下列为opt38冻结样本的历史归因，保留原始结论；当前分项已在上面的“新优化分支”实测，输入上传也已优化。此处“下一次测量”不表示仍未执行。

本轮只分析 opt38 已保存的 9 个正式计时组，共 108 次完整 forward（45 次 prefill、63 次 decode）；保留全部慢样本，没有修改模型、重新运行 NPU 或改变数值门槛。原始客户端组耗时合计 44.709 s，按互不重叠的主机时间区间分解如下。

| 区间 | 累计耗时 | 占客户端总耗时 |
|---|---:|---:|
| Session 内准备输入，含一次 Embedding 查表 | 0.381 s | 0.85% |
| 准备完成至 LM-head 设备调用开始 | **39.507 s** | **88.36%** |
| NPU 最终 RMSNorm＋LM-head 调用 | 0.289 s | 0.65% |
| LM-head 完成至 Session 返回 | **4.010 s** | **8.97%** |
| Session 外及未单列边界 | 0.522 s | 1.17% |

63 次 decode 的完整 forward 中位数为 336.66 ms，准备输入约 1.68 ms、head 前约 308.56 ms、设备 head 约 1.18 ms、head 后约 25.26 ms；各区间分别取中位数，不能要求其精确相加。设备 head 的 1 ms 与此前约 20 ms 的组件数值范围不同：组件包含 logits 读回与拼接；这里的 head 后还含完整 hidden 读回和有限值检查。

**首要未分解区间：**head 前包含主机向 8 个 rank 上传输入、构造调用参数、89 次主干提交及完成等待，以及期间的设备计算、通信和运行时调度。源码每步有 14 个动态输入、112 次 `copy_to`，合计 9,078,336 字节；最多保留两个待完成句柄。调用次数和字节数已确认，但没有逐项计时，不能把这 88.36% 写成纯 NPU 计算、通信，或认定全是 Python 下发。

**次要未分解区间：**head 后包括两类设备读回、主机拼接、有限值检查和活动请求取行。两个读回各涉及 8 个 shard，padded logits 为 9,928,704 字节、完整 32 行 hidden 为 2,097,152 字节；拼接后的完整 logits 为 9,912,320 字节。现有约 25 ms 不能直接当作 PCIe 拷贝时间，必须把 copy、拼接和检查分开测量。

**波动跨并发组：**第二轮 C4 比第一轮多 5.012 s 后，第三轮 C1/C2/C4 相比各自第一轮仍分别多 537/328/395 ms，增加部分几乎都落在 head 前。较慢调用跨约 20 s 出现，中间夹有正常调用，尚不能证明持续停顿或同一根因。所有正式组队列等待最大仅 3.36 ms，现有记录不支持把秒级差异归因于 Serving 排队。

**已确认但收益较小的重复工作：**Serving Worker 在 prefill/decode 打包时先查表生成 `input_embeddings`／`hidden_states`，GLM Runner 只消费 token IDs，随后 Session 再查表。日志中的 108 次外部多余查表累计 243.15 ms，平均每步 2.25 ms、占本轮客户端总耗时 0.54%。这不是主瓶颈；对应既有 opt30，应在主干与回收测量之后处理，不以虚假的设备 Embedding 能力标志绕过接口。

下一次测量先记录每类上传、每个主干程序的提交与 `handle.result()` 等待，以及两类读回、拼接和检查；主机等待时间仍需与单次执行的设备时间线、系统调度记录交叉核对。保留两句柄上限、相同输入与检查，先定位再改算法或调度。当前 resident Session 拒绝会重放缓存的两遍 swimlane 流程，不能直接开启该流程采集真实增量请求。

完整分析与原始输入 SHA 保存于 Lib 的 `build_output/glm53-perf-iterations/npu-head-opt38/performance-analysis-20261026/{stall-analysis.json,path-analysis.json}`。

## 接下来怎么做

新一轮在 `feat/next_glm53_optimize` 分支推进，按预期整网收益调整顺序。先取得分项耗时，再估算收益；调用数或通信字节数下降不直接等于整网提速。下列候选逐项实现、逐项验证，未验收的改动不计入已采纳数量。

| 优先级 | 优化项 | 预期收益和验收重点 | 当前状态 |
|---|---|---|---|
| P0 | 当前主干耗时归因 | TP8基线已测整步627.913ms，主干80.83%、上传13.27%、独立NPU LM-head0.22%；TP16输入合并后的当前整步1062.945ms、主干961.762ms，未采匹配旧基线。只记主机窗口，不冒称纯设备耗时；状态相关forward不重复回放。 | TP8基线及TP8／TP16输入合并后的分项均完成，原始数据和独立审计已保留。 |
| P1，工具链根因已定位，待修复 | 合并主干设备提交或同步（opt18） | submit约占整步37.56%，result等待约42.94%；优先研究89次主干提交降至47次，保持缓存写入、依赖和请求槽位语义。两个窗口不能直接视为全部可消除成本。 | PyPTO `ExpandMixedKernel`复制参数时遗漏动态shape引用的重映射，已提[issue #3203](https://github.com/hw-native-sys/pypto/issues/3203)。31行独立复现及静态对照完成；KDA／DSA默认融合编译仍有78／12个Error104，尚未修复或进入NPU计时。 |
| P1，下一可执行项 | 单TP8的C2／C4卡效率对照 | 先确认native batch能否用8卡接近或超过双实例吞吐，再决定是否做单入口多副本。当前没有相同NPU-head基线的正式结果，不预设收益。 | 待实施；另建匹配协议，保留同卡数和每卡吞吐两种口径，不临时扩大本轮负载。 |
| P1，部署对照已测 | 固定16卡：单TP16与两个独立TP8实例 | 相同总并发C2下，正式吞吐1.926905→2.580460 token/s，**提高33.92%**。两个TP8各用8卡／EP8／DP1及独立端口，模型权重驻留增加。 | r2整体exit0，两配置各320个计时accepted token、400行完整FP32指纹通过；主审、第二独立审计及最终资源复查全部通过。当前GLM单入口DP2仍未实现，不把本项当作生产多副本功能已完成。 |
| P1，已完成本轮验证 | 合并小输入上传并复用共享暂存 | 两块同dtype缓冲将14N次上传降为2N，保持各输入dtype、布局、生命周期和精确内容。TP8主机窗口降低8.82%；TP16仅报告当前1062.94ms，不算提升比例。 | TP8／TP16整网和独立审计通过；每拓扑58个完整FP32向量及4请求回归一致。当前生产源只保留这一项新优化，未混入parked候选。 |
| P2，暂缓整网 | MoE唯一owner行恢复 | 独立NPU正确性和计时已完成，但小token绝对收益较小；不能用87.5%／93.75%的理论payload读取减少代替整网提速。同步协议完整保留，独立decode的owner仍须恒0。 | TP8／TP16的40轮连续改输入、2个通用归约及16组各105次调用全部通过；完整补丁和原始数据保留，生产源已恢复，不与输入上传候选混测。 |
| P3，暂缓整网 | EOS后的排队decode剪除 | 主要减少短请求结束后的占卡及对下一请求的阻塞，不改善普通长decode的单步计算。保持Engine深度2，并正确处理ignore_eos、min_tokens、混批及复用。 | CPU目标157项通过，原基线全unit836项覆盖通过；同步5d200b后CLI定向67项通过。完整补丁已保留，生产源已恢复；尚未NPU或性能验收，暂不采纳。 |

上传或通信项若实测占比小，应下调优先级。先完成一个高收益点的相同输入、相同配置A/B及原有正确性门槛，再推进下一项；EOS候选保留待验，不抢占主干大收益项的设备测量顺序。

**卡效率的后续对照：**当前双TP8／单TP16试验只回答“固定16卡、C2时两种部署谁更快”。它不证明双TP8优于只用8卡的单TP8：单实例可通过native batch摊薄提交成本。下一项应在当前NPU-head／输入合并基线上补单TP8的C2／C4正式吞吐，再决定是否优先实现单入口多副本；多副本的C4对照另建协议，不临时扩展本轮冻结试验。

当前NPU-head路径的独立head窗口仅约0.2%，因此CPU head后端（opt36）、CPU投影活跃行（opt37）和围绕旧CPU head设计的输出重叠（opt14B）暂缓。opt31的完整前缀复用主要服务重复prompt的首字延迟，按命中负载单独验收；它不能提高普通未命中的decode速度。Indexer／attention候选先根据当前实际时间占比评估，不将旧组件百分比直接作为整网预期。

opt18本轮复验使用原探针和旧Lib `2f8c054` 的只读源快照，将模型后续NZ／K128改动与工具链效果隔离；4案均无Worker或设备执行。源码哈希、完整错误、两个控制的native产物及新stack记录见新Lib worktree的 `build_output/glm53-next-optimize/fusion-recheck/SUMMARY.json`。没有修改编译器、绕过验证或把失败写成可直接测试的融合实现。

为避免把历史接口限制直接套用当前模型，又从当前 `c13033c` 主机实现的实际AST提取NZ／K128 L2注解、production L3参数和新增scratch，分别生成KDA／DSA的host-pair及单L2 true-inline组合与同ABI单attention控制，共8案。4个控制均完成codegen、PTOAS和真实native构建；4个pair仍分别出现78／12条Error104（KDA的`prefill_kda.py` scratch_rows、DSA的`mla_epilog.py`动态维度等式），无Worker／NPU。证据见同级 `fusion-recheck/current-interface/CURRENT_INTERFACE_SUMMARY.json`（SHA256 `0292eb2e0e2840ef1ad1df3dd357b4e324b1b95c5cc99aeaf21fae22e4708f96`）。这说明当前测试的两种融合受阻，不等于所有融合方式都不可能。

**融合阻塞的依赖核对及更新（2026-10-28）：**按用户要求，runtime和ISA执行目标已改为各自最新主干快照。以下统一为北京时间（UTC+8），年份均为2026；Git组件使用提交时间，PTOAS使用release发布时间，不是环境安装时间。

先保留原NPU验收基线与PyPTO默认依赖的对照，**这张表的默认依赖不是本轮实际执行目标**：

| 组件 | 原NPU实测基线及日期 | PyPTO主干默认依赖及日期 | 新旧差距 |
|---|---|---|---|
| PyPTO | `ce0eb7c`，10-27 09:35 | `69dbfe2`，10-28 10:19 | 6笔提交，约25小时 |
| runtime／Simpler | `d8f997f`，10-20 19:44 | `531765c`，10-27 14:13 | 42笔提交，约7天 |
| pto-isa | `b1d460e`，10-19 15:45 | `dcebd22`，10-27 12:40 | 49笔提交，约8天 |
| PTOAS | `0.67`，10-08 10:35 | `0.70`，10-26 19:41 | 发布相差约18天 |

提交数包含合并和文档提交，不等于修复数量。此前`6307a2e`（10-27 21:04）仍指定PTOAS0.67；随后[PR #3171](https://github.com/hw-native-sys/pypto/pull/3171)于10-28 09:23合入，在PyPTO侧修正MX绑定兼容处理并改为指定0.70。[当前该快照的工具链配置](https://github.com/hw-native-sys/pypto/blob/69dbfe2cb20f36ebbb12cbab68f7a1f672c0e771/toolchain/versions.env)

**本轮实际构建及复测的环境：**

| 组件 | 使用版本 | 提交／发布时间（北京时间） |
|---|---|---|
| PyPTO | `69dbfe2` | 2026-10-28 10:19 |
| runtime／Simpler | `9c27de9` | 2026-10-28 10:59 |
| pto-isa | `1b54ac7` | 2026-10-28 10:17 |
| PTOAS | `0.70` | 2026-10-26 19:41（发布） |

三个源码组件均从独立worktree构建，未复制旧环境的项目二进制。runtime的唯一源码改动是将`pto_isa.pin`从`dcebd22`同步到`1b54ac7`。实际加载的runtime二进制版本为`9c27de9`，与源码HEAD一致；四种runtime构建元数据的ISA pin和实际checkout均为`1b54ac7`。PyPTO仍会提示runtime偏离其默认gitlink`531765c`，这是本次主动选用最新主干的结果；提示保留，版本匹配校验没有关闭。CANN9.0.0、Torch2.14.0+cpu保持原配置。

验证结果：

- **普通生产程序：TP8、TP16各7项，共14项通过codegen和真实native构建**，包含六个主干程序与独立NPU LM-head；两拓扑的完整六程序manifest重载也通过。826条二进制记录及其中420条ELF记录重新回读核验。
- **版本／元数据相关单测65项通过**：PyPTO runtime pin与host-control参数40项、runtime ISA管理25项；前一组保留1条既有`ir.compile(platform=...)`弃用告警。
- **融合问题仍在**：原50个模型源文件和4个probe保持不变，测试驱动AST只替换4个版本字符串。4个同ABI非融合对照通过codegen和native构建；hostpair／inline两种融合写法的KDA仍各有78条Error104，DSA仍各有12条，与旧栈一致。新增依赖没有解决该IR TypeCheck失败；依赖复测当时尚未隔离首个错误转换，后续定位见下文。
- 新环境Serving CLI安装及`--help`退出码0通过。激活入口为本机package目录下的`glm53-latest-deps-20261028/activate.sh`，CLI及Python实际指向新环境。
- **全部为CPU侧验证，没有创建NPU Worker、读取权重payload或执行设备推理。新栈的NPU功能、精度与性能尚未回归**；旧栈的1.93／2.58 token/s结果不能作为新栈结果。原验收环境的二进制指纹保持不变。

生产补测首轮误用了早期脚本的L2旁文件重载方式，TP8各程序codegen成功后被`compiled_meta.json`缺失拦住；该失败原件保留在`production-compile-r1`，没有改写成通过。修正脚本为实际分布式native组装路径后，`production-compile-r2`两拓扑14项通过，模型／golden／native成功要求未变。

完整证据位于新Lib worktree的`build_output/glm53-next-optimize/latest-deps-20261028/`，含8案融合复测、普通生产程序两轮原件及总汇`VALIDATION.json`（SHA256 `b8d2ba2cffd48bdb8f110289e34d808b5e8424e6f7f92bbb80095ac7aa7b2845`）。环境构建、单测、CLI及旧二进制指纹记录在本机package目录的`glm53-latest-deps-20261028/artifacts/`。本轮是依赖更新和编译兼容性验证，不增加21项历史性能试验或12项采纳计数。Runtime完整pre-commit尝试被无关Markdown hook的npm初始化错误`EALLOWGIT`拦住，未改动npm限制，也不报完整hook集合通过；`pto_isa.pin`相关的大文件、EOF及尾部空格三项检查和报告文件pre-commit均通过。

**融合编译根因已定位（2026-10-28）：**已提交[PyPTO issue #3203](https://github.com/hw-native-sys/pypto/issues/3203)。`ExpandMixedKernel`拆分Cube／Vector函数时，新建了标量参数，却直接复用了Tensor参数原有类型；Tensor的shape仍引用旧标量，循环携带值的类型则被`DeepClone`改为引用新标量，因此名称相同、变量身份不同。31行独立复现中，变换前两者均引用变量12，变换后Tensor初值仍引用12，循环类型改为55；这里的编号仅用于该次运行内的身份对照。

| 逐pass检查的实际案例 | `ExpandMixedKernel`之前 | 该pass之后 | `SplitVectorKernel`之后 |
|---|---|---|---|
| DSA attention＋MoE | 0项TypeCheck错误 | 8项 | 12项 |
| DSA attention-only对照 | 0项 | 8项 | 12项 |
| KDA attention＋MoE | 0项 | 48项 | 78项 |

因此，先前attention-only默认编译通过不代表其全部中间IR正确。组合MoE后存在被选中的Buffer kernel，`LowerTileToBuffer`末尾的整程序TypeCheck才把较早产生的坏IR暴露出来；不是MoE数值计算或EP通信造成shape变化。新增逐pass检查只观察并记录，没有关闭原有校验或改动模型。

独立动态shape复现在旧的完整pin组合`ce0eb7c/d8f997f/b1d460e/0.67`和新组合`69dbfe2/9c27de9/1b54ac7/0.70`均出现2项Error104；仅把Tensor的动态行数改为固定32的对照在两环境均通过。这已排除本次主动更新runtime／ISA才引入问题的解释。完整源码、错误、变量身份和逐pass证据位于新Lib的`build_output/glm53-next-optimize/latest-deps-20261028/error104-investigation/ROOT_CAUSE.json`；issue正文包含可独立运行的全部复现源码，不依赖本机文件或模型权重。

当前状态是**已定位并提issue，尚未修复编译器**。后续需统一重映射参数类型中的动态维度，补充回归，再重跑融合编译和NPU正确性／性能；不按变量名字放宽检查，也不把CPU定位算成性能收益。原生产模型与编译器源码均未因本次定位修改。

**MoE owner恢复的独立结果：**同197、同一份冻结输入和原全rank求和Golden，TP8／TP16各4种token规模、两种模式，每组5次预热及100次采样。一次被测dispatch包含两次collective；单token的主机调用中位数，TP8从2.263ms降至2.086ms，TP16从3.671ms降至3.279ms。该差值不能直接外推整网；TP16还有约50ms长尾，单token主机均值反而上升，全部样本保留。40轮连续epoch改输入、空行／尾块／重写／正负零的原始BF16字节和canary均通过；16组共1680次实际性能调用逐一核验，另两次通用False模式独立验证原门槛。源码前后不变，正常退出并释放全16卡及全局锁。

本项暂缓整网集成；83ms上传段优化已完成，下一步优先验证固定16卡的双TP8吞吐。不增加历史采纳计数。完整验证记录见新Lib `build_output/glm53-next-optimize/owner-restore/operator-ab-r1-audit/FINAL_AUDIT.json`（SHA256 `7f6364d8745e4202a3a1816d88153f6a8bdc2e4512eab4b183d8a91ab3db6289`），候选为同级 `owner-restore/MODEL_ONLY.patch`；EOS补丁见本仓 `build_output/glm53-next-optimize/parked-eos/PATCH.diff`。这些候选均保留可恢复的完整源和证据，没有混入下一项的实际生产测量。

首轮未产生任何计时数据。独立重放已证明，真实格式 `| NPU Chip | PID | ... |` 被旧解析器漏读，而原CPU夹具误把PID写在同一列；这是夹具和采集错误，不能据此判定实际绑卡失败。`/dev/davinciN` FD为空的具体原因未捕获，不猜测驱动行为；真实npu-smi进程、精确历史身份和worker设备日志作为后续核验依据。退出后HBM回收需有界等待，仍保留原始阈值，不能抬门槛让试验通过。证据为本仓 `build_output/glm53-next-optimize/two-tp8-throughput/`，首轮fresh资源审计 `FRESH_FINAL_RESOURCES.json` 及原失败报告继续保留。

双TP8属于部署配置研究，不宣称已支持单入口DP2：当前GLM CLI采用`overlapped`（单replica），kernel拓扑要求局部DP1、TP=EP。现有可行入口是两个独立CLI，分别`--tp 8 --ep 8 --dp 1 --devices 0,1,2,3,4,5,6,7`与`--devices 8,9,10,11,12,13,14,15`，端口各自独立；后8设备映射和两实例同时运行仍需实际检查。实验关闭prefix；以后增加路由时，前缀／会话亲和也需单独设计，不能将本次配置试验当作多副本缓存功能已验收。每实例host线程配置为16，双实例的配置总量为32，单TP16为16；本轮固定卡数和总并发，不固定CPU线程配置总量。结果属于整套部署方案对照，不能把全部差异单独归因为TP通信。

正式实现逐项验收；CPU 分析、组件夹具准备可以并行。每次整网性能对比只改变一个优化机制。

| 顺序 | 工作 | 完成标志 |
|---|---|---|
| 本轮已完成 | **K128集成及当前源码TP8／TP16 HTTP功能验收** | 两拓扑各16个必要响应案例及断连检查、完整logits／缓存／4请求native批处理／自然EOS／关闭均通过并完成第二独立审计；组件、CPU构建和静态检查证据保留。正式匹配性能对照仍待完成，原R2失败、`</think>`及EOS后一轮未交付decode观察不改写。 |
| 历史已完成 | **opt31：prefix cache 功能验收** | 对应冻结版本的 TP8、TP16 均通过；完整 logits 和六类缓存逐字节比较，覆盖完整命中、部分命中、16 槽、采样策略、取消及清理。性能对照另列下方。 |
| 待确认 | **确认 opt38 的数值参考并决定采纳** | 实现与可运行验证已完成；原 CPU 逐点失败和 FP64 通过同时保留，等待用户确认以哪个参考验收，不自动放宽标准。 |
| 阶段已完成 | **opt39：主机分项归因，继续调查主干成本** | TP8基线及TP8／TP16输入合并后的主干提交／等待、读回、拼接与检查均已分项采集。当前优先减少主干成本，间歇波动的设备／系统根因仍未完全确定；保留原12组及新观测样本，不删慢轮次。 |
| 后续候选 | **补 opt31 的独立性能对照** | 功能关卡完成后，固定相同 LM-head 后端和负载，只切换完整前缀 logits 复用；完整保存开关两侧的相同阶段数据，再判断收益。历史 CPU-head 控制不能直接对比当前 NPU-head 候选。 |
| 后续候选 | **完成 opt26A Indexer 准备范围优化** | 在已有 TP8 整网通过的基础上，补 TP16，再与既有基线做单项 Serving 对照。组件约 64% 的收益不能直接当作整网收益。 |
| 后续候选 | **推进 opt14B 设备执行与 CPU 输出处理重叠** | 保证输出缓冲区、请求代次和采样依赖正确，完成真实 NPU 生命周期及故障回归，再采 Serving 性能。 |
| 后续候选 | **验证 opt35、opt36、opt37 组件候选** | 分别验证跳过全 padding attention 块、CPU head 后端、活跃请求行投影。先过既有数值门槛；有效的候选再逐项进入整网验证。 |
| 配套验证 | **补 opt17B 更长输出和回答质量检查** | 先在现有 512-token 上限内增加 32/64-token 输出、自然 EOS、中文/英文及并发/缓存一致性；扩展上下文时另建兼容编译产物和匹配基线。 |
| 后续 | **按瓶颈推进元数据、拷贝、权重和算子优化** | 按下方完整队列逐项执行，收益不足则否决并保留记录。编译器受阻项保留复现和恢复条件。 |

## 已完成试验总表

截至本次更新，共完成 **21 项试验：12 项采纳、8 项否决或回滚、1 项配置研究**。编号保留；“已完成”包含有明确否决结论的试验。另有一项独立的权重 BYPASS 组件试验被否决，未计入这 21 项。

| ID | 改动作用 | 结论 | 关键实测或验收 | 保留的实现提交 |
|---|---|---|---|---|
| opt01 | 去掉无 query 的槽位预留，将编译容量 192 降到 128 | 采纳 | C4 吞吐 0.523→0.762 tokens/s；TP8/TP16 输出回归通过 | Serving `d3e4df7` |
| opt02 | 在设备侧清理请求缓存 | 采纳 | C4 0.762→0.927；六类缓存逐字节校验通过 | Lib `991f7a7` |
| opt03 | 保留通信窗口，按程序连续推进 collective epoch | 采纳 | C4 0.927→2.607；TP8/TP16、失败派发后的禁用与清理通过 | Lib `2ac35be` |
| opt04 | decode 前预留尚未返回的输出 token 预算 | 采纳 | C4 2.607→3.292；减少完成后的多余 forward，804 项单元测试通过 | Serving `cc2076e` |
| trial05_gpu_head | 在 NPU 上以 FP32 投影完整词表 | 否决并回滚 | 4,956,160 个值中 9 个未过原有逐点 `rtol=atol=1e-5` 门槛；32 个 top-1 虽全部一致，仍未进入 Serving 计时 | 无 |
| opt05 | 每个 Session 只把 CPU LM-head 权重扩展为 FP32 一次并保留 | 采纳 | C4 3.292→3.489；增加约 2.36 GiB 常驻主机内存 | Lib `1001ea5` |
| opt06 | 无本地 token 的 owner 跳过 shared expert 计算 | 采纳 | C4 3.489→3.557；routed expert、EP 通信和 token 布局恢复继续执行 | Lib `a7cfb96` |
| opt07 | 固定请求映射只在 Session 启动时上传 | 采纳 | C4 3.557→3.699；每 rank 的动态 H2D 调用 17→14 | Lib `0a53b67` |
| opt08 | worker 继承共享主机 I/O 缓冲区 | 否决并回滚 | C4 3.699→2.840；独立复制变快，Serving 整体变慢 | 无 |
| opt09 | 上传完成后，owner 的 Torch 线程数设为 16，关闭时恢复 | 采纳，显式启用 | C4 3.699→4.081；实际调用线程检查和输出回归通过 | Lib `aca3ebe`；Serving `ca54f67` |
| opt10 | shared INT8 权重在上传时一次性打包为 NZ | 采纳，保留归因限定 | 另一主机同机对照 C4 3.246→4.077；shared 组件变快，整网差值同时含 CPU 波动 | Lib `b0f6135` |
| opt11 | 整批 prefill 均未结束时跳过 CPU head | 否决并回滚 | C4 4.077→2.570；少做 27 次 head，但剩余 head 时间大增，原因未定位 | 无 |
| opt12 | routed INT8 权重改用 NZ | 否决并回滚 | 组件延迟下降 5.52%–9.83%；同机 C4 4.077→4.006，未显示 Serving 收益 | 无 |
| opt13 | 最多保留两个待完成的设备 submit，完成后才提交 epoch | 采纳 | 同机 C4 4.077→4.910（+20.43%）；TP8/TP16 保存输出和异常排空回归通过 | Lib `bd589fb` |
| opt14A | 在 command lane 提前准备 decode 元数据 | 否决并回滚 | 数值及生命周期通过；160/8 的 C4 3.500→3.206（−8.40%），未观测到未完成 native 程序与准备过程的重叠 | 无，恢复 opt13+15 |
| opt15 | 跨请求复用已确认的 MLA/Indexer/KDA 状态快照 | 采纳，显式启用 | 匹配的重复前缀 160/8 负载 C4 3.500→4.682（+33.77%）；80 行完整 logits 及 TP8/TP16 状态回归通过 | Lib `bee01da`；Serving `486c4b1` |
| opt16B | 固定 Torch16，分别显式设置 OpenBLAS4/8/16/32 | 否决，未应用 | 428 次完整输出均逐字节一致；head 中位数 167.186/167.002/167.060/167.023 ms，差异最多 0.11% | 无 |
| opt17 | 对比对齐的 prefill chunk32/64/128 | 配置研究完成，默认保留 32 | 128 在所测 C1 下 +35.23%，C2/C4 为 −16.95%/−31.57%；64 三组均变慢 | 无正式源码改动 |
| opt19 | query 总数≤16 时选择独立编译的 CAP64 attention | 否决，六个改动文件均回滚 | TP8/TP16 和 80 行 Serving 输出逐字节一致；C1/C2/C4 吞吐 −0.24%/−4.08%/+0.09%，无稳定收益 | 无，恢复 opt33 |
| opt25 | 短上下文的 Indexer 排序叶子 2048→512 | 否决，独立补丁未应用 | 13 对 native 用例通过；目标 visible32/128 延迟 88.27→90.96、91.07→91.27 μs，未变快 | 无 |
| opt33 | CPU head 只投影实际返回的 16 行 | 采纳 | 每次 forward 仍执行 head，并保留全部 rank 的 32 行读回及有限值检查；历史 54 行、24 份 prefix payload 和 Serving80 行逐字节一致；160/8 C4 3.500→5.057（+44.49%） | Lib `2e075a2` |

独立保留的修正另记：Lib `b9c4523` 只修复 routed standalone 输出方向声明，不保留 routed NZ；Lib `be033dc` 修复产物重放 API，19 项针对性测试和 4 项实际元数据重载通过。独立权重 BYPASS 虽通过数值检查，但大多数 shared/routed 形状的完整 device wall time 变差，已否决。

## 验证中项目的详细进度

### opt31：完整前缀 logits 复用，TP8／TP16 功能验收通过

**实现已接入原 Serving 仓，默认关闭。** 当前统一添加 `--enable-prefix-caching`，即可同时支持部分前缀命中，以及对完全相同且命中已保存对齐终点的 prompt 复用原始、完整 FP32 logits。首次采样使用新请求的采样策略；之后恢复正常 decode。首 token 不调用主干或 LM-head，但仍真实执行设备槽位清理和六类状态恢复。下述历史设备验收使用原双开关版本；本次仅统一命令行开关，不重写历史设备验收结论。

验收使用官网 W8A8 权重、context512、budget128、chunk32、最多16请求、每个正常请求生成8个token。每种拓扑各启动一个开启缓存的 Engine、加载一次权重，以其**真实冷启动、没有恢复缓存的请求**为参考，与后续实际命中逐字节比较。两者使用同一 NPU LM-head，不与旧 CPU-head 或另一拓扑混比。没有 primer、预热组或计时组；独立开关性能比较留待后续。

| 功能关卡 | TP16 / EP16 | TP8 / EP8 |
|---|---|---|
| 128-token 完整命中；首采样后继续7次 decode | 通过；首采样0主干、0head、0 Session forward，collective epoch不推进；真实reset/restore | 同样通过 |
| 160-token 部分命中 | 通过；真冷启动从0开始，命中恢复128后仅计算剩余32 token | 同样通过 |
| 完整输出回归 | 29对轨迹、232行逐字节比较通过，每行154880个FP32值，token及采样参数一致；总计保留299行，含取消后的不可见旧行 | 同样通过；各自使用本拓扑参考，不跨拓扑混比 |
| 状态恢复 | 31次恢复，每次六类缓存×16rank，2976项检查；按编译产物物理层步幅比较含填充的完整逻辑槽，累计10,408,984,576字节 | 31次×六类×8rank，共1488项，7,492,599,808字节；逐字节通过 |
| 请求布局与缓存隔离 | 16个逻辑槽、混合命中／未命中批次、源槽重置后跨槽恢复、两个bank淘汰和generation切换均通过；故意修改采样输入行不污染缓存 | 同样通过；TP8也覆盖16个请求槽 |
| 策略与生命周期 | 随机采样及seed、top-p/top-k、最短输出EOS屏蔽、stop字符串；首ACK前取消并同ID重启，旧预算1→新预算8及旧预算8→新预算8均通过 | 同样通过 |
| 资源关闭和来源 | 运行结束lease及pending全归零，worker正常退出0，CPU logits和bank清空，146份源文件hash不变 | 同样通过；197结束后16个设备均无进程 |

**CPU 回归：**182项相关单测通过，另有26项深层协议检查通过；ruff、文件头、英文代码检查及diff检查通过。全量Serving单测在补充8项回归前为810通过／4失败，4项失败均为未修改的DeepSeek测试硬编码访问用户已删除的Serving内嵌`pypto-lib`；未恢复该冗余目录。

**失败记录保留：**本机TP8任务`task_20261026_044817_56469028166`在首个冷启动forward就报告`hdc disconnect`、ACL507901／507018和`TENSOR_WAIT_TIMEOUT`，尚未进入任何缓存捕获或命中。根因未定，不能归因于prefix cache；已保存日志并终止本任务，转197原样重跑，未手工重置卡或终止其他任务。

**历史试验保留：**旧R1因夹具扣住ACK又等待新请求调度而死锁；R2修正ACK顺序后，OFF完成12组测量，ON自身验证通过，但跨引擎比较错误地包含了OFF额外8行primer。202个共同行完整输出相同；未删行、修改旧参考或覆盖失败报告。旧数据在197的`build_output/glm53-exact-prefix-opt31-r2-197/`。本轮另建只做功能验收的目录，解决比较阶段不一致，同时补原先缺失的TP16和旧预算8硬件取消覆盖。

**支持边界：**context512默认KV page128与KDA16共同对齐，只命中实际捕获过的完整终点；并非任意32／64-token前缀都可恢复。两个CPU完整logits行增加1,239,040字节，设备快照容量仍为两个bank。旧预算8的取消发生在首个ACK之前，终结prefill屏障阻止旧decode；不宣称覆盖了已有输出后的抢占重算、旧decode在途取消、grammar masking或自然EOS终止。上述缓存对照不替代独立整网golden，也不改变opt38数值参考待确认的状态。

**性能仍待测：**当前只是功能验收，不计入上面的21项已完成性能试验、不替换吞吐基线，也不从零query直接换算加速比。下一次需固定同一head后端做匹配的开关测量。

**新证据目录：**Lib的`build_output/glm53-perf-iterations/prefix-acceptance-opt31-20261026/`，含冻结harness、CPU回归、本机失败日志、197原始数据副本和汇总`ACCEPTANCE.json`；197执行目录为同层`prefix-acceptance-opt31-20261026-197/`，TP16为`native-tp16-r1`，TP8为`native-tp8-r2`。TP16报告SHA256：`8453072abf1427e3c687f5bacf8d9152c5804eb17b075472762d3e244259c179`；TP8：`a3530e10e21f6fe25265c70c217b866e3e3ac4ab28f6127ca764d4da0d2269e3`。

### 验证中：opt14B，设备执行与 CPU 输出处理分离

- **改什么：**设备执行、等待及各 rank 输出回传完成后，保留独立的 CPU 隐藏状态快照，交给输出队列完成现有 LM-head 和采样。
- **为什么：**尝试让下一个独立请求或已知 token 的 prefill 主干，与前一个请求的 CPU 输出处理重叠；同一请求的 decode 仍须等待采样结果。
- **与 opt14A 的区别：**这是带私有输出快照和生命周期管理的新方案；不是重新启用已因整网收益不足而回退的元数据重叠方案。
- **已完成：**离线原型覆盖两份快照、背压、依赖 decode 拒绝、前缀捕获顺序以及错误和关闭时的排空；使用替代的 native 执行验证协议。
- **保留的失败：**早期发现 ACK 已可见、旧输出任务却仍保留前缀快照缓冲区的生产者标记，阻挡后续复用。R2 在发布 ACK 前释放该标记；CPU 输出缓冲区及权重仍保留到最终清理，旧任务清理也不能误清新任务的标记。
- **CPU 结果：**独立复查通过 Session/Worker 协议、采样策略、12 项 ACK 交接生命周期用例和 79 项已有测试；这些结果不证明真实 NPU 与 CPU 已重叠。
- **版本状态：**依赖 opt19 的条件适配版已归档；opt19 回退后，应从 R2 基础继续，并显式适配后续获保留的 opt31、opt26A 行为。
- **还缺什么：**真实 TP8/TP16 正确性、实际执行区间重叠、默认关闭路径、取消和抢占、完整关闭，以及 Serving 性能比较。
- **结论边界：**未合入生产，也尚无整网加速结论；早期四项 CPU 失败缺少原始源码快照，不能用当前代码倒推其完整前后行为。
- **关键证据：**`build_output/glm53-perf-iterations/worker-pipeline-opt14B/revision2/READY.json`，SHA 前缀 `cb6d374c`。

### 验证中：opt26A，减少 Indexer 不会被评分读取的准备工作

- **改什么：**静态限制 pool 收集和 scale 打包范围，只准备评分轴可能访问的逻辑前缀；当前几何下收集页数由 4096 减至 1040，scale 工作项由 256 减至 65。
- **保持什么：**公开参数、物理存储、scratch 形状、评分计算、查询和 golden 均保留；按实际活跃请求进一步压缩页范围属于尚未实现的 opt26B。
- **组件实测：**A3 单卡组件在可见范围 32/128 时，有效执行时间中位数约由 996/1003 微秒降至 355/359 微秒，减少约 64%；未裁剪的控制项基本持平。
- **指标边界：**上述数字来自设备侧 orchestration/scheduler 有效时间，不是整网吞吐或客户端延迟，也不能据此推算 Serving 收益。
- **TP8 整网正确性：**一次真实 Session 完成 60 次完整 forward 和一次独立故障部分执行；旧输出、前缀输出、状态、带非零填充的切换及两份完整 128 行流缓冲检查均逐位通过。
- **故障及清理：**队列后继排空、禁止失败后的继续提交、状态不错误提交及资源释放检查通过；旧参考没有重新计算，生产代码未修改。
- **当前进度：**TP16 的源代码、运行产物及验证方案已准备，尚未执行；TP8 正确性通过不等于整项已验收。
- **还缺什么：**完成 TP16 整网验证，再与相同来源、相同 160/8 工作负载的 Serving 基线比较；须使用匹配的新 DSA 产物，不能混用旧编译缓存。
- **关键证据：**`build_output/glm53-perf-iterations/indexer-pool-preparation-opt26A/` 的 `native-component/` 和 `full-model-integration/`；TP8 最终审计 SHA 前缀 `538d036e`。

### 验证中：opt35，跳过 prefill attention 中完全无效的填充块

- **改什么：**在 tile 初始化前计算连续有效块的上界，让完全填充的块跳过 gather、QK、PV 及无用的在线 softmax 工作。
- **为什么：**当前路径仍会遍历全部 17 个块；短窗口中真正有效的块较少，剩余工作可能没有有效结果。
- **保留的语义：**有效块的更新顺序、BF16 舍入位置及 golden 不变；全屏蔽输入仍执行一个块，部分块、多块及 2051 宽尾部需要保持原行为。
- **CPU 覆盖：**已检查全部 2052 种有效前缀长度，并核对计算体及参数使用方式。
- **编译结果：**8 组原版/候选的 host/PTOAS 构建，以及 16 个 AIC/AIV CANN 二进制编译、链接均通过；不能把较早的 host 编译成功等同于二进制成功。
- **产物要求：**公开 tensor 参数及 DSA 参数数量不变，但提升首次使用位置改变了两个编译内部参数的顺序，必须使用配套新编译的 orchestration 和二进制。
- **组件验证方案：**CPU 侧夹具已冻结，计划先完成 22 个用例、44 次原版/候选数值调用，再对 5 种 TP8 几何分别执行 5 次预热和 100 次计时，合计 1094 次 native 调用。
- **运行范围：**计划使用单卡、单 Worker；其中 TP16 仅表示组件采用的本地 head 几何，不代表分布式 16 卡验证。
- **当前进度：**编译和夹具 CPU 检查通过，尚未执行 NPU；没有加载模型权重，也没有修改生产代码或 golden。
- **还缺什么：**真实 NPU 数值验证、组件性能、完整模型和 Serving 验证；目前不能给出实际加速结论。
- **关键证据：**opt35 的 `component-validation/READY.json` 已更新，SHA 前缀 `5b8f46b4`；较早的编译证据仍保留。

### 验证中：opt36，比较 CPU LM-head 的 oneDNN 后端

- **改什么：**尝试常驻一份 oneDNN 不透明格式的 FP32 词表权重，用公开的 `to_mkldnn` / `functional.linear` 路径与现有 CPU 实现比较。
- **为什么：**探索完整 CPU 输出投影的执行成本；常驻权重不代表后端一定不会再做内部准备，不能预先把“少打包”当作收益。
- **必须保持：**现有 FP32 RMSNorm、完整 154880 维 logits 及冻结的参考；不能通过改变 golden 或放宽数值门槛继续候选。
- **已完成：**完整探针方案及 20 项文件指纹已冻结；实现对比确认 AST 只有 return 表达式变化，小形状检查、相差 1 ULP 时的拒绝检查和异常后的恢复检查均在 CPU 上通过。
- **尚未完成：**尚未加载真实 head 权重，没有计算完整冻结参考，也没有计时；小规模结果不能代表完整权重精度、峰值内存或性能。
- **下一步精度门槛：**仅加载已发布的 head 权重，先把两个 16 行分组分别与已有完整 32 行参考逐位比较；不通过即停止性能试验。
- **后续采集：**若精度通过，再按预先声明的交替次序采集两种后端，计入每次输入/输出格式转换，并分开记录一次性准备时间和常驻内存。
- **结论边界：**未进行完整权重或 Serving 性能验收，没有可报告的加速，也没有生产或 NPU 代码改动。
- **关键证据：**`cpu-head-backend-opt36/READY.json`，SHA 前缀 `6d09128b`；不能把探针冻结或小形状 CPU 检查写成模型实测。

### 待实施：opt37，仅投影 chunk 接口实际消费的请求行

- **改什么：**让 `forward_chunks` 按实际请求槽执行 CPU 输出投影；opt33 已从 32 行缩减到 16 行，而该接口随后还会从 16 行中选取所需请求。
- **为什么：**单请求或少量并发时，其余行可能不被消费；这是继续减少行数的候选，并非省略整个 forward 的 LM-head。
- **当前进度：**仅完成候选登记和调用链分析，尚未实现，也没有测得性能收益。
- **先做什么：**使用已有完整参考，分别检查 1/2/4/8 行的 CPU 投影；改变 GEMM 行数可能改变累加过程，必须先验证输出逐位一致。
- **不能改变：**每次 forward 仍需计算其应有的 head；保留完整 32 行、各 rank 的设备回传和保护检查，以及原有完整 batch 返回接口。
- **请求布局：**必须覆盖非连续槽位、请求重排、取消和槽位复用，不能假设有效行总是连续前几行。
- **还缺什么：**候选实现、CPU 完整数值门槛、TP8/TP16 native 验证，以及与保留基线相匹配的 Serving 性能采集。
- **结论边界：**目前不能写成 opt33 的后续收益已经落地，也不能引用 DeepSeek 的行选择原则代替 GLM-A3 实测。
- **关键证据：**原报告队列表中的 opt37，以及 GLM `forward_chunks` / `_run_step` 调用链；暂无可验收的性能报告。

## 剩余优化队列

以下项目均未完成生产验收，或仍属于原因调查、后续验证及设计候选。已采纳的 opt13、opt15、opt33 继续保留；opt14A、opt19 已否决并回退，不能直接重新启用。opt17 的分块试验已经完成，默认仍为 chunk32；长输出、长上下文另由 opt17B 负责。

**执行顺序以上面的“接下来怎么做”为准。** 当前固定16卡的双TP8对照已完成，下一项先补单TP8的C2／C4卡效率；主干提交融合等待编译问题解除。下表保留历史候选的依赖关系，不能再将“已有实现先收尾”作为最高优先级：opt31性能、opt26A整网及opt14B重叠，均应根据当前NPU-head基线的实际收益空间再安排。opt35、opt36 可以先完成各自的算子或 CPU 门槛，但必须满足资源和证据依赖：本机编译未结束时不同时采集 CPU head 性能。其余项目逐点推进，每次只改变一个机制，分别记录数值、性能和采纳／否决结论。分组表示优先级与依赖，不表示组内项目可以直接同时实施或压测，也不承诺收益。

### 已有实现待收尾的历史候选

| 编号 | 目标 | 当前状态 | 完成条件 |
|---|---|---|---|
| **共享 BF16 linear K128（本轮新增）** | 将逐元素K1推进改为K128分块，配套两处必要列tile适配，保持数值边界和原 golden／门槛。 | **已正式集成，TP8／TP16当前源码HTTP功能验收通过；正式匹配性能待测。** 组件／调用者、两拓扑全12程序CPU／native／重载及两仓静态检查通过。新raw诊断窗口单列，不计入21项历史性能试验或12项采纳。 | 本轮所需HTTP功能已过；后续在相同主机／负载和统计口径下完成正式性能对照。更广泛长文质量、`</think>`展示及EOS后在途尾部工作另列，不把单组件约162.7倍当作整网收益。 |
| **opt38** | 最终 RMSNorm 和完整词表投影使用 PTO3/NPU，保持 FP32 归一化边界和通用采样接口。 | **实现与测量完成，数值参考待确认。** TP8/TP16 组件、旧夹具、56 次完整 forward、Serving 12 组均已执行；FP64 对照通过，旧 CPU 逐点微差保留。 | 根据用户确认的参考标准决定采纳；不把生成相同或更快自动等同为旧 CPU 逐点标准已通过。 |
| **opt39** | 定位 NPU head 后跨并发组的 forward 抖动与主要耗时。 | **108 次正式 forward 的已保存数据分析完成。** head 前未分解区间占客户端总耗时 88.36%，head 后占 8.97%；慢轮之后的 C1/C2/C4 也有波动，具体根因未定。 | 新增上传、主干 submit/result、两类读回、拼接、检查分项及系统／设备时间证据；定位后再实施单一改动，保留全部正式样本。 |
| **opt31** | 对完整、对齐且完全相同的前缀复用终点FP32 logits，消除完整命中后的模型forward。 | **TP8、TP16功能验收通过；性能待测。** 128完整命中、160部分命中、16槽、混批、完整FP32行、六类状态及两种首ACK前同ID取消通过；旧失败报告保留。 | 固定同一head后端、相同阶段完整数据，补独立开关性能对照。grammar masking、已有输出的抢占重算与在途旧decode取消另行验证。 |
| **opt26A** | 先把 Indexer 的准备工作限制到实际 score 轴，减少无关 gather／pack。 | 独立算子测量通过，active32／128 的组件中位数分别下降约64.37%／64.19%；**TP8 完整模型60次 forward 及输出、状态、失败清理检查已过**。TP16 仅完成准备，TP16 与 Serving 尚未完成验收，未采纳正式集成。 | 完成 TP16 的完整 logits、六类缓存、物理步幅、尾部和失败清理检查，再进行同机、相同工作负载的 Serving 对照。只有整网测量才能决定是否采纳；组件约64%的结果不能换算为整网收益。 |
| **opt14B** | 用私有输出生命周期分离设备下发和 CPU 输出回收，让独立请求或已知 token 的非终结 prefill 有机会在前一个 CPU head 尚未回收时推进。 | R2 原型完成 CPU 审查，native 验证脚本仅完成 CPU 准备；**尚无真实设备重叠、Serving 性能或正式验收**。依赖已采纳基线；基于已否决 opt19 的条件 R3 已归档。 | 在届时已采纳的代码上显式语义适配，尤其处理 opt31／opt26 的变化。保持最多两个私有输出票据、真实采样依赖、prefix 捕获先于后继修改、精确 ACK、取消／抢占／同 ID 重启／槽复用及异常关闭。完成 TP8／TP16 完整输出与状态、真实 Worker 路径及匹配 Serving 测量。同请求 decode 必须等采样；公共调用区间相交或未回收的完成句柄不能当作硬件执行重叠。保留每次完整 head 和全 rank 的32行读回／有限值检查，不重试原样 opt14A。 |

### 需要先完成 CPU／算子门槛的候选

| 编号 | 目标 | 当前状态 | 完成条件 |
|---|---|---|---|
| **opt16** | 定位 CPU LM-head 延迟波动的原因，再针对实际瓶颈优化。 | **原因仍未定位。** NUMA 小试验中位数仅变化约0.22%，显式 BLAS4／8／16／32 的 opt16B 最大差约0.11%，均没有有用收益；opt16B 已否决。 | 固定 FP32 运算和输入，区分权重内存放置、后端执行、等待与运行环境影响，形成可复现的因果证据。线程 getter 不同、启动4线程转为 owner16线程，或 CPU 时间／墙钟比值，都不能单独证明原因或物理核心使用数。新机制须另立对照，不原样重复已否决试验。 |
| **opt35** | 跳过 prefill attention 中完全填充的块，保留所有有效块的原在线 softmax 顺序。 | **只有编译和 CPU 夹具准备证据。** 已有8个 host／PTOAS 构建、16个 CANN 二进制编译／链接通过；CPU 覆盖有效前缀长度和边界。尚未进行 native 数值、组件计时或整网／Serving 验收。 | 使用一致的新 DSA、编排和二进制产物，先做真实设备数值与边界检查，覆盖全 mask、部分块、多块和2051宽尾部，再做组件及整网测量。内部参数位置变化后不能混用旧产物；不绕过编译器验证或修改工具链 pin。 |
| **opt36** | 比较保留一次的 FP32 opaque 权重与公开 oneDNN linear，评估另一种 CPU vocabulary head 后端。 | **仅小接口和探针准备通过，真实 head 权重尚未计算，也没有真实头计时。** 完整 FP32 norm AST 保留，仅替换最后的返回计算；公开 API、小权重往返、拒绝门槛和异常恢复已检查。 | 在本机编译结束、避免同时采集其他 CPU 性能数据的条件下执行。只加载一次发布版 head；先验证整个权重 unpack 逐字节一致，再验证冻结32行 oracle 的两个16行半区、全部154880维 FP32 输出。任一字节失配立即保存实际输出并拒绝，不进入计时。全过后才做每后端5次 warmup、100次计时，50次 AB／50次 BA；每调用格式转换计入完整 helper 时间，一次准备耗时和内存另记。不能从组件结果直接声称 Serving 收益。 |
| **opt37** | 只投影 `forward_chunks` 实际消费的请求槽，进一步减少 opt33 之后仍存在的16行 CPU head 工作。 | **尚未实现，尚无数值或收益结论。** | 先将1／2／4／8行的完整 CPU 输出与冻结全量 oracle 比较；改变 GEMM 行数也可能改变累加结果，门槛不能放宽。覆盖不连续／重排槽、取消和槽复用；保持每次 forward 的 head、全 rank 的32行设备读回／检查及旧完整 batch 接口，再完成 native 和匹配 Serving 验收。 |

### Host 准备与数据搬运候选

| 编号 | 目标 | 当前状态 | 完成条件 |
|---|---|---|---|
| **opt21** | 保留不可变 CPU 元数据和 KDA 常量，减少每步重复构造。 | 隔离原型和 CPU 检查已过；准备阶段减少约0.119–0.140 ms／调用。尚无 Serving 接受结果。 | 对多种 chunk／请求布局比较所有准备张量的字节、shape 和别名生命周期，确认不缓存可变请求数据；分别测量准备时间和完整 Serving 时间。 |
| **opt22** | 提前绑定固定张量参数和层序号，每步只更新新 collective epoch。 | 待实现／验证；运行时已有注册逻辑不能重复计为新优化。 | 证明参数次序、ping-pong 绑定和每次下发的独立新标量存储正确；覆盖计划／ABI 变化、失败提交、确认后 epoch 发布及清理，再测量实际收益。 |
| **opt23** | 仅在内容改变时上传剩余动态元数据。 | 待实现；依赖清晰的静态／动态元数据及请求生命周期划分。 | 检查各 rank 变化检测、零到活跃、token／position 更新、取消和槽复用；统计真实 copy 数和驻留字节，保持输出与状态一致后再做 Serving 对照。 |
| **opt30** | 消除通用 Worker 和 GLM Session 对同一输入的重复 CPU embedding lookup。 | 发现明确源码入口，尚未完成实现及性能验收。 | 建立显式 token 输入契约，保持 prefill／decode token 和输出不变；用实际调用计数证明每次 forward 只 lookup 一次，再比较整网时间。CPU lookup 的优化不能写成设备 embedding 已实现。 |

### 需要先证明适用形状与布局的候选

| 编号 | 目标 | 当前状态 | 完成条件 |
|---|---|---|---|
| **opt24** | 批量处理 Gate 归一化行并复用 gamma 加载。 | 候选，待实现及测量。 | 保留 BF16 路由边界、FP32 INT8 量化和专家完全同分时的既定顺序；覆盖近临界选择及所有 golden 门槛，测量完整 MoE 和 Serving，而非只报告局部加载次数。 |
| **opt26B** | 在 opt26A 之后，进一步只准备活跃请求可见的页面。 | **尚未实现**；依赖 opt26A 结果及新动态路径的编译可行性。 | 先验证动态页映射、活跃／失活切换、尾部与 mask、物理步幅和一般回退，再完成组件、TP8／TP16 和 Serving 检查。不能将 opt26A 的组件收益作为本项结果。 |
| **opt27** | 在多个16行激活 tile 之间复用 shared-expert 权重 tile。 | 待证明适用负载；当前冻结 CAP128 基准的 owner 只有16行，不能证明本项收益。 | 先选择真实 owner 行数大于16的形状，保留有符号 INT8 NZ 寻址和输出算术，完成组件测量；若进入不同容量的模型，还需相应编译／布局验证及匹配整网对照。 |
| **opt28** | 跨重启缓存已验证的预打包权重和运行时产物。 | 候选，尚无启动收益验收。 | 对 checkpoint、量化、布局、拓扑和工具链完整指纹化；拒绝旧缓存，约束磁盘／内存占用，并验证分阶段字节一致。单独报告启动／加载耗时，不能声称仅靠启动缓存提升稳定生成吞吐。 |
| **opt29** | 在相邻 MLA 输出列 tile 之间复用输出激活。 | 候选，需在 A3 上验证。 | 保留 FP32 TP partial 和归约后的舍入边界，在有代表性的 prefill／decode 形状上测量组件，随后通过整层、TP8／TP16 和 Serving 验收；A5 的参考结果不能替代 A3 结论。 |
| **opt32** | 45层 Serving 主干不驻留未使用的 MTP 权重、额外 MoE／router 层及未映射栈。 | 仅完成 header 级大小分析；预计未用 payload 为 TP8 每 rank 1011.70 MiB、TP16 每 rank 566.14 MiB，实际 HBM／启动节省未测。 | 新建明确的 backbone-only 驻留计划，同时保留 MTP／组件使用者的完整加载能力。改变栈深需更新 manifest／ABI 并重新编译，不靠未初始化 padding 或旧 runtime 绕过。核对各活跃层 staged bytes、TP8／TP16 完整 logits／缓存和初始化／HBM；启动收益与稳定 tokens/s 分开报告。 |
| **opt34** | 合并已验证连续的 BF16 cache 行 gather，优先研究 Indexer 的四行 pool。 | 候选，尚无 GLM-A3 组件或整网收益。排序后的完整128行连续段并不常见。 | 优先优化主干实际使用的 prefill 路径；逐段证明连续性和页面对应关系，保持选中顺序，并对任意顺序、重复、负值、跨页及2051宽尾部保留严格回退。BF16／softmax 边界及 scoring 算术不变；先测可用比例与 A3 组件时间，再做模型及 Serving 门槛。 |

### 配套验证及等待设计／工具链条件的项目

| 编号 | 目标 | 当前状态 | 完成条件 |
|---|---|---|---|
| **opt17B** | 扩展输出长度和上下文覆盖，检查更接近持续生成的负载。 | 待建立独立的新工作负载；不重新打开已完成的默认 chunk32／64／128 决策。 | 保持明确配置限制及16-token KDA 分块边界；检查完整 logits、生成 ID、准入公平性、内存和 TTFT／TPOT／吞吐。新上下文或输出长度需要各自匹配对照，不能沿用160输入／8输出的收益结论。 |
| **opt20** | 整合设备 embedding、final norm 和 vocabulary projection。 | **需重设计。** 早期 NPU 全词表 FP32 投影已因数值门槛不通过而拒绝，不能直接重新启用。 | 保留 FP32 final norm 语义、通用采样需要的完整 logits 和原数值门槛；top-1 一致不能代替完整输出验收。greedy 专化如要实施，须有明确受限接口和独立验证，再做拓扑／整网测量。 |
| **opt18** | 合并 attention 与 FFN，研究将每次 forward 的89次提交减少到47次。 | **被固定版本编译器阻塞。** 合并 KDA／DSA 入口触发 Error104 维度等式检查，单 attention 对照能编译。 | 有独立验证的编译器修复后，先重跑最小复现及控制例；编译与数值通过后才进入设备计时。不能绕过验证、私改 runtime 或将编译失败写成可直接测试的优化。 |
| **opt19B** | 仅计算活跃 query 行的 attention projection。 | **动态方案被编译器阻塞**：`InferTileMemorySpace`／`AccCompactValid`；CAP16 还受现有 Indexer projection 约束。静态 CAP64 的 opt19 已完成并拒绝。 | 需要已验证的编译能力，或有明确差异的新机制，再证明布局、别名／输出生命周期、尾部和 collective epoch 正确，完成 native 及 Serving 对照。不能将已否决 CAP64 试验改名重跑。 |

### GLM-5.3 recipe 对照：新增候选，尚未实现或测量

2026-10-29只读对照本地`cann-recipes-infer`的`a06336361c2614c7bcaebcb9046181a9e23008f7`，重点读取`models/glm_5_3/`、共享MoE量化模块和executor图执行路径。GLM-5.3目录最近一次提交为`74cfba8`（2026-09-16），是950平台样例。本轮没有启动模型、设备任务或性能采集，没有修改recipe或Lib实现；以下是源码证据和候选方向，不增加已完成试验／采纳计数。

**先区分部署前提：**recipe默认950DT、Hybrid HiF8／MXFP8／MXFP4、Attention DP8＋TP1、MoE EP8；我们当前验证的是A3、INT8 W8A8、Attention TP8／TP16＋DP1、MoE EP8／EP16。recipe README的8K上下文、每卡BS1、16.60ms/步和8卡481.93 token/s，是文档报告的decode指标；每卡BS1对应全局8请求，不能直接与我们的C1／C4、短上下文HTTP或主机forward窗口相除。没有复测或确认该表的采集环境与当前源快照完全一致。

| 新编号／建议顺序 | recipe的实际做法与源码位置 | 我们当前的对应差异 | 后续工作与验收条件 |
|---|---|---|---|
| **opt40，可先做的小改动** | `modeling_glm53.py:920`在加载后调用`kda.py:206`的`build_conv_weight()`，卷积权重的转置／连续化只做一次。 | Serving生产链`prefill_fwd.py:576`每次KDA调用都执行`stack_conv_weight`；34个KDA层每步都重新组装固定权重。 | 将生产所需tap-major、q/k/v拼接布局在权重驻留时一次性准备，更新权重布局和ABI后移除逐步重排。先比较打包后的每个字节与现有设备布局，覆盖TP8／TP16；重新编译，再测完整forward和Serving。预期减少固定任务和访存，尚无收益百分比。 |
| **opt41，高优先级的计算路径候选** | `kda.py:272`按prefill／decode分支；`:369`的decode调用单步卷积更新和`fused_recurrent_kda`，不构造chunk三角逆。 | Serving `npu_runner.py:240`将decode的每请求单token送到同一`forward_chunks`；Lib `prefill_fwd.py:581`／`:584`仍调用`kda_conv_prefill`和`prefill_kda`。后者为短query也建立chunk scratch、三角逆和状态scan。独立`decode_kda.py:98`及`kda_conv_decode`已存在，`decode_fwd.py:257`／`:259`也有消费者，但未接入当前Serving生产路径。 | 先审数值契约，再为纯decode批次设计专用生产入口；混合prefill／decode批保留明确分流。组件、连续状态、重排／空槽／复用、prefix恢复和整网golden均需验证。现有chunk scan会将部分状态和GEMM操作数舍入BF16，单步FP32递推不能未经验证直接替换，也不能假定旧完整logits会逐字节相同。 |
| **opt42，高收益方向，依赖编译器／runtime能力** | `executor/utils/graph_utils.py:66`用`torch.compile(..., fullgraph=True, backend="npugraph_ex")`；`model_worker.py:446`只对decode调用compiled model，prefill保留eager。 | 我们已经编译并复用6种程序，但每步仍按`production_launch_plan`发起89次主干提交，队列最多两个在途。减少程序文件数量没有减少每步提交数量。 | 借鉴稳定任务图／依赖和输入地址复用，在PTO3中研究整层、分段及整步下发；不是给现有代码加一个torch图开关。先完成opt18的89→47融合及Error104回归，再评估更大组合或公开runtime任务图接口。动态epoch、请求槽／取消、缓存写入和完整logits语义必须保留；235ms submit窗口不是全部可消除开销。 |
| **opt43，结构性高潜力，先评估内存及调度** | `modeling_glm53.py:858`要求Attention／Dense／MoE／shared的TP均为1；配置world_size8，框架派生Attention DP8、MoE EP8。每rank处理自己的请求，专家跨8rank共同服务。 | 我们的Attention／Dense使用TP8或TP16，MoE前后有token owner压缩／恢复及跨rank归约；生产入口当前只接受TP8/EP8/DP1和TP16/EP16/DP1。 | 研究A3下更小Attention TP与更大DP、同时保留EP的配置。先核算复制的非专家权重、KDA状态／KV和workspace的HBM，再改loader、请求归属、拓扑与collective；不能仅改`--tp 1`。它不同于两个独立TP8实例，必须在相同总卡数、总请求数及计时范围下测量，也不保证单请求一定更快。 |
| **opt44，MoE粗化与实际重叠，收益待归因** | `modeling_glm53.py:301`decode使用dispatch/combine v2；`:343`侧流计算shared，事件约束汇合；共享FFN的down在`:97`等待routed GMM事件。MoE量化模块使用grouped matmul及融合激活／再量化。 | 我们已做gate/up合并和INT8路由，但dispatch／重排／专家计算／return仍有多段任务。shared在源码中后提交；已有生成C++的shared任务只读取gate产生的`x_int8`，并不能由后提交就断言设备全串行。 | 先从已有或后续单次执行时间线确认shared与routed／通信的真实依赖、核心竞争和重叠，再调整PTO3任务顺序、粗化量化epilog和通信阶段。保留EP协议和事件生命周期；不以函数搬位置、Python调用区间相交或换用官方Torch算子当作PTO3已提速。950 MXFP4算子及MC2实现不能假定可直接复用于A3 INT8。 |
| **opt45，mHC融合，先确认实际代价** | `models/modules.py:93`调用`npu_hc_pre`，C++入口`ops/ascendc/torch_ops_extension/custom_ops/csrc/npu_hc_pre.cpp:214`走`aclnnHcPre`融合实现；post也有独立融合op。 | 我们的`mhc_mixes`、`mhc_pre`和后续norm有多个显式中间缓冲与阶段，每层Attention／FFN均调用。PTO3本身可能已经合并部分任务，不能按Python行数推算kernel数。 | 检查实际生成任务、局部buffer和访存；优先在PTO3中融合可证明等价的统计／系数／collapse路径，保留FP32归一化、20轮Sinkhorn、BF16边界和既有数值门槛。先完成组件及整层验证，再决定是否进入Serving对照。 |

**实施顺序建议：**优先研究opt41及减少主干下发的opt42，opt43先做HBM／并行设计；opt40作为小范围预打包候选，具体定序结合后续vLLM对照。opt44／45按设备时间线决定排序，不因recipe用了融合算子就预设整网收益。这次recipe审阅只是分析，原已测基线和候选状态保持原口径。

**值得借鉴但不能整段照抄的部分：**recipe把step metadata在一次forward内生成后供所有层共享，decode避免`.cpu().tolist()`；我们已有共享输入缓冲、固定元数据驻留和增量pool缓存，不能再记为新完成项。recipe的KDA prefill在`kda.py:321`显式新建零初态，`modeling_glm53.py:772`的`has_initial_state`也从零开始；替换我们支持chunked prefill／prefix restore的路径时必须保留已提交状态。recipe `indexer.py:214`的decode专用lightning indexer与我们INT8＋scale的池缓存表示不同，只能借鉴阶段划分和数据驻留，不能直接替换调用。本地KDA底层由`cannbot-dsl==0.0.3`依赖提供，本轮没有安装／审阅该包实现或验证它的A3支持。该recipe同样显式拒绝MTP（`modeling_glm53.py:861`），不提供可直接迁移的GLM MTP流程。

### 性能追赶参照：同模型、同A3硬件的vLLM Ascend

2026-10-29核对本地checkpoint配置和官方文档后，首选追赶对象确定为**vLLM Ascend上的GLM-5.3-Flash W8A8、单机A3**。现已完成短负载、同条件控制及优化组合的真实测量，协议和结果分节记录。

| 实现位置 | 实际模型／部署格式 | 硬件与当前证据 |
|---|---|---|
| recipe `models/glm_5` | 原版GLM-5，A3样例包含W8A8 | README明确面向Atlas A3；不是45层KDA＋mHC的Flash模型。 |
| recipe `models/glm_5_3` | GLM-5.3-Flash，默认Hybrid HiF8／MXFP8／MXFP4 | 默认950DT、8rank；其16.60ms/步等文档结果只作为不同平台的参考。 |
| PTO3 `models/glm5_3_flash` | GLM-5.3-Flash文本主干，使用发布版INT8 W8A8 | 已验证A3 TP8/EP8/DP1和TP16/EP16/DP1，每逻辑NPU64GiB。 |
| vLLM Ascend `GLM5.3-Flash.md` | 同一GLM-5.3-Flash；A3模型为`Eco-Tech/GLM-5.3-Flash-w8a8` | 官方单机A3为64GB×16逻辑NPU、TP16、DP1、开启EP。同页950DT使用不同的W8A8-MXFP8模型。 |

PTO3本地发布版权重的`config.json`为`glm5_next`／`Glm5NextForConditionalGeneration`，文本主干45层、hidden4096、288专家／top8；`quant_model_description.json`为`W8A8_DYNAMIC`。部署记录确认其来源为`Eco-Tech/GLM-5.3-Flash-w8a8`，revision `4873e6c9bebe101cd472e5939b04a9d8948e2a52`。后续对照应直接复用同一已校验的本地权重目录，避免同名hub模型更新造成差异。A3的8张双芯物理卡对应16个逻辑设备；PTO3的TP8测试使用8个逻辑设备，不是整台16逻辑设备A3的同卡数对照。

官方依据为[GLM-5.3-Flash部署指南](https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/GLM5.3-Flash.html#glm53-flash-5-1-a3)。本轮读取的repo快照为`vllm-project/vllm-ascend@f14377bf1ad8ad1584de0a8a017741e245d6e846`，源码证据见本仓`build_output/glm53-next-optimize/reference-target-20261029/SOURCE_INDEX.json`及其正文副本。指南的Performance Evaluation章节只链接AISBench／vLLM benchmark方法，未提供我们所需单机A3、无MTP、相同请求长度及并发的吞吐表；CI的130k PD用例也不能替代这种基准。

**第一阶段目标：**PTO3＋pypto-serving追平同机、同权重、同负载、关闭MTP的vLLM Ascend，在同一HTTP客户端口径下同时比较TTFT、TPOT和聚合输出TPS。先用两边都能承载的TP16／EP16／DP1、16逻辑NPU配置；8逻辑设备对照需另行验证vLLM容量及配置。固定纯文本、相同tokenizer和输入token IDs、相同实际输出长度、seed和采样／EOS策略，关闭prefix命中；优先保持latent KV为BF16，不把额外C8量化作为相同精度的控制。两引擎可以使用各自原生图执行能力，但须记录真实图模式和软件版本。不能把单步主机窗口与HTTP吞吐相除。

已沿用16设备C2、11输入／8输出的负载取得首轮vLLM结果：14.856511 token/s、TTFT中位数0.678174s、请求耗时中位数1.056874s。PTO3已有记录为1.926905 token/s、1.058365s、8.054267s。vLLM客户端以HTTP DONE计完整窗口，旧PTO3记录额外等待内部release／drain；软件栈及部分中间算术也不同，因此倍率仅作跨引擎追赶参考。原始计时及条件见下文。更长输出及更大并发的持续生成另建匹配协议，不沿用这一短输出数值。

**第二阶段目标：**再追赶vLLM开启MTP、C8及优化DP／TP配置后的产品性能，同时明确PTO3需要补齐的功能。官方当前A3单机示例开启MTP5及INT8 KV cache；我们尚无MTP，因此该默认命令不作为第一阶段的无MTP控制。recipe的950DT／MXFP4吞吐保留为远期跨平台参考，不用来计算当前差距倍数。

### vLLM Ascend源码逐项对照与新增优化候选

**审阅范围：**固定`vllm-ascend@f14377bf1ad8ad1584de0a8a017741e245d6e846`（提交时间2026-10-29 12:50:02 UTC），审阅GLM模型、KDA、MLA、KPool、MoE／W8A8和ACL Graph共24个文件。下载副本的SHA256全部与索引一致，证据为本仓`build_output/glm53-next-optimize/vllm-source-review-20261029/SOURCES_{CORE,OPS,BACKENDS}.json`。只读取第三方源码，没有导入或执行下载的Python文件。下面的“差异”是源码结论，不能替代任务时间线或实测收益。

| 优先级／关联项目 | vLLM实际实现 | PTO3现状与可采取的动作 |
|---|---|---|
| **高：opt41，专用KDA decode** | `models/glm5next/kda.py:680`对decode调用recurrent，对prefill调用chunk。实际主算子由`ops/kda.py`导入`fla_npu.ops.ascendc`，recurrent在算子内处理q/k归一化、gate和beta sigmoid；不能因为仓内有Triton同名文件就认定主路径用Triton。 | 当前Serving decode仍经过`forward_chunks`的chunk KDA，独立单步实现尚未接入。应优先研究纯decode入口，保留混批、初态、prefix恢复及BF16舍入契约；不能只替换函数名。输出回写在vLLM也仍有独立helper，不能声称整个KDA仅一个kernel。 |
| **高：新增opt46，六路输入投影合并** | `kda.py:288`／`:424`将同一hidden输入的q、k、v、b、f_a、g_a权重合并，一次投影后split；f_a／g_a在head-sharding ranks间保持复制。f_b、g_b仍分别投影，因此完整投影链是8路GEMM变成3路。 | `kda_projection.py:162`等仍分别计算上述八路。可以在驻留阶段打包六路权重，使用一个合并输出矩阵，再按既定布局取视图。先保持已有dtype、K维分块累加和BF16输出边界，检查beta padding及TP8／TP16的复制／切分。当前多个GEMM已位于同一SPMD组，不能把6→1 GEMM直接换算为Host提交次数或整网倍数。 |
| **高但有工具链依赖：opt42／opt19B** | `compilation/acl_graph.py:156`以batch descriptor缓存图，`:300`执行replay；官方A3示例使用FULL_DECODE_ONLY。按批次选择执行形状，并复用稳定地址的输入／输出。部分FULL路径在replay前仍有同步，不是零Host开销。 | `production_launch_plan()`仍是89次主干提交，辅助NPU LM-head另计；多请求decode的投影还按CAP128工作。分别研究更少提交和活跃query形状；Error104及动态tile相关编译阻塞仍存在，不能将图包装器直接套到现有runtime，也不能把已否决的静态CAP64重记成新方案。 |
| **中：新增opt47，同输入的MLA／Indexer投影合并** | `attention.py:183`合并q_a和kv_a；`model.py:748`的loader映射也将Indexer的wk与weights_proj合并。 | 目前对应投影分开。先分别研究相同dtype下的权重拼接、单次GEMM及split，保留各自TP／复制关系和累加边界。它与改变MLA量化计算是两个独立试验，不能合在一项中归因。 |
| **中，数值改动较大：新增opt48，MLA原生W8A8** | MLA线性层在checkpoint对应权重量化时保留quant_config，由W8A8 dynamic linear执行动态激活量化和quant matmul；kv_b显式保留BF16。 | 当前MLA加载器会将相关量化权重解量化成BF16，主干使用BF16线性计算。若改为原生W8A8，需要逐个核对实际checkpoint量化成员、scale、smooth、动态量化输入和输出舍入，再做组件及整网数值检查。不能归为纯布局优化，也不直接降低golden门槛。 |
| **中：opt44，MoE量化epilog与通信** | W8A8 dynamic路径支持GMM1＋SwiGLU＋重新量化融合，并传入swiglu_limit；通信有all-to-all、MC2和FUSED_MC2等条件分支。shared是否切TP取决于sequence-parallel配置。 | 我们已有gate/up合并INT8和EP路由，shared保持完整权重。只对实际时间线中占时的任务研究PTO3融合／重叠；不能假定A3默认启用全部融合通信，也不因vLLM的可选TP配置改变我们既定MoE契约。 |
| **小范围：opt40，固定卷积权重预打包** | `kda.py:530`在首次使用时准备合并卷积权重，之后复用。 | 与recipe给出的方向一致，生产路径每步34个KDA层仍调用`stack_conv_weight`。在权重驻留阶段一次准备即可作为独立小试验；不与六路GEMM合并混测。 |
| **待时间线定序：opt45，mHC阶段粗化** | `hc_pre`调用`npu_hc_pre_v2`，随后仍独立RMSNorm；`hc_post_pre`虽然名字含“post_pre”，实际先调用hc_post再hc_pre，并非一个跨边界融合kernel。 | 核对PTO3已生成的任务后，再合并统计、系数和collapse等有明确开销的阶段。不能仅凭vLLM函数名宣称已经有可照搬的post＋pre＋norm融合。 |

主要源码：[KDA模型与投影](https://github.com/vllm-project/vllm-ascend/blob/f14377bf1ad8ad1584de0a8a017741e245d6e846/vllm_ascend/models/glm5next/kda.py)、[AscendC KDA调用契约](https://github.com/vllm-project/vllm-ascend/blob/f14377bf1ad8ad1584de0a8a017741e245d6e846/vllm_ascend/ops/kda.py)、[MLA投影](https://github.com/vllm-project/vllm-ascend/blob/f14377bf1ad8ad1584de0a8a017741e245d6e846/vllm_ascend/models/glm5next/attention.py)、[模型、mHC与loader](https://github.com/vllm-project/vllm-ascend/blob/f14377bf1ad8ad1584de0a8a017741e245d6e846/vllm_ascend/models/glm5next/model.py)、[W8A8动态量化](https://github.com/vllm-project/vllm-ascend/blob/f14377bf1ad8ad1584de0a8a017741e245d6e846/vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py)、[ACL Graph执行](https://github.com/vllm-project/vllm-ascend/blob/f14377bf1ad8ad1584de0a8a017741e245d6e846/vllm_ascend/compilation/acl_graph.py)。

**需要单独标明的精度契约：**vLLM `sparse_attn_indexer_kpool.py:100`要求KPool压缩cache为BF16，未完成的raw tail及gate分数为FP32；我们当前的pool cache为INT8＋FP32 scale，并使用Hadamard量化打分。同一W8A8权重不代表所有cache／中间计算格式一致。该差异需要在性能报告中披露，并通过相同prompt的输出和必要的logits对照判断影响；本轮源码审阅不能将其判为精度bug或擅自改cache格式。vLLM prefill支持已有KDA初态，其模型代码可作为chunked prefill／prefix恢复的参考；recipe零初态样例不应直接替换这类流程。

**新增项目状态与实施次序：**opt46／47／48均为新候选，尚未修改kernel、构建或实测，不增加21项已完成试验。模型侧先推进opt41与opt46；减少主干下发的opt42具有较高潜力，优先推动编译器阻塞修复；opt47随后评估，opt44／45按时间线成本排序。opt40是预打包小项，排在主要decode／下发差距之后；opt48改变MLA量化路径，单独准备数值基线。每项仍按组件、状态／整层、TP8／TP16、匹配Serving协议逐项验收，不预填收益百分比。

**实测环境：**197启动前确认A3的16逻辑NPU无计算进程、driver为26.0.rc1，发布版权重100文件完整。原`vllm-ascend-dev-image:latest`创建于2026-01-21，内部vLLM为0.13.0、vllm-ascend为0.13.0rc2开发版且缺少fla_npu依赖，不能用作这个模型的参考。已独立准备官方`v0.30.0rc2-a3`镜像（2026-10-13 16:30:59 UTC创建、展开22.35GiB），原PTO3环境保留。官方与下载镜像站的index及arm64 manifest digest分别核对为`sha256:0676d2dbdd480c2c456aacbeee486681c7c33a81aa83faa1edc9168a0f4f7fb6`、`sha256:89f41d4b45df14461cb861e8844cc3ae2945295f1e4ca3888ec8a4725d792f93`。镜像内实查vLLM 0.30.0、vllm-ascend 0.30.0rc2、torch 2.10.0+cpu、torch-npu 2.10.0.post7、Transformers 5.14.1，`fla_npu`及`cann_ops_transformer`均存在。release源码已检查含KPool、合并KDA投影及AscendC KDA；不能把这份release镜像与10月29日的源码审阅快照混称同一提交。

**实际尝试与证据：**197的`<artifact-root>/glm53-vllm-reference-20261029/`保存准备、启动及原始日志；本仓对应`build_output/glm53-next-optimize/vllm-reference-20261029/`。R1的BF16 16×16 matmul真实设备检查通过，随后因新版CLI移除`--disable-log-requests`而退出，未进入整网计时；容器正常关闭、全部NPU释放。R1的13个文件已冻结并核对SHA256，保存在`attempt-r1/`，没有覆盖失败记录。R2仅修正该参数并参数化run ID，配置同一权重、TP16／EP16／DP1、无MTP／prefix、BF16 latent KV、FULL_DECODE_ONLY和纯文本并发2，整网和HTTP计时均完成。图捕获完成后，14:26:39 UTC实际记录`ACL graph replay is active`；启动时的通用Mamba卷积回退警告所指路径与GLM的独立wrapper不同，镜像内GLM卷积wrapper SHA256与release源码一致，调用`fla_npu.ops.ascendc`。无须依据这条通用警告将本次GLM判为eager执行。

### vLLM Ascend首轮实测：197、16逻辑NPU、相同短负载

**2026-10-29的真实结果，不是文档引用值。** 两边复用相同发布版W8A8 checkpoint和tokenizer，prompt为`Write a clear explanation of photosynthesis: Plants use sunlight`，11个输入token；关闭MTP／prefix，greedy、seed0、忽略EOS、固定8输出。vLLM实际完成5组预热、20组计时，每组2请求，40个计时请求共320输出token；从首个计时组发起到最后HTTP DONE为21.539377s。排除镜像准备、权重加载、图捕获及预热。本节是外部引擎参考测量，不计入21项PTO3优化试验。

| 实现与数据来源 | 总逻辑NPU／并发 | 聚合输出token/s | TTFT中位数（s） | HTTP请求耗时中位数（s） |
|---|---|---:|---:|---:|
| PTO3：已有单TP16／EP16记录 | 16／C2 | 1.926905 | 1.058365 | 8.054267 |
| PTO3：已有两个独立TP8／EP8记录 | 16／总C2，各实例C1 | 2.580460 | 0.806077 | 6.105944 |
| vLLM Ascend：本轮TP16／EP16，实际decode图replay | 16／C2 | **14.856511** | **0.678174** | **1.056874** |

vLLM的TPOT中位数为**52.290381ms**，按每个请求首／末非空输出片段时间差除以7计算；40个计时请求都收到8个非空输出片段和`finish_reason=length`。TTFT p99为0.968348s、TPOT p99为90.850257ms、HTTP请求耗时p99为1.144636s；这是40个请求的样本统计，尚无多轮稳定性或长输出结论。

**作为追赶参照：**同拓扑的吞吐参考比为14.856511／1.926905＝**7.71倍**；对已有两个TP8实例的总吞吐为**5.76倍**。PTO3仍沿用前次实际记录，没有为本节重跑；旧PTO3窗口含内部release／drain观察，vLLM窗口以HTTP DONE结束。两者还使用不同软件栈、MLA量化和KPool缓存算术，所以这些比值应保留条件，与某一PTO3优化点的匹配收益分开记录。第一阶段的具体短负载目标可定为本轮14.86 token/s，更长输出／更高并发需要另测。

**软件及数值范围：**PTO3已有数据对应PyPTO `ce0eb7c`（2026-10-27）、CANN9.0.0、runtime `d8f997f`、ISA `b1d460e`／PTOAS0.67；vLLM镜像实查CANN9.2.0、`flash-linear-attention-npu-a3==26.9.1+deva4a7958`，core为`ced6857`（2026-09-21），Ascend为`8fde6d4`（2026-10-13）。两边driver均为197的26.0.rc1。vLLM 39／40个计时请求输出`, water, and carbon dioxide to make`，与PTO3已有8个token一致；1个将最后的`make`换为`produce`，前7个token相同。输出token数和语义续写检查通过；未做跨引擎完整FP32 logits精度验收，不据此宣称逐字节数值等价。

**验收及清理：**`OFFLINE_AUDIT.json`从原始SSE重新解析usage／finish、重算320个输出、完整客户端窗口和中位数，并核对启动参数、模型就绪、实际图日志和关闭状态。它由当前agent离线复算，未宣称独立第二agent审计。R2正常退出0，容器停止；关闭后的`npu-smi`确认16逻辑设备无计算进程。原始数据为`measurement/report.json`、`server.log`、`run-report.json`、`runtime-source-metadata.json`和`tokenizer-check.json`，本地与197均保留。

**优化定序据此更新：**首字延迟的差距小于完整请求耗时的差距，应优先定位decode链路：opt41专用KDA单步递推、opt42减少主干下发／图执行能力、opt46合并KDA投影。opt42仍受编译器问题阻塞，先推动该问题并保留模型侧可独立推进的项目；opt40等小项随后实施。上述是结合已有PTO3 profile、源码差异和本次整体数据作出的优先级判断，尚无逐kernel时间线证明每项能贡献多少收益。

### 缓存对齐范围与vLLM优化配置补测

前面的“KV格式对齐”特指V0基准的MLA主latent KV均为BF16。各类缓存的数据类型对照如下；类型相同还不代表内存布局、中间计算或逐字节结果相同。

| 缓存 | PTO3 GLM-5.3-Flash | V0实测vLLM（未开C8） | 数据类型是否一致 |
|---|---|---|---|
| MLA主latent KV | BF16 | BF16 | 是 |
| Indexer原始key／gate | FP32 | FP32 | 是；PTO3保留分页历史，vLLM使用请求尾部ring |
| Indexer池化key | INT8＋FP32 scale | BF16 | **否** |
| KDA短卷积状态 | BF16 | BF16 | 是 |
| KDA递归状态 | FP32 | FP32 | 是 |

PTO3实际分配见Lib `prefill_fwd.py:2638`，MLA页大小为128；V0日志记录vLLM为对齐MLA／KDA／Indexer页而选择384-token attention页。vLLM实际core `ced6857`的`MambaStateDtypeCalculator.kda_state_dtype()`在auto时为递归状态选择FP32；Ascend release `8fde6d4`的GLM KDA调用该方法。因此存储类型只有部分对齐，不能把所有cache统称为BF16，也不能把INT8 Indexer池缓存等同于主KV的C8。

**2026-10-29新增授权：**逐项开启vLLM优化，补测其产品配置的性能。沿用已校验的同一image／权重及197的16逻辑NPU，PTO3源码和此前V0记录保持冻结。这些是外部参考试验，不增加21项PTO3优化完成计数。

| 参考编号 | 配置／工作负载 | 当前结果或状态 |
|---|---|---|
| **V0** | BF16主KV，无MTP／FlashComm1／prefix；C2、11输入／8输出 | 已完成：14.856511 token/s，TTFT中位数0.678174s，TPOT中位数52.290ms。 |
| **V1** | 仅改主KV为INT8（C8），其余保持V0 | 已完成：14.722844 token/s，TTFT中位数0.661058s，TPOT中位数54.026ms；40个计时请求／320输出均完成，正常关闭。相对V0本轮吞吐约−0.90%，这组短上下文未见收益；尚无多轮稳定性结论。 |
| **V2** | 在V1上增加MTP5，主／draft均申请图执行；同一短C2负载 | 已完成：15.864418 token/s，TTFT中位数0.622219s，有效TPOT中位数17.532ms；40计时请求／320输出，离线复算通过。比V1本轮约＋7.75%；含预热的计数为100次draft、500个提议token、293个接受的draft token，均不作为API吞吐分母／分子。 |
| **V3** | 在V2上增加FlashComm1；保持短C2负载 | 原小容量尝试因6／12图桶不满足TP16倍数而失败，未进入计时；CAP32以48／96／192桶完成后续实测，结果见V4及混合文本表。 |
| **V4** | 组合配置，扩大容量；160输入／128输出，C1／2／4／8／16／32 | 关闭／开启FlashComm1两组均完成。重复文本C32为770.08／679.43 TPS；另补混合文本C32为503.24／453.23 TPS，分开记录，不能跨负载计算优化倍率。 |
| **V5** | 增加prefix caching，超过实际attention页边界的公共前缀 | 已完成1296输入／128输出C2关闭／开启对照，实际命中约91.71%；TTFT p50为0.794→0.652秒，TPS为102.43→92.13，本批未取得聚合吞吐收益。 |

原始记录在本仓`build_output/glm53-next-optimize/vllm-optimized-20261029/`及197的`<artifact-root>/glm53-vllm-optimized-20261029/`。每次启动均有独立参数、源码hash、日志、SSE／usage及NPU关闭记录；使用197共享排他锁串行运行，保留失败尝试。V1已完成原始SSE／usage和客户端窗口的离线复算，并核对实际int8 KV、无MTP和图replay日志；C8路径自动将attention页大小调整为640，后续prefix负载据此暂设1296输入（2页＋16 token），组合配置启动后再次核对页大小。

### 同机同输入的 PTO3／vLLM 直接对照

2026-10-29 UTC完成。两侧均为197单机A3、16逻辑NPU（8张双芯物理卡）、TP16／EP16／DP1，同一发布版W8A8权重；MLA主KV为BF16、KDA卷积状态BF16／递归状态FP32，关闭MTP和prefix。各自保留原生执行能力，vLLM使用实际FULL_DECODE_ONLY图replay，shared overlap开启，FlashComm1关闭。

使用完全相同的CPU-only HTTP客户端、字符串prompt及发布版tokenizer IDs；160输入／32输出，1／2／4并发，每档1组预热、3组计时，分别96／192／384个计时输出token。两侧context512、step budget128、长prefill阈值32、max sequences4，greedy／seed0／ignore EOS。TPS为首组计时dispatch至最后HTTP DONE内的实际输出token数，不含启动、预热及关闭，也不再额外等待旧PTO控制中的内部release／drain观察。

| 并发 | PTO3 输出TPS | vLLM 输出TPS | vLLM／PTO3 |
|---:|---:|---:|---:|
| 1 | 0.799538 | 10.191591 | 12.75倍 |
| 2 | 1.593137 | 17.163558 | 10.77倍 |
| 4 | 3.114144 | 30.300921 | 9.73倍 |

| 引擎／并发 | TTFT p50／p99（s） | 有效TPOT p50／p99（ms） | HTTP请求耗时p50（s） |
|---|---:|---:|---:|
| PTO3 C1 | 6.078／6.082 | 1095.60／1096.99 | 40.032 |
| vLLM C1 | 2.501／2.508 | 22.35／22.38 | 3.192 |
| PTO3 C2 | 6.383／6.391 | 1090.64／1095.09 | 40.201 |
| vLLM C2 | 2.733／3.062 | 31.87／40.16 | 3.695 |
| PTO3 C4 | 6.889／9.837 | 1090.21／1113.56 | 41.048 |
| vLLM C4 | 3.370／3.497 | 26.36／58.20 | 4.203 |

**本轮结论：**聚合吞吐相差9.73～12.75倍。PTO3有效TPOT约1.09秒，vLLM为22～32毫秒；首字延迟差距约2.0～2.4倍，持续decode是更明显的差距。该指标包含调度、IPC、整网、输出及HTTP，不是单个kernel的时间，不能据此直接把全部差距归因于某一算子。

**匹配范围：**同一份W8A8 checkpoint不代表每个投影的算术相同。PTO3部分Attention投影加载后以BF16计算，vLLM保留原生量化路径；Indexer池化key仍为PTO3 INT8＋FP32 scale／vLLM BF16，页布局不同。软件栈沿用前述冻结的ce0／CANN9.0与vLLM 0.30.0／Ascend 0.30.0rc2／CANN9.2，未将尚未NPU验收的新PyPTO候选混入。197冻结副本的5个修改过的模型核心文件与当前优化分支SHA256一致。

原始SSE、usage、时钟和参数在`build_output/glm53-next-optimize/fair-comparison-20261029/`，`COMPARISON.json`及两侧`OFFLINE_AUDIT.json`已重新解析并核对完全相同的输入hash／prompt顺序。逐字输出一致的计时请求为C1 2/3、C2 1/6、C4 2/12；这是性能对照，不宣称跨引擎logits或文本完全等价。每档样本较少，p99仅描述本批观察，不作为生产SLO。

PTO首次尝试在模型启动前因前一个vLLM任务退出后HBM尚未释放至空闲阈值而中断，失败保留在`pto-r1/`。重试`pto-r2/`增加有界空闲等待，测量输入及计算实现不变，三档均通过且资源释放；真实受控SIGTERM退出码为−15，未使用强制关闭。

### vLLM优化组合：中英文混合文本与重复文本分开记录

2026-10-29 UTC完成关闭FlashComm1的混合文本系列，2026-10-30 UTC完成开启系列的重试。两侧均为同一197 A3／16逻辑NPU、发布版W8A8、TP16／EP16／DP1，C8主KV、MTP5、主模型实际FULL_DECODE_ONLY图replay（MTP draft同时申请图执行）、shared overlap开启；context2048、max sequences32、step budget8192、memory utilization0.92、prefix关闭。FlashComm1开启时实际图桶为48／96／192，关闭时为6／12／24／48／96／192。

输入固定160 token、输出固定128 token，采用8类中英文问题模板及不同参考编号，每档2组预热、5组计时；同一档开关两侧的输入文本／token IDs／顺序完全一致。与上一节的小块prefill、32输出对照是不同负载，只在各自表内比较。

| 并发 | FlashComm1关闭 TPS | FlashComm1开启 TPS | 开启相对关闭 |
|---:|---:|---:|---:|
| 1 | 46.91 | 32.56 | -30.59% |
| 2 | 67.61 | 47.76 | -29.35% |
| 4 | 99.05 | 110.05 | +11.11% |
| 8 | 151.64 | 164.80 | +8.68% |
| 16 | 234.83 | 199.12 | -15.21% |
| 32 | 503.24 | 453.23 | -9.94% |

**本轮最高：503.24输出token/s，32并发，C8＋MTP5、FlashComm1关闭。**160个计时请求各生成128 token，共20,480 token／40.69655秒；TTFT p50／p99为1.463／3.493秒，有效TPOT p50／p99为39.83／56.99毫秒，HTTP请求耗时p50为6.660秒。该档MTP提议接受率约33.90%（计数含预热），吞吐只计算API最终接受的输出，不把提议token计入。FlashComm1在4／8并发本轮略有收益，1／2／16／32并发下降，不能作为全负载通用收益。

**重复文本系列另列：**同样160输入／128输出时，32并发关闭／开启FlashComm1分别为770.08／679.43 token/s，但并发请求文本相同，输出反复续写光合作用语句，MTP接受率约99.6%。这个数值属于较容易预测的合成压力负载，不能代替混合文本结果，也不能与PTO3的不同长度／并发数值相除。8并发重复文本从202.37提高至302.38 token/s，也仅代表该负载。

**Prefix组合已实测：**1296输入（超过C8实际640-token页的两个边界）／128输出、2并发，MTP5／C8／FlashComm1关闭，mamba cache mode为align。关闭／开启prefix的TPS为102.43／92.13；TTFT p50为0.794／0.652秒。实际命中16,640／18,144个查询token（约91.71%，计数含预热），128-token续写文本与关闭一致；本批较慢请求使聚合吞吐下降，不能把首字p50改善写成稳定吞吐增益。

**测量边界：**这是固定长度raw completion压力测试，输入包含通用背景和参考编号，部分输出续写编号或复述提示。它不替代自然EOS、聊天质量、真实业务吞吐或跨引擎logits精度验收。每档只有一批测量，不能称为全配置上限或统计显著收益；PD分离、其他拓扑及最新源码未纳入本轮。

原始记录在`build_output/glm53-next-optimize/vllm-natural-20261029/`，`COMPARISON.json`、两组`OFFLINE_AUDIT.json`、原始SSE、usage、计时与MTP counters均已复算，开启组已核对真实FlashComm1日志。早先小容量图桶不匹配、以及释放延迟导致启动前检查中断的尝试原样保留；完成的重试只增加启动前有界空闲等待，计算和请求不变。

### DeepSeek 参考提交与可借鉴机制

这些提交是机制参考，不是 GLM 的收益预测。复用前必须重新核对硬件、量化格式、shape、状态布局、通信和执行契约。

| GLM 项目 | DeepSeek 参考提交 | 借鉴范围与限制 |
|---|---|---|
| opt13／opt14B | Serving `1aed42d`、`b1c19a3`、`9db614a`；Lib V4.1 `de44572` | 有界异步提交、槽内输出生命周期及 prepare／dispatch／reclaim 分工；必须适配 GLM 工作区和 collective 顺序。opt14A 原机制已否决，不能据此重启。 |
| opt15／opt31 | Serving `5e1d58b` | 共享前缀所有权和状态重建原则。GLM 仍需 Worker 确认的 KDA 快照与精确终点；这不是现成的 logits cache 实现，通用 KV hash 也不足以替代状态快照。 |
| opt16／opt17B | Serving DSpark `e114182`；Lib V4 `eec9235` | 元数据批处理、完整 slab 输入及 copy 复用。chunk 上限试验是 GLM 自己的配置研究，不是这些提交已证明的分块收益。 |
| opt18 | Lib V4.1 `3652e16`、`de44572` | 有界组合与减少 Host 提交；GLM 编译阻塞仍需单独解决。 |
| opt20 | Lib V4.1 `3627f8f`；Serving DSpark `bc8a830` | 设备输出路径和受限 greedy 专化；不能替代通用完整 logits 的精度验收。 |
| opt21／opt22 | Serving `a74a514` | 不可变元数据复用、固定参数预绑定；请求相关数据和新 epoch 不能被错误缓存。 |
| opt23 | Serving `0ae4fd8` | 内容改变后才更新；需保留活跃切换和槽复用语义。 |
| opt24／opt27 | Lib `eeda7fb`、`e3673ca` | MoE 事务、通信、活跃行成本及权重 tile 复用；按 GLM 实际 owner 形状分别测量。 |
| opt26A／opt26B | Lib `560e871`；Serving DSpark `e114182` | Indexer 准备范围、活跃页及每步元数据成本；26A 和26B 是不同阶段，不能共享收益结论。 |
| opt28 | Serving `39487b7` | 启动阶段准备和缓存原则；不等于 opt32 的 backbone-only 驻留实现，也不证明稳定生成吞吐提升。 |
| opt29 | Lib `2d70dc9`、`a47e652` | MLA 激活／操作数复用；A5 参考结果不代表 GLM-A3 收益。 |
| 布局候选回顾 | Lib V4 `3ff92eb`；V4.1 `a47e652` | dtype 正确的 NZ、流式权重访问和操作数复用。shared NZ 已采纳；routed NZ 和宽泛 BYPASS 已否决，不能重复计为待做新优化。 |
| opt34／opt35 | Lib `eb52f27`，PR #1505 | 先验证行有效性再合并 gather，以及减少无用 softmax／core 工作的原则。需重新证明 GLM 有效块顺序、mask、尾部及回退。 |
| opt30／opt32／opt36／opt37 | GLM 源码及已采纳的驻留权重／head 行数优化；结合 DeepSeek token 输入、驻留 head、末尾 query 行等原则 | 这些是 GLM 的具体缺口和新候选，没有可直接套用的 GLM-A3 实测收益。opt36 使用本机公开 Torch API；opt37 仍未实现。 |

**A5／A3 范围限定：** DeepSeek 某些提交的 A5、Ascend950DT／TP1 测量，不能作为 GLM 在 A3、TP8／TP16 上的性能结论。`eb52f27` 的 E2M1 反量化变化也不适用于 GLM 的 BF16 latent cache。相同思路只有在本机工具链、实际 shape 和状态契约下重新通过编译、数值及匹配测量，才能计入 GLM 报告；不绕过编译器、修改 pin 或承诺跨平台收益。

## 历史测量：分系列阅读

<details markdown="1">
<summary>展开历史性能表、原始轮次及取舍依据</summary>

C1/C2/C4 表示 1/2/4 个并发客户端；TPS 为聚合输出 tokens/s。下表保留原报告精度，“—”表示原文未记录该指标，未补算。**不同主机、输入长度或实现检查点的增益不能相乘；同系列中的每一项也只对其注明的控制版本有效。**

Serving 固定协议为 TP8/EP8/DP1、context512、step budget128，默认 chunk32、关闭 prefix；每档并发先预热 1 组，再计时 3 组；预热原始数据保留，性能统计排除预热。输入/输出长度见各系列。TTFT 从客户端提交开始；TPOT 是逐请求 7 个相邻输出间隔的平均值，再对请求取中位数。吞吐以全部计时输出 token 数除以客户端组总耗时，包含 prefill、调度/IPC、CPU embedding/head、sampling 和清理；不含 HTTP/网络、启动、预热与关闭。

较慢轮次全部保留，三轮观察不构成统计显著性结论。Torch 线程数是调用现场的配置值，不代表同样数量的物理核心始终在执行。保存输出的逐字节回归是与先前 PTO3 实现对照；独立算子仍用各自 golden，并未据此宣称独立的完整整网 CPU-golden 或长文本质量验收。

原始逐轮记录、任务号及保留的失败报告见Serving：`build_output/glm53-chunked-validation/chinese-report/reference-before-zh.md`。归档不再作为另一份待办文档更新。

### 系列 A：原主机，128 输入 / 8 输出，opt01–09

起点使用 8 个 64-GiB A3 逻辑设备，编译容量为 192；opt01 降为 128。opt08 被回滚，因此 opt09 的控制是 opt07。opt09 显式设置 `PYPTO_GLM_HOST_THREADS=16`：启动及 rank worker 保留 4 线程，上传后 owner 使用 16，关闭时恢复。

| 阶段 | 并发 | TTFT 中位数（s） | TPOT 中位数（s） | TPS | 相对本阶段控制的增幅 | 相对原始基线的增幅 |
|---|---:|---:|---:|---:|---:|---:|
| 原始基线 | 1 | 27.57 | 4.64 | 0.135 | — | — |
| 原始基线 | 2 | 26.05 | 4.67 | 0.277 | — | — |
| 原始基线 | 4 | 28.75 | 4.65 | 0.523 | — | — |
| opt01 | 1 | 19.11 | 3.03 | 0.160 | 18.24% | 18.24% |
| opt01 | 2 | 18.04 | 3.10 | 0.414 | 49.47% | 49.47% |
| opt01 | 4 | 21.24 | 3.12 | 0.762 | 45.66% | 45.66% |
| opt02 | 1 | 15.01 | 2.98 | 0.225 | 40.84% | 66.53% |
| opt02 | 2 | 14.19 | 2.87 | 0.462 | 11.47% | 66.62% |
| opt02 | 4 | 14.30 | 2.90 | 0.927 | 21.53% | 77.02% |
| opt03 | 1 | 5.22 | 1.04 | 0.651 | 189.39% | 381.92% |
| opt03 | 2 | 4.93 | 0.86 | 1.348 | 191.80% | 386.20% |
| opt03 | 4 | 4.82 | 0.91 | 2.607 | 181.31% | 397.96% |
| opt04 | 1 | 3.61 | 0.84 | 0.843 | 29.48% | 524.00% |
| opt04 | 2 | 3.63 | 0.83 | 1.681 | 24.72% | 506.37% |
| opt04 | 4 | 3.79 | 0.84 | 3.292 | 26.29% | 528.88% |
| opt05 | 1 | 3.31 | 0.78 | 0.909 | 7.81% | 572.73% |
| opt05 | 2 | 3.38 | 0.78 | 1.803 | 7.25% | 550.36% |
| opt05 | 4 | 3.65 | 0.79 | 3.489 | 5.98% | 566.48% |
| opt06 | 1 | 3.27 | 0.77 | 0.925 | 1.85% | 585.19% |
| opt06 | 2 | 3.34 | 0.77 | 1.838 | 1.94% | 563.00% |
| opt06 | 4 | 3.49 | 0.79 | 3.557 | 1.94% | 579.42% |
| opt07 | 1 | 3.24 | 0.74 | 0.953 | 2.96% | 605.50% |
| opt07 | 2 | 3.24 | 0.75 | 1.890 | 2.84% | 581.82% |
| opt07 | 4 | 3.44 | 0.74 | 3.699 | 4.01% | 606.63% |
| opt08（否决，控制 opt07） | 1 | — | — | 0.798 | — | — |
| opt08（否决，控制 opt07） | 2 | — | — | 1.604 | — | — |
| opt08（否决，控制 opt07） | 4 | — | — | 2.840 | — | — |
| opt09（控制 opt07） | 1 | 2.87 | 0.67 | 1.054 | 10.61% | 680.36% |
| opt09（控制 opt07） | 2 | 2.94 | 0.67 | 2.094 | 10.78% | 655.35% |
| opt09（控制 opt07） | 4 | 3.12 | 0.67 | 4.081 | 10.33% | 679.66% |

opt09 相对原始基线的吞吐约为 C1/C2/C4 的 **7.80/7.55/7.80 倍**。保留以下关键限定：

- opt01 的第二轮 C1 曾耗时 71.43 s，其中两个 CPU head API span 为 17.91 s、9.04 s；这轮未剔除，原因未定位。
- opt02 的独立 reset API 中位数：单槽 923.10→1.85 ms、两个稀疏槽 1850.58→1.39 ms（3 次预热、10 次计时）。这是主机 API 时间，不能直接当设备 kernel 时间。
- opt03 保留读取 fence 和全部 rank 的完成边界；异常 epoch 发布后会阻止后续状态操作。
- opt04 将每组 forward 由 12 降到 11，预热/计时 forward 由 144 降到 132；八个生成 token 和客户端协议不变。
- opt05 增加约 2.36 GiB 常驻 FP32 权重，去掉每次 head 的重复 widening；这里仍是当时的 32 行 CPU 投影。NPU 全 FP32 词表投影因 9 个逐点超差被否决，未降低门槛。
- opt06 仅跳过无本地 owner token 的 shared 计算。opt07 保留固定的 `block_table`、`kda_state_rows`、`pool_blocks`，每 rank 少 3 次 H2D、少传 16,704 B；动态物理分页要重新检查这一假设。
- opt08 独立 TP16 copy 阶段虽由 346.92→19.56 ms，计时窗口内 CPU head 却由 30.03→54.24 s，抵消约 8.40 s 复制相关节省；未定位 CPU 变慢原因，保留普通传输实现。

### 系列 B：另一台 A3 主机，128 输入 / 8 输出，opt10–13

opt10 在这台主机上使用新的 opt09 控制；不能接到系列 A 的 7.8 倍后继续乘。仍为 chunk32、启动 4 / owner16 线程、1 组预热和 3 组计时。

| 并发 | 同机 opt09 控制 TPS | opt10 shared NZ TPS | 观察增幅 | opt10 TTFT（s） | opt10 TPOT（s） |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.825 | 1.054 | +27.74% | 2.85 | 0.68 |
| 2 | 1.666 | 2.075 | +24.54% | 2.97 | 0.68 |
| 4 | 3.246 | 4.077 | +25.58% | 3.11 | 0.67 |

CPU head span 39.82→24.99 s，解释了总客户端时间减少 18.02 s 中的大部分；MoE FFN API 为 12.59→11.47 s。因此不能把整个宏观差值都归因于 NZ。独立 shared expert 组件在 16/17 行下的 device effective 中位数下降 **12.32%/16.39%**（5 次预热、100 次计时），支持保留布局改动。

| 候选 | 控制 | C1 TPS | C2 TPS | C4 TPS | 结论 |
|---|---|---:|---:|---:|---|
| opt11 跳过未完成 prefill 的 CPU head | opt10：1.054/2.075/4.077 | 0.759 | 1.568 | 2.570 | 否决 |
| opt12 routed NZ | 同一 opt10 控制 | 1.039 | 2.061 | 4.006 | 否决；分别 −1.40%/−0.64%/−1.75% |

opt11 减少了 99 次计时 head 中的 27 次；被移除的调用原耗时 6.70 s，但保留的 72 次由 18.29→52.30 s，客户端总时间 69.46→99.59 s。计算公式不变，原因未确认；每次 forward 执行 CPU head 的实现被保留。

opt12 独立 routed 组件在 16/32/128 行下延迟下降 9.83%/8.76%/5.52%，并通过最后一个 expert 及混合 expert 寻址检查；整网 MoE API 11.47→11.60 s、CPU head 24.99→25.18 s。组件收益未转为 Serving 收益，因此正式实现保留 shared NZ、routed ND。

opt13 使用同机 opt10 shared-NZ 控制：

| 并发 | 控制 TPS | opt13 TPS | 观察增幅 | TTFT 控制→候选（s） | TPOT 控制→候选（s） |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.054 | 1.245 | +18.09% | 2.848 → 2.397 | 0.683 → 0.571 |
| 2 | 2.075 | 2.485 | +19.77% | 2.968 → 2.412 | 0.679 → 0.572 |
| 4 | 4.077 | 4.910 | +20.43% | 3.109 → 2.463 | 0.674 → 0.578 |

opt13 最多保留两个 submit handle，成功完成后才推进已提交 epoch；读回和失败清理前排空队列，保留原始错误。TP8/TP16 的 25/29 份保存输出均逐字节一致。99 次计时 head span 总和为 24.99→25.24 s，客户端组总时间为 69.46→58.15 s。原有 `runtime.run` 观察器无法给 async `submit` 提供可比的逐程序 span，因此这些增幅是 Serving wall time 结果，不能表述成独立 kernel 加速或完整 worker pipeline 已完成。

### 系列 C：197，同机 160 输入 / 8 输出

共同条件为 context512、budget128、owner Torch16；默认 chunk32、prefix off。下表中 opt15/14A/17/33 均对照同一历史 32 行 head、chunk32、prefix-off 控制；opt19 的控制改为 opt33 的 16 行 head。opt15 的候选开启重复前缀命中，opt17 的候选只改 chunk。

| 项目 | 并发 | 控制→候选 TPS | 观察变化 | TTFT 控制→候选（s） | TPOT 控制→候选（s） |
|---|---:|---:|---:|---:|---:|
| opt15 prefix hit | 1 | 0.760→1.229 | +61.62% | 3.997 → 0.811 | 0.767 → 0.706 |
| opt15 prefix hit | 2 | 1.498→2.168 | +44.69% | 4.519 → 0.805 | 0.748 → 0.864 |
| opt15 prefix hit | 4 | 3.500→4.682 | +33.77% | 4.136 → 0.918 | 0.717 → 0.776 |
| opt14A（否决） | 1 | 0.760244→0.840571 | +10.57% | 3.997 →4.033 | 0.767 →0.782 |
| opt14A（否决） | 2 | 1.498230→1.614646 | +7.77% | 4.519 →4.208 | 0.748 →0.814 |
| opt14A（否决） | 4 | 3.499924→3.205955 | -8.40% | 4.136 →4.412 | 0.717 →0.818 |
| opt17 chunk64 | 1 | 0.760244→0.686392 | -9.71% | 3.997→2.941 | 0.767→1.333 |
| opt17 chunk64 | 2 | 1.498230→1.123868 | -24.99% | 4.519→5.032 | 0.748→0.950 |
| opt17 chunk64 | 4 | 3.499924→1.575020 | -55.00% | 4.136→7.119 | 0.717→1.386 |
| opt17 chunk128 | 1 | 0.760244→1.028072 | +35.23% | 3.997→1.740 | 0.767→0.837 |
| opt17 chunk128 | 2 | 1.498230→1.244312 | -16.95% | 4.519→2.134 | 0.748→1.563 |
| opt17 chunk128 | 4 | 3.499924→2.395166 | -31.57% | 4.136→4.255 | 0.717→1.091 |
| opt33 head16 | 1 | 0.760244→1.305986 | +71.79% | 3.997→2.571 | 0.767→0.504 |
| opt33 head16 | 2 | 1.498230→2.604624 | +73.85% | 4.519→2.636 | 0.748→0.502 |
| opt33 head16 | 4 | 3.499924→5.056876 | +44.49% | 4.136→2.774 | 0.717→0.511 |
| opt19 CAP64（否决） | 1 | 1.305986→1.302885 | -0.24% | 2.571→2.640 | 0.504→0.499 |
| opt19 CAP64（否决） | 2 | 2.604624→2.498481 | -4.08% | 2.636→2.690 | 0.502→0.502 |
| opt19 CAP64（否决） | 4 | 5.056876→5.061597 | +0.09% | 2.774→2.792 | 0.511→0.504 |

以下为原报告已经展开的全部三轮 TPS，按原执行次序保留；未剔除较慢轮次：

| 版本 | C1 第 1/2/3 轮 | C2 第 1/2/3 轮 | C4 第 1/2/3 轮 |
|---|---|---|---|
| 历史 chunk32 / prefix-off 控制 | 0.633989 / 0.827829 / 0.861467 | 1.551614 / 1.763983 / 1.264265 | 3.754169 / 3.495276 / 3.282019 |
| opt15 prefix hit | 0.974786 / 1.390283 / 1.435720 | 1.781750 / 2.536469 / 2.334258 | 4.047342 / 5.139860 / 5.021646 |
| opt14A | 0.875063 / 0.807851 / 0.841483 | 1.538006 / 1.614908 / 1.699034 | 3.125195 / 3.160043 / 3.340827 |
| opt17 chunk64 | 0.649570 / 0.652029 / 0.770698 | 0.765093 / 1.320203 / 1.653272 | 1.457702 / 1.310395 / 2.194918 |
| opt17 chunk128 | 1.093382 / 1.045962 / 0.954715 | 0.992967 / 1.191603 / 1.770906 | 2.695717 / 1.959527 / 2.693700 |
| opt33 head16 | 1.292518 / 1.312206 / 1.313446 | 2.620005 / 2.613582 / 2.580628 | 5.009008 / 5.105572 / 5.056969 |
| opt19 CAP64 | 1.297079 / 1.310316 / 1.301331 | 2.338095 / 2.588355 / 2.586084 | 5.078110 / 5.043643 / 5.063158 |

- **opt15：**21/21 次计时请求确实恢复了 128 个 token。每个并发组的三轮完整 forward/head 为 36→24；处理的 query token 为 501→117、1002→234、2004→468，合计少 2688 个。80 行完整 FP32 logits 与冷跑对照逐字节一致。收益限于重复前缀的 TTFT/吞吐；C2/C4 的 TPOT 反而上升，不能写成 decode 普遍变快。一次 capture 和 38 次 restore 的同步 API span 合计 75.94 ms，包含主机开销。
- **opt14A：**TP8/TP16 及真实 Serving 数值、请求生命周期通过；C4 吞吐与延迟变差，故回滚。120 个宽泛的“metadata/前驱 step”区间重叠没有证明未完成 native 程序或 CPU head 的重叠。两个原始 Serving 报告均保留 exit1：一个后处理误判合法的延后准备，另一个误把相同生成配置的 tuple/list 序列化形式当成不一致；保存数据复核确认 80 行 logits、全部历史及 12 组计时有效，不重跑模型、不改写失败报告。opt14B 是另一个待验机制。
- **opt17：**两种拓扑各 156 次 full forward 通过。chunk64 的计时 forward/head 总数为 30/30/45，chunk128 为 27/33/42，控制均为 36。64 虽减少 C1/C2 调用，head 总耗时却升至 24.275/31.755/44.446 s；128 为 13.109/26.304/23.633 s，控制为 18.281/18.776/14.592 s。保留默认32；128 只在所测单请求 160/8 负载上表现更好，不构成通用建议。后续 opt33 改了 head 检查点，不能直接沿用这个 chunk128 结论；长输出/上下文另列 opt17B。
- **opt33：**保留每次 forward 的 head、全 rank 的 32 行读回及有限值检查，只将 CPU 投影输入切为实际返回的 16 行。每个并发仍执行 36 次计时 head；head span 中位数 377.32/376.80/371.66→188.97/185.69/191.29 ms，总和 18.281/18.776/14.592→6.775/6.719/6.899 s。大部分观察到的差值在 head，其他部分也有变化，且历史控制未与候选交错计时；不把宏观变化当作独立因果百分比。其报告 `d11af93f…` 成为后续相同负载、默认32、prefix-off 的新控制。
- **opt19：**CAP64 的注册、epoch、流尾部和缓存状态通过 TP8/TP16，Serving80 行及完整生成历史也一致；但 C2 变慢，C4 的 +0.09% 没有建立有效收益。三并发各仍有 36 次计时 head，耗时总和 6.974/7.016/6.978 s；较慢的首轮 C2 完整保留。已回滚，恢复已采纳的 opt33；没有为寻找更好结果重测。

这些 head span 和“inclusive forward 减 head”都包含主机执行、复制或等待，不能作为独立设备 kernel 时间，也没有解决 opt16 的波动根因。

### CPU 与独立组件：与 Serving 系列分开

**opt16 的 NUMA/亲和性探针**仅加载 LM-head，固定字节 oracle。20 次计时/分支的中位数为未绑定 237.287 ms、绑定 236.765 ms（约 0.22%）；范围分别为 233.410–868.971、233.729–373.700 ms，所有样本保留。51 次完整输出逐字节一致。此前观察器的 `torch.equal` 会创建额外线程，已改为精确 raw-byte 校验；早期失败尝试均保留。该 CPU 协议对双方均设置 `OPENBLAS_NUM_THREADS=1`，不假定等价于 Serving 环境。NUMA 页面随阶段变化、THP 保持不变；未保留亲和性改动，根因仍未定位。

**opt16B：**Torch16 固定；每个 BLAS 设置先验收两份 16 行 oracle，再做 5 次预热、100 次计时，四个位置各出现 25 次。428 次完整输出均逐字节一致。

| BLAS getter | head wall 中位数（ms） | 最小 / 最大（ms） | 进程 CPU-time / wall 中位数 |
|---:|---:|---:|---:|
| 4 | 167.186 | 163.634 / 281.973 | 7.265 |
| 8 | 167.002 | 164.003 / 281.838 | 7.265 |
| 16 | 167.060 | 164.009 / 283.147 | 7.253 |
| 32 | 167.023 | 162.833 / 281.551 | 7.251 |

最多 0.11% 的差异没有实用收益，未应用配置钩子或进入 Serving 试验。CPU/wall 比值包含进程线程，不能当作物理核心数；它也没有证明“四核瓶颈”。

**opt25：**13 对 native 用例的 selection/validity/expanded-position 字节全部一致。随后同一已准备的 worker 内共 840 次调用，4 个负载各版本 5 次预热、100 次计时，AB/BA 平衡。采用官方 L2 `TraceInvocation.effective_us` 中位数，所有 raw 样本保留。

| visible 数量 | 旧 2048 leaf（μs） | 新 512 leaf（μs） | wall 变化 |
|---:|---:|---:|---:|
| 0，无排序控制 | 87.181 | 85.6495 | −1.76% |
| 32，目标范围 | 88.2705 | 90.959 | +3.05% |
| 128，目标范围 | 91.070 | 91.2705 | +0.22% |
| 512，扩展范围 | 104.0205 | 103.3705 | −0.62% |

512 超出 context512 部署的 128 个完整 pool 范围；目标 32/128 未变快，故组件阶段即否决，没有加载整网或宣称 Serving 收益。

**opt33 的独立 CPU 组件测量：**原主机、Torch16、显式 OpenBLAS1，5 次预热/20 次计时，AB/BA 交错；完整 RMSNorm+GEMM 中位数 364.82→305.61 ms（−16.23%），范围 231.45–390.40 与 165.89–327.26 ms。两份 16 行结果覆盖原32行 oracle 的全部 4,956,160 个值，50 份保留输出也逐字节一致。它与后续 197 Serving 测量分开，不能用 −16.23% 替代或解释整网的全部差值。

### 原始记录索引

以下目录为原报告已记录的产物入口；保留其 raw 采样、完整输出与原始失败记录，不另建一套可变计划。系列 A/B 未在正文展开的逐轮数值不补写，可继续通过Serving：`build_output/glm53-chunked-validation/chinese-report/reference-before-zh.md`追溯。

| 项目 | 原始记录入口或标识 |
|---|---|
| opt14A / opt15 | 只读归档中相应的完整历史表与三轮数据；opt14A 的原始 exit1/失败报告不改写 |
| opt17 chunk32 控制 | 报告 SHA256 `76eae991b0094ebf439344a20d1440ef6c0915f89b6c20ea3bd171692f8cbc81` |
| opt17 chunk64 | Lib：`build_output/glm53-perf-iterations/chunk-sweep-opt17/remote-c64-completed/case/`；报告 `b87c79259ddb80e84126d19c92431ca7ef107d4a40b9f1b5143de27027a52bc4` |
| opt17 chunk128 | Lib：`build_output/glm53-perf-iterations/chunk-sweep-opt17/remote-c128-completed/case/`；报告 `fff5b469b1d4ad6fbf6c53de256219efb728d298fce492c2b6a9c194762db929`；实验父目录 `FINAL_HANDOFF.json` 汇总全部结果 |
| opt19 Serving | Lib：`build_output/glm53-perf-iterations/static-attention-opt19-serving-preparation/completed-r2/`；报告 `c468aa75e069d2f3982d480d00a02e481cc91df5b7f1c83b3bd8a186f61db45b`，复核 `bec190d32158bf2ae42cac96bb485f2a233f959690a0e43a999c1a6adb5e74e7` |
| opt25 | 数值任务 `task_20261024_213712_412152426268`；唯一性能任务 `task_20261024_214544_41858109455`，完整 840 次 raw 调用保留 |
| opt16B | Lib：`build_output/glm53-perf-iterations/cpu-blas-threads-opt16B/`；报告 `4f4fdcfaf866283abafd153841a103f4589fc1dc8f2b0d3f1e7808f37521fabf`，raw 采样 `5e8a0634fae3d0bef68dffe40778d251df9ead47508cd65ee75bf63cc276e6d9` |
| opt33 CPU 组件 | Lib：`build_output/glm53-perf-iterations/cpu-head-rows-opt33/report.json`；SHA256 `539b9ddfbb3922cc19bff459470c0e35dcec34f1f1d68846ac535160f1f19f5f` |
| opt33 Serving | Lib：`build_output/glm53-perf-iterations/cpu-head-rows-opt33/serving-197/`；新控制报告 `d11af93fae291654d1ffb6a51c61835c589ea6f520a4264e1a30ba05ab3b16c2`；head 归因记录 `67a613849a096b01bc939d34b22bcaba390ede2b4e659c967476c60e73444580` |

</details>

## 每项优化怎样验收

状态统一为：**待实施 → 实施中 → 数值通过 → 性能已测 → 采纳或否决**。遇到编译器或接口限制时标为“受阻”，记录最小复现和恢复条件。

| 关卡 | 需要记录什么 |
|---|---|
| 明确改动 | 改动的位置、解决的瓶颈、参考的 DeepSeek 机制、源代码或提交标识。 |
| 数值与生命周期 | 拓扑、shape、冻结输入、原有误差门槛、完整 logits/输出、缓存状态；相关的取消、槽位复用、异常排空和关闭行为。 |
| 公平性能对照 | 同一主机和设备、同一权重与负载；每次只验收一个正式机制。保留既有兼容产物，DSL、shape 或 ABI 改变时重新编译。 |
| 采样 | Serving 默认每并发预热 1 组、采样 3 组，全部轮次保留。组件试验使用各自预先冻结的轮次和交错顺序。启动、权重加载和关闭时间单列。 |
| 结果 | 首字延迟、token 间隔、总吞吐；条件允许时补组件/设备时间、拷贝与调用次数、内存。注明 CPU 波动、主机变化等归因限制。 |
| 结论 | 有收益且正确则采纳并记录提交；无收益则否决、还原候选并保留数据；未完成则列出缺失关卡和下一步。 |

新的缓存命中场景、输入/输出长度或上下文上限，分别建立匹配对照。原始失败记录、所有采样轮次和冻结参考结果保留；不会通过反复挑选基线、丢弃慢轮次或放宽数值门槛来获得收益结论。

### 后续更新格式

每完成一项，在本文件更新对应总表，并追加以下记录：

```text
日期 / 优化 ID：
改动与参考提交：
数值验证：拓扑、输入、门槛、输出/缓存/异常检查及结果
性能设置：主机、权重、长度、并发、预热、采样、对照来源
实测：TTFT / token 间隔 / 总吞吐，原始轮次所在路径
结论：采纳（提交）/ 否决（已还原）/ 受阻（恢复条件）
剩余事项与下一项：
```

## 数据和历史记录在哪里

以下路径以各自仓库根目录为起点。原始 JSON、张量、日志和编译产物是本地验证证据，不作为源码提交。

| 内容 | 位置 |
|---|---|
| 本中文总报告 | Serving：`docs/models/glm53-optimization.md`。后续统一更新这里。 |
| 运行命令和参数 | Serving：[GLM 模型使用说明](glm53.md)。 |
| 试验台账 | Serving：`build_output/glm53-chunked-validation/performance-iterations.json`。 |
| 原机器早期性能系列 | Serving：`build_output/glm53-chunked-validation/` 下各步报告和原始采样。 |
| 最新 opt33 Serving 基线 | Lib：`build_output/glm53-perf-iterations/cpu-head-rows-opt33/serving-197/report.json`，SHA256 `d11af93fae291654d1ffb6a51c61835c589ea6f520a4264e1a30ba05ab3b16c2`。 |
| opt13、opt15 及后续组件/整网证据 | Lib：`build_output/glm53-perf-iterations/` 下对应优化目录。具体文件见各项说明。 |
| 整理前完整进度日志 | Serving：`build_output/glm53-chunked-validation/chinese-report/reference-before-zh.md`。原 2247 行逐字节归档，SHA256 `deac715dab4c024cfe014d61b6133228bba5ffc3795c0cbabaac1b42acd73860`。原始跟踪内容同时保留在 Git 历史中。 |

中文整理保留优化编号、采纳/否决结论、性能系列、原始数据和失败事实；不再逐条重复加载、进程检查和源文件校验的流水日志。详细审计信息仍保存在上述原始证据中。
