# DPLB 每日进度追踪

按日期追加记录，保留历史进度；同一天有新结果时追加“补充更新”。各条记录中的“下一步”代表当时的安排，以最新记录为准。

## 2026-11-18

### 已完成

- 完成第一阶段 Host DFX 开发，支持记录 DPLB 的请求分配、前缀命中和分组负载变化。
- 完成 Host 测试和 16 卡真实 NPU 验证。

### 当前效果

- 服务正常完成 10 个请求，两轮 DFX 采集均完整通过。
- 记录到 4 次前缀命中，分组负载能够正常分配并在请求结束后归零。
- 当前已具备 DPLB 运行过程的可观测和问题定位能力。

### 下一步

- 对比 DPLB 开关前后的性能和采集开销。
- 补充高并发、抢占、取消和容量不足等边界场景。
- 根据采集数据继续优化 DPLB 分组策略。

## 2026-11-19（前次记录）

### 已完成

- 完成请求分配观测和 M2 Host 调度快照，能够关联每步计划、分组负载与 worker 返回。
- Host 回归累计 820 项通过；本轮 16 卡 NPU 的 off/load 各完成 10 个请求，load 的 49 个 Step 全部配对。
- 固定串行顺序复核已完成：off/load 各 10 个请求，load 的 105 个 Step 全部配对；文本严格一致率为 8/10。

### 当前效果

- 已能解释请求分到哪组、每步计划处理什么，并修复异步结果与采集窗口错配问题。
- 上次设备异常定位到新旧构建产物错配，已用私有构建目录规避；本轮未复现。
- 并发复核文本一致率为 9/10，串行复核为 8/10；usage 和 finish reason 均一致，尚未通过严格文本等价验收。

### 下一步

- 区分模型在相同输入下的自然生成差异与 DFX 影响，必要时增加确定性输出或 token 级比对；继续补齐 M2 的 runner/L3 调用关联。
- 后续接入设备计时，再开展采集开销和 DPLB 策略收益实验。

## 2026-11-19（补充更新：对齐整体开发计划）

### 已完成

- 完成 M1 请求分配观测和 M2 Host 调度快照，能够关联分组决策、每步计划与 worker 返回。
- Host 回归分批累计去重 820 项通过；16 卡并发、串行对照均完成，off/load 每轮各完成 10 个请求、320 个输出 token，load 分别完成 49、105 个 Step 配对。

### 当前效果

- 已能解释请求分到哪组、每步计划处理什么；修复了请求身份、异步结果和采集窗口错配问题。
- 设备异常定位到新旧构建产物错配，已用私有构建目录规避，重跑未复现。
- 并发对照 9/10、串行对照 8/10 文本一致；生成差异原因待查，严格输出等价、采集开销和 DPLB 性能收益仍未验证。

### 下一步

- 主线补齐 M2：串起 Step、worker/runner 准备、L3 调用及执行槽位/复用代次，完成从调度到实际提交的关联。
- 开展 M4 最小设备计时原型；用受控重复实验和 token 级证据继续定位输出差异。
- M2 稳定后补 M3 分析报告，设备计时接通后进入 M5 开销验收和 DPLB 开关收益实验。

## 2026-11-19（补充更新：提交 PR1）

- **已完成**：基于最新 main 整理并提交 [PR1 草稿 #347](https://github.com/hw-native-sys/pypto-serving/pull/347)，交付 M1 Host 分配与归属诊断；#336 策略迁移保留为独立前置 commit，M2 留在原开发分支。
- **当前效果**：647 项 Serving/CLI 测试通过；GitHub 静态检查和文档构建通过，设备 CI 正在排队或运行。尚未完成新基线的 NPU 验收。
- **下一步**：评审策略前置提交和 M1，跟进设备 CI；前置策略合入后精简 PR1，再继续 PR2 的 runner/L3 调用关联。

## 2026-11-19（补充更新：SWA 设备计时原型）

- **已完成**：在独立分支 `feat/dplb-device-timing-prototype` 增加真实 SWA 子程序的 8 卡实验入口和离线分析，支持核对 runtime 身份、统计逐 rank 与逻辑 DP 组时间；73 项相关测试和 CPU 编译通过。
- **当前效果**：已具备采集、关联和缺失数据识别能力；实机任务 `task_20261119_121454_27173883306` 正在排队，尚无实机验收结论。当前范围是独立 SWA 子程序，不代表完整 attention/MoE 或 Serving 逐层计时。
- **下一步**：回收 8 卡数值与原始计时验证，补齐 Serving Step → runner/L3 的关联和融合图语义边界，再进入完整设备成本采集。

## 2026-11-19（补充更新：SWA 设备计时实机验证）

- **已完成**：8 卡真实 NPU SWA 子程序计时实验；8 个 rank 的 golden 全部通过，采集开关前后有效输出逐位一致，每个 rank 生成 74 条 AICore 原始记录。
- **当前效果**：两个演示 DP 组的观测 `attention_busy_union_max_us` 为 48.62µs、44.46µs，`attention_envelope_max_us` 为 59.18µs、52.46µs。raw 文件未携带 retained-run coverage envelope，已明确标记 coverage unavailable，结果仅为 observed。
- **下一步**：继续 M2 主线，补 Serving Step → runner/L3 invocation 和融合图层语义；之后再做完整 attention/MoE/通信/等待计时与开销验收。

## 2026-11-19（补充更新：M2 跨层 invocation 关联）

### 已完成

- 为每个 Serving Step 增加 `invocation_id`、执行槽位和槽位复用代次，并通过 IPC 传到 worker。
- 将 invocation 上下文传到 Prefill/Decode batch、executor、runner 和 L3 的同步/异步 profile span。
- DFX `step_plan` 与 `step_outcome` 使用同一 invocation，避免异步结果串联到错误的 Step。

### 当前效果

- 已建立 Serving → worker/runner → L3 的 Host 关联链路，能定位一次 Step 对应的 L3 调用。
- `zm_pypto` 环境下 serving 相关 570 项、model/common 75 项测试通过；Ruff 和文件头检查通过。
- 当前只完成 Host identity bridge，尚未拿到 runtime rank/task、attention/MoE 设备时间。

### 下一步

- 将 M4 P1 timing adapter 接入一次真实融合调用，补齐 runtime rank、clock domain 和 coverage 关联。
- 在此基础上实现 M3 分析报告，再开展完整设备 timing、DFX 开销和 DPLB off/on 收益实验。

## 2026-11-20（M4 P2：真实 fused L3 语义关联）

### 已完成

- 为真实 fused L3 编译 handle 增加显式 `attention_block_v1`、attention 类型和覆盖层号元数据。
- 将语义元数据与 `invocation_id`、Step、执行槽位和复用代次一起写入同步 run、异步 submit、worker wait 的 profile span。
- 将 M4 P1 timing adapter 接入 Serving invocation manifest，校验 runtime rank、dispatch identity、rank-local clock 和 coverage，再汇总逻辑 DP 组的 rank duration。
- 新增 14 项 timing adapter 单测；`zm_pypto` 环境下 Serving 相关 583 项、DeepSeek/Qwen executor 116 项测试通过。

### 当前效果

- 已建立 `Serving Step → fused L3 invocation → semantic scope → runtime rank/task → attention raw task` 的可验证数据合同，并在一次 8 卡 Serving 调用中完成 rank/task join。
- adapter 保留原始任务、时钟域和覆盖状态。缺失或不完整时输出 `partial` 或 `observed`，不会把未知时间当成零。
- 真实 raw records 已回收并输出 `observed`，`timing` 配置仍保持不可用；通信、等待和 full-step 仍为空。

### 下一步

- 修正请求超时和 profile 收尾后，再运行一次开启 chip-swimlane 的 Serving fused 调用，补齐完整 profile trace。
- 基于已生成的 invocation manifest 补 M3 慢组、rank skew 和覆盖报告。
- 完成真实采集后再评估双执行影响、DFX 开销和 DPLB off/on 收益。

## 2026-11-20（M4 P2 补充更新：8 卡 Serving NPU raw join）

### 已完成

- 提交 8 卡最小 Serving fused 调用，服务完成模型加载、KV 初始化和单次 completion 请求。
- 回收 8 个 rank 的 `chip_swimlane_records.json`、`dispatch_identity` 和 `dispatch_program` sidecar，使用 run 5 identity 组成一次逻辑 DP 组 manifest。
- 离线 adapter 完成真实 raw records 的 rank、dispatch identity、clock domain 和逻辑 DP 组汇总。

### 当前效果

- 真实 NPU 结果为 `observed`，8 rank 的 `attention_busy_union_max_us` 为 92081.88，`attention_envelope_max_us` 为 121479.96。
- raw 文件没有 retained-run coverage envelope，通信、MoE、等待和 full-step 时间继续保持不可用，结果不能升级为 `complete`。
- 请求客户端在 DFX 输出落盘后因 30 秒超时退出，没有生成完整 profile trace 和 response 文件；这属于实验脚本收尾问题。

### 下一步

- 将 completion 请求超时提高并补齐 profile stop 收尾，重新采集完整 trace。
- 基于已验证的 invocation manifest 进入 M3，输出慢组、rank skew 和 coverage 报告。

## 2026-11-20（M4 P2 补充更新：完整 profile trace NPU 重跑）

### 已完成

- 任务 `task_20261120_123742_305833311126` 在 8 卡 NPU 上以 exit=0 完成。completion 请求返回 HTTP 200，prompt 4、输出 2，服务端记录 e2e 367.84 秒。
- `/start_profile` 和 `/stop_profile` 均成功，profile stop 合并 69 个事件，生成完整 `trace.json` 和 `response.json`。Host trace 包含 3 个 invocation，fused span 带有 `attention_block_v1`、43 个 layer id 以及 SWA/CSA/HCA 语义。
- runtime 侧 8 个 rank 均生成 d2 dispatch 的 identity/program sidecar 和 AICore raw records。50 MHz rank-local clock 下原始记录全部有效，两个逻辑 DP 组均可汇总。

### 当前效果

- 本次重跑解决了上一轮 30 秒客户端超时导致的 trace 缺失问题，已完成“Serving 请求 → Host invocation/profile → runtime rank/task”的实机 smoke 验证。
- raw timing 仍为 `observed`。group 0 的 busy union 最大值为 82987.64µs、envelope 最大值为 115684.08µs，group 1 分别为 24745.44µs、103015.92µs。
- runtime raw 文件没有 retained-run coverage envelope，通信、MoE、等待和 full-step 时间继续保持不可用。profile manifest 仍显示 1 个 open request，进程退出时有 8 个 shared memory 泄漏警告。

### 下一步

- 将本次 invocation 和 8 rank raw records 固化为 M3 慢组、rank skew、coverage 报告。
- 补齐 runtime coverage 和请求释放事件，再验证完整设备计时。
- 单独测量 profile、load 和 timing 的 DFX 开销，最后进行 DPLB off/on 受控收益对比。

## 2026-11-20（M3 首版离线分析报告）

### 已完成

- 基于任务 `task_20261120_123742_305833311126` 的 profile trace、scheduled-load JSONL、8 个 rank raw records 和 sidecar，生成 M3 JSON/Markdown 报告。
- 报告串联了请求、3 个 invocation、Prefill/Decode 阶段、placement 选择、逻辑 DP 组、rank-local AICore 时间和 coverage 状态。
- 输出 group 0、group 1 的 busy union、envelope、均值和 rank skew，保留 raw records、时钟频率和不可用指标说明。

### 当前效果

- 报告状态为 `observed`。8 个 rank 共读取 862688 条 raw records，全部有效。
- group 0 的 busy union max/mean/skew 为 82987.64/79483.05/7089.92µs，group 1 为 24745.44/24329.78/806.62µs。
- coverage 仍为 unavailable，通信、MoE、等待和 full-step 暂不可计算。报告记录了 `open_request_count=1` 和 8 个 shared memory 泄漏警告。

### 下一步

- 补 runtime coverage metadata 和请求释放事件，完成 M4 P2 收尾。
- 在此报告基础上增加 idle/wait 和完整 coverage 指标。
- 进入 DFX off/load/timing 开销对照。

报告文件位于 `artifacts/dplb-m4-p2/20261120-serving-fused-r6/m3-report/`。
