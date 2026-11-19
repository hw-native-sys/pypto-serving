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
