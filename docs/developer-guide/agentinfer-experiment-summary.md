# AgentInfer DeepSeek 多 Agent 实验总结

## 结论

本次实验使用 4 个并发的真实 OpenCode Agent，通过标准 OpenAI-compatible Chat API 访问 DeepSeek V4 服务。服务以 `--agentinfer-mode enforce` 启动，4 个 Agent 分别使用独立 workspace 和独立 session。

实验整体通过：4 个 Agent 都完成了代码修改和测试，4 个 AgentInfer Program 均被正确创建，多轮请求保持了各自的 session 关联，服务端没有出现 HCCL、Fabric 或 ring 错误。

本次实验验证了并发 Agent 的 session/Program 隔离和多轮生命周期管理，但没有触发 KV 容量不足时的排队等待分支。

## 实验目标

验证以下真实链路是否能够工作：

```text
OpenCode Agent
  -> OpenAI-compatible Chat API
  -> AgentInfer session / Program
  -> AgentInfer admission
  -> DeepSeek V4 生成工具调用
  -> OpenCode 执行 bash
  -> 同一 session 继续请求
  -> 任务测试完成
```

实验不修改 OpenCode，不给客户端增加适配层，也不手工注入 `agent_hint`。Agent 通过标准 API 和标准 `--dir` 参数访问服务。

## 环境与配置

实验分支和代码状态：

- 分支：`feat/agentinfer-pr2`
- HEAD：`b19c3c5200216a18c8b455dbb9a52aebf2942aab`
- 模型：`/data/models/dsv4-flash-w8a8`
- Agent：OpenCode CLI
- Agent 数量：4
- 物理设备：由 `task-submit` 分配为 NPU 8-15
- 服务内部设备参数：`--devices 0,1,2,3,4,5,6,7`

服务的关键参数如下：

```text
--backend npu
--platform a2a3
--dp 8 --ep 8 --tp 1
--block-size 128
--max-model-len 4096
--max-num-seqs 8
--npu-memory-utilization 0.94
--max-num-batched-tokens 1024
--num-speculative-tokens 1
--enable-prefix-caching
--ring-dep-pool 131072
--ring-task-window 131072
--ring-heap 2147483648
--agentinfer-mode enforce
```

Agent 使用的模型上下文为 4096，输出上限为 512。4 个 Agent 各自拥有一个独立的任务目录，避免互相修改文件。

## Agent 任务

任务使用 `status_count` fixture。每个 Agent 都执行同一个真实编码任务：

1. 阅读 `README.md` 和 `report.py`。
2. 修复 `count_statuses` 中把 `error` 统计为成功的问题。
3. 只能修改 `report.py`。
4. 执行 `python -m unittest -v`。
5. 在同一个 session 中再次执行测试，确认修改仍然有效。

原始错误逻辑如下：

```python
successful = sum(record["status"] == "error" for record in records)
```

Agent 将其修改为按 `status == "ok"` 统计成功请求。

## 执行过程

### 服务启动

`task-submit` 分配 8 张 NPU 后启动服务。DeepSeek V4 完成 43 层权重加载，8 个 chip worker 全部 ready，随后完成驻留权重上传和 KV Cache 初始化。

服务端关键启动日志包括：

```text
allocated DeepSeekV4 cache: slots=20, max_seq_len=4096
AgentInfer admission configured
Application startup complete
```

### 首轮 Agent 请求

4 个 OpenCode 进程并发向同一个服务地址发送请求，但每个进程拥有不同的 session。服务端为每个 session 创建一个 `:lead` Program，并根据 AgentInfer 策略执行放行。

首个请求触发了 `direct_admission_promote`，后续请求出现了 `scheduled_progress_promote`。这说明多个 session 同时存在时，Progress-TTL 仍然能够为请求维护进度优先级。

### 工具调用和 follow-up

每个 Agent 都通过 bash 读取文件、修改 `report.py` 并执行单测。首轮完成后，脚本使用原 session ID 发起 follow-up，服务端通过 `lifecycle=continue` 将请求关联回原来的 Program。

## 实验结果

### Agent 结果

| Agent | Session | 首轮 | Follow-up | 同一 Session | 独立单测 | 目标文件修改 | 受保护文件未变 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `ses_e7f7a0589ffeZg4gsFm6h81i53` | 通过 | 通过 | 是 | 通过 | 是 | 是 |
| 2 | `ses_e7f7a0566ffewDSKOH223PNaZf` | 通过 | 通过 | 是 | 通过 | 是 | 是 |
| 3 | `ses_e7f7a0657ffeFW47MHQLJfX2mS` | 通过 | 通过 | 是 | 通过 | 是 | 是 |
| 4 | `ses_e7f7a052bffeKnWfC1Wb6PsAKy` | 通过 | 通过 | 是 | 通过 | 是 | 是 |

### 服务端结果

| 指标 | 结果 |
| --- | --- |
| AgentInfer 配置 | 成功 |
| AgentInfer Program | 4 个 |
| Chat API HTTP 200 | 32 次 |
| `lifecycle=continue` | 20 次 |
| Prefix Cache 命中 | 3072 tokens，2 次 |
| 请求等待后放行 | 0 次 |
| 服务端错误 | 0 次 |
| 服务退出和设备释放 | 成功 |

4 个 Agent 都使用了独立 Program，且 follow-up 没有串到其它 session。Prefix Cache 出现命中，说明并发请求之间存在可复用的前缀缓存。

## 验证到的内容

本次实验可以证明：

- `--agentinfer-mode enforce` 能够在 DeepSeek V4 服务中正常工作。
- 标准 OpenCode Chat API 请求能够被 AgentInfer 识别。
- 多个 Agent session 能够映射到多个独立 Program。
- 不同 session 的多轮请求不会互相串 session。
- Progress-TTL 能够执行直接放行和后续进度提升。
- Agent 能够真实调用工具、修改目标文件并运行测试。
- 多个 Agent 的 workspace 彼此隔离。
- DeepSeek V4 MTP、prefix cache 和 AgentInfer 可以同时工作。

## 尚未覆盖的内容

本次实验中所有请求最终都被直接放行，服务端统计为 `admitted_after_wait_count=0`。因此以下场景还需要单独压测：

- KV 容量不足时请求进入等待队列。
- 等待请求在资源释放后重新放行。
- `force_resume_min` 和 `force_resume_max` 超时恢复。
- terminal 通知触发 Program 释放。
- 大于当前 KV 容量的 Agent 数量和长上下文请求。

这些场景需要进一步增加并发 Agent 数量、缩小可用 KV 容量，或构造更长的持续对话。

## 失败轮次定位

实验前曾出现过几类启动或执行失败：

- `max-model-len=8192` 时，运行时静态内存池申请失败。
- `npu-memory-utilization=0.85` 时，模型驻留后无法放下一个 KV slot。
- 一次运行中，首个 prefill 的 HCCL Fabric window 分配返回 `207001`，导致跨卡通信失败。
- 早期 OpenCode 运行目录错误，读取了仓库根目录 README，导致上下文超过 4096；加入标准 `--dir` 后解决。

最终实验使用 4096 上下文、0.94 显存利用率和新的 8 卡设备组成功完成。上述失败均发生在设备资源、运行时或实验调用路径上，没有发现 AgentInfer 调度代码错误。

## 实验产物

本机保存了完整实验输出：

```text
/data/zhaomin/agent/pr-preparation-20261016/pr2-validation/deepseek-opencode-multi-agent-20261027/summary.json
/data/zhaomin/agent/pr-preparation-20261016/pr2-validation/deepseek-opencode-multi-agent-20261027/server.log
```

每个 Agent 的 OpenCode 事件、工具调用和错误输出分别保存在 `agent-1` 到 `agent-4` 子目录中。
