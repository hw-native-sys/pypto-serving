# Simpler Integration

## Which ranks launch Simpler

Only **replica** ranks initialize a Simpler runtime instance. Coordinator and request manager ranks are pure host-side and never touch device memory or Simpler APIs.

Device assignment: `deviceId = instanceId - numPartitions`

With two partitions and one replica each, ranks 0 and 1 are coordinators, ranks 2 and 3 are replicas owning devices 0 and 1 respectively.

## Initialization

Each replica rank initializes Simpler in `main()` before the serving engine starts:

```cpp
hllm::simpler::RuntimeBinaries bins{hostLib, aicpuLib, aicoreKernel, dispatcherLib, simplerLogLib};
auto rt = std::make_unique<hllm::simpler::SimplerRuntime>();
rt->init(bins, deviceId);
mnist::loadKernels(*rt, artifactDir);
```

`RuntimeBinaries` holds five paths to compiled runtime artifacts:

| Field | File | Purpose |
|---|---|---|
| `host` | `libhost_runtime.so` | Core device runtime |
| `aicpu` | `libaicpu_kernel.so` | CPU-side kernel support |
| `aicore` | `aicore_kernel.o` | NPU core kernel binary |
| `dispatcher` | `libsimpler_aicpu_dispatcher.so` | On-device dispatcher |
| `simplerLog` | `libsimpler_log.so` | Logging (preloaded RTLD_GLOBAL) |

Model Support is responsible for providing and locating these artifacts. The platform does not interpret or validate them.

## `processFc` — the task execution callback

The replica module accepts a user-provided function with signature:

```cpp
std::function<void(serving::modules::roles::TaskContext &context)>
```

The platform calls this function once per job, after all input dependencies have arrived. Inside the function, Model Support:

1. Reads inputs from `context.getInput(edgeName)` — returns a `LocalMemorySlot` containing the host buffer
2. Stages inputs to device with `rt->toDevice(ptr, bytes)`
3. Dispatches a Simpler kernel with `rt->run(callableId, args, config)`
4. Stages outputs back with `rt->toHost(hostPtr, devPtr, bytes)`
5. Registers outputs with `context.setOutput(edgeName, ptr, size)`

The platform guarantees that:
- All declared inputs are present and ready before the call
- Outputs registered via `setOutput` are forwarded to the coordinator after the call returns
- The function is called at most once per job; re-entrancy is not required

## `SimplerRuntime` API surface

```cpp
void   init(const RuntimeBinaries &bins, int deviceId);
void   loadCallable(const void *blob, size_t size, uint32_t id);
void   run(uint32_t callableId, ChipStorageTaskArgs &args, ChipCallConfig &config);
void  *toDevice(const void *hostPtr, size_t bytes);
void   toHost(void *hostPtr, const void *devPtr, size_t bytes);
void  *alloc(size_t bytes);
void   free(void *devPtr);
void   finalize();
```

## Channel model and hot path

All channels in the current platform are host-side. Tensor payloads passed through `processFc` are staged through host memory (`toDevice` / `toHost`). This is correct for control traffic and small tensors but adds latency on the hot path for large prefill/decode tensors.

**In-device tensor channels** (direct NPU-to-NPU transfer without host staging) are the next milestone. When implemented, the `processFc` interface will remain the same — Model Support will simply receive device-memory `LocalMemorySlot` handles instead of host buffers, and the staging calls become unnecessary.
