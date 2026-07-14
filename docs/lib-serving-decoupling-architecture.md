# Lib-Serving Decoupling Architecture Design

## 1. Background and Problem

From the repository's intended architecture, `pypto-serving` owns the online serving runtime, including model loading, worker lifecycle management, request scheduling, KV cache management, kernel compilation orchestration, and the HTTP/CLI entry points. `pypto-lib` owns model kernels, the kernel ABI, model-specific layouts, and lower-level execution logic.

Today, `pypto-serving` directly copies a large amount of `pypto-lib` internal knowledge in order to drive the Qwen3-14B kernels. This includes kernel file paths, file names, function names, host-side JIT wrapper argument order, compile dummy arguments, runtime argument tuples, weight layout keys, kernel constant names, and model name inference rules.

This design made it possible to integrate one concrete model quickly, but it also created hard-coded coupling between the two repositories. Whenever the `pypto-lib` kernel ABI changes, `pypto-serving` must be updated in lockstep; otherwise, failures usually surface only during compilation or runtime.

The main forms of coupling are:

- Kernel file path coupling: serving constructs paths under `pypto-lib/models/...`.
- Kernel file name coupling: serving loads modules by file names such as `prefill_fwd.py` and `decode_layer.py`.
- Host-side JIT wrapper signature coupling: serving maintains wrappers that must match the kernel ABI.
- Compile dummy argument coupling: serving builds model-specific dummy tensors for JIT compilation.
- Runtime argument order coupling: serving constructs tuples in the order required by kernel signatures.
- Weight layout coupling: serving knows the weight keys, transposes, padding rules, and layer stacking rules required by kernels.
- Model inference coupling: serving guesses the contract from model names, paths, or architecture strings.
- Kernel constant coupling: serving reads module constants such as `BATCH`, `VOCAB`, and `NUM_LAYERS`.

These forms of coupling create the following architectural problems:

- The two repositories must be changed together, and changes in one repository can easily break the other.
- Serving duplicates lib kernel ABI knowledge, creating a second source of truth.
- New model integration tends to accumulate `if model_name...` branches in serving.
- Compatibility failures lack structured diagnostics and often appear as import, shape, or runtime errors.
- Models beyond Qwen3, such as DeepSeek-V4 Flash/Pro, will amplify the existing coupling.

## 2. Goals and Non-Goals

Design goals:

- Make `pypto-lib` the single source of truth for the model serving ABI.
- Keep `pypto-serving` responsible only for serving runtime orchestration, without owning model kernel ABI details.
- Preserve the current Qwen3-14B serving path.
- Leave extension points for DeepSeek-V4 Flash/Pro and future models.
- Drive compatibility checks from contract contents instead of manually maintained model version numbers.
- When an ABI is incompatible, report the specific contract, field, and suggested action.

Non-goals:

- This design does not require replacing the submodule with a package. The submodule can remain the current mechanism for exact reproducibility.
- This design does not move shared scheduler, HTTP, CLI, or worker logic into `pypto-lib`.
- This design does not express model kernel ABI details on the serving side.
- This design does not finalize a cross-model `ModelConfig` schema.
- This design does not require all future models to share the same execution graph, weight layout, or runtime argument structure.

## 3. Overall Architecture

Before the refactor, `pypto-serving` directly reads internal `pypto-lib` implementation details:

```mermaid
flowchart LR
    subgraph ServingBefore["pypto-serving"]
        SPath["construct lib kernel path"]
        SImport["import kernel files by filename"]
        SConst["read kernel constants"]
        SHost["own host-side JIT wrapper signatures"]
        SCompile["build compile dummy args"]
        SRuntime["build runtime arg tuples"]
        SWeight["prepare model-specific weight layout"]
    end

    subgraph LibBefore["pypto-lib"]
        Kernels["actual kernel implementation"]
    end

    SPath --> Kernels
    SImport --> Kernels
    SConst --> Kernels
    SHost --> Kernels
    SCompile --> Kernels
    SRuntime --> Kernels
    SWeight --> Kernels
```

After the refactor, `pypto-serving` consumes only structured contracts exposed by `pypto-lib` through a contract registry:

```mermaid
flowchart LR
    subgraph ServingAfter["pypto-serving"]
        Loader["model loader"]
        Worker["worker / executor"]
        Scheduler["scheduler / KV cache"]
        Surface["HTTP / CLI"]
        Helpers["contract helper APIs"]
    end

    subgraph LibAfter["pypto-lib"]
        Registry["serving contract registry"]
        Contract["ModelServingContract"]
        Identity["model identity"]
        Limits["capabilities / limits"]
        Graph["execution graph"]
        KernelSpec["KernelSpec metadata"]
        Host["host-side JIT wrappers"]
        CompileArgs["compile args builders"]
        RuntimeArgs["runtime args builders"]
        Weights["weight layout preparation"]
        KernelLoader["kernel loader"]
        Validators["validators"]
    end

    Helpers --> Registry --> Contract
    Contract --> Identity
    Contract --> Limits
    Contract --> Graph
    Contract --> KernelSpec
    Contract --> Host
    Contract --> CompileArgs
    Contract --> RuntimeArgs
    Contract --> Weights
    Contract --> KernelLoader
    Contract --> Validators
```

Repository responsibility boundaries:

`pypto-lib` owns:

- model-specific serving ABI
- model-owned serving stage specifications
- kernel file layout
- kernel function mapping
- kernel constant interpretation
- host-side JIT wrapper signatures
- compile dummy argument construction
- runtime raw argument tuple construction
- weight layout preparation
- model/kernel compatibility validation

`pypto-serving` owns:

- model loading orchestration
- runtime limits and request-facing configuration
- executor selection
- kernel compilation orchestration
- batching and scheduling
- KV cache allocation
- worker lifecycle
- HTTP and CLI surfaces

Cross-boundary constraints:

- Serving does not directly concatenate lib kernel paths.
- Serving does not import kernels by file name.
- Serving does not read model-specific kernel constants.
- Serving does not own model-specific host-side JIT wrapper signatures.
- Serving does not duplicate kernel argument tuple ordering.
- Serving does not duplicate model-specific weight layouts.
- Serving does not construct model-specific compile dummy tensors.

## 4. Detailed Design

### 4.1 Contract Registry and Model Selection

The contract registry is the model selection boundary between serving and lib. It receives an explicit family/variant or model metadata parsed from the model directory, then returns the matching `ModelServingContract`.

Input forms:

- Explicit family / variant, such as `("qwen3", "14b")`.
- Loaded model configuration metadata, such as architecture, model type, hidden size, layer count, attention/MoE/quantization fields.

Output forms:

- The matched `ModelServingContract`.
- If the model family is planned but not yet implemented, a clear not implemented error.
- If the model is unsupported, an unsupported contract error.

Matching principles:

- Each contract defines the structured fields required for matching.
- Serving does not infer the model family from the model name, directory name, or string containment checks.
- Unsupported model errors should include diagnostic information such as `model_id`, architecture, and `model_type`.

Multi-model extension:

- Qwen3-14B is an independent contract.
- DeepSeek-V4 Flash and DeepSeek-V4 Pro should be independent contracts.
- New models are integrated by adding new lib contracts.
- Serving should add at most an executor class mapping, not kernel ABI details.

### 4.2 `ModelServingContract`

`ModelServingContract` is the top-level description of a model serving ABI. It
describes the model identity, runtime capabilities, execution graph, kernel
stages, and lib-owned hooks exposed by a model contract to serving.

Phase 1 required fields:

- `schema_version`: the format version of the contract schema.
- `model`: stable model identity, including family, variant, size, quantization, and similar fields.
- `capabilities`: capabilities supported by the model contract, such as paged KV, chunked prefill, and device greedy sampling.
- `limits`: model/kernel limits, such as batch size, maximum sequence length, vocabulary size, and page size.
- `execution`: the stage graph for serving phases such as prefill and decode.
- `kernels`: mapping from logical kernel stages to `KernelSpec`.
- `kernel_binder`: binds loaded kernel functions to lib-owned host wrappers.
- `prepare_weights`: model-specific weight layout preparation function.
- `load_kernels`: lib-owned kernel loader.
- `validate_kernels`: lib-owned kernel/module/model validation function.

Future extension fields:

- `resources`: resources required for model execution.
- `cache`: KV cache layout and page metadata.
- `weights`: structured weight layout metadata, if serving later needs generic introspection.

The key point of `ModelServingContract` is to expose only the structured ABI information that serving must know. The actual kernel files, constant names, and internal module organization remain implementation details owned by lib.

### 4.3 `KernelSpec`

`KernelSpec` describes a logical kernel stage. Serving does not directly care about kernel file names or the module that contains a Python function; it only cares about the ABI metadata exposed by this stage for a serving phase.

Phase 1 required fields:

- `name`: logical stage name, such as `prefill` or `decode`.
- `public_name`: stable name for logging and profiling, such as `qwen3.prefill`.
- `args`: ordered tensor argument metadata, including name, dtype, shape expression, and direction.
- `host_jit_fn`: lib-owned host-side JIT wrapper function.
- `compile_args_builder`: builder for compile dummy arguments.
- `runtime_args_builder`: builder that maps logical runtime inputs to the raw kernel ABI tuple.

Future extension fields:

- `outputs`: logical outputs produced by the kernel stage.
- `capabilities`: capabilities provided by the stage.
- `validate_runtime`: optional validator for runtime configuration and actual argument tensors.

The purpose of `KernelSpec` is to keep ABI changes inside the lib contract. Serving uses `KernelSpec` to orchestrate compilation and execution, but it does not duplicate signatures.

`KernelSpec` should live next to the kernel implementation that owns the ABI. A
Qwen3-14B implementation should not keep a parallel `models/qwen3/serving/*`
mirror that restates kernel signatures far away from the kernel entry points.
For example:

```text
models/qwen3/14b/
  constants.py
  config.py
  prefill_fwd.py        # may expose PREFILL_STAGE
  decode_layer.py       # may expose DECODE_STAGE
  greedy_sample.py      # may expose GREEDY_SAMPLE_STAGE
  token_embed.py        # may expose TOKEN_EMBED_STAGE
  serving_contract.py   # aggregates the stage specs into ModelServingContract
  serving_host.py       # optional colocated host-side wrappers
  serving_weights.py    # optional colocated weight layout preparation
```

The important rule is locality: when a kernel entry function changes, the stage
spec and its contract tests should be in the same variant-owned area as the
kernel, so the kernel author sees and updates the ABI contract at the same time.

### 4.3.1 `TensorArgSpec`

`TensorArgSpec` describes one tensor argument in a kernel stage. It should
describe a shape contract rather than one incidental compile-time tensor shape.
Compile dummy arguments and runtime actual arguments are two instantiations of
the same `TensorArgSpec`.

Phase 1 required fields:

- `name`: parameter name in the host/kernel ABI.
- `dtype`: element dtype, such as `bf16`, `fp32`, or `int32`.
- `direction`: `in`, `out`, or `inout`.
- `shape`: ordered dimension expressions represented as `int` values or string expressions.

Future extension fields:

- `role`: semantic role, such as `runtime`, `weight`, `cache`, `output`, `scratch`, or `static`.
- `source`: logical source used by the runtime argument builder, such as `inputs.seq_lens`, `weights.decode_wq`, `static.rope_cos`, or `runtime.k_cache`.
- `constraints`: structured validation rules for this argument.

For Phase 1, shape expressions are intentionally simple strings. They are not a
general expression language owned by serving; they are lib-owned labels used for
metadata, diagnostics, fingerprinting, and focused contract tests. Lib-owned
compile/runtime builders and validators remain responsible for evaluating the
actual model-specific shape rules.

```python
TensorArgSpec(
    name="hidden_states",
    dtype="bf16",
    direction="in",
    shape=("PREFILL_TOKENS", "H"),
)

TensorArgSpec(
    name="k_cache",
    dtype="bf16",
    direction="inout",
    shape=("KV_CACHE_ROWS", "D"),
)
```

The concrete dimension environment is supplied by model metadata, kernel
constants, and runtime configuration. For example:

```text
B = kernel batch limit
H = model hidden size
L = number of layers
D = attention head dimension
MAX_SEQ = kernel maximum sequence length
PAGE_SIZE = KV cache page size
PREFILL_TOKENS = runtime prefill token count
KV_CACHE_ROWS = runtime KV cache row capacity
```

The contract must distinguish fixed dimensions from dynamic dimensions. A
runtime tensor may vary only on dimensions that the lib-owned ABI marks dynamic
and that the kernel implementation actually handles dynamically, either through
explicit tensor annotations such as `pl.Tensor[[M, H], pl.BF16]` or through
lib-owned `bind_dynamic` usage. Bare `pl.Tensor` annotations are sufficient for
PyPTO to specialize from compile dummy tensors, but they are not sufficient as a
cross-repository serving ABI source because serving would need to know the dummy
tensor shape before lib has described it.

In Phase 1, dynamic-range validation can live in lib-owned validator code rather
than in `TensorArgSpec` itself. A later schema revision may replace string shape
expressions with structured dimension objects if generic validation or automatic
argument allocation becomes necessary.

When the PyPTO kernel entry signature already contains complete
`pl.Tensor[[shape], dtype]` annotations, lib may derive `TensorArgSpec` from the
function signature. This is an implementation convenience, not a serving
responsibility. If the signature is incomplete, the kernel module must expose an
explicit colocated stage spec or complete the annotations. The serving contract
must never rely on serving-side dummy arguments as the first source of ABI
truth.

### 4.4 `LoadedKernelModules`

`LoadedKernelModules` is the return value from the lib kernel loader. It contains loaded kernel functions and extracted kernel constants.

Fields:

- `functions`: mapping from logical function name to callable.
- `constants`: structured constants dictionary.

Ownership:

- Kernel directory layout is owned by lib.
- Kernel filename mapping is owned by lib.
- Kernel constant names and semantics are owned by lib.
- Serving only receives structured loading results and follows the repository responsibility boundaries defined in Section 3.

### 4.5 `ServingRequirements`

`ServingRequirements` is serving's generic requirement declaration for a contract. It is not a model ABI version; it describes the conditions that a contract must satisfy for the serving runtime.

Fields:

- `schema_version`: the contract schema that serving can parse.
- `required_phases`: phases required by serving, such as `prefill` and `decode`.
- `required_capabilities`: mandatory capabilities, such as paged KV.
- `optional_capabilities`: capabilities that serving can use but does not require.

Validation semantics:

- A schema mismatch means serving cannot parse the contract.
- A missing phase means the execution graph cannot satisfy the serving flow.
- A missing capability means the contract cannot satisfy the current serving runtime mode.

### 4.6 `ExecutionGraphSpec`

`ExecutionGraphSpec` describes the mapping from serving phases to kernel stages.

Example:

```text
prefill -> prefill, greedy_sample
decode  -> decode
```

This structure allows future models to have different execution graphs:

- Multi-stage prefill.
- Separate sample or embedding stages.
- MoE- or MLA-specific stages.
- Different graphs for different variants.

Serving invokes logical stages by phase instead of assuming a fixed set of four kernels.

### 4.7 Kernel Loading

Kernel loading is fully owned by the lib contract.

The lib-side loader is responsible for:

- Locating the kernel directory.
- Maintaining the mapping from logical stages to file names, modules, and functions.
- Loading kernel functions.
- Reading and interpreting kernel constants.
- Returning `LoadedKernelModules`.

Serving-side API calls:

```python
loaded_kernels = contract.load_kernels()
contract.validate_kernels(contract, loaded_kernels, model)
contract.kernel_binder(**loaded_kernels.functions)
```

Kernel constant mismatch errors are produced by the lib validator. Serving only calls the loader, validator, and binder, while following the repository responsibility boundaries defined in Section 3.

### 4.8 Host-Side JIT Wrappers and Compile Arguments

A host-side JIT wrapper refers to the PyPTO `pl.jit.host` layer: the host-side JIT boundary that connects Python orchestration logic to PyPTO kernel calls. It is owned by lib. The wrapper signature must match the kernel ABI, so it belongs to the lib-side serving ABI rather than serving runtime orchestration logic.

Design:

- Lib defines host-side JIT wrappers.
- Lib receives the actual loaded kernel functions through a binder.
- Serving obtains the host JIT function from `KernelSpec.host_jit_fn`.
- Serving does not maintain wrapper signatures.

Compile dummy arguments are generated by lib builders:

```python
stage = contract.kernels["prefill"]
dummy_args = stage.compile_args_builder(model_config, runtime_config)
compiled = stage.host_jit_fn.compile(*dummy_args, config=run_config)
```

Compile dummy arguments are not a separate ABI. They are concrete
`torch.Tensor` instances generated from the same `TensorArgSpec` used for
runtime dispatch. Their contents are irrelevant; their metadata is used by
PyPTO specialization and compilation.

The compile profile must satisfy the stage argument contract:

- argument count and order match the host JIT wrapper;
- direction, dtype, and rank match the stage ABI;
- static dimensions match the kernel contract;
- dynamic dimensions use a valid compile-time profile for the runtime mode;
- shared dynamic dimensions are bound consistently.

PyPTO can infer tensor shape and dtype from the dummy tensors passed to
`.compile()`. That inference is a lower-level JIT specialization mechanism; it
must not be the serving/lib contract boundary. The contract must exist before
dummy tensors are built, otherwise serving would need to know the model-specific
ABI in order to ask lib what the ABI is.

Design benefits:

- Kernel signature changes require changes only in the lib contract.
- The serving compile flow remains generic.
- Compile-time shape and dtype rules are not duplicated in serving.

### 4.9 Runtime Arguments and Execution Flow

The serving runner constructs only logical inputs, such as prefill hidden states, sequence lengths, block tables, slot mappings, static weight records, and KV cache pages.

The lib-side runtime argument builder is responsible for:

- Mapping logical inputs to the raw kernel ABI tuple.
- Maintaining kernel argument order.
- Handling model-specific optional outputs.
- Handling contract-specific parameters such as device sampling and device embedding.

Serving dispatch flow:

```python
stage = contract.kernels["prefill"]
args = stage.runtime_args_builder(inputs, static, **runtime_objects)
compiled_callable(*args)
```

Runtime actual arguments must satisfy the same `TensorArgSpec` used to build
compile dummy arguments. Dummy arguments provide a valid compile-time profile;
actual arguments carry real request data, prepared weights, cache pages, and
output buffers while still satisfying the same ABI contract.

The intended flow is:

```text
ABI spec
  -> compile-time dummy args
  -> runtime actual args
```

Serving may know the logical concepts, but raw argument ordering stays in the
lib runtime argument builder. Runtime ABI changes should affect only the lib
builder and contract tests.

Runtime shape variability is allowed only where the lib-owned ABI marks a
dimension dynamic and the kernel implementation supports that dynamic dimension.
If a runtime tensor is outside the supported range, the lib-owned validator
should raise a structured compatibility error before dispatch. Examples:

- `user_batch <= kernel_batch`
- `prefill_tokens <= user_batch * max_seq`
- `max_seq_len <= kernel_max_seq`
- `page_size == kernel_page_size`
- `vocab == kernel_padded_vocab`
- `real_vocab == model_vocab`
- `num_layers == kernel_num_layers`
- `kv_cache_rows` is sufficient for the allocated pages, layers, KV heads, and page size.

This means a serving instance can provide tensors sized for its actual request
batch and cache allocation, but lib owns the rules that decide whether those
tensors are valid for the compiled stage.

### 4.10 Weight Layout

Weight layout is part of the model kernel ABI, so it is owned by the lib contract.

Weight layout means the shape, order, dtype, padding, transposition, stacking,
and memory-export form that the kernel expects for model weights. Hugging
Face-style loaded weights are usually not already in this form. For Qwen3, for
example, per-layer weights may need to be transposed and stacked into tensors
such as `decode_wq`, `decode_wk`, `decode_wv`, `decode_w_gate`, and
`decode_w_down`; embedding and LM-head weights may need vocabulary padding; norm
weights may need shape normalization such as `[1, H]` or `[L, H]`.

Lib-side weight preparation is responsible for:

- Kernel weight key names.
- Tensor transposition.
- Vocabulary padding.
- Layer stacking.
- Shared memory export policy.
- Destructive or free-after-pack behavior that may release original tensors.

Serving is responsible for:

- Loading model weights.
- Holding the runtime model record.
- Providing a tensor exporter, such as shared-memory placement.
- Calling `contract.prepare_weights`.

Qwen3 and future models may have completely different layouts. Serving should not duplicate Qwen3 layout rules.

### 4.11 ABI Compatibility and Version Strategy

The submodule commit SHA remains valuable because it provides reproducibility. A serving commit can precisely record which lib SHA it used. However, the submodule commit SHA should not be the only runtime compatibility mechanism.

Responsibilities of `schema_version`:

- Describe the contract schema shape.
- Indicate whether serving can parse the contract fields.
- It does not represent the version of a specific model kernel ABI.

Responsibilities of `abi_fingerprint`:

- Compute a stable hash from public ABI metadata.
- Change when kernel arguments, execution graph, limits, weight metadata, or similar public ABI data changes.
- Support logging, diagnostics, and compatibility error localization.

`abi_fingerprint` is mainly diagnostic. It should be written to logs and included in compatibility errors, making it easier to align a failed serving process with the public ABI metadata it actually observed. Hard runtime compatibility should be determined by structured checks: schema parsing, `ServingRequirements`, model/runtime validation, and lib-owned kernel validators. By default, the fingerprint should not become a fixed allowlist gate, because that would reintroduce the complexity of manually maintained version tables.

The following diagram shows only the compatibility check path, not the full startup flow. The full startup flow is described in Section 5.

```mermaid
flowchart TD
    Start["load selected contract"]
    Sha["log submodule commit SHA"]
    Schema["schema_version check"]
    Requirements["ServingRequirements check"]
    Runtime["model/runtime validation"]
    Kernels["lib-owned kernel validator"]
    Fingerprint["log abi_fingerprint"]
    Compile["compile and register model"]
    Error["structured compatibility error"]

    Start --> Sha --> Schema --> Requirements --> Runtime --> Kernels --> Fingerprint --> Compile
    Schema -- fail --> Error
    Requirements -- fail --> Error
    Runtime -- fail --> Error
    Kernels -- fail --> Error
    Fingerprint -. "diagnostic only" .-> Error
```

Compatibility checks include:

- Contract schema.
- Required phases and capabilities.
- Kernel argument metadata.
- Weight layout metadata.
- Cache layout metadata.
- Model/runtime metadata.
- Loaded kernel constants.

Error messages should include:

- Model family / variant.
- Schema version.
- ABI fingerprint.
- Incompatible field.
- Suggested action, such as updating the submodule, updating serving, or rebuilding kernels.

### 4.12 Multi-Model Extension

The Qwen3-14B contract describes the current dense transformer serving path:

- paged KV
- chunked prefill
- device greedy sampling
- device embedding
- Qwen3-specific weight layout
- Qwen3-specific runtime argument builders

DeepSeek-V4 Flash/Pro should be integrated through independent contracts:

- Independent family / variant.
- MoE / MLA specific metadata.
- Potentially different execution graph.
- Independent kernel loader.
- Independent weight layout.
- Independent runtime argument builder.
- Independent validator.

Future model integration path:

1. Add colocated kernel stage specs in `pypto-lib`, next to the model variant's kernel entry points.
2. Add a model contract aggregator in `pypto-lib` for the model variant.
3. Add a kernel loader and validator in `pypto-lib`.
4. Add host wrappers, compile dummy argument builders, runtime argument builders, and weight layout in `pypto-lib`.
5. Add an executor mapping in `pypto-serving` only if the model requires a new executor class.
6. Do not add kernel ABI detail code in serving.

### 4.13 Model Config Boundary

`ModelConfig` is currently the bridge between the model loader and the contract registry. It carries model metadata parsed from `config.json` for contract matching and runtime validation.

Design constraints:

- Serving should not infer the model family from model names.
- Contract matching should be based on structured metadata.
- Serving should not add `if` branches for every model naming style.

Open design points:

- The `config.json` fields differ significantly across Qwen3, DeepSeek-V4, and quantized models.
- This design does not yet decide whether to expand `ModelConfig` into a wide schema, preserve the raw config, or move more parsing and matching into lib.

Design requirements:

- The final approach must avoid per-model hard-coded `if` chains on the serving side.
- The final approach must allow each contract to inspect the structured fields it needs.

### 4.14 Contract Instance Lifecycle

A contract instance consists of immutable metadata and hook references. The registry may create a new contract object or return an equivalent prebuilt object; after selection, serving treats it as read-only.

The contract lifecycle is bound to model registration. It is not a per-request object and should not store mutable request state. Request state remains in the serving scheduler, runner, and KV cache structures. During model registration, the worker selects the contract. After executor compilation completes, the runner holds a reference to the contract so it can continue using runtime argument builders during prefill and decode. The full startup steps are described in Section 5.

### 4.15 Executor Abstraction

The serving executor abstraction remains owned by `pypto-serving`. The contract does not replace the executor. The contract provides model-related ABI knowledge; the executor provides runtime orchestration capability.

Executor responsibilities:

- Create PyPTO run configurations.
- Invoke JIT compilation.
- Allocate and share runtime buffers.
- Integrate device workers.
- Create model runners.
- Expose serving runtime capabilities such as device sampling and device embedding.

Contract responsibilities inside the executor flow:

- Provide host JIT functions.
- Provide compile dummy arguments.
- Prepare model-specific weights.
- Map logical runtime inputs to raw ABI tuples.
- Validate loaded kernels using model/runtime metadata.

Adding a new model should not automatically require a new executor class. A new executor class is needed only when device orchestration, runner behavior, or runtime integration differs. If the difference is limited to the kernel ABI, weight layout, execution graph, or argument builders, it should be expressed through a new lib contract.

## 5. Startup and Runtime Flow

Startup flow:

```mermaid
sequenceDiagram
    participant W as Worker
    participant L as ModelLoader
    participant R as pypto-lib Registry
    participant C as ModelServingContract
    participant E as Executor
    participant Runner as ModelRunner

    W->>L: load model weights and config
    L-->>W: LoadedModel
    W->>R: select contract from model config
    R-->>W: ModelServingContract
    W->>W: validate ServingRequirements
    W->>C: load kernels
    C-->>W: LoadedKernelModules
    W->>C: validate kernels with model and runtime information
    W->>C: bind kernel functions to host-side JIT wrappers
    W->>E: create executor for contract
    E->>C: get host_jit_fn and compile dummy args from stage spec
    E->>E: compile host-side JIT wrappers
    E->>C: prepare weight layout
    E->>Runner: create runner with compiled kernels and contract
    Runner-->>W: model registration complete
```

The compile path is intentionally lib-driven:

```text
Executor -> KernelSpec.compile_args_builder(model_config, runtime_config)
         -> dummy torch.Tensor tuple
         -> KernelSpec.host_jit_fn.compile(*dummy_args)
```

The executor does not construct model-specific dummy tensors itself.

Prefill/decode flow:

```mermaid
sequenceDiagram
    participant S as Scheduler
    participant Runner as ModelRunner
    participant C as ModelServingContract
    participant K as Compiled Kernel

    S-->>Runner: scheduled logical batch
    Runner->>Runner: construct logical input objects
    Runner->>C: validate and construct runtime args for stage
    C-->>Runner: raw kernel ABI tuple
    Runner->>K: call compiled callable
    K-->>Runner: logits / sampled ids / hidden states
    Runner-->>S: serving outputs
```

The runtime path is also lib-driven at the raw ABI boundary:

```text
Runner logical inputs + prepared weights + runtime buffers
  -> KernelSpec.runtime_args_builder(...)
  -> raw argument tuple in kernel order
  -> compiled callable
```

Serving owns the logical inputs and buffer lifecycle; lib owns the conversion
from those logical objects to the kernel ABI tuple.

Failure ownership:

- Unsupported model: registry.
- Incompatible contract: serving requirement validation.
- Kernel constant mismatch: lib validator.
- Runtime limit mismatch: contract/runtime validation.
- Compilation failure: serving executor orchestration.

## 6. Test Strategy

`pypto-lib` tests:

- Contract metadata tests.
- ABI fingerprint stability tests.
- Stage spec and kernel entry signature consistency tests.
- TensorArgSpec validation tests for static dimensions, dynamic dimensions, dtype, rank, and direction.
- Compile dummy argument construction tests.
- Runtime raw argument construction tests.
- Weight layout tests.
- Kernel loader tests.
- Kernel validator tests.
- Registry matcher tests.

`pypto-serving` tests:

- Contract compatibility tests.
- Executor selection tests.
- Source-text coupling absence tests.
- Absence of model-specific compile dummy tensor construction in serving.
- Batching and scheduler regression tests.
- Qwen3 E2E tests.

Cross-repository validation:

- `pypto-lib` PRs run lib contract and fingerprint tests.
- `pypto-lib` PRs can optionally run serving compatibility tests based on the PR SHA.
- `pypto-serving` PRs run serving tests against the pinned lib submodule.

## 7. Acceptance Criteria

### 7.1 Qwen3 Migration Complete

- Serving does not construct lib kernel paths.
- Serving does not import kernel files by file name.
- Serving does not read model-specific kernel constants.
- Serving does not own model-specific host-side JIT wrapper signatures.
- Serving does not construct model-specific runtime raw argument tuples.
- Serving does not construct model-specific compile dummy tensors.
- Serving does not duplicate the Qwen3 weight layout.
- Serving does not assert against lib source text.
- The lib contract is the single source of truth for the serving ABI.
- Qwen3 stage specs are colocated with Qwen3-14B kernel entry points or in the same variant-owned directory.
- Qwen3 serving E2E passes.

### 7.2 Future Model Extension Readiness

- New model integration primarily adds lib stage specs, contract aggregators, loaders, validators, builders, and weight layouts.
- When model differences are limited to ABI, graph, weights, or runtime argument layout, serving does not need new kernel ABI details.
- Serving needs a new executor mapping only when device orchestration or runner behavior differs.
- For heterogeneous `config.json` files, the `ModelConfig` boundary remains a clear open issue; future model extension readiness does not mean this schema has been finalized.
