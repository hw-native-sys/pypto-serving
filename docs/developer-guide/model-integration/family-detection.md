# Family Detection and Model Loading

The first step in integrating a new model is teaching the serving stack to recognise and load its checkpoint format. Two mechanisms work together: `detect_model_family()` for quick model-type identification, and `ModelFormatLoader` for loading checkpoint data.

## Model Family Type

The `ModelFamily` type (`pypto_serving/model/model_family.py:17`) is a simple literal:

```python
ModelFamily = Literal["deepseek_v4", "qwen"]
```

The `detect_model_family()` function reads `config.json` from the model directory and returns the family:

```python
def detect_model_family(config_data: dict[str, object]) -> ModelFamily:
    return "deepseek_v4" if is_deepseek_v4_config(config_data) else "qwen"
```

To add a new family, extend `ModelFamily` and add detection logic. For example:

```python
# In model_family.py
ModelFamily = Literal["deepseek_v4", "qwen", "my_model"]

def is_my_model_config(config_data: dict[str, object]) -> bool:
    model_type = str(config_data.get("model_type") or "").lower()
    return model_type == "my_model"

def detect_model_family(config_data: dict[str, object]) -> ModelFamily:
    if is_deepseek_v4_config(config_data):
        return "deepseek_v4"
    if is_my_model_config(config_data):
        return "my_model"
    return "qwen"
```

## Model Format Loader Protocol

The `ModelFormatLoader` protocol (`pypto_serving/model/model_loader.py:58`) defines how a loader is discovered and invoked:

```python
class ModelFormatLoader(Protocol):
    format_names: tuple[str, ...]

    def supports_format(self, model_format: str) -> bool:
        """Return whether this loader handles the explicit format name."""

    def can_load(self, model_path: Path) -> bool:
        """Return whether this loader can infer support for a model path."""

    def load(self, request: ModelLoadRequest) -> LoadedModel:
        """Load tensors, tokenizer, and metadata for one model request."""
```

### format_names

A tuple of string identifiers for the format. Used when the user passes `--model-format` explicitly. Example:

```python
format_names = ("huggingface", "hf")  # HuggingFaceDirectoryLoader
format_names = ("deepseek_v4_w8a8", "deepseek-v4-w8a8", "dsv4-w8a8")  # DeepSeekV4W8A8DirectoryLoader
```

### can_load()

Auto-detection: return `True` if this loader can handle the directory. The `SafetensorsDirectoryLoader` base class (`model_loader.py:76`) provides a shared implementation:

```python
class SafetensorsDirectoryLoader:
    def can_load(self, model_path: Path) -> bool:
        if not (model_path / "config.json").exists():
            return False
        return self._recognises(model_path)

    def _recognises(self, model_path: Path) -> bool:
        """Family-specific detection, given that config.json exists."""
        raise NotImplementedError
```

### load()

Returns a `LoadedModel` dataclass containing:

| Field | Type | Description |
|-------|------|-------------|
| `model_id` | `str` | Model identifier |
| `model_dir` | `str` | Path to checkpoint directory |
| `config` | `ModelConfig` | Parsed model architecture metadata |
| `tokenizer` | `TokenizerAdapter` | Loaded tokenizer |
| `layer_specs` | `list[LayerSpec]` | Per-layer shape specifications |
| `runtime_model` | `RuntimeModel` | Runtime tensors (embed, norm, lm_head) |

## Model Loader Registry

The `ModelLoader` class (`model_loader.py:435`) is the registry that selects and invokes format loaders:

```python
class ModelLoader:
    def __init__(self, format_loaders=None):
        self._format_loaders = format_loaders or [
            DeepSeekV4W8A8DirectoryLoader(),
            HuggingFaceDirectoryLoader(),
        ]

    def register(self, format_loader: ModelFormatLoader) -> None:
        self._format_loaders.append(format_loader)

    def load(self, model_id, model_dir, ...) -> LoadedModel:
        loader = self._select_loader(request)
        return loader.load(request)
```

When `model_format` is specified, the registry matches by `format_names`. Otherwise, it tries each loader's `can_load()` in registration order and returns the first match.

## Examples

### HuggingFaceDirectoryLoader

A general-purpose loader for Qwen-style decoder-only models (`model_loader.py:235`):

- Detection: accepts any directory with `*.safetensors` files (or a `model.safetensors.index.json`)
- Load: reads `config.json`, builds `ModelConfig`, loads tokenizer, reads global weights (embed, norm, lm_head), leaves per-layer weights for lazy staging
- Supported architectures: `Qwen2ForCausalLM`, `Qwen3ForCausalLM`, `Qwen2Model`, `Qwen3Model`

### DeepSeekV4W8A8DirectoryLoader

A stricter loader for the quantized DeepSeek V4 checkpoint (`model_loader.py:313`):

- Detection: requires `model.safetensors.index.json` + `config.json` that names DeepSeekV4
- Validates: `quantization_config.quant_method == "compressed-tensors"`, checks `compress_ratios`, validates required tensor names
- Load: reads metadata without materializing quantized weights (placeholder tensors for embed/norm/lm_head)

## Adding a New Loader

1. Create a loader class inheriting from `SafetensorsDirectoryLoader`
2. Set `format_names` for explicit format selection
3. Implement `_recognises()` for auto-detection
4. Implement `load()` to return a `LoadedModel`
5. Register it: `ModelLoader().register(MyLoader())`

```python
class MyModelDirectoryLoader(SafetensorsDirectoryLoader):
    format_names = ("my_model",)

    def _recognises(self, model_path: Path) -> bool:
        config_data = read_model_config(model_path)
        return config_data.get("model_type") == "my_model"

    def load(self, request: ModelLoadRequest) -> LoadedModel:
        # ... read config, load tokenizer, build LoadedModel ...
        pass
```