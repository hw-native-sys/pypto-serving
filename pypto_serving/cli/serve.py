# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""HTTP serving process lifecycle."""

from __future__ import annotations

import logging

from pypto_serving.config.types import GenerateConfig
from pypto_serving.observability.access_log import create_uvicorn_log_config
from pypto_serving.serving.engine.async_engine import AsyncLLMEngine, EngineConfig
from pypto_serving.serving.utils.env import configure_runtime_logging
from pypto_serving.tools.profile import configure_profiler, get_profiler, merge_profile


def serve(config: EngineConfig, generation: GenerateConfig, *, host: str, port: int) -> None:
    """Create the HTTP app and run its engine for the uvicorn lifecycle."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    configure_runtime_logging()
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError("Serving mode requires uvicorn. Install with: pip install uvicorn") from exc

    from pypto_serving.model.tokenizer import load_tokenizer
    from pypto_serving.serving.server.server import create_serving_app

    configure_profiler(config.profile_config, process_name="pypto-serving-api", initially_active=False)
    engine = AsyncLLMEngine(
        config=config, tokenizer=load_tokenizer(config.model_dir, integration_id=config.integration_id),
    )
    app = create_serving_app(engine, config.model_id, generation)

    @app.on_event("startup")
    async def startup() -> None:
        """Start the inference engine when the HTTP app starts."""
        await engine.start()

    @app.on_event("shutdown")
    async def shutdown() -> None:
        """Stop the engine and merge profiling output on app shutdown."""
        await engine.stop()
        merge_profile()

    parallel = config.parallel_config
    if parallel is None:
        device_groups = [list(config.worker_device_ids())]
        parallelism = f"replicas=1, worker_group_size={len(device_groups[0])}"
    else:
        device_groups = [list(group) for group in parallel.replica_device_groups]
        parallelism = (
            f"mode={parallel.placement_mode}, replicas={parallel.num_replicas}, "
            f"dp={parallel.data_parallel_size}, tp={parallel.tensor_parallel_size}, "
            f"ep={parallel.expert_parallel_size}"
        )
    runtime = config.runtime_config
    model_prefill_limit = runtime.max_prefill_tokens_per_request if runtime is not None else None
    prefill_limits = [config.max_num_scheduled_tokens]
    if config.enable_chunk_prefill and config.long_prefill_token_threshold > 0:
        prefill_limits.append(config.long_prefill_token_threshold)
    if model_prefill_limit is not None:
        prefill_limits.append(model_prefill_limit)

    print(f"Starting PyPTO serving on {host}:{port}")
    print(f"  Model: {config.model_id} (loaded in worker process)")
    print(f"  Platform: {config.platform}, Device groups: {device_groups}")
    print(f"  Parallelism: {parallelism}")
    print(f"  Max running requests: {config.max_num_running_reqs}")
    print(f"  Max scheduled tokens/iter: {config.max_num_scheduled_tokens}")
    print(f"  Configured prefill chunk size: {config.long_prefill_token_threshold}")
    print(f"  Model prefill token limit: {model_prefill_limit if model_prefill_limit is not None else 'unlimited'}")
    print(f"  Effective prefill chunk size: {min(prefill_limits)}")
    if runtime is not None and runtime.num_speculative_tokens > 0:
        support = "supported" if runtime.supports_chunked_prefill_with_speculation else "unsupported"
        print(f"  Chunked prefill with speculative decoding: {support}")
    print(f"  Prefix cache: {'enabled' if config.enable_prefix_cache else 'disabled'}")
    print(f"  Chunk prefill: {'enabled' if config.enable_chunk_prefill else 'disabled'}")
    endpoints = "/v1/completions, /v1/chat/completions, /v1/models, /health, /metrics"
    if get_profiler().enabled:
        endpoints += ", /start_profile, /stop_profile"
    print(f"  Endpoints: {endpoints}")
    uvicorn.run(
        app, host=host, port=port, log_level="info",
        log_config=create_uvicorn_log_config(["/metrics", "/metrics/json"]),
    )
