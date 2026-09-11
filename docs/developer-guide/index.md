# Developer Guide

Developer Guide documents the serving internals, model integration workflow, and model-specific runtime notes. Start with the architecture page for the request and worker flow, then use the focused sections below for implementation work.

## Serving Internals

These pages describe the shared serving stack and the data movement that every model integration builds on:

- [Architecture](architecture.md): API, scheduler, worker, executor, and kernel execution flow.
- [Weight Staging](weight-staging.md): checkpoint tensor staging, rule tables, slab allocation, and invariants.

## Model Integration

Use the model integration guide when adding a new model family or changing a model-specific executor:

- [Model Integration Overview](model-integration/index.md): end-to-end checklist and prerequisites.
- [Family Detection and Model Loading](model-integration/family-detection.md): loader discovery, checkpoint detection, and `LoadedModel` construction.
- [Weight Spec Rules](model-integration/weight-spec.md): declarative weight rules and staging contract.
- [Executor and Runner](model-integration/executor-runner.md): PyPTO executor lifecycle, runner responsibilities, and task arguments.
- [Testing](model-integration/testing.md): unit-test patterns and NPU validation scope.

## Model Runtime Notes

These pages are model- or topology-specific notes. They are useful when working on that runtime path, but they are not prerequisites for every model integration:

- [DeepSeek V4 Runtime](deepseek-v4-runtime.md): eight-rank DeepSeek V4 serving behavior, cache layout, and MTP state.
- [DeepSeek V4 DSpark](deepseek-v4-dspark.md): DSpark topology, kernel constraints, and validation notes.
