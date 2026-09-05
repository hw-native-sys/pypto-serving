# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import json
import os
import subprocess
import sys
import threading
import time
from types import ModuleType

import pytest

from pypto_serving.config.types import KVCacheGroupSpec, KVCacheSpec
from pypto_serving.serving.external_cache import (
    ExternalCacheNamespace,
    ExternalPrefixCacheConfig,
    ExternalKVBuffer,
    ExternalKVCheckpointManifest,
    ExternalKVWorkerConnector,
    ExternalPrefixCacheIndex,
    ExternalKVTransfer,
    build_deepseek_checkpoint_manifest,
    checkpoint_prefix_digest,
    latest_checkpoint_token_count,
    stable_token_digest,
)
from pypto_serving.serving.external_cache.mooncake import (
    MooncakeStoreBackend,
    create_mooncake_backend,
)
from pypto_serving.serving.external_cache.chip_bridge import SimplerMooncakeConnector
from pypto_serving.serving.external_cache.config import MooncakeClientConfig
from pypto_serving.serving.memory.kv_cache import KvCacheManager
from pypto_serving.serving.sched.scheduler import Request, RequestStatus, Scheduler, SchedulerConfig


def _group(
    name,
    *,
    block_size=128,
    compress_ratio=1,
    sliding_window=None,
    eagle=False,
    page_size_bytes=4096,
):
    return KVCacheGroupSpec(
        name=name,
        layer_indices=(0, 1),
        spec=KVCacheSpec(
            block_size=block_size,
            page_size_bytes=page_size_bytes,
            compress_ratio=compress_ratio,
        ),
        max_blocks_per_seq=128,
        num_partitions=8,
        sliding_window=sliding_window,
        is_eagle_group=eagle,
    )


def _deepseek_groups(*, eagle=False):
    return (
        _group("ori", sliding_window=256, eagle=eagle),
        _group("cmp_c128", compress_ratio=128),
        _group("cmp_c4", compress_ratio=4),
        _group("idx", compress_ratio=4, page_size_bytes=2048),
        _group("hca_state", block_size=8, sliding_window=128, page_size_bytes=1024),
        _group("csa_state", block_size=4, sliding_window=4, page_size_bytes=512),
        _group("csa_inner_state", block_size=4, sliding_window=4, page_size_bytes=256),
    )


def _namespace(groups, *, mtp_enabled=False, data_parallel_size=8):
    return ExternalCacheNamespace.for_deepseek_v4(
        model_id="deepseek-v4",
        model_revision="weights-sha256",
        tokenizer_revision="tokenizer-sha256",
        kv_dtype="mixed",
        tensor_parallel_size=8,
        world_size=8,
        parallel_config={
            "data_parallel_size": data_parallel_size,
            "expert_parallel_size": 8,
            "placement_mode": "overlapped",
            "tensor_parallel_size": 1,
            "world_size": 8,
        },
        mtp_enabled=mtp_enabled,
        group_specs=groups,
    )


def _group_hashes(groups, token_count=256):
    return {
        group.name: list(range(1, token_count // group.spec.token_capacity + 1))
        for group in groups
    }


def test_prefix_block_hash_is_stable_across_python_hash_seeds():
    script = (
        "from pypto_serving.serving.memory.kv_cache import NONE_HASH, hash_block_tokens; "
        "print(hash_block_tokens(NONE_HASH, (1, 2, 3)))"
    )
    outputs = []
    for seed in ("1", "987654"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        outputs.append(subprocess.check_output([sys.executable, "-c", script], env=env, text=True))
    assert outputs[0] == outputs[1]


def test_external_cache_config_parses_sizes_and_policy(tmp_path):
    path = tmp_path / "external-cache.json"
    path.write_text(
        json.dumps(
            {
                "model_revision": "model-commit",
                "tokenizer_revision": "tokenizer-commit",
                "min_tokens": 256,
                "load_timeout_ms": 4000,
                "save_timeout_ms": 5000,
                "transfer_concurrency": 3,
                "max_pending_saves": 4,
                "max_pending_save_blocks": 512,
                "failure_policy": "fail_startup",
                "mooncake": {
                    "metadata_server": "127.0.0.1:2379",
                    "master_server_address": "127.0.0.1:50051",
                    "protocol": "ascend",
                    "ascend_buffer_pool": "4:8",
                    "global_segment_size": "2GB",
                    "local_buffer_size": "32MB",
                    "enable_ssd_offload": True,
                    "ssd_offload_path": "/var/lib/mooncake",
                    "tenant_id": "serving-test",
                },
            }
        ),
        encoding="utf-8",
    )

    config = ExternalPrefixCacheConfig.from_file(path)

    assert config.model_revision == "model-commit"
    assert config.min_tokens == 256
    assert config.load_timeout_ms == 4000
    assert config.save_timeout_ms == 5000
    assert config.transfer_concurrency == 3
    assert config.max_pending_saves == 4
    assert config.max_pending_save_blocks == 512
    assert config.failure_policy == "fail_startup"
    assert config.mooncake.global_segment_size == 2 * 1024**3
    assert config.mooncake.local_buffer_size == 32 * 1024**2
    assert config.mooncake.ascend_buffer_pool == "4:8"
    assert config.mooncake.enable_ssd_offload
    assert config.mooncake.ssd_offload_path == "/var/lib/mooncake"
    assert config.mooncake.tenant_id == "serving-test"


@pytest.mark.parametrize(
    "field",
    [
        "min_tokens",
        "load_timeout_ms",
        "save_timeout_ms",
        "transfer_concurrency",
        "max_pending_saves",
        "max_pending_save_blocks",
    ],
)
def test_external_cache_config_rejects_boolean_numeric_fields(tmp_path, field):
    path = tmp_path / "external-cache.json"
    data = {
        "model_revision": "model-commit",
        "tokenizer_revision": "tokenizer-commit",
        "mooncake": {
            "metadata_server": "127.0.0.1:2379",
            "master_server_address": "127.0.0.1:50051",
        },
        field: True,
    }
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        ExternalPrefixCacheConfig.from_file(path)


def test_mooncake_config_rejects_relative_ssd_path():
    with pytest.raises(ValueError, match="absolute"):
        MooncakeClientConfig.from_mapping(
            {
                "metadata_server": "127.0.0.1:2379",
                "master_server_address": "127.0.0.1:50051",
                "enable_ssd_offload": True,
                "ssd_offload_path": "relative/cache",
            }
        )


@pytest.mark.parametrize("value", ["4", "4:8:16", "four:8", "0:8", 8])
def test_mooncake_config_rejects_invalid_ascend_buffer_pool(value):
    with pytest.raises(ValueError, match="ascend_buffer_pool"):
        MooncakeClientConfig.from_mapping(
            {
                "metadata_server": "127.0.0.1:2379",
                "master_server_address": "127.0.0.1:50051",
                "ascend_buffer_pool": value,
            }
        )


def test_external_namespace_covers_layout_parallelism_and_mtp():
    groups = _deepseek_groups()
    namespace = _namespace(groups)

    assert namespace.digest == _namespace(tuple(reversed(groups))).digest
    assert namespace.digest != _namespace(groups, mtp_enabled=True).digest
    assert namespace.digest != _namespace(groups, data_parallel_size=1).digest
    changed = list(groups)
    changed[0] = _group("ori", sliding_window=128)
    assert namespace.digest != _namespace(changed).digest


def test_mtp_checkpoint_digest_includes_boundary_lookahead_token():
    first = list(range(129))
    second = first.copy()
    second[128] = 999

    assert checkpoint_prefix_digest(first, 128, mtp_enabled=False) == checkpoint_prefix_digest(
        second,
        128,
        mtp_enabled=False,
    )
    assert checkpoint_prefix_digest(first, 128, mtp_enabled=True) != checkpoint_prefix_digest(
        second,
        128,
        mtp_enabled=True,
    )


def test_latest_checkpoint_uses_all_group_publication_boundaries():
    groups = _deepseek_groups(eagle=True)
    counts = {
        "ori": 1,
        "cmp_c128": 2,
        "cmp_c4": 2,
        "idx": 2,
        "hca_state": 32,
        "csa_state": 64,
        "csa_inner_state": 64,
    }

    assert latest_checkpoint_token_count(groups, counts) == 128
    counts["ori"] = 0
    assert latest_checkpoint_token_count(groups, counts) == 0


def test_manifest_contains_full_history_and_only_rolling_tails():
    groups = _deepseek_groups()
    prefix_digest = stable_token_digest(range(256))
    manifest = build_deepseek_checkpoint_manifest(
        _namespace(groups),
        prefix_digest=prefix_digest,
        token_count=256,
        source_partition=3,
        group_specs=groups,
        group_block_hashes=_group_hashes(groups),
    )

    positions = {}
    for item in manifest.objects:
        positions.setdefault(item.group_name, []).append(item.logical_block_index)
    assert positions["ori"] == [0, 1]
    assert positions["cmp_c128"] == [0, 1]
    assert positions["hca_state"] == list(range(16, 32))
    assert positions["csa_state"] == [63]
    assert positions["csa_inner_state"] == [63]
    assert manifest.manifest_key.endswith(f"/{prefix_digest}/p3/t256")
    assert ExternalKVCheckpointManifest.from_bytes(manifest.to_bytes()) == manifest


def test_manifest_rejects_partial_group_state():
    groups = _deepseek_groups()
    hashes = _group_hashes(groups)
    hashes["idx"] = hashes["idx"][:1]

    with pytest.raises(ValueError, match="requires 2"):
        build_deepseek_checkpoint_manifest(
            _namespace(groups),
            prefix_digest=stable_token_digest(range(256)),
            token_count=256,
            source_partition=0,
            group_specs=groups,
            group_block_hashes=hashes,
        )


class _FakeMooncakeStore:
    def __init__(self):
        self.put_args = None
        self.get_args = None

    def batch_is_exist(self, keys):
        return [1 if key == "present" else 0 for key in keys]

    def batch_put_from_multi_buffers(self, keys, addresses, sizes):
        self.put_args = keys, addresses, sizes
        return [0] * len(keys)

    def batch_get_into_multi_buffers(self, keys, addresses, sizes):
        self.get_args = keys, addresses, sizes
        return [0] * len(keys)

    def put(self, key, payload):
        self.put_args = key, payload
        return len(payload)


def test_create_mooncake_backend_enables_rank_local_ssd_path(monkeypatch, tmp_path):
    setup_calls = []
    put_calls = []
    engines = []
    initialize_buffer_pools = []
    monkeypatch.delenv("ASCEND_BUFFER_POOL", raising=False)

    class FakeReplicateConfig:
        preferred_segment = ""
        prefer_alloc_in_same_node = False

    class FakeTransferEngine:
        def __init__(self):
            self.initialize_args = None
            self.registered = []
            self.unregistered = []
            engines.append(self)

        def initialize(self, *args):
            self.initialize_args = args
            initialize_buffer_pools.append(os.environ.get("ASCEND_BUFFER_POOL"))
            return 0

        @staticmethod
        def get_rpc_port():
            return 12345

        def get_engine(self):
            return self

        def register_memory(self, address, size):
            self.registered.append((address, size))
            return 0

        def unregister_memory(self, address):
            self.unregistered.append(address)
            return 0

    class FakeDistributedStore(_FakeMooncakeStore):
        def setup(
            self,
            *args,
            enable_ssd_offload=False,
            ssd_offload_path="",
            tenant_id="default",
            engine=None,
        ):
            setup_calls.append((args, enable_ssd_offload, ssd_offload_path, tenant_id, engine))
            return 0

        def register_buffer(self, _address, _size):
            return 0

        def batch_put_from_multi_buffers(self, keys, addresses, sizes, config):
            put_calls.append((keys, addresses, sizes, config))
            return [0] * len(keys)

    mooncake_module = ModuleType("mooncake")
    store_module = ModuleType("mooncake.store")
    engine_module = ModuleType("mooncake.engine")
    store_module.MooncakeDistributedStore = FakeDistributedStore
    store_module.ReplicateConfig = FakeReplicateConfig
    engine_module.TransferEngine = FakeTransferEngine
    mooncake_module.store = store_module
    mooncake_module.engine = engine_module
    monkeypatch.setitem(sys.modules, "mooncake", mooncake_module)
    monkeypatch.setitem(sys.modules, "mooncake.store", store_module)
    monkeypatch.setitem(sys.modules, "mooncake.engine", engine_module)
    ssd_root = tmp_path / "external-kv"
    config = MooncakeClientConfig(
        metadata_server="127.0.0.1:2379",
        master_server_address="127.0.0.1:50051",
        local_hostname="127.0.0.1",
        ascend_buffer_pool="4:8",
        enable_ssd_offload=True,
        ssd_offload_path=str(ssd_root),
        tenant_id="serving-test",
    )

    backend = create_mooncake_backend(config, contribute_memory=True, storage_rank=3)

    assert isinstance(backend, MooncakeStoreBackend)
    args, enabled, path, tenant, engine = setup_calls[0]
    assert args[0] == "127.0.0.1:12345"
    assert args[2] == 0
    assert args[3] == 64 * 1024**2
    assert enabled
    assert path == str(ssd_root / "rank_3")
    assert tenant == "serving-test"
    assert engine is engines[0]
    assert engines[0].initialize_args == ("127.0.0.1", "P2PHANDSHAKE", "ascend", "")
    assert initialize_buffer_pools == ["4:8"]
    assert (ssd_root / "rank_3").is_dir()

    buffers = (ExternalKVBuffer(0x200000, 0x400000),)
    backend.register_buffers(buffers)
    assert engines[0].registered == [(0x200000, 0x400000)]
    transfer = ExternalKVTransfer("rank-local", buffers)
    assert backend.put((transfer,)) == (0,)
    _, _, _, replicate_config = put_calls[0]
    assert replicate_config.preferred_segment == "127.0.0.1:12345"
    assert replicate_config.prefer_alloc_in_same_node is True
    backend.close()
    assert engines[0].unregistered == [0x200000]


def test_scheduler_mooncake_client_uses_zero_memory_ascend_control_path(monkeypatch):
    setup_calls = []
    engines = []

    class FakeTransferEngine:
        def __init__(self):
            self.initialize_args = None
            engines.append(self)

        def initialize(self, *args):
            self.initialize_args = args
            return 0

        @staticmethod
        def get_rpc_port():
            return 12345

        def get_engine(self):
            return self

    class FakeDistributedStore(_FakeMooncakeStore):
        def setup(self, *args, engine=None):
            setup_calls.append((args, engine))
            return 0

        def register_buffer(self, _address, _size):
            return 0

    mooncake_module = ModuleType("mooncake")
    store_module = ModuleType("mooncake.store")
    engine_module = ModuleType("mooncake.engine")
    store_module.MooncakeDistributedStore = FakeDistributedStore
    engine_module.TransferEngine = FakeTransferEngine
    mooncake_module.store = store_module
    mooncake_module.engine = engine_module
    monkeypatch.setitem(sys.modules, "mooncake", mooncake_module)
    monkeypatch.setitem(sys.modules, "mooncake.store", store_module)
    monkeypatch.setitem(sys.modules, "mooncake.engine", engine_module)
    config = MooncakeClientConfig(
        metadata_server="127.0.0.1:2379",
        master_server_address="127.0.0.1:50051",
        local_hostname="127.0.0.1",
        global_segment_size=1024,
        local_buffer_size=2048,
    )

    create_mooncake_backend(config, contribute_memory=False)

    args, engine = setup_calls[0]
    assert args[0] == "127.0.0.1:12345"
    assert args[2:5] == (0, 0, "ascend")
    assert engine is engines[0]
    assert engines[0].initialize_args == ("127.0.0.1", "P2PHANDSHAKE", "ascend", "")


def test_mooncake_backend_rejects_missing_rank_local_placement_before_setup(monkeypatch):
    constructed = []

    class FakeDistributedStore(_FakeMooncakeStore):
        def __init__(self):
            constructed.append(self)

    mooncake_module = ModuleType("mooncake")
    store_module = ModuleType("mooncake.store")
    store_module.MooncakeDistributedStore = FakeDistributedStore
    mooncake_module.store = store_module
    monkeypatch.setitem(sys.modules, "mooncake", mooncake_module)
    monkeypatch.setitem(sys.modules, "mooncake.store", store_module)

    with pytest.raises(RuntimeError, match="ReplicateConfig"):
        create_mooncake_backend(
            MooncakeClientConfig(
                metadata_server="127.0.0.1:2379",
                master_server_address="127.0.0.1:50051",
            ),
            contribute_memory=True,
        )

    assert not constructed


def test_mooncake_backend_preserves_scatter_gather_objects():
    store = _FakeMooncakeStore()
    registered = []
    backend = MooncakeStoreBackend(
        store,
        register_buffer=lambda addresses, sizes: registered.append((addresses, sizes)),
    )
    buffers = (ExternalKVBuffer(0x1000, 64), ExternalKVBuffer(0x2000, 32))
    transfer = ExternalKVTransfer("cache-key", buffers)

    backend.register_buffers(buffers)
    with pytest.raises(ValueError, match="already registered"):
        backend.register_buffers((ExternalKVBuffer(0x1000, 128),))
    assert backend.exists(("present", "missing")) == (True, False)
    assert backend.put((transfer,)) == (0,)
    assert backend.get((transfer,)) == (0,)
    assert registered == [([0x1000, 0x2000], [64, 32])]
    assert store.put_args == (["cache-key"], [[0x1000, 0x2000]], [[64, 32]])
    assert store.get_args == store.put_args


class _FakeExternalBackend:
    def __init__(self, *, present_suffix=None, present_keys=None, put_results=None):
        self.present_suffix = present_suffix
        self.present_keys = set(present_keys or ())
        self.put_results = list(put_results or [])
        self.put_batches = []
        self.byte_puts = []

    def register_buffers(self, buffers):
        pass

    def exists(self, keys):
        return tuple(
            key in self.present_keys
            or (self.present_suffix is not None and key.endswith(self.present_suffix))
            for key in keys
        )

    def put(self, transfers):
        self.put_batches.append(tuple(transfer.key for transfer in transfers))
        if self.put_results:
            return tuple(self.put_results.pop(0))
        return (0,) * len(transfers)

    def get(self, transfers):
        return (0,) * len(transfers)

    def put_bytes(self, key, payload):
        self.byte_puts.append((key, payload))
        return len(payload)


def test_scheduler_index_selects_longest_manifest_beyond_local_hit():
    groups = _deepseek_groups()
    backend = _FakeExternalBackend(present_suffix="/p2/t256")
    index = ExternalPrefixCacheIndex(
        backend,
        _namespace(groups),
        alignment=128,
        num_partitions=8,
        min_tokens=128,
    )

    result = index.lookup(range(400), local_hit_tokens=128, max_hit_tokens=399)

    assert result is not None
    assert result.token_count == 256
    assert result.source_partition == 2
    assert result.prefix_digest == stable_token_digest(range(256))


def _poll_one(connector):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        completed = connector.poll_completed()
        if completed:
            return completed[0]
        time.sleep(0.001)
    raise AssertionError("external cache transfer did not complete")


def test_worker_connector_commits_manifest_after_data_objects():
    backend = _FakeExternalBackend()
    connector = ExternalKVWorkerConnector(backend, max_workers=1)
    data = (
        ExternalKVTransfer("data-1", (ExternalKVBuffer(0x1000, 64),)),
        ExternalKVTransfer("data-2", (ExternalKVBuffer(0x2000, 64),)),
    )
    try:
        connector.start_save(
            "save-1",
            data,
            manifest_key="manifest",
            manifest_payload=b"committed",
        )
        completion = _poll_one(connector)
    finally:
        connector.close()

    assert completion.succeeded
    assert completion.status_codes == (0, 0, len(b"committed"))
    assert backend.put_batches == [("data-1", "data-2")]
    assert backend.byte_puts == [("manifest", b"committed")]


def test_worker_connector_does_not_commit_partial_checkpoint():
    backend = _FakeExternalBackend(put_results=[(0, -1)])
    connector = ExternalKVWorkerConnector(backend, max_workers=1)
    data = (
        ExternalKVTransfer("data-1", (ExternalKVBuffer(0x1000, 64),)),
        ExternalKVTransfer("data-2", (ExternalKVBuffer(0x2000, 64),)),
    )
    try:
        connector.start_save(
            "save-1",
            data,
            manifest_key="manifest",
            manifest_payload=b"committed",
        )
        completion = _poll_one(connector)
    finally:
        connector.close()

    assert not completion.succeeded
    assert completion.status_codes == (0, -1)
    assert backend.put_batches == [("data-1", "data-2")]
    assert backend.byte_puts == []


def test_worker_connector_skips_existing_immutable_data_objects():
    backend = _FakeExternalBackend(present_keys={"data-1"})
    connector = ExternalKVWorkerConnector(backend, max_workers=1)
    data = (
        ExternalKVTransfer("data-1", (ExternalKVBuffer(0x1000, 64),)),
        ExternalKVTransfer("data-2", (ExternalKVBuffer(0x2000, 64),)),
    )
    try:
        connector.start_save(
            "save-1",
            data,
            manifest_key="manifest",
            manifest_payload=b"committed",
        )
        completion = _poll_one(connector)
    finally:
        connector.close()

    assert completion.succeeded
    assert backend.put_batches == [("data-2",)]
    assert backend.byte_puts == [("manifest", b"committed")]


def test_worker_connector_reports_cancel_only_after_running_transfer_stops():
    entered = threading.Event()
    release = threading.Event()

    class BlockingBackend(_FakeExternalBackend):
        def get(self, transfers):
            entered.set()
            assert release.wait(timeout=2)
            return (0,) * len(transfers)

    connector = ExternalKVWorkerConnector(BlockingBackend(), max_workers=1)
    transfer = ExternalKVTransfer("data", (ExternalKVBuffer(0x1000, 64),))
    try:
        connector.start_load("load-1", (transfer,))
        assert entered.wait(timeout=2)
        assert connector.cancel("load-1")
        assert connector.poll_completed() == ()
        release.set()
        completion = _poll_one(connector)
    finally:
        release.set()
        connector.close()

    assert completion.cancelled
    assert not completion.succeeded


def test_simpler_chip_connector_maps_partition_to_physical_device():
    commands = []
    poll_count = 0

    def control(_name, payload):
        nonlocal poll_count
        command = json.loads(payload)
        commands.append(command)
        if command["operation"] == "poll":
            poll_count += 1
            if poll_count == 1:
                raise RuntimeError("PYPTO_EXTERNAL_PENDING:load-1")

    config = ExternalPrefixCacheConfig(
        mooncake=MooncakeClientConfig("127.0.0.1:2379", "127.0.0.1:50051"),
        model_revision="model",
        tokenizer_revision="tokenizer",
        transfer_concurrency=1,
    )
    connector = SimplerMooncakeConnector(control, config, (4, 7))
    try:
        connector.register_buffers(
            (ExternalKVBuffer(0x200000, 0x400000),),
            partition=1,
        )
        transfer = ExternalKVTransfer("data", (ExternalKVBuffer(0x1000, 64),))
        connector.start_load("load-1", (transfer,), partition=1)
        completion = _poll_one(connector)
    finally:
        connector.close()

    assert completion.succeeded
    registration = next(command for command in commands if command["operation"] == "register")
    assert registration["partition"] == 1
    assert registration["device_id"] == 7
    assert registration["buffers"] == [{"address": 0x200000, "size_bytes": 0x400000}]
    start = next(command for command in commands if command["operation"] == "load")
    assert start["partition"] == 1
    assert start["device_id"] == 7
    assert start["transfers"][0]["buffers"] == [
        {"address": 0x1000, "size_bytes": 64}
    ]


def test_simpler_chip_connector_cleans_up_partial_initialization():
    operations = []

    def control(_name, payload):
        operation = json.loads(payload)["operation"]
        operations.append(operation)
        if operation == "init":
            raise RuntimeError("one child failed")

    config = ExternalPrefixCacheConfig(
        mooncake=MooncakeClientConfig("127.0.0.1:2379", "127.0.0.1:50051"),
        model_revision="model",
        tokenizer_revision="tokenizer",
        transfer_concurrency=1,
    )

    with pytest.raises(RuntimeError, match="one child failed"):
        SimplerMooncakeConnector(control, config, (0, 1))

    assert operations == ["init", "close"]


def test_simpler_chip_connector_waits_for_cancelled_child_terminal_state():
    cancelled = threading.Event()

    def control(_name, payload):
        command = json.loads(payload)
        if command["operation"] == "cancel":
            cancelled.set()
        elif command["operation"] == "poll" and not cancelled.is_set():
            raise RuntimeError("PYPTO_EXTERNAL_PENDING:load-1")

    config = ExternalPrefixCacheConfig(
        mooncake=MooncakeClientConfig("127.0.0.1:2379", "127.0.0.1:50051"),
        model_revision="model",
        tokenizer_revision="tokenizer",
        transfer_concurrency=1,
    )
    connector = SimplerMooncakeConnector(control, config, (0,))
    try:
        transfer = ExternalKVTransfer("data", (ExternalKVBuffer(0x1000, 64),))
        connector.start_load("load-1", (transfer,), partition=0)
        assert connector.cancel("load-1")
        completion = _poll_one(connector)
    finally:
        connector.close()

    assert cancelled.is_set()
    assert completion.cancelled
    assert not completion.succeeded


def _external_scheduler(backend):
    groups = _deepseek_groups()
    manager = KvCacheManager(block_size=128, enable_prefix_cache=True)
    manager.init_groups(groups, max_batch_size=1)
    index = ExternalPrefixCacheIndex(
        backend,
        _namespace(groups),
        alignment=manager.group_prefix_cache_alignment,
        num_partitions=8,
        min_tokens=128,
    )
    scheduler = Scheduler(
        SchedulerConfig(
            max_num_scheduled_tokens=128,
            max_prefill_tokens_per_request=128,
            max_seq_len=512,
        ),
        manager,
        external_cache_index=index,
    )
    return scheduler, manager


def test_scheduler_waits_for_atomic_external_load_before_prefill():
    backend = _FakeExternalBackend(present_suffix="/p2/t256")
    scheduler, manager = _external_scheduler(backend)
    request = Request("external-hit", list(range(257)), max_new_tokens=1)
    scheduler.add_request(request)

    load_output = scheduler.schedule()

    assert not load_output.scheduled_requests
    assert load_output.poll_external_cache
    assert len(load_output.external_cache_loads) == 1
    load = load_output.external_cache_loads[0]
    assert load.checkpoint_token_count == 256
    assert load.source_partition == 2
    assert request.status is RequestStatus.WAITING_FOR_REMOTE_KV
    assert request.num_computed_tokens == 0
    assert scheduler.external_cache_stats.lookup_hits == 1
    assert scheduler.external_cache_stats.pinned_hbm_blocks > 0
    assert scheduler.external_cache_stats.peak_pinned_hbm_blocks > 0

    scheduler.finish_external_cache_load(
        request.request_id,
        job_id=load.job_id,
        succeeded=True,
    )
    prefill_output = scheduler.schedule()

    assert request.status is RequestStatus.RUNNING
    assert request.num_computed_tokens == 256
    assert len(prefill_output.scheduled_requests) == 1
    assert prefill_output.scheduled_requests[0].num_computed_tokens == 256
    assert prefill_output.scheduled_requests[0].num_new_tokens == 1
    assert scheduler.external_cache_stats.load_successes == 1
    assert scheduler.external_cache_stats.load_bytes > 0
    assert scheduler.external_cache_stats.pinned_hbm_blocks == 0
    assert manager.acquire_group_prefix_blocks(
        "second",
        request.group_block_hashes,
        max_cache_hit_tokens=256,
    )[1] == 256


def test_scheduler_rolls_back_failed_external_load_to_cold_prefill():
    backend = _FakeExternalBackend(present_suffix="/p2/t256")
    scheduler, manager = _external_scheduler(backend)
    request = Request("external-failure", list(range(257)), max_new_tokens=1)
    scheduler.add_request(request)
    load = scheduler.schedule().external_cache_loads[0]

    backend.present_suffix = None
    scheduler.finish_external_cache_load(
        request.request_id,
        job_id=load.job_id,
        succeeded=False,
    )
    cold_output = scheduler.schedule()

    assert request.status is RequestStatus.RUNNING
    assert request.num_computed_tokens == 0
    assert cold_output.scheduled_requests[0].num_new_tokens == 128
    assert request.request_id not in scheduler.waiting_for_remote_kv
    assert manager.group_request_partition(request.request_id) is not None
    assert scheduler.external_cache_stats.load_failures == 1
    assert scheduler.external_cache_stats.load_fallbacks == 1
    assert scheduler.external_cache_stats.pinned_hbm_blocks == 0


def test_scheduler_treats_external_lookup_error_as_one_cold_miss():
    class FailingLookupBackend(_FakeExternalBackend):
        def exists(self, keys):
            raise RuntimeError("metadata unavailable")

    scheduler, _manager = _external_scheduler(FailingLookupBackend())
    request = Request("external-lookup-error", list(range(257)), max_new_tokens=1)
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert request.external_cache_lookup_attempted
    assert request.status is RequestStatus.RUNNING
    assert output.scheduled_requests[0].num_computed_tokens == 0
    assert output.scheduled_requests[0].num_new_tokens == 128


def test_scheduler_falls_back_on_load_timeout_and_quarantines_dma_pages():
    backend = _FakeExternalBackend(present_suffix="/p2/t256")
    scheduler, manager = _external_scheduler(backend)
    scheduler.external_cache_index._load_timeout_seconds = 0.001
    request = Request("external-timeout", list(range(257)), max_new_tokens=1)
    scheduler.add_request(request)
    load = scheduler.schedule().external_cache_loads[0]
    request.external_cache_load_started_at = time.monotonic() - 1

    timeout_output = scheduler.schedule()

    assert timeout_output.external_cache_cancellations == [load.job_id]
    assert request.request_id not in scheduler.waiting_for_remote_kv
    assert load.job_id in scheduler._external_cache_load_quarantines
    assert manager.group_request_partition(request.request_id) is not None
    assert timeout_output.scheduled_requests[0].num_computed_tokens == 0
    assert scheduler.external_cache_stats.load_timeouts == 1
    assert scheduler.external_cache_stats.load_fallbacks == 1
    assert scheduler.external_cache_stats.pinned_hbm_blocks > 0

    scheduler.finish_external_cache_load(
        request.request_id,
        job_id=load.job_id,
        succeeded=True,
    )

    assert load.job_id not in scheduler._external_cache_load_quarantines
    assert request.status is RequestStatus.RUNNING
    assert scheduler.external_cache_stats.pinned_hbm_blocks == 0


def test_scheduler_save_timeout_cancels_without_unpinning_dma_pages():
    backend = _FakeExternalBackend()
    scheduler, _manager = _external_scheduler(backend)
    scheduler.external_cache_index._save_timeout_seconds = 0.001
    request = Request("external-save-timeout", list(range(257)), max_new_tokens=1)
    scheduler.add_request(request)
    first_chunk = scheduler.schedule()
    scheduler.update_from_output(first_chunk, {})
    save = scheduler.schedule().external_cache_saves[0]
    snapshot = scheduler._external_cache_save_snapshots[save.job_id]
    snapshot.dispatched_at = time.monotonic() - 1
    pinned = scheduler.external_cache_stats.pinned_hbm_blocks

    timeout_output = scheduler.schedule()

    assert timeout_output.external_cache_cancellations == [save.job_id]
    assert scheduler.external_cache_stats.save_timeouts == 1
    assert scheduler.external_cache_stats.pinned_hbm_blocks == pinned
    assert scheduler._external_cache_disabled

    scheduler.finish_external_cache_save(save.job_id, succeeded=False)
    assert scheduler.external_cache_stats.pinned_hbm_blocks == 0


def test_scheduler_limits_pending_save_snapshots_and_pinned_pages():
    backend = _FakeExternalBackend()
    scheduler, _manager = _external_scheduler(backend)
    scheduler.external_cache_index._max_pending_saves = 1
    request = Request("external-save-backpressure", list(range(385)), max_new_tokens=1)
    scheduler.add_request(request)

    first_chunk = scheduler.schedule()
    scheduler.update_from_output(first_chunk, {})
    second_chunk = scheduler.schedule()
    first_save = second_chunk.external_cache_saves[0]
    pinned = scheduler.external_cache_stats.pinned_hbm_blocks
    scheduler.update_from_output(second_chunk, {})

    assert len(scheduler._external_cache_save_snapshots) == 1
    assert not scheduler._pending_external_cache_saves
    assert scheduler.external_cache_stats.pinned_hbm_blocks == pinned
    assert scheduler.external_cache_stats.save_dropped == 1

    scheduler.finish_external_cache_save(first_save.job_id, succeeded=True)
    assert scheduler.external_cache_stats.pinned_hbm_blocks == 0


def test_scheduler_rejects_save_that_exceeds_pinned_page_budget():
    backend = _FakeExternalBackend()
    scheduler, _manager = _external_scheduler(backend)
    scheduler.external_cache_index._max_pending_save_blocks = 1
    request = Request("external-save-page-budget", list(range(257)), max_new_tokens=1)
    scheduler.add_request(request)

    first_chunk = scheduler.schedule()
    scheduler.update_from_output(first_chunk, {})

    assert not scheduler._external_cache_save_snapshots
    assert not scheduler._pending_external_cache_saves
    assert scheduler.external_cache_stats.pinned_hbm_blocks == 0
    assert scheduler.external_cache_stats.save_dropped == 1


def test_scheduler_abort_waits_for_external_write_fence():
    backend = _FakeExternalBackend(present_suffix="/p2/t256")
    scheduler, manager = _external_scheduler(backend)
    request = Request("external-abort", list(range(257)), max_new_tokens=1)
    scheduler.add_request(request)
    load = scheduler.schedule().external_cache_loads[0]

    scheduler.abort_request(request.request_id)

    assert request.request_id in scheduler.waiting_for_remote_kv
    assert scheduler.schedule().external_cache_cancellations == [load.job_id]
    scheduler.finish_external_cache_load(
        request.request_id,
        job_id=load.job_id,
        succeeded=False,
    )
    assert request.request_id not in scheduler.requests
    assert manager.group_request_partition(request.request_id) is None


def test_scheduler_pins_checkpoint_pages_until_save_completion():
    backend = _FakeExternalBackend()
    scheduler, manager = _external_scheduler(backend)
    request = Request("external-save", list(range(257)), max_new_tokens=1)
    scheduler.add_request(request)
    first_chunk = scheduler.schedule()

    scheduler.update_from_output(first_chunk, {})
    save_output = scheduler.schedule()

    assert len(save_output.external_cache_saves) == 1
    save = save_output.external_cache_saves[0]
    assert save.checkpoint_token_count == 128
    assert save.source_partition == request.cache_partition
    assert ExternalKVCheckpointManifest.from_bytes(save.manifest_payload).manifest_key == save.manifest_key
    assert scheduler.external_cache_stats.pinned_hbm_blocks > 0
    assert scheduler.external_cache_stats.peak_pinned_hbm_blocks > 0

    scheduler.finish_request(request.request_id, RequestStatus.FINISHED_ABORTED)
    assert scheduler.has_work()
    scheduler.finish_external_cache_save(save.job_id, succeeded=True)
    assert save.job_id not in scheduler._external_cache_save_snapshots
    assert scheduler.external_cache_stats.save_successes == 1
    assert scheduler.external_cache_stats.save_bytes > 0
    assert scheduler.external_cache_stats.pinned_hbm_blocks == 0
