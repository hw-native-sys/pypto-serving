# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Start one real DSpark role and validate its Phase D cache/owner contract.

This probe deliberately stops before peer exchange, transfer, or Decode. Run P
and D serially on the same 16-device host; never run both roles concurrently.
"""

from __future__ import annotations

import argparse
import asyncio
import json

from pypto_serving.cli.main import build_parser, build_serving_engine_config
from pypto_serving.model.tokenizer import load_tokenizer
from pypto_serving.serving.engine.async_engine import ReplicaEngineCore
from pypto_serving.serving.memory.kv_cache import GroupReservationState
from pypto_serving.serving.pd.config import (
    PD_LOGICAL_GROUPS,
    PD_PHYSICAL_REGIONS,
    PDCapabilities,
    PDRole,
)
from pypto_serving.serving.pd.connector import DecodeConnector
from pypto_serving.serving.pd.planner import ChunkTransferPlanner, PChunkLifecycle
from pypto_serving.serving.pd.protocol import (
    AbortHandoff,
    HandoffKey,
    ReserveAccepted,
    ReserveRequest,
)
from pypto_serving.serving.pd.session import validate_rank_registrations
from pypto_serving.serving.pd.worker_api import (
    OP_PREPARE_REGISTRY,
    WorkerRegistryBundle,
    decode_worker_payload,
)
from pypto_serving.transfer.types import CompletionCertainty


def _engine_config(args: argparse.Namespace):
    cli = build_parser().parse_args(
        [
            "--model",
            args.model,
            "--served-model-name",
            "dsv4-flash-dspark-w8a8",
            "--backend",
            "npu",
            "--platform",
            "a2a3",
            "--devices",
            ",".join(str(device) for device in range(16)),
            "--dp",
            "4",
            "--ep",
            "16",
            "--tp",
            "4",
            "--block-size",
            "32",
            "--max-model-len",
            "1024",
            "--max-num-seqs",
            "8",
            "--max-num-batched-tokens",
            "8192",
            "--long-prefill-token-threshold",
            "128",
            "--speculative-config",
            '{"method":"dspark","num_speculative_tokens":7}',
            "--no-enable-prefix-caching",
            "--ring-heap",
            "2147483648,2147483648,4294967296,8589934592",
            "--use-compile-cache",
            "--pd-role",
            args.role,
            "--pd-node-id",
            f"pd-probe-{args.role}",
            "--pd-peer-node-id",
            f"synthetic-peer-{args.role}",
            "--pd-run-id",
            args.run_id,
            "--pd-control-host",
            "127.0.0.1",
            "--pd-peer-host",
            "127.0.0.1",
            "--pd-transfer-hostname",
            args.transfer_hostname,
            "--pd-model-revision",
            args.model_revision,
            "--pd-journal-path",
            f"/tmp/{args.run_id}-{args.role}-probe-pd-journal.jsonl",
        ]
    )
    return build_serving_engine_config(cli)


def _assert_disjoint(first: dict[str, tuple[int, ...]], second: dict[str, list[int]]) -> None:
    for group in PD_LOGICAL_GROUPS:
        if set(first[group]) & set(second[group]):
            raise AssertionError(f"reserved {group} pages leaked into normal allocation")


async def _run(args: argparse.Namespace) -> None:
    config = _engine_config(args)
    tokenizer = load_tokenizer(config.model_dir)
    core = ReplicaEngineCore(config, tokenizer)
    await core.start()
    try:
        wire = await core.call_pd_worker(OP_PREPARE_REGISTRY)
        bundle = decode_worker_payload(wire, WorkerRegistryBundle)
        registry = bundle.registry()
        if bundle.model_revision != args.model_revision:
            raise AssertionError("worker exported an unexpected model revision")
        if bundle.topology != (16, 4):
            raise AssertionError(f"unexpected DSpark topology {bundle.topology}")
        if registry.fingerprint != bundle.registry_fingerprint:
            raise AssertionError("registry fingerprint changed during serialization")
        if registry.layout_fingerprint != bundle.layout_fingerprint:
            raise AssertionError("layout fingerprint changed during serialization")
        if tuple(core.kv_cache_manager.group_names) != PD_LOGICAL_GROUPS:
            raise AssertionError("scheduler logical cache groups differ from PD contract")
        if {component.component_id for component in registry.components} != set(
            PD_PHYSICAL_REGIONS
        ):
            raise AssertionError("runner registry does not expose eight physical regions")
        validate_rank_registrations(bundle.ranks, expected_count=16)
        for rank in bundle.ranks:
            for region in rank.regions:
                if region.extent != registry.entry(region.component_id).extent:
                    raise AssertionError("owner region extent differs from cache geometry")

        key = HandoffKey("phase-d-probe", "handoff-1", 1, 1, 1)
        if config.pd_config.role is PDRole.DECODE:
            capabilities = PDCapabilities(
                model_revision=bundle.model_revision,
                registry_fingerprint=bundle.registry_fingerprint,
                layout_fingerprint=bundle.layout_fingerprint,
                topology=bundle.topology,
            )
            connector = DecodeConnector(
                core.kv_cache_manager,
                capabilities,
                registry,
                bundle.ranks,
            )
            accepted = connector.reserve(
                ReserveRequest(key, 64, 128, bundle.layout_fingerprint)
            )
            if not isinstance(accepted, ReserveAccepted):
                raise AssertionError(f"real D reservation was rejected: {accepted}")
            reservation = core.kv_cache_manager.group_cache_reservation(
                accepted.reservation_id
            )
            if (
                reservation is None
                or reservation.state is not GroupReservationState.CONSTRUCTING
                or not reservation.write_authorized
            ):
                raise AssertionError("D reservation did not reach authorized CONSTRUCTING")
            try:
                core.kv_cache_manager.adopt_group_cache(accepted.reservation_id)
            except ValueError:
                pass
            else:
                raise AssertionError("D adopted cache before manifest commit")
            ordinary = core.kv_cache_manager.ensure_group_blocks(
                "phase-d-scratch-probe",
                1,
                partition=accepted.partition,
            )
            _assert_disjoint(reservation.block_ids_by_group, ordinary)
            core.kv_cache_manager.release_all_group_requests("phase-d-scratch-probe")
            connector.abort(
                AbortHandoff(key, "PROBE_COMPLETE", True),
                deterministic=True,
            )
            released = core.kv_cache_manager.group_cache_reservation(
                accepted.reservation_id
            )
            if released is None or released.state is not GroupReservationState.RELEASED:
                raise AssertionError("D deterministic probe reservation did not release")
            partition = accepted.partition
        else:
            blocks = core.kv_cache_manager.ensure_group_blocks(
                "phase-d-probe",
                192,
                partition=0,
            )
            source = {name: tuple(ids) for name, ids in blocks.items()}
            rank_ids = tuple(range(4))
            tables = {rank_id: source for rank_id in rank_ids}
            plan = ChunkTransferPlanner(registry, core.kv_cache_manager.group_specs).plan_chunk(
                key,
                chunk_id=0,
                start_token=0,
                end_token=64,
                final=True,
                rank_ids=rank_ids,
                source_blocks_by_rank=tables,
                destination_blocks_by_rank=tables,
            )
            if set(plan.copies_by_rank) != set(rank_ids):
                raise AssertionError("P plan does not cover its TP4 cache partition")
            if {unit.component_id for unit in plan.expected_units} != set(
                PD_PHYSICAL_REGIONS
            ):
                raise AssertionError("P final plan omits a physical region")
            lifecycle = PChunkLifecycle(core.kv_cache_manager)
            guard = lifecycle.begin(
                "phase-d-probe",
                0,
                source,
                0,
            )
            ordinary = core.kv_cache_manager.ensure_group_blocks(
                "phase-d-scratch-probe",
                1,
                partition=0,
            )
            _assert_disjoint(source, ordinary)
            core.kv_cache_manager.release_all_group_requests("phase-d-scratch-probe")
            lifecycle.settle(
                guard,
                CompletionCertainty.NOT_SUBMITTED,
            )
            core.kv_cache_manager.release_all_group_requests("phase-d-probe")
            partition = 0

        print(
            json.dumps(
                {
                    "kind": "phase_d_cache_registry_probe",
                    "status": "PASS",
                    "role": args.role,
                    "topology": bundle.topology,
                    "logical_groups": len(core.kv_cache_manager.group_names),
                    "physical_regions": len(registry.components),
                    "owner_ranks": len(bundle.ranks),
                    "partition": partition,
                    "registry_fingerprint": bundle.registry_fingerprint,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        await core.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("prefill", "decode"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--transfer-hostname", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--run-id", required=True)
    asyncio.run(_run(parser.parse_args()))
