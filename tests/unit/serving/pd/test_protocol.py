# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import socket

import pytest

from pypto_serving.model.deepseek_dspark.pd_adapter import DSV4_DSPARK_K7_CONTRACT
from pypto_serving.serving.pd.config import PDCapabilities, PDRole
from pypto_serving.serving.pd.protocol import (
    ChunkManifest,
    ContinuationMetadata,
    DecodeOutputWire,
    FramedChannel,
    HandoffKey,
    PrefixMatchSpec,
    QueryHandoff,
    decode_message,
    encode_message,
    exchange_and_validate_hello,
    make_hello,
    make_prefix_match_spec,
    validate_prefix_match_spec,
)
from pypto_serving.serving.reasoning import OutputParserSpec


def _capabilities(model_revision: str = "ds-v4-test") -> PDCapabilities:
    return PDCapabilities(
        adapter_id=DSV4_DSPARK_K7_CONTRACT.adapter_id,
        contract_version=DSV4_DSPARK_K7_CONTRACT.version,
        contract_digest=DSV4_DSPARK_K7_CONTRACT.digest,
        continuation_schema=DSV4_DSPARK_K7_CONTRACT.continuation_schema,
        model_revision=model_revision,
        registry_fingerprint="f" * 64,
        layout_fingerprint="l" * 64,
        topology=(16, 4),
        logical_groups=DSV4_DSPARK_K7_CONTRACT.logical_groups,
        physical_regions=DSV4_DSPARK_K7_CONTRACT.physical_regions,
    )


def _channels():
    left, right = socket.socketpair()
    left.settimeout(2)
    right.settimeout(2)
    return (
        left,
        right,
        FramedChannel(
            left,
            local_node_id="prefill",
            peer_node_id="decode",
        ),
        FramedChannel(
            right,
            local_node_id="decode",
            peer_node_id="prefill",
        ),
    )


def test_hello_and_message_round_trip() -> None:
    left, right, prefill, decode = _channels()
    p_hello = make_hello(
        node_id="prefill",
        role=PDRole.PREFILL,
        run_id="run-1",
        capabilities=_capabilities(),
    )
    d_hello = make_hello(
        node_id="decode",
        role=PDRole.DECODE,
        run_id="run-1",
        capabilities=_capabilities(),
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        p_result = pool.submit(
            exchange_and_validate_hello,
            prefill,
            p_hello,
            expected_peer_node_id="decode",
            expected_peer_role=PDRole.DECODE,
        )
        d_result = pool.submit(
            exchange_and_validate_hello,
            decode,
            d_hello,
            expected_peer_node_id="prefill",
            expected_peer_role=PDRole.PREFILL,
        )
        assert p_result.result().node_id == "decode"
        assert d_result.result().node_id == "prefill"

    key = HandoffKey("request", "handoff", 1, 1, 1)
    prefill.send(QueryHandoff(key))
    assert decode.receive() == QueryHandoff(key)
    left.close()
    right.close()


def test_capability_mismatch_is_rejected() -> None:
    left, right, prefill, decode = _channels()
    p_hello = make_hello(
        node_id="prefill",
        role=PDRole.PREFILL,
        run_id="run-1",
        capabilities=_capabilities(),
    )
    d_hello = make_hello(
        node_id="decode",
        role=PDRole.DECODE,
        run_id="run-1",
        capabilities=_capabilities("other-model"),
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = (
            pool.submit(
                exchange_and_validate_hello,
                prefill,
                p_hello,
                expected_peer_node_id="decode",
                expected_peer_role=PDRole.DECODE,
            ),
            pool.submit(
                exchange_and_validate_hello,
                decode,
                d_hello,
                expected_peer_node_id="prefill",
                expected_peer_role=PDRole.PREFILL,
            ),
        )
        for result in results:
            with pytest.raises(ValueError, match="model_revision"):
                result.result()
    left.close()
    right.close()


def test_contract_digest_mismatch_is_rejected() -> None:
    local = _capabilities()
    peer = object.__new__(PDCapabilities)
    for name, value in vars(local).items():
        object.__setattr__(peer, name, value)
    object.__setattr__(peer, "contract_digest", "0" * 64)

    assert local.compatibility_error(peer) == (
        "PD capability mismatch: contract_digest"
    )


def test_prefix_cache_profile_mismatch_is_rejected() -> None:
    local = replace(_capabilities(), prefix_cache_mode="d_only")
    peer = replace(_capabilities(), prefix_cache_mode="disabled")
    assert local.compatibility_error(peer) == (
        "PD capability mismatch: prefix_cache_mode"
    )


def test_prefix_match_spec_is_canonical_bounded_and_tamper_evident() -> None:
    spec = make_prefix_match_spec(
        token_count=128,
        alignment=128,
        contract_digest=DSV4_DSPARK_K7_CONTRACT.digest,
        group_block_hashes={"ori": [b"a" * 32, b"b" * 32]},
    )
    validate_prefix_match_spec(spec)
    tampered = PrefixMatchSpec(
        spec.schema_version,
        spec.token_count,
        spec.alignment,
        spec.contract_digest,
        spec.groups,
        "0" * 64,
    )
    with pytest.raises(ValueError, match="tampered"):
        validate_prefix_match_spec(tampered)


def test_old_continuation_schema_is_rejected_before_handoff() -> None:
    local = _capabilities()
    peer = replace(local, continuation_schema="deepseek-v4-dspark-k7/v1")

    assert local.compatibility_error(peer) == (
        "PD capability mismatch: continuation_schema"
    )


def test_capacity_specific_registry_fingerprint_is_not_a_peer_compatibility_gate() -> None:
    local = _capabilities()
    peer = object.__new__(PDCapabilities)
    for name, value in vars(local).items():
        object.__setattr__(peer, name, value)
    object.__setattr__(peer, "registry_fingerprint", "d" * 64)

    assert local.compatibility_error(peer) is None


def test_control_message_codec_rejects_empty_input() -> None:
    message = QueryHandoff(HandoffKey("request", "handoff", 1, 1, 1))
    assert decode_message(encode_message(message)) == message
    with pytest.raises(ValueError, match="size"):
        decode_message(b"")


def test_parser_spec_and_reasoning_survive_pd_wire_round_trip() -> None:
    key = HandoffKey("request", "handoff", 1, 1, 1)
    continuation = ContinuationMetadata(
        prompt_token_ids=(1, 2, 3),
        max_new_tokens=4,
        temperature=0.0,
        top_p=1.0,
        top_k=None,
        seed=None,
        stop_strings=(),
        eos_token_id=2,
        output_parser_spec=OutputParserSpec(
            "deepseek_v4",
            "reasoning",
            include_reasoning=False,
        ),
    )
    manifest = ChunkManifest(
        key=key,
        chunk_id=0,
        start_token=0,
        end_token=3,
        final=True,
        manifest_hash="m" * 64,
        expected_units=(),
        copies_by_rank={},
        source_prefix_hit_tokens=128,
        first_token=100,
        continuation=continuation,
    )
    decoded_manifest = decode_message(encode_message(manifest))
    assert decoded_manifest.continuation == continuation
    assert decoded_manifest.source_prefix_hit_tokens == 128

    output = DecodeOutputWire(
        key=key,
        token_id=101,
        text="answer",
        reasoning="hidden on request but valid on the wire",
        finished=True,
        finish_reason="FINISHED_LENGTH",
        prompt_tokens=3,
        completion_tokens=2,
    )
    assert decode_message(encode_message(output)) == output
