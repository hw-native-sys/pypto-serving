# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from concurrent.futures import ThreadPoolExecutor
import socket

import pytest

from pypto_serving.serving.pd.config import PDCapabilities, PDRole
from pypto_serving.serving.pd.protocol import (
    FramedChannel,
    HandoffKey,
    QueryHandoff,
    decode_message,
    encode_message,
    exchange_and_validate_hello,
    make_hello,
)


def _capabilities(model_revision: str = "ds-v4-test") -> PDCapabilities:
    return PDCapabilities(
        model_revision=model_revision,
        registry_fingerprint="f" * 64,
        layout_fingerprint="l" * 64,
        topology=(16, 4),
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


def test_k7_capability_mismatch_is_rejected() -> None:
    local = _capabilities()
    peer = object.__new__(PDCapabilities)
    for name, value in vars(local).items():
        object.__setattr__(peer, name, value)
    object.__setattr__(peer, "decode_speculative_tokens", 0)

    assert local.compatibility_error(peer) == (
        "PD capability mismatch: decode_speculative_tokens"
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
