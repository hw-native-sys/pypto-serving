# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One versioned, bounded control representation; no native addresses reach Router."""

import hashlib
import json
import struct
import uuid

import msgspec

VERSION = 1
MAX_FRAME = 16 * 1024 * 1024


def digest(value):
    return hashlib.sha256(json.dumps(msgspec.to_builtins(value), sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class Component(msgspec.Struct, frozen=True):
    name: str
    group: str
    atomic: str
    final: bool = False


class Contract(msgspec.Struct, frozen=True):
    adapter: str
    model: str
    layout: str
    topology: tuple[int, int]
    components: tuple[Component, ...]


class Node(msgspec.Struct, frozen=True):
    node_id: str
    role: str
    boot: str
    contract: Contract
    registry: str
    provider: str = "mooncake"
    prefix: bool = True
    healthy: bool = True
    version: int = VERSION

    @property
    def key(self):
        return self.node_id, self.boot, self.registry

    def compatible(self, other):
        return (self.healthy and other.healthy and self.version == other.version == VERSION
                and {self.role, other.role} == {"prefill", "decode"}
                and self.provider == other.provider and self.contract == other.contract)


class Binding(msgspec.Struct, frozen=True):
    request: str
    prefill: tuple[str, str, str]
    decode: tuple[str, str, str]
    input_digest: str

    @classmethod
    def create(cls, p, d, payload):
        if not p.compatible(d):
            raise ValueError("incompatible PD nodes")
        return cls(uuid.uuid4().hex, p.key, d.key, digest(payload))

    def check(self, node, role):
        if getattr(self, role) != node.key or node.role != role:
            raise ValueError("request refers to a stale or different node")


class Prefix(msgspec.Struct, frozen=True):
    tokens: int
    alignment: int
    hashes: dict[str, tuple[str, ...]]


class Continuation(msgspec.Struct, frozen=True):
    tokens: tuple[int, ...]
    config: dict
    parser: dict | None
    constraint: dict | None
    prefix: Prefix


class Placement(msgspec.Struct, frozen=True):
    request: str
    hit: int
    partition: int
    blocks: dict[str, tuple[int, ...]]
    state_slots: dict[str, int] = msgspec.field(default_factory=dict)


class Copy(msgspec.Struct, frozen=True):
    component: str
    source_rank: int
    destination_rank: int
    source_offset: int
    destination_offset: int
    length: int


class Plan(msgspec.Struct, frozen=True):
    binding: Binding
    continuation_digest: str
    chunk: int
    start: int
    end: int
    final: bool
    mapping: tuple[tuple[int, int], ...]
    copies: tuple[Copy, ...]
    units: tuple[tuple[int, str], ...]
    first_token: int | None = None


class Completion(msgspec.Struct, frozen=True):
    plan_digest: str
    completed: tuple[tuple[int, str], ...]

    def check(self, plan):
        if (self.plan_digest != digest(plan) or len(set(self.completed)) != len(self.completed)
                or set(self.completed) != set(plan.units)):
            raise ValueError("incomplete or mismatched transfer completion")


async def send(writer, value):
    payload = msgspec.msgpack.encode(value)
    if len(payload) > MAX_FRAME:
        raise ValueError("PD control frame too large")
    writer.write(struct.pack("!I", len(payload)) + payload)
    await writer.drain()


async def receive(reader):
    size, = struct.unpack("!I", await reader.readexactly(4))
    if not 0 < size <= MAX_FRAME:
        raise ValueError("invalid PD control frame length")
    return msgspec.msgpack.decode(await reader.readexactly(size))


def decode(value, cls):
    return msgspec.convert(value, type=cls, strict=True)
