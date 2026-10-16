# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Routing policy is independent of HTTP and of physical cache placement."""


class RoundRobin:
    def __init__(self):
        self.cursor = 0

    def select(self, pairs):
        if not pairs:
            raise RuntimeError("no compatible healthy P/D pair")
        pair = pairs[self.cursor % len(pairs)]
        self.cursor += 1
        return pair


class Directory:
    def __init__(self, endpoints, policy=None):
        self.endpoints = {e.node_id: e for e in endpoints}
        self.nodes = {}
        self.policy = policy or RoundRobin()

    def update(self, node):
        if node.node_id not in self.endpoints:
            raise ValueError("unconfigured node")
        self.nodes[node.node_id] = node

    def pairs(self, model=None):
        p = [n for n in self.nodes.values() if n.role == "prefill"]
        d = [n for n in self.nodes.values() if n.role == "decode"]
        # Rotate the D offset after each P cycle, rather than pinning P_i to D_i.
        pairs = [(p[i], d[(i + offset) % len(d)])
                 for offset in range(len(d)) for i in range(len(p))
                 if p[i].compatible(d[(i + offset) % len(d)])
                 and (model is None or p[i].contract.model == model)]
        return pairs

    def select(self, model=None):
        return self.policy.select(self.pairs(model))
