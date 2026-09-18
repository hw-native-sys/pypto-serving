# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Cross-host CPU-only smoke for the fixed PD control channel."""

from __future__ import annotations

import argparse
import asyncio
import json

from pypto_serving.serving.pd.config import PDCapabilities, PDConfig, PD_PHYSICAL_REGIONS, PDRole
from pypto_serving.serving.pd.protocol import (
    HandoffKey,
    HandoffStatus,
    QueryHandoff,
    RankRegistration,
    RegionRegistration,
    RegistryAdvertisement,
)
from pypto_serving.serving.pd.session import open_control_session


def _registry(topology: tuple[int, ...]) -> RegistryAdvertisement:
    return RegistryAdvertisement(
        model_revision="phase-d-host-probe",
        topology=topology,
        registry_fingerprint="f" * 64,
        layout_fingerprint="l" * 64,
        ranks=tuple(
            RankRegistration(
                rank_id=rank_id,
                owner_generation=1,
                endpoint_generation=1,
                worker_id=f"probe-{rank_id}",
                regions=tuple(
                    RegionRegistration(component, 1, 64, b"cpu-probe")
                    for component in PD_PHYSICAL_REGIONS
                ),
            )
            for rank_id in range(topology[0])
        ),
    )


async def _run(args) -> None:
    role = PDRole(args.role)
    topology = (16, 4)
    config = PDConfig(
        role=role,
        node_id=args.node_id,
        peer_node_id=args.peer_node_id,
        run_id=args.run_id,
        control_host=args.local_host,
        control_port=args.port,
        peer_host=args.peer_host,
        auth_secret_env=args.secret_env,
        transfer_hostname=args.local_host,
        model_revision="phase-d-host-probe",
        connect_timeout_seconds=args.timeout,
    )
    capabilities = PDCapabilities(
        model_revision="phase-d-host-probe",
        registry_fingerprint="f" * 64,
        layout_fingerprint="l" * 64,
        topology=topology,
    )
    session = await open_control_session(config, capabilities)
    try:
        peer_registry = await session.exchange_registry(_registry(topology))
        key = HandoffKey("host-probe", "host-probe-handoff", 1, 1, 1)
        if role is PDRole.PREFILL:
            await session.send(QueryHandoff(key))
            response = await session.receive()
            if not isinstance(response, HandoffStatus) or response.state != "PROBE_OK":
                raise RuntimeError("Decode peer returned an invalid probe response")
        else:
            request = await session.receive()
            if not isinstance(request, QueryHandoff) or request.key != key:
                raise RuntimeError("Prefill peer sent an invalid probe request")
            await session.send(HandoffStatus(key, "PROBE_OK"))
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "role": role.value,
                    "node_id": config.node_id,
                    "peer_node_id": session.peer_hello.node_id,
                    "peer_registry": peer_registry.registry_fingerprint,
                    "topology": topology,
                    "regions_per_rank": len(PD_PHYSICAL_REGIONS),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        await session.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("prefill", "decode"), required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--peer-node-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--local-host", required=True)
    parser.add_argument("--peer-host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--secret-env", default="PYPTO_PD_AUTH_SECRET")
    parser.add_argument("--timeout", type=float, default=30.0)
    asyncio.run(_run(parser.parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
