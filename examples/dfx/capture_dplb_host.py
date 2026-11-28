# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Generate a metadata-only Host example; no model or device timing is measured."""

import argparse
from pathlib import Path

from pypto_serving.config.types import KVCacheGroupSpec, KVCacheSpec
from pypto_serving.observability.dplb_dfx import DplbDfxConfig, DplbDfxRecorder
from pypto_serving.serving.memory.kv_cache import KvCacheManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manager = KvCacheManager(enable_dplb=True)
    manager.init_groups(
        (
            KVCacheGroupSpec(
                "kv",
                (0,),
                KVCacheSpec(16, 16, 1),
                256,
                num_partitions=2,
            ),
        ),
        max_batch_size=8,
        primary_num_blocks=256,
    )
    recorder = DplbDfxRecorder(DplbDfxConfig(level="load"), args.output)
    manager.dplb_dfx = recorder
    recorder.start(example="metadata_only", policy=manager._dfx_policy())
    try:
        manager.dfx_baseline()
        # Full-prompt load is visible even though only the first chunk is allocated.
        manager.ensure_group_blocks("long", 16, partition=0, load_tokens=2048)
        manager.ensure_group_blocks("short", 16, partition=1, load_tokens=32)
        manager.ensure_group_blocks("incoming", 16, load_tokens=1536)
        manager.release_all_group_requests("incoming", reason="preempted")
        manager.ensure_group_blocks("incoming", 32, load_tokens=1536)
        for request_id in ("long", "short", "incoming"):
            manager.release_all_group_requests(request_id)
    finally:
        if not recorder.close():
            raise RuntimeError("Host example writer did not finish")
    print(recorder.sink.path)
    print(recorder.sink.path.with_suffix(".manifest.json"))


if __name__ == "__main__":
    main()
