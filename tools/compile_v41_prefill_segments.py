# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Generate selected V4.1 prefill code without device binary assembly or execution."""
import argparse


def main():
    from pypto_serving.model.deepseek_v41.prefill_segment import PREFILL_ARGUMENTS, compile_prefill_segments
    from pypto_serving.model.deepseek_v41.swa_segment import SegmentTopology

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib-root", required=True)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--dp", type=int, default=2)
    parser.add_argument("--local-capacity", type=int, default=16)
    parser.add_argument("--modes", nargs="+", choices=tuple(PREFILL_ARGUMENTS), required=True)
    parser.add_argument("--build-dir", default="build_output/v41-prefill-composites")
    args = parser.parse_args()
    from pypto.ir import DistributedConfig
    from pypto.runtime import RunConfig
    from pypto_serving.model.common.compiler.compiler import KernelCompiler

    topology = SegmentTopology(tp=args.tp, dp=args.dp, local_capacity=args.local_capacity)
    config = RunConfig(platform="a5", distributed_config=DistributedConfig(device_ids=list(range(topology.world))))
    compiler = KernelCompiler(run_config=config, cache_dir=args.build_dir)
    attention, _ = compile_prefill_segments(compiler, args.lib_root, topology, args.modes)
    print("CODEGEN PASS:", ", ".join(attention), "+ packed-FP4 MoE; no binary assembly/device execution", flush=True)


if __name__ == "__main__":
    main()
