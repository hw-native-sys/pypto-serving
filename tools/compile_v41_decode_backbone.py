# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Codegen-only check of the 40-layer V4.1 decode serving boundary."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib-root", required=True)
    parser.add_argument("--build-dir", default="build_output/v41-serving-decode")
    args = parser.parse_args()

    from pypto.ir import DistributedConfig
    from pypto.runtime import RunConfig

    from pypto_serving.model.common.compiler.compiler import KernelCompiler
    from pypto_serving.model.deepseek_v41.decode_backbone import compile_decode_backbone
    from pypto_serving.model.deepseek_v41.swa_segment import SegmentTopology

    topology = SegmentTopology.for_decode()
    config = RunConfig(platform="a5", distributed_config=DistributedConfig(device_ids=list(range(topology.world))))
    compiler = KernelCompiler(run_config=config, cache_dir=args.build_dir)
    _, arguments = compile_decode_backbone(compiler, args.lib_root, topology)
    print(f"CODEGEN PASS: 40-layer decode, {len(arguments)} ABI arguments")


if __name__ == "__main__":
    main()
