# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CPU-only layouts consumed by V4.1 lib composite weights.

Matches pypto-lib 4c3eab2 quantization.pack_mx_b_scale and
pack_mxfp4_weight_tiles, with bounded FP4 tile scratch. No compiler imports.
"""

import torch


def pack_mx_scale(codes):
    """Logical [K/32,N] E8M0 bytes to MX_B_NN physical order."""
    groups, width = codes.shape
    if groups % 2 or width % 16:
        raise ValueError("MX_B_NN requires K divisible by 64 and N divisible by 16")
    return codes.reshape(groups // 2, 2, width // 16, 16).permute(2, 0, 3, 1).contiguous().reshape(
        groups, width,
    )


def pack_fp4_tiles(payload, tile=256):
    """Reorder packed [N,K/2] to lib [K*N/256,128], without float expansion."""
    source = payload.contiguous().view(torch.uint8)
    n, half_k = source.shape
    k = half_k * 2
    if k % tile or n % tile:
        raise ValueError("routed FP4 weights require complete 256x256 tiles")
    output = torch.empty((n // tile, k // tile, tile, tile // 2), dtype=torch.uint8)
    for nb in range(n // tile):
        for kb in range(k // tile):
            block = source[nb*tile:(nb+1)*tile, kb*(tile//2):(kb+1)*(tile//2)]
            codes = torch.empty((tile, tile), dtype=torch.uint8)
            codes[:, 0::2] = block & 15
            codes[:, 1::2] = block >> 4
            transposed = codes.T
            output[nb, kb] = transposed[:, 0::2] | (transposed[:, 1::2] << 4)
    return output.reshape(k * n // tile, tile // 2)


def fp8_input_major(payload, scale):
    """Checkpoint FP8 block32 [N,K] to [K,N] plus E8M0 MX_B_NN backing."""
    n, k = payload.shape
    if k % 64 or n % 32:
        raise ValueError("FP8 native projection requires K divisible by 64 and N by 32")
    codes = scale.contiguous().view(torch.uint8)
    if tuple(codes.shape) != (n // 32, k // 32):
        raise ValueError("FP8 block scales do not match the projection")
    return payload.T.contiguous(), pack_mx_scale(codes.T.repeat_interleave(32, dim=1)).view(
        torch.float8_e8m0fnu,
    )


def dequantize_output_groups(payload, scale, groups):
    """The wo_a ABI uses grouped BF16, with checkpoint block32 scales applied."""
    n, k = payload.shape
    if n % groups or n % 32 or k % 32:
        raise ValueError("wo_a requires complete output groups and 32x32 quantization blocks")
    codes = scale.contiguous().view(torch.uint8)
    if tuple(codes.shape) != (n // 32, k // 32):
        raise ValueError("wo_a scales do not match the projection")
    # Convert one output block at a time instead of materializing a full FP32 matrix.
    output = torch.empty((n, k), dtype=torch.bfloat16)
    for row in range(0, n, 32):
        factors = torch.exp2(codes[row // 32].float() - 127).repeat_interleave(32)
        output[row:row+32] = (payload[row:row+32].float() * factors).to(torch.bfloat16)
    return output.reshape(groups, n // groups, k)
