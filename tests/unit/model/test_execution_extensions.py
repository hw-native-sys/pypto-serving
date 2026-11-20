# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from pypto_serving.model.deepseek_lowlat.decoding import (
    LowlatDecodeExtension, decode_decode_extension, encode_decode_extension,
)
from pypto_serving.serving.model_extension import RequestExecutionContext

def test_request_context_is_shared_admission_state_without_model_fields():
    context = RequestExecutionContext("request", 128, 64, execution_generation=3)
    assert context.prompt_tokens == 128
    assert not hasattr(context, "prefix_resume")


def test_model_extension_is_an_opaque_bytes_payload():
    payload = LowlatDecodeExtension((7,), (b"receipt",))
    encoded = encode_decode_extension(payload)
    assert isinstance(encoded, bytes)
    assert decode_decode_extension(encoded) == payload
