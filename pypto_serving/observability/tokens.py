# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


import hashlib


def token_ids_sha256(token_ids: tuple[int, ...]) -> str:
    """Return the canonical address-free digest used by all serving planes."""
    if not token_ids:
        return ""
    material = b"".join(
        int(token).to_bytes(8, "big", signed=True) for token in token_ids
    )
    return hashlib.sha256(material).hexdigest()
