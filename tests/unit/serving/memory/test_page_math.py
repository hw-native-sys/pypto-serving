# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# -----------------------------------------------------------------------------------------------------------

import pytest

from pypto_serving.serving.memory.page_math import ceil_div, page_address, token_to_page


def test_page_arithmetic_boundaries():
    assert ceil_div(0, 128) == 0
    assert ceil_div(129, 128) == 2
    assert token_to_page(127, 128) == 0
    assert token_to_page(128, 128) == 1
    assert page_address(3, 129, 128) == 385


@pytest.mark.parametrize("value", [-1, 0])
def test_invalid_page_size(value):
    with pytest.raises(ValueError):
        ceil_div(1, value)
