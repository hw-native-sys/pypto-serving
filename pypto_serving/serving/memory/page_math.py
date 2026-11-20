# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# -----------------------------------------------------------------------------------------------------------

"""Checked arithmetic for token/page conversions."""

from __future__ import annotations


def _check_page_tokens(page_tokens: int) -> None:
    if type(page_tokens) is not int or page_tokens <= 0:
        raise ValueError("page_tokens must be a positive integer")


def ceil_div(value: int, divisor: int) -> int:
    """Return integer ceiling division for non-negative values."""
    if type(value) is not int or value < 0:
        raise ValueError("value must be a non-negative integer")
    _check_page_tokens(divisor)
    return (value + divisor - 1) // divisor


def token_to_page(position: int, page_tokens: int) -> int:
    """Return the page containing a non-negative token position."""
    if type(position) is not int or position < 0:
        raise ValueError("position must be a non-negative integer")
    _check_page_tokens(page_tokens)
    return position // page_tokens



def page_address(page_id: int, position: int, page_tokens: int) -> int:
    """Return the flat row address for a page and token position."""
    if type(page_id) is not int or page_id < 0:
        raise ValueError("page_id must be a non-negative integer")
    if type(position) is not int or position < 0:
        raise ValueError("position must be a non-negative integer")
    _check_page_tokens(page_tokens)
    return page_id * page_tokens + position % page_tokens
