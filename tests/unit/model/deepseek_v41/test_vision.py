# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Image request transport and absolute-position prefill regressions."""

import base64
import io
from types import SimpleNamespace

import msgspec
from PIL import Image
import pytest
import torch

from pypto_serving.model.deepseek_v41.encoding import DeepSeekV41TokenizerAdapter
from pypto_serving.model.deepseek_v41.vision import (
    V41VisionResources, image_grid, image_patches, vision_ring_heaps,
)
from pypto_serving.model.tokenizer import MultimodalPrompt
from pypto_serving.serving.server.ipc import NewRequestData
from pypto_serving.serving.memory.kv_cache import KvCacheManager
from pypto_serving.serving.sched.scheduler import Request, Scheduler, SchedulerConfig


CONFIG = dict(patch_size=2, downsample_ratio=3, min_pixels=4, max_image_tokens=32, max_wh_ratio=None)


def test_vision_workspace_fits_active_scopes_without_broadcasting_layer_heap():
    cfg = SimpleNamespace(dim=1024, intermediate=2816, downsample=3, language_dim=5120)
    assert vision_ring_heaps(cfg, 51, 74) == (16 << 20, 32 << 20, 512 << 20, 16 << 20)
    assert vision_ring_heaps(cfg, 33, 47) == (16 << 20, 16 << 20, 256 << 20, 16 << 20)
    large = vision_ring_heaps(cfg, 100, 100)
    assert large[1] > 32 << 20
    assert large[2] > 512 << 20
    assert large[0] == large[3] == 16 << 20


def image_bytes(color):
    output = io.BytesIO()
    Image.new("RGB", (2, 2), color).save(output, format="PNG")
    return output.getvalue()


class Backend:
    def encode(self, text, **kwargs):
        return list(text.encode())


def prompt(color):
    adapter = DeepSeekV41TokenizerAdapter(Backend())
    adapter.model_config = dict(vision_config=CONFIG, image_token_id=129264)
    url = "data:image/png;base64," + base64.b64encode(image_bytes(color)).decode()
    return adapter.apply_chat_template([dict(role="user", content=[
        dict(type="text", text="before "), dict(type="image_url", image_url=dict(url=url)),
        dict(type="text", text=" after")])])


def test_processor_patch_channel_order_and_bf16_normalization():
    patches = image_patches(image_bytes("red"), CONFIG, (1, 1))
    assert patches.shape == (1, 3, 2, 2)
    assert patches.dtype == torch.bfloat16
    assert torch.equal(patches[0, 0], torch.ones(2, 2))
    assert torch.equal(patches[0, 1:], -torch.ones(2, 2, 2))
    for width, height in ((1, 10000), (10000, 1), (2000, 3000)):
        h, w = image_grid(width, height, CONFIG)
        assert min(h, w) > 0
        assert ((h + 2) // 3) * (((w + 2) // 3) + 1) + 2 <= CONFIG["max_image_tokens"]


def test_expanded_prompt_and_wire_records_keep_distinct_images():
    red, blue = prompt("red"), prompt("blue")
    assert isinstance(red, MultimodalPrompt)
    assert red.token_ids == blue.token_ids
    assert red.cache_salt != blue.cache_salt
    image = red.images[0]
    assert image["length"] == 4
    assert red.token_ids[image["start"]:image["start"] + 4] == (129264,) * 4
    record = NewRequestData("r", list(red.token_ids), 0, 1, None, multimodal_inputs=list(red.images))
    restored = msgspec.msgpack.decode(msgspec.msgpack.encode(record), type=NewRequestData)
    assert restored.multimodal_inputs == list(red.images)
    with pytest.raises(ValueError, match="base64"):
        adapter = DeepSeekV41TokenizerAdapter(Backend())
        adapter.model_config = dict(vision_config=CONFIG, image_token_id=129264)
        adapter.apply_chat_template([dict(role="user", content=[
            dict(type="image_url", image_url=dict(url="data:image/png;base64,%%%"))])])


def test_chunks_intersect_absolute_image_spans_and_release():
    resources = V41VisionResources.__new__(V41VisionResources)
    resources.image_token_id = 99
    span = torch.arange(30).reshape(10, 3).bfloat16()
    resources.requests = {"image": ([], [(4, span)])}
    request = SimpleNamespace(request_id="image", start=9, end=17, token_ids=(99,) * 5 + (1,) * 3)
    other = SimpleNamespace(request_id="text", start=0, end=2, token_ids=(2, 3))
    source = torch.zeros(10, 3, dtype=torch.bfloat16)
    actual = resources.inject(SimpleNamespace(requests=(request, other)), source)
    assert torch.equal(actual[:5], span[5:])
    assert not bool(actual[5:].count_nonzero())
    assert not bool(source.count_nonzero())
    assert resources.mask("image", 0, 6).tolist() == [False] * 4 + [True] * 2
    resources.release_request("image")
    assert not bool(resources.mask("image", 0, 16).any())


def test_routing_bias_uses_packed_request_offsets_and_padding():
    resources = V41VisionResources.__new__(V41VisionResources)
    resources.topology = SimpleNamespace(world=2, dp=1, tp=2, capacity=8)
    resources.layers, resources.experts = 2, 4
    resources.text_bias = torch.zeros(2, 4)
    resources.vision_bias = torch.ones(2, 4)
    resources.routing_bias = torch.empty(2, 2 * 8 * 4)
    resources.requests = {"image": ([], [(4, torch.zeros(3, 1))])}
    first = SimpleNamespace(request_id="text", partition=0, start=0, token_ids=(1, 2))
    second = SimpleNamespace(request_id="image", partition=0, start=3, token_ids=(1, 99, 99, 99))
    resources.stage_bias(SimpleNamespace(requests=(first, second)))
    expected = torch.tensor([0, 0, 0, 1, 1, 1, 0, 0]).float()
    actual = resources.routing_bias.view(2, 2, 8, 4)
    assert torch.equal(actual, expected[None, None, :, None].expand_as(actual))


def test_image_identity_is_included_in_scheduler_prefix_hashes():
    manager = KvCacheManager(num_blocks=8, block_size=2, enable_prefix_cache=True)
    scheduler = Scheduler(SchedulerConfig(enable_prefix_cache=True), manager)
    requests = [Request(name, [1, 2, 3, 4], max_new_tokens=1, cache_salt=salt)
                for name, salt in (("red", b"red"), ("blue", b"blue"), ("red-again", b"red"))]
    for request in requests:
        scheduler.add_request(request)
    assert requests[0].block_hashes != requests[1].block_hashes
    assert requests[0].block_hashes == requests[2].block_hashes
    assert requests[0].block_hashes != manager.compute_block_hashes([1, 2, 3, 4])
