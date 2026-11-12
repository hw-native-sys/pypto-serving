# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""V4.1 image preprocessing, request spans and lib vision encoding."""

from __future__ import annotations

import base64
import hashlib
import importlib
import io
import logging
import math
import os
import uuid
from dataclasses import replace
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch

from pypto_serving.model.tokenizer import MultimodalPrompt

logger = logging.getLogger(__name__)


def load_image_bytes(url):
    if not isinstance(url, str):
        raise ValueError("image_url.url must be a string")
    if url.startswith("data:"):
        header, separator, payload = url.partition(",")
        if not separator or ";base64" not in header:
            raise ValueError("image data URLs must use base64 encoding")
        try:
            return base64.b64decode(payload, validate=True)
        except ValueError as exc:
            raise ValueError("invalid base64 image data") from exc
    if url.startswith(("http://", "https://")):
        with urlopen(url, timeout=30) as response:
            return response.read()
    raise ValueError("image URLs must use http, https or base64 data URLs")


def image_grid(width, height, config):
    """Apply the released processor's resize and total image-token budget."""
    patch, downsample = config["patch_size"], config["downsample_ratio"]
    maximum = config["max_image_tokens"]
    max_ratio = config.get("max_wh_ratio")
    if max_ratio is not None and width > height * max_ratio:
        width = height * max_ratio
    if 0 < width * height < config["min_pixels"]:
        scale = math.sqrt(config["min_pixels"] / (width * height))
        width, height = int(width * scale), int(height * scale)
    target_w = math.ceil(width / patch) * patch
    target_h = math.ceil(height / patch) * patch
    grid_h = math.ceil((target_h // patch) / downsample)
    grid_w = math.ceil((target_w // patch) / downsample)
    if grid_h * (grid_w + 1) + 2 > maximum:
        ratio = height / width
        float_w = math.sqrt((maximum - 2) / ratio + 0.25) - 0.5
        float_h = float_w * ratio
        cell = patch * downsample
        if float_w < 1:
            target_h, target_w = (maximum - 2) // 2 * cell, cell
        elif float_h < 1:
            target_h, target_w = cell, (maximum - 3) * cell
        else:
            scale = min(math.floor(float_w) * cell / width, math.floor(float_h) * cell / height)
            target_h = math.floor(height * scale / patch) * patch
            target_w = math.floor(width * scale / patch) * patch
    if min(target_h, target_w) <= 0:
        raise ValueError("image dimensions cannot fit the configured vision token budget")
    return target_h // patch, target_w // patch


def image_span_length(h, w, downsample):
    return math.ceil(h / downsample) * (math.ceil(w / downsample) + 1) + 2


def image_patches(data, config, grid):
    from PIL import Image, ImageOps

    patch = config["patch_size"]
    h, w = grid
    with Image.open(io.BytesIO(data)) as source:
        image = source.convert("RGB")
    if image_grid(image.width, image.height, config) != (h, w):
        raise ValueError("image grid differs from the admitted prompt")
    max_ratio = config.get("max_wh_ratio")
    size = (w * patch, h * patch)
    if max_ratio is not None and image.width >= max_ratio * image.height:
        image = image.resize(size)
    else:
        image = ImageOps.pad(image, size, color=(127, 127, 127))
    pixels = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
    pixels = ((pixels - 0.5) / 0.5).bfloat16()
    return pixels.reshape(3, h, patch, w, patch).permute(1, 3, 0, 2, 4).reshape(h * w, 3, patch, patch)


def prepare_chat(messages, adapter, options):
    from PIL import Image
    from .encoding import IMAGE_TOKEN

    config = getattr(adapter, "model_config", {}).get("vision_config")
    if not config:
        raise ValueError("the selected checkpoint has no vision configuration")
    image_token = adapter.model_config["image_token_id"]
    records, rendered_messages = [], []
    marker_prefix = f"PYPTO_IMAGE_{uuid.uuid4().hex}_"
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            rendered_messages.append(dict(message))
            continue
        parts = []
        for block in content:
            if not isinstance(block, dict):
                raise ValueError("message content blocks must be objects")
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif block.get("type") == "image_url" and message.get("role") == "user":
                value = block.get("image_url")
                if not isinstance(value, dict):
                    raise ValueError("image_url must contain a URL object")
                data = load_image_bytes(value.get("url"))
                with Image.open(io.BytesIO(data)) as image:
                    h, w = image_grid(image.width, image.height, config)
                marker = f"{marker_prefix}{len(records)}_END"
                records.append(dict(data=data, grid_h=h, grid_w=w, marker=marker))
                parts.append(marker)
            else:
                raise ValueError("V4.1 supports text blocks and user image_url blocks")
        rendered_messages.append(dict(message, content="".join(parts)))
    text = adapter.apply_chat_template(rendered_messages, **dict(options, tokenize=False))
    tokens, images, cursor = [], [], 0
    digest = hashlib.sha256()
    for record in records:
        position = text.index(record["marker"], cursor)
        tokens.extend(adapter.encode(text[cursor:position]))
        count = image_span_length(record["grid_h"], record["grid_w"], config["downsample_ratio"])
        image = {name: value for name, value in record.items() if name != "marker"}
        image.update(start=len(tokens), length=count)
        images.append(image)
        digest.update(record["data"])
        digest.update(f"/{len(tokens)}/{count}/".encode())
        tokens.extend([image_token] * count)
        cursor = position + len(record["marker"])
    tokens.extend(adapter.encode(text[cursor:]))
    for record in records:
        text = text.replace(record["marker"], IMAGE_TOKEN)
    if not images:
        return tokens if options.get("tokenize", False) else text
    result = MultimodalPrompt(text, tuple(tokens), tuple(images), digest.digest())
    return list(result.token_ids) if options.get("tokenize", False) else result


def vision_ring_heaps(cfg, height, width):
    """Size the lib vision program's outer and per-layer allocation scopes."""
    rows = height * width
    aligned = math.ceil(height / cfg.downsample) * math.ceil(width / cfg.downsample)
    # make_vision_program keeps two feature buffers and three aligner buffers
    # in scope 1. Scope 2 holds the layer intermediates and tiled attention scratch.
    # The onboard executor opens the implicit scope 0 before entering the program.
    outer = 2 * (2 * rows * cfg.dim + aligned * (cfg.dim * cfg.downsample ** 2 + 2 * cfg.language_dim))
    layer = 2 * rows * (12 * cfg.dim + 3 * cfg.intermediate)
    layer += 4 * 32 * 16 * math.ceil(rows / 64) * 64
    # Layer allocations overlap while the previous layer's tasks retire. Keep
    # two layers plus alignment/metadata room before rounding to runtime powers of two.
    return tuple(max(16 << 20, 1 << (size + (1 << 20) - 1).bit_length())
                 for size in (0, outer, 2 * layer, 0))


class V41VisionResources:
    """Request-owned image embeddings and pre-fork routing-bias storage."""

    def __init__(self, model_dir, raw, topology, device_id, loader):
        self.model_dir, self.config = Path(model_dir), raw["vision_config"]
        self.topology, self.device_id = topology, device_id
        self.image_token_id = raw["image_token_id"]
        text = raw["text_config"]
        self.layers, self.experts = text["num_hidden_layers"], text["n_routed_experts"]
        self.dim = text["hidden_size"]
        self.loader = loader
        self.text_bias = self.vision_bias = self.special = None
        shape = (topology.world, self.layers * topology.capacity * self.experts)
        self.routing_bias = torch.empty(shape, dtype=torch.float32).share_memory_()
        self.requests = {}
        self.encoder = self.weights = None

    @property
    def inherited_host_tensors(self):
        return (self.routing_bias,)

    def register(self, request_id, records, prompt_length):
        if request_id in self.requests:
            if self.requests[request_id][0] != records:
                raise ValueError("image inputs changed during an active request")
            return
        if self.encoder is None:
            from pypto.backend import BackendType
            from pypto.runtime import ExecutionMode, RunConfig

            module = importlib.import_module("models.deepseek_v4_1_flash.vision")
            self.text_bias = torch.stack([self.loader.store.load_tensor(f"layers.{i}.ffn.gate.bias")
                                          for i in range(self.layers)]).float()
            self.vision_bias = torch.stack([self.loader.store.load_tensor(f"layers.{i}.ffn.gate.bias_vl")
                                            for i in range(self.layers)]).float()
            self.special = {name: self.loader.store.load_tensor(name).bfloat16()
                            for name in ("image_start", "image_end", "image_newline")}
            cfg = module.VisionConfig(
                dim=self.config["hidden_size"], layers=self.config["num_hidden_layers"],
                heads=self.config["num_attention_heads"], intermediate=self.config["intermediate_size"],
                patch_size=self.config["patch_size"], language_dim=self.dim,
                downsample=self.config["downsample_ratio"], rope_theta=self.config["rope_theta"])
            self.weights = module.load_vision_weights(self.model_dir, cfg)
            self.encoder = module.VisionEncoder(cfg, run_config=RunConfig(
                backend_type=BackendType.Ascend950, execution_mode=ExecutionMode.ONBOARD,
                device_id=self.device_id))
        spans, previous_end = [], 0
        for record in records:
            h, w = record["grid_h"], record["grid_w"]
            count = image_span_length(h, w, self.config["downsample_ratio"])
            start = record["start"]
            if record["length"] != count or not previous_end <= start < start + count <= prompt_length:
                raise ValueError("image spans must be ordered, disjoint and contained in the prompt")
            patches = image_patches(record["data"], self.config, (h, w))
            override = os.environ.get("PYPTO_V41_VISION_RING_HEAP")
            heaps = int(override) if override is not None else vision_ring_heaps(self.encoder.cfg, h, w)
            self.encoder.run_config = replace(self.encoder.run_config, ring_heap=heaps)
            logger.info("V4.1 vision workspace request %s grid=%sx%s ring_heap=%s", request_id, h, w, heaps)
            _, aligned = self.encoder(patches, self.weights, h, w)
            lh, lw = math.ceil(h / self.config["downsample_ratio"]), math.ceil(w / self.config["downsample_ratio"])
            if aligned.dtype != torch.bfloat16 or tuple(aligned.shape) != (lh * lw, self.dim):
                raise ValueError("lib vision output has an incompatible shape or dtype")
            if not bool(torch.isfinite(aligned).all()):
                raise ValueError("lib vision output contains non-finite embeddings")
            span = torch.empty(count, self.dim, dtype=torch.bfloat16)
            span[0], span[-1] = self.special["image_start"], self.special["image_end"]
            grid = span[1:-1].view(lh, lw + 1, self.dim)
            grid[:, :lw] = aligned.view(lh, lw, self.dim)
            grid[:, lw] = self.special["image_newline"]
            spans.append((start, span))
            logger.info("V4.1 vision request %s grid=%sx%s aligned=%s span=[%s,%s) dtype=%s",
                        request_id, h, w, tuple(aligned.shape), start, start + count, aligned.dtype)
            previous_end = start + count
        self.requests[request_id] = (records, spans)

    def mask(self, request_id, start, count):
        mask = torch.zeros(count, dtype=torch.bool)
        for position, span in self.requests.get(request_id, (None, ()))[1]:
            begin, end = max(start, position), min(start + count, position + len(span))
            if begin < end:
                mask[begin - start:end - start] = True
        return mask

    def inject(self, step, embeddings):
        result, offset = embeddings.clone(), 0
        for request in step.requests:
            count = len(request.token_ids)
            mask = self.mask(request.request_id, request.start, count)
            ids = torch.tensor(request.token_ids)
            if bool((ids[mask] != self.image_token_id).any()):
                raise ValueError("image embedding rows must correspond to image placeholder IDs")
            if bool(mask.any()):
                logger.info("V4.1 image injection request %s chunk=[%s,%s) image_rows=%s",
                            request.request_id, request.start, request.end, int(mask.sum()))
            for position, span in self.requests.get(request.request_id, (None, ()))[1]:
                begin, end = max(request.start, position), min(request.end, position + len(span))
                if begin < end:
                    result[offset + begin - request.start:offset + end - request.start] = span[begin - position:end - position]
            offset += count
        return result

    def stage_bias(self, step):
        masks = torch.zeros(self.topology.world, self.topology.capacity, dtype=torch.bool)
        offsets = [0] * self.topology.dp
        for request in step.requests:
            count, begin = len(request.token_ids), offsets[request.partition]
            mask = self.mask(request.request_id, request.start, count)
            for tp_rank in range(self.topology.tp):
                rank = request.partition * self.topology.tp + tp_rank
                masks[rank, begin:begin + count] = mask
            offsets[request.partition] += count
        target = self.routing_bias.view(self.topology.world, self.layers, self.topology.capacity, self.experts)
        torch.where(masks[:, None, :, None], self.vision_bias[None, :, None, :],
                    self.text_bias[None, :, None, :], out=target)

    def release_request(self, request_id):
        self.requests.pop(request_id, None)

    def close(self):
        self.requests.clear()
        self.encoder = self.weights = None
