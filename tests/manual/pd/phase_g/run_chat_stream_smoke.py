# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run one real external-Router PD chat streaming regression."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import urllib.request


def _get(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-url", required=True)
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="dsv4-flash-dspark-w8a8")
    parser.add_argument("--prompt", default="紫禁城")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.max_tokens < 1:
        raise ValueError("max-tokens must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": True,
    }
    request = urllib.request.Request(
        f"{args.router_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        method="POST",
        headers={"content-type": "application/json"},
    )
    started_ns = time.time_ns()
    monotonic_ns = time.monotonic_ns()
    frames = []
    done = False
    content_type = ""
    with urllib.request.urlopen(request, timeout=args.timeout_seconds) as response:
        content_type = response.headers.get("content-type", "")
        for wire in response:
            line = wire.decode().strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                done = True
                continue
            frames.append(json.loads(data))
    elapsed_ns = time.monotonic_ns() - monotonic_ns

    errors = []
    if not content_type.startswith("text/event-stream"):
        errors.append("Router response is not text/event-stream")
    if not done:
        errors.append("SSE stream has no [DONE] terminator")
    if not frames:
        errors.append("SSE stream has no JSON frames")
    if any(frame.get("object") == "error" for frame in frames):
        errors.append("SSE stream contains an error frame")

    choice_frames = [frame for frame in frames if frame.get("choices")]
    usage_frames = [frame for frame in frames if not frame.get("choices")]
    if len(usage_frames) != 1 or not usage_frames[0].get("usage"):
        errors.append("SSE stream must contain exactly one terminal usage frame")
    request_ids = {frame.get("id", "") for frame in frames}
    request_ids.discard("")
    if len(request_ids) != 1:
        errors.append("SSE frames do not share one request id")
    text = "".join(
        frame["choices"][0].get("delta", {}).get("content", "")
        for frame in choice_frames
    )
    if not text:
        errors.append("chat stream produced no assistant text")
    finish_reasons = [
        frame["choices"][0].get("finish_reason")
        for frame in choice_frames
        if frame["choices"][0].get("finish_reason")
    ]
    if len(finish_reasons) != 1:
        errors.append("chat stream must contain one terminal finish reason")

    snapshots = {
        "router_metrics": _get(f"{args.router_url.rstrip('/')}/metrics", 30),
        "router_recovery": _get(f"{args.router_url.rstrip('/')}/recovery", 30),
        "prefill_capacity": _get(
            f"{args.prefill_url.rstrip('/')}/internal/pd/capacity", 30
        ),
        "decode_capacity": _get(
            f"{args.decode_url.rstrip('/')}/internal/pd/capacity", 30
        ),
    }
    if snapshots["router_recovery"].get("phase") != "RUNNING":
        errors.append("Router recovery gate is not RUNNING")
    for role in ("prefill_capacity", "decode_capacity"):
        capacity = snapshots[role]
        for field in (
            "active_handoffs",
            "queued_handoffs",
            "prepared_requests",
            "reservations",
            "quarantined_reservations",
            "inflight_transfer_bytes",
        ):
            if capacity.get(field, 0):
                errors.append(f"{role} retained {field}")

    request_id = next(iter(request_ids), "")
    terminal = next(
        (
            item
            for item in snapshots["router_metrics"].get("recent_terminal", ())
            if item.get("request_id") == request_id
        ),
        None,
    )
    if terminal is None:
        errors.append("Router has no terminal metrics for the chat request")
    elif not terminal.get("token_ids_sha256"):
        errors.append("Router terminal metrics have no token-ID digest")

    result = {
        "schema_version": 1,
        "started_ns": started_ns,
        "elapsed_ns": elapsed_ns,
        "request": payload,
        "request_id": request_id,
        "content_type": content_type,
        "assistant_text": text,
        "finish_reasons": finish_reasons,
        "frames": frames,
        "done": done,
        "snapshots": snapshots,
        "errors": errors,
        "result": "PASS" if not errors else "FAIL",
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"result": result["result"], "errors": errors}))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
