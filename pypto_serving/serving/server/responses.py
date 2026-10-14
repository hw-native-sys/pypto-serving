# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Public response formatting for any normal token-output iterator."""

from __future__ import annotations

from contextlib import aclosing
import json
import time

from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from pypto_serving.observability.tokens import token_ids_sha256
from pypto_serving.tools.profile import profile_span, profile_instant
from .api_types import (
    ChatCompletionChoice, ChatCompletionResponse, CompletionChoice,
    CompletionResponse, ResponseUsage,
)
from .chat_format import chat_delta, chat_finish_reason, chat_message, map_finish_reason


def _sse(value):
    return "data: " + json.dumps(value.model_dump()) + "\n\n"


async def stream_outputs(outputs, request_id, model, *, kind, parsed=False, parallel_tool_calls=True):
    previous_text = previous_reasoning = ""
    try:
        with profile_span("http.stream_chat_completion" if kind == "chat" else "http.stream_completion",
                          cat="request", args={"request_id": request_id}):
            async with aclosing(outputs):
                async for output in outputs:
                    if parsed:
                        content, reasoning = output.text_delta, output.reasoning_delta
                    else:
                        if not output.text.startswith(previous_text) or not output.reasoning.startswith(previous_reasoning):
                            raise RuntimeError("generation changed already published output")
                        content = output.text[len(previous_text):]
                        reasoning = output.reasoning[len(previous_reasoning):]
                        previous_text, previous_reasoning = output.text, output.reasoning
                    if kind == "chat":
                        chunk = ChatCompletionResponse(
                            id=request_id, object="chat.completion.chunk", created=int(time.time()), model=model,
                            choices=[ChatCompletionChoice(
                                delta=chat_delta(content, reasoning, output.tool_call_deltas,
                                                 parallel_tool_calls=parallel_tool_calls),
                                finish_reason=chat_finish_reason(output) if output.finished else None,
                            )],
                        )
                    else:
                        chunk = CompletionResponse(
                            id=request_id, created=int(time.time()), model=model,
                            choices=[CompletionChoice(text=content,
                                finish_reason=map_finish_reason(output.finish_reason) if output.finished else None)],
                        )
                    yield _sse(chunk)
                    if output.finished:
                        response_type = ChatCompletionResponse if kind == "chat" else CompletionResponse
                        usage = response_type(
                            id=request_id, created=int(time.time()), model=model, choices=[],
                            usage=ResponseUsage(prompt_tokens=output.prompt_tokens,
                                completion_tokens=output.completion_tokens,
                                total_tokens=output.prompt_tokens + output.completion_tokens),
                            **({"object": "chat.completion.chunk"} if kind == "chat" else {}),
                        )
                        yield _sse(usage)
                        profile_instant(
                            "http.stream_chat.finished" if kind == "chat" else "http.stream_completion.finished",
                            cat="request", args={"request_id": request_id, "finish_reason": output.finish_reason},
                        )
                        yield "data: [DONE]\n\n"
                        return
                raise EOFError("generation ended without a terminal output")
    except Exception as exc:
        error = {"error": {"message": str(exc),
            "type": "invalid_model_output" if isinstance(exc, ValueError) else "generation_error",
            "code": 400 if isinstance(exc, ValueError) else 503}}
        yield "data: " + json.dumps(error) + "\n\n"
        yield "data: [DONE]\n\n"


async def output_response(outputs, request_id, model, *, kind, stream, parsed=False, parallel_tool_calls=True):
    with profile_span("http.chat_completions" if kind == "chat" else "http.completions",
                      cat="request", args={"request_id": request_id, "stream": stream}):
        return await _output_response(outputs, request_id, model, kind=kind, stream=stream,
                                      parsed=parsed, parallel_tool_calls=parallel_tool_calls)


async def _output_response(outputs, request_id, model, *, kind, stream, parsed, parallel_tool_calls):
    if stream:
        return StreamingResponse(
            stream_outputs(outputs, request_id, model, kind=kind, parsed=parsed,
                           parallel_tool_calls=parallel_tool_calls),
            media_type="text/event-stream", background=BackgroundTask(outputs.aclose),
        )
    final = None
    text = reasoning = ""
    async with aclosing(outputs):
        async for output in outputs:
            text = output.text or text
            reasoning = output.reasoning or reasoning
            if output.finished:
                final = output
                break
    if final is None:
        raise EOFError("generation ended without a terminal output")
    usage = ResponseUsage(prompt_tokens=final.prompt_tokens, completion_tokens=final.completion_tokens,
                          total_tokens=final.prompt_tokens + final.completion_tokens)
    if kind == "chat":
        response = ChatCompletionResponse(
            id=request_id, created=int(time.time()), model=model, usage=usage,
            choices=[ChatCompletionChoice(
                message=chat_message(text, reasoning, final.tool_calls, parallel_tool_calls=parallel_tool_calls),
                finish_reason=chat_finish_reason(final),
            )],
        )
    else:
        response = CompletionResponse(id=request_id, created=int(time.time()), model=model, usage=usage,
            choices=[CompletionChoice(text=text, finish_reason=map_finish_reason(final.finish_reason))])
    return JSONResponse(response.model_dump(),
                        headers={"x-pypto-token-ids-sha256": token_ids_sha256(final.token_ids)})
