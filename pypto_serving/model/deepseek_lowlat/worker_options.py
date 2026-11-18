# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Forward public Worker depths with the pinned original factory preserved."""

from contextlib import contextmanager
import ast
import __future__
import functools
import hashlib
import inspect
import threading


_LOCK = threading.Lock()


@contextmanager
def native_worker_options():
    from pypto.runtime import distributed_runner

    with _LOCK:
        original = distributed_runner._construct_worker
        if tuple(inspect.signature(original).parameters) != (
                "dc", "platform", "runtime_name", "num_sub", "enable_sdma", "startup_timeout_s"):
            raise ValueError("pinned Worker factory signature changed")
        source = inspect.getsource(original)
        tree = ast.parse(source)
        calls = [n.value for n in ast.walk(tree) if isinstance(n, ast.Return)
                 and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                 and n.value.func.id == "Worker"]
        if (len(calls) != 1 or len(calls[0].keywords) != 1
                or calls[0].keywords[0].arg is not None
                or not isinstance(calls[0].keywords[0].value, ast.Name)
                or calls[0].keywords[0].value.id != "worker_config"):
            raise ValueError("pinned Worker option forwarding changed")
        options = {"pipeline_depth": 2, "launch_depth": 1}
        calls[0].keywords += [ast.keyword(arg=k, value=ast.Constant(value=v)) for k, v in options.items()]
        ast.fix_missing_locations(tree)
        namespace = dict(original.__globals__)
        exec(compile(tree, __file__+":public_options", "exec",
                     flags=__future__.annotations.compiler_flag, dont_inherit=True), namespace)
        adapted = functools.update_wrapper(namespace[original.__name__], original)
        distributed_runner._construct_worker = adapted
        try:
            yield dict(public_options=options, original_factory_sha256=hashlib.sha256(source.encode()).hexdigest(),
                       sdk_files_modified=False)
        finally:
            if distributed_runner._construct_worker is not adapted:
                raise RuntimeError("Worker factory context lost ownership")
            distributed_runner._construct_worker = original
