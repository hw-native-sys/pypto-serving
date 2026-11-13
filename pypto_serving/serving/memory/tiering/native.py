# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Build the small owner copy loop before creating any runtime owners."""
import os
from pathlib import Path
import shlex
import subprocess
from tempfile import TemporaryDirectory


def build_copy_library():
    toolkit = Path(os.environ["ASCEND_HOME_PATH"])
    directory = TemporaryDirectory(prefix="pypto-kv-copy-")
    try:
        command = shlex.split(os.environ.get("CXX", "g++")) + [
            "-std=c++17", "-O2", "-shared", "-fPIC", str(Path(__file__).with_name("copy.cpp")),
            "-I" + str(toolkit / "include"), "-L" + str(toolkit / "lib64"), "-lascendcl",
            "-o", str(Path(directory.name) / "libcache_copy.so"),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        directory.cleanup()
        raise RuntimeError(f"cannot build DDR copy helper: {error.stderr.strip()}") from error
    except BaseException:
        directory.cleanup()
        raise
    return directory
