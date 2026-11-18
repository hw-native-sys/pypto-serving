# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check module identities across errors and late relative helper imports."""

from pathlib import Path
import sys
from types import ModuleType

import pytest

from pypto_serving.model.deepseek_lowlat.kernel_loader import LowlatPrograms, isolated_profile


PREFIX = "models.deepseek_v4_flash_lowlat"


def fake_parent(monkeypatch):
    parent = ModuleType("models")
    monkeypatch.setitem(sys.modules, "models", parent)
    for name in list(sys.modules):
        if name == PREFIX or name.startswith(PREFIX + "."):
            monkeypatch.delitem(sys.modules, name)
    return parent


def test_failed_import_restores_existing_profile_and_process_globals(monkeypatch):
    parent = fake_parent(monkeypatch)
    old = ModuleType(PREFIX)
    old_config = ModuleType(PREFIX + ".config")
    parent.deepseek_v4_flash_lowlat = old
    monkeypatch.setitem(sys.modules, PREFIX, old)
    monkeypatch.setitem(sys.modules, PREFIX + ".config", old_config)
    before_argv = sys.argv
    before_path = list(sys.path)
    transient = ModuleType(PREFIX + ".new_helper")
    with pytest.raises(RuntimeError, match="import failed"):
        with isolated_profile() as captured:
            assert PREFIX not in sys.modules
            assert sys.argv[sys.argv.index("--tp") + 1] == "8"
            sys.modules[transient.__name__] = transient
            raise RuntimeError("import failed")
    assert captured[transient.__name__] is transient
    assert transient.__name__ not in sys.modules
    assert sys.modules[PREFIX] is parent.deepseek_v4_flash_lowlat is old
    assert sys.modules[PREFIX + ".config"] is old_config
    assert sys.argv is before_argv
    assert sys.path == before_path


def test_program_context_retains_lazy_helpers_without_publishing_them(monkeypatch):
    parent = fake_parent(monkeypatch)
    private = ModuleType(PREFIX)
    helper = ModuleType(PREFIX + ".lazy_helper")
    programs = LowlatPrograms(None, None, {}, {}, {PREFIX: private}, Path("."), None, None)
    with programs.context():
        assert parent.deepseek_v4_flash_lowlat is private
        sys.modules[helper.__name__] = helper
    assert not hasattr(parent, "deepseek_v4_flash_lowlat")
    assert PREFIX not in sys.modules and helper.__name__ not in sys.modules
    with programs.context():
        assert sys.modules[helper.__name__] is helper
        assert sys.modules[PREFIX] is private
    assert programs.modules[helper.__name__] is helper
