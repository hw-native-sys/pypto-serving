# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""External control-plane Router for PyPTO PD serving."""

from .app import create_router_app
from .config import RouterConfig

__all__ = ["RouterConfig", "create_router_app"]
