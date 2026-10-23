# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
# Modified by PyPTO Serving: internal imports use the bundled scheduling package.

"""Host-neutral scheduling contracts shared by AgentCache and AgentRouter."""

from pypto_serving.serving.agentinfer_core.scheduling.admission_outcome import AdmissionDisposition, AdmissionOutcome
from pypto_serving.serving.agentinfer_core.scheduling.backend import BackendInfo, BackendPoolInfo, DispatchTarget, DpRankInfo
from pypto_serving.serving.agentinfer_core.scheduling.domain import (
    ProgramRef,
    ProgramState,
    ProgramStatus,
    ProgramTokenObservation,
    ProgramView,
    SharedPrefixAttribution,
    TokenObservationSource,
)
from pypto_serving.serving.agentinfer_core.scheduling.events import SchedulingEvent, SchedulingEventKind, StrategyDiagnostic
from pypto_serving.serving.agentinfer_core.scheduling.factors import StrategyFactors
from pypto_serving.serving.agentinfer_core.scheduling.headers import AgentRequestIdentity, select_router_headers
from pypto_serving.serving.agentinfer_core.scheduling.identity import AgentIdentity, MetadataError, encode_agent_identity, parse_agent_identity
from pypto_serving.serving.agentinfer_core.scheduling.lifecycle import ProgramLifecycle
from pypto_serving.serving.agentinfer_core.scheduling.observability import SchedulerObservabilityConfig
from pypto_serving.serving.agentinfer_core.scheduling.program_registry import ProgramRegistry, StaleProgramReferenceError
from pypto_serving.serving.agentinfer_core.scheduling.program_runtime import RuntimeProgram
from pypto_serving.serving.agentinfer_core.scheduling.request_pool import RequestPool, RequestPoolEntry, RequestPoolStatus
from pypto_serving.serving.agentinfer_core.scheduling.runtime import ProgramScheduler
from pypto_serving.serving.agentinfer_core.scheduling.snapshot import SchedulingSnapshot
from pypto_serving.serving.agentinfer_core.scheduling.strategy import SchedulingStrategy
from pypto_serving.serving.agentinfer_core.scheduling.transitions import (
    TransitionController,
    TransitionKind,
    TransitionRequest,
    TransitionResult,
)

__all__ = [
    "AdmissionDisposition",
    "AdmissionOutcome",
    "AgentIdentity",
    "AgentRequestIdentity",
    "BackendInfo",
    "BackendPoolInfo",
    "DispatchTarget",
    "DpRankInfo",
    "MetadataError",
    "ProgramLifecycle",
    "ProgramRegistry",
    "ProgramRef",
    "ProgramState",
    "ProgramStatus",
    "ProgramTokenObservation",
    "ProgramView",
    "ProgramScheduler",
    "RequestPool",
    "RequestPoolEntry",
    "RequestPoolStatus",
    "RuntimeProgram",
    "SchedulingEvent",
    "SchedulingEventKind",
    "SchedulingSnapshot",
    "SchedulingStrategy",
    "SchedulerObservabilityConfig",
    "SharedPrefixAttribution",
    "StrategyDiagnostic",
    "StrategyFactors",
    "StaleProgramReferenceError",
    "TokenObservationSource",
    "TransitionController",
    "TransitionKind",
    "TransitionRequest",
    "TransitionResult",
    "encode_agent_identity",
    "parse_agent_identity",
    "select_router_headers",
]
