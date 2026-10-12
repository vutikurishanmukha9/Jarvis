"""Compatibility layer between Deep Agents and installed LangChain / LangGraph versions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ExtendedModelResponse shim
try:
    from langchain.agents.middleware.types import ExtendedModelResponse
except ImportError:
    @dataclass
    class ExtendedModelResponse:
        model_response: Any = None
        command: Any = None

# InputAgentState / OutputAgentState shims
try:
    from langchain.agents.middleware.types import InputAgentState, OutputAgentState
except ImportError:
    try:
        from langchain.agents.middleware.types import _InputAgentState as InputAgentState, _OutputAgentState as OutputAgentState
    except ImportError:
        InputAgentState = Any  # type: ignore[assignment, misc]
        OutputAgentState = Any  # type: ignore[assignment, misc]

# ContextOverflowError shim
try:
    from langchain_core.exceptions import ContextOverflowError
except ImportError:
    class ContextOverflowError(Exception):  # type: ignore[no-redef]
        pass
