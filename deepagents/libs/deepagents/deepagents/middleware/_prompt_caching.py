"""Provider-specific prompt-caching middleware helpers."""

import logging
from importlib import import_module
from typing import Any, cast

from langchain.agents.middleware.types import AgentMiddleware
logger = logging.getLogger(__name__)


def _create_anthropic_prompt_caching_middleware() -> AgentMiddleware[Any, Any] | None:
    """Create Anthropic prompt caching middleware when `langchain-anthropic` is installed."""
    try:
        from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware

        return cast("AgentMiddleware[Any, Any]", AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"))
    except ImportError:
        return None


def _create_bedrock_prompt_caching_middleware() -> AgentMiddleware[Any, Any] | None:
    """Create Bedrock prompt caching middleware when `langchain-aws` is installed."""
    module_name = "langchain_aws.middleware.prompt_caching"
    try:
        module = import_module(module_name)
    except ImportError as exc:
        if exc.name not in {"langchain_aws", "langchain_aws.middleware", module_name}:
            raise
        logger.debug("Bedrock prompt caching middleware is unavailable.", exc_info=exc)
        return None
    middleware_cls = module.BedrockPromptCachingMiddleware
    return cast("AgentMiddleware[Any, Any]", middleware_cls(unsupported_model_behavior="ignore"))


def _create_fireworks_prompt_caching_middleware() -> AgentMiddleware[Any, Any] | None:
    """Create Fireworks prompt caching middleware when `langchain-fireworks` is installed."""
    module_name = "langchain_fireworks.middleware.prompt_caching"
    try:
        module = import_module(module_name)
    except ImportError as exc:
        if exc.name not in {"langchain_fireworks", "langchain_fireworks.middleware", module_name}:
            raise
        logger.debug("Fireworks prompt caching middleware is unavailable.", exc_info=exc)
        return None
    middleware_cls = module.FireworksPromptCachingMiddleware
    return cast("AgentMiddleware[Any, Any]", middleware_cls(unsupported_model_behavior="ignore"))


def append_prompt_caching_middleware(middleware: list[AgentMiddleware[Any, Any]]) -> None:
    """Append provider-specific prompt caching middleware."""
    anthropic_middleware = _create_anthropic_prompt_caching_middleware()
    if anthropic_middleware is not None:
        middleware.append(anthropic_middleware)
    bedrock_middleware = _create_bedrock_prompt_caching_middleware()
    if bedrock_middleware is not None:
        middleware.append(bedrock_middleware)
    fireworks_middleware = _create_fireworks_prompt_caching_middleware()
    if fireworks_middleware is not None:
        middleware.append(fireworks_middleware)
