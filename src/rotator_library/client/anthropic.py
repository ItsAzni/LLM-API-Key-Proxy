# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Anthropic API compatibility handler for RotatingClient.

This module provides Anthropic SDK compatibility methods that allow using
Anthropic's Messages API format with the credential rotation system.
"""

import json
import logging
import uuid
from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional, Callable, cast

from ..anthropic_compat import (
    AnthropicMessagesRequest,
    AnthropicCountTokensRequest,
    translate_anthropic_request,
    openai_to_anthropic_response,
    anthropic_streaming_wrapper,
    anthropic_to_openai_messages,
    anthropic_to_openai_tools,
)
from ..transaction_logger import TransactionLogger

if TYPE_CHECKING:
    from .rotating_client import RotatingClient

lib_logger = logging.getLogger("rotator_library")


class AnthropicHandler:
    """
    Handler for Anthropic API compatibility methods.

    This class provides methods to handle Anthropic Messages API requests
    by translating them to OpenAI format, processing through the client's
    acompletion method, and converting responses back to Anthropic format.

    Example:
        handler = AnthropicHandler(client)
        response = await handler.messages(request, raw_request)
    """

    def __init__(self, client: "RotatingClient"):
        """
        Initialize the Anthropic handler.

        Args:
            client: The RotatingClient instance to use for completions
        """
        self._client = client

    async def messages(
        self,
        request: AnthropicMessagesRequest,
        raw_request: Optional[Any] = None,
        pre_request_callback: Optional[Callable[..., Any]] = None,
    ) -> Any:
        """
        Handle Anthropic Messages API requests.

        This method accepts requests in Anthropic's format, translates them to
        OpenAI format internally, processes them through the existing acompletion
        method, and returns responses in Anthropic's format.

        Args:
            request: An AnthropicMessagesRequest object
            raw_request: Optional raw request object for disconnect checks
            pre_request_callback: Optional async callback before each API request

        Returns:
            For non-streaming: dict in Anthropic Messages format
            For streaming: AsyncGenerator yielding Anthropic SSE format strings
        """
        request_id = f"msg_{uuid.uuid4().hex[:24]}"
        original_model = request.model

        # Extract provider from model for logging
        provider = original_model.split("/")[0] if "/" in original_model else "unknown"

        # Create Anthropic transaction logger if request logging is enabled
        anthropic_logger = None
        if self._client.enable_request_logging:
            anthropic_logger = TransactionLogger(
                provider,
                original_model,
                enabled=True,
                api_format="ant",
            )
            # Log original Anthropic request
            anthropic_logger.log_request(
                request.model_dump(exclude_none=True),
                filename="anthropic_request.json",
            )

        # Translate Anthropic request to OpenAI format
        openai_request = translate_anthropic_request(request)

        # Forward selected Anthropic headers from the incoming request so provider
        # plugins can preserve behavior like prompt-caching scope betas.
        if raw_request is not None and hasattr(raw_request, "headers"):
            forwarded_headers = {}
            anthropic_beta = raw_request.headers.get("anthropic-beta")
            anthropic_version = raw_request.headers.get("anthropic-version")
            if anthropic_beta:
                forwarded_headers["anthropic-beta"] = anthropic_beta
            if anthropic_version:
                forwarded_headers["anthropic-version"] = anthropic_version

            if forwarded_headers:
                existing = openai_request.get("extra_headers")
                if isinstance(existing, dict):
                    forwarded_headers = {**existing, **forwarded_headers}
                openai_request["extra_headers"] = forwarded_headers

        # DEBUG: trace thinking config from Anthropic request
        import logging as _logging

        _logging.getLogger("rotator_library").info(
            "[AnthropicHandler] request.thinking=%s, max_tokens=%s, model=%s",
            request.thinking,
            request.max_tokens,
            request.model,
        )

        # Pass thinking config for providers that handle it natively (e.g. GitLab Duo)
        # This preserves the original budget_tokens without lossy reasoning_effort roundtrip
        # Note: type can be "enabled", "adaptive", etc. — anything not "disabled" means thinking is on
        if request.thinking and request.thinking.type != "disabled":
            openai_request["thinking_type"] = request.thinking.type
            if request.thinking.budget_tokens is not None:
                openai_request["thinking_budget"] = request.thinking.budget_tokens

        # Pass output_config.effort for adaptive thinking (4.6 models)
        # Determine effort: from output_config if present, otherwise default for 4.6 models
        effort = None
        if request.output_config and request.output_config.effort:
            effort = request.output_config.effort

        model_lower = request.model.lower()
        is_46_model = "4.6" in model_lower or "4-6" in model_lower
        is_opus_model = "opus" in model_lower

        # For Claude 4.6 models: default to "high" if no effort specified
        # (matches Anthropic's default for adaptive thinking)
        if (
            is_46_model
            and effort is None
            and request.thinking
            and request.thinking.type != "disabled"
        ):
            effort = "high"

        if effort:
            # Keep the 4.6 high→max upgrade, but preserve medium as medium.
            if is_46_model:
                effort_upgrade = {"high": "max"}
                original = effort
                effort = effort_upgrade.get(effort, effort)
                if effort != original:
                    lib_logger.info(
                        f"[AnthropicHandler] Upgraded effort {original}→{effort} for 4.6 model {request.model}"
                    )

            openai_request["effort"] = effort

        # Pass parent log directory to acompletion for nested logging
        if anthropic_logger and anthropic_logger.log_dir:
            openai_request["_parent_log_dir"] = anthropic_logger.log_dir

        if request.stream:
            # Streaming response
            response_generator = await self._client.acompletion(
                request=raw_request,
                pre_request_callback=pre_request_callback,
                **openai_request,
            )

            # Create disconnect checker if raw_request provided
            is_disconnected = None
            if raw_request is not None and hasattr(raw_request, "is_disconnected"):
                is_disconnected = raw_request.is_disconnected

            # Return the streaming wrapper
            # Note: For streaming, the anthropic response logging happens in the wrapper
            return anthropic_streaming_wrapper(
                openai_stream=response_generator,
                original_model=original_model,
                request_id=request_id,
                is_disconnected=is_disconnected,
                transaction_logger=anthropic_logger,
            )
        else:
            # Non-streaming response
            response = await self._client.acompletion(
                request=raw_request,
                pre_request_callback=pre_request_callback,
                **openai_request,
            )
            response_obj = cast(Any, response)

            # Convert OpenAI response to Anthropic format
            openai_response = (
                response_obj.model_dump()
                if hasattr(response_obj, "model_dump")
                else dict(response_obj)
            )
            anthropic_response = openai_to_anthropic_response(
                openai_response, original_model
            )

            # Override the ID with our request ID
            anthropic_response["id"] = request_id

            # Log Anthropic response
            if anthropic_logger:
                anthropic_logger.log_response(
                    anthropic_response,
                    filename="anthropic_response.json",
                )

            return anthropic_response

    async def count_tokens(
        self,
        request: AnthropicCountTokensRequest,
    ) -> dict:
        """
        Handle Anthropic count_tokens API requests.

        Counts the number of tokens that would be used by a Messages API request.
        This is useful for estimating costs and managing context windows.

        Args:
            request: An AnthropicCountTokensRequest object

        Returns:
            Dict with input_tokens count in Anthropic format
        """
        anthropic_request = request.model_dump(exclude_none=True)

        openai_messages = anthropic_to_openai_messages(
            anthropic_request.get("messages", []), anthropic_request.get("system")
        )

        # Count tokens for messages
        message_tokens = self._client.token_count(
            model=request.model,
            messages=openai_messages,
        )

        # Count tokens for tools if present
        tool_tokens = 0
        if request.tools:
            # Tools add tokens based on their definitions
            # Convert to JSON string and count tokens for tool definitions
            openai_tools = anthropic_to_openai_tools(
                [tool.model_dump() for tool in request.tools]
            )
            if openai_tools:
                # Serialize tools to count their token contribution
                tools_text = json.dumps(openai_tools)
                tool_tokens = self._client.token_count(
                    model=request.model,
                    text=tools_text,
                )

        total_tokens = message_tokens + tool_tokens

        return {"input_tokens": total_tokens}
