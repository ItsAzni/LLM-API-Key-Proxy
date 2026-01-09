"""
Ollama API compatibility layer.

This module provides translation between Ollama's native API format and OpenAI's format,
enabling Raycast AI and other Ollama clients to use the proxy.
"""

from .models import (
    OllamaChatMessage,
    OllamaChatRequest,
    OllamaChunkResponse,
    OllamaTagsResponse,
    OllamaShowRequest,
    OllamaShowResponse,
    OllamaModelInfo,
    OllamaModelDetails,
)
from .translator import (
    ollama_to_openai_request,
    openai_to_ollama_chunk,
    generate_model_display_name,
)
from .streaming import ollama_streaming_wrapper

__all__ = [
    # Models
    "OllamaChatMessage",
    "OllamaChatRequest",
    "OllamaChunkResponse",
    "OllamaTagsResponse",
    "OllamaShowRequest",
    "OllamaShowResponse",
    "OllamaModelInfo",
    "OllamaModelDetails",
    # Translator
    "ollama_to_openai_request",
    "openai_to_ollama_chunk",
    "generate_model_display_name",
    # Streaming
    "ollama_streaming_wrapper",
]
