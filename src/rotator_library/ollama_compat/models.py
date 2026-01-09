"""
Pydantic models for the Ollama API.

These models define the request and response formats for Ollama's native API,
enabling compatibility with Raycast AI and other Ollama clients.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
import hashlib
import json


# --- Tool Call Format ---
class OllamaToolCallFunction(BaseModel):
    """Ollama tool call function with arguments as object (not JSON string)."""

    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class OllamaToolCall(BaseModel):
    """Ollama tool call format."""

    function: OllamaToolCallFunction


# --- Chat Message ---
class OllamaChatMessage(BaseModel):
    """Ollama chat message format."""

    role: str  # "user", "assistant", "system", "tool"
    content: str = ""
    images: Optional[List[str]] = None  # Base64 encoded images
    tool_calls: Optional[List[OllamaToolCall]] = None


# --- Tool Definition ---
class OllamaToolFunction(BaseModel):
    """Ollama tool function definition."""

    name: str
    description: str = ""
    parameters: Dict[str, Any] = Field(default_factory=dict)


class OllamaTool(BaseModel):
    """Ollama tool definition (supports local_tool type for Raycast)."""

    type: str = "function"  # "function" or "local_tool"
    function: Optional[OllamaToolFunction] = None


# --- Chat Request ---
class OllamaChatRequest(BaseModel):
    """Ollama /api/chat request format."""

    model: str
    messages: List[OllamaChatMessage]
    tools: Optional[List[OllamaTool]] = None
    stream: bool = True  # Ollama defaults to streaming
    options: Optional[Dict[str, Any]] = None  # Temperature, top_p, etc.


# --- Chat Response (Streaming Chunk) ---
class OllamaMessageContent(BaseModel):
    """Ollama message content in response."""

    role: str = "assistant"
    content: str = ""
    thinking: Optional[str] = None  # For reasoning/thinking models
    tool_calls: Optional[List[OllamaToolCall]] = None


class OllamaChunkResponse(BaseModel):
    """Ollama streaming chunk response format (NDJSON)."""

    model: str
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    message: OllamaMessageContent
    done: bool = False
    done_reason: Optional[str] = None  # "stop", "tool_calls", "length"

    # Optional performance stats (only on final chunk)
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


# --- Model List (/api/tags) ---
class OllamaModelDetails(BaseModel):
    """Ollama model details."""

    parent_model: str = ""
    format: str = "gguf"
    family: str = "llama"
    families: List[str] = Field(default_factory=lambda: ["llama"])
    parameter_size: str = "7B"
    quantization_level: str = "Q4_K_M"


class OllamaModelInfo(BaseModel):
    """Ollama model info for /api/tags response."""

    name: str  # Display name (what user sees)
    model: str  # Internal model ID (provider/model format)
    modified_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    size: int = 500000000  # Fake size for compatibility
    digest: Optional[str] = None
    details: OllamaModelDetails = Field(default_factory=OllamaModelDetails)

    def __init__(self, **data):
        super().__init__(**data)
        if self.digest is None:
            # Generate consistent digest from name and model
            hash_input = json.dumps({"name": self.name, "model": self.model})
            self.digest = hashlib.sha256(hash_input.encode()).hexdigest()


class OllamaTagsResponse(BaseModel):
    """Ollama /api/tags response format."""

    models: List[OllamaModelInfo]


# --- Model Info (/api/show) ---
class OllamaShowRequest(BaseModel):
    """Ollama /api/show request format."""

    model: str  # Can be display name or internal ID


class OllamaModelInfoDetails(BaseModel):
    """Extended model info for /api/show response."""

    general_architecture: str = Field(default="llama", alias="general.architecture")
    general_file_type: int = Field(default=2, alias="general.file_type")
    general_parameter_count: int = Field(
        default=7000000000, alias="general.parameter_count"
    )
    llama_context_length: int = Field(default=128000, alias="llama.context_length")
    llama_embedding_length: int = Field(default=4096, alias="llama.embedding_length")
    tokenizer_ggml_model: str = Field(default="gpt2", alias="tokenizer.ggml.model")

    class Config:
        populate_by_name = True


class OllamaShowResponse(BaseModel):
    """Ollama /api/show response format."""

    modelfile: str = ""
    parameters: str = ""
    template: str = "{{ .Prompt }}"
    details: OllamaModelDetails = Field(default_factory=OllamaModelDetails)
    model_info: Dict[str, Any] = Field(default_factory=dict)
    capabilities: List[str] = Field(default_factory=lambda: ["completion"])
