# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

import httpx
import logging
from typing import List
from .provider_interface import ProviderInterface

lib_logger = logging.getLogger("rotator_library")


class TrybonsProvider(ProviderInterface):
    """
    Provider implementation for the TryBons API.

    TryBons is an Anthropic-compatible API at go.trybons.ai.
    Routes through LiteLLM's native anthropic support with custom api_base.
    No message conversion needed - endpoint speaks Anthropic Messages API natively.
    """

    skip_cost_calculation = True  # LiteLLM doesn't have cost data for TryBons

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetches the list of available models from the TryBons API.
        """
        try:
            response = await client.get(
                "https://go.trybons.ai/v1/models",
                headers={"x-api-key": api_key},
            )
            response.raise_for_status()
            data = response.json()
            return [f"trybons/{model['id']}" for model in data.get("data", [])]
        except Exception as e:
            lib_logger.error(f"Failed to fetch TryBons models: {e}")
            return []
