# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

import httpx
import logging
from typing import List
from .provider_interface import ProviderInterface

lib_logger = logging.getLogger("rotator_library")


class BaseHttpProvider(ProviderInterface):
    """
    Shared base class for simple HTTP-based providers that fetch models
    from a single endpoint using Bearer token authentication.

    Subclasses only need to set:
      - provider_prefix: used as model name prefix (e.g., "openai")
      - api_base_url: the models endpoint URL
      - response_key: JSON key containing the model list ("data" or "models")
      - model_id_key: JSON key for the model identifier within each item
    """

    provider_prefix: str = ""
    api_base_url: str = ""
    response_key: str = "data"
    model_id_key: str = "id"

    async def get_models(
        self, api_key: str, client: httpx.AsyncClient
    ) -> List[str]:
        try:
            response = await client.get(
                self.api_base_url,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            items = response.json().get(self.response_key, [])
            return [
                f"{self.provider_prefix}/{item[self.model_id_key]}" for item in items
            ]
        except httpx.RequestError as e:
            lib_logger.error(f"Failed to fetch {self.provider_prefix} models: {e}")
            return []
