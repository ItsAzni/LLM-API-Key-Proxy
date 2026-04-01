# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

from .base_http_provider import BaseHttpProvider


class OpenRouterProvider(BaseHttpProvider):
    provider_prefix = "openrouter"
    api_base_url = "https://openrouter.ai/api/v1/models"
