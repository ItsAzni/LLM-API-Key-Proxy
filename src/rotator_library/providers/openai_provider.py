# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

from .base_http_provider import BaseHttpProvider


class OpenAIProvider(BaseHttpProvider):
    provider_prefix = "openai"
    api_base_url = "https://api.openai.com/v1/models"
