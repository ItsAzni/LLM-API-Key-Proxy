# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

from .base_http_provider import BaseHttpProvider


class GroqProvider(BaseHttpProvider):
    provider_prefix = "groq"
    api_base_url = "https://api.groq.com/openai/v1/models"
