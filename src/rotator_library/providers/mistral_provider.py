# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

from .base_http_provider import BaseHttpProvider


class MistralProvider(BaseHttpProvider):
    provider_prefix = "mistral"
    api_base_url = "https://api.mistral.ai/v1/models"
