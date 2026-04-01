# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

from .base_http_provider import BaseHttpProvider


class CohereProvider(BaseHttpProvider):
    provider_prefix = "cohere"
    api_base_url = "https://api.cohere.ai/v1/models"
    response_key = "models"
    model_id_key = "name"
