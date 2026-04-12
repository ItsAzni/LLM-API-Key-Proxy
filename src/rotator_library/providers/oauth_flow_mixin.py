# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

# src/rotator_library/providers/oauth_flow_mixin.py

import logging
from typing import Dict, Any, Union

lib_logger = logging.getLogger("rotator_library")


class OAuthFlowMixin:
    """Mixin providing shared OAuth token initialization logic.

    Used by GoogleOAuthBase, IFlowAuthBase, and QwenAuthBase.
    Requires the consuming class to provide:
    - ENV_PREFIX: str class attribute
    - _load_credentials(path): async method
    - _is_token_expired(creds): method
    - _should_force_interactive(creds, force_interactive): method (from OAuthMixin)
    - _get_display_name(creds_or_path): method (from OAuthMixin)
    - _refresh_token(path, creds, force=False): async method
    - _execute_interactive_oauth(path, creds, display_name, provider_name, timeout): async method (from OAuthMixin)
    """

    async def initialize_token(
        self,
        creds_or_path: Union[Dict[str, Any], str],
        force_interactive: bool = False,
    ) -> Dict[str, Any]:
        path = creds_or_path if isinstance(creds_or_path, str) else None

        display_name = self._get_display_name(creds_or_path)

        lib_logger.debug(f"Initializing {self.ENV_PREFIX} token for '{display_name}'...")
        try:
            creds = (
                await self._load_credentials(creds_or_path) if path else creds_or_path
            )
            token_expired = self._is_token_expired(creds)
            needs_interactive, reason = self._should_force_interactive(
                creds, force_interactive=force_interactive
            )

            if token_expired and not needs_interactive:
                needs_interactive = True
                reason = "token is expired"

            if needs_interactive:
                if reason == "token is expired" and creds.get("refresh_token"):
                    try:
                        return await self._refresh_token(path, creds)
                    except Exception as e:
                        lib_logger.warning(
                            f"Automatic token refresh for '{display_name}' failed: {e}. Proceeding to interactive login."
                        )

                lib_logger.warning(
                    f"{self.ENV_PREFIX} OAuth token for '{display_name}' needs setup: {reason}."
                )

                return await self._execute_interactive_oauth(
                    path=path,
                    creds=creds,
                    display_name=display_name,
                    provider_name=self.ENV_PREFIX,
                    timeout=300.0,
                )

            lib_logger.info(
                f"{self.ENV_PREFIX} OAuth token at '{display_name}' is valid."
            )
            return creds
        except Exception as e:
            raise ValueError(
                f"Failed to initialize {self.ENV_PREFIX} OAuth for '{path}': {e}"
            )
