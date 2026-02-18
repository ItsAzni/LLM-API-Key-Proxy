# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Per-group concurrent request limit checker.

Caps how many concurrent requests a specific quota group can use
on a credential, scoped by tier priority. This allows reserving
concurrency slots on high-priority credentials for specific workloads.
"""

from typing import Dict, Optional

from ..types import CredentialState, LimitCheckResult, LimitResult
from .base import LimitChecker


class GroupConcurrentChecker(LimitChecker):
    """
    Checks per-group concurrent request limits.

    Blocks a credential when a specific quota group has reached its
    per-group concurrency cap for that credential's priority tier.

    Example config: {1: {"g3-flash": 1}}
    Means: on priority-1 (ultra) credentials, g3-flash can only have
    1 concurrent request, leaving remaining slots for other groups.
    """

    def __init__(self, group_concurrent_caps: Dict[int, Dict[str, int]]):
        self._caps = group_concurrent_caps

    @property
    def name(self) -> str:
        return "group_concurrent"

    def check(
        self,
        state: CredentialState,
        model: str,
        quota_group: Optional[str] = None,
    ) -> LimitCheckResult:
        """
        Check if a quota group has reached its per-group concurrent cap.

        Args:
            state: Credential state to check
            model: Model being requested
            quota_group: Quota group for this model

        Returns:
            LimitCheckResult indicating pass/fail
        """
        if not self._caps:
            return LimitCheckResult.ok()

        # Look up caps for this credential's priority
        priority_caps = self._caps.get(state.priority)
        if not priority_caps:
            return LimitCheckResult.ok()

        # Check if there's a cap for the relevant group
        group_key = quota_group or model
        cap = priority_caps.get(group_key)
        if cap is None:
            return LimitCheckResult.ok()

        # Check current group usage on this credential
        current = state.active_requests_by_group.get(group_key, 0)
        if current >= cap:
            return LimitCheckResult.blocked(
                result=LimitResult.BLOCKED_GROUP_CONCURRENT,
                reason=(
                    f"Group '{group_key}' at per-group concurrent cap: "
                    f"{current}/{cap} (priority {state.priority})"
                ),
                blocked_until=None,
            )

        return LimitCheckResult.ok()
