/**
 * Utility functions for formatting quota data.
 * Updated for modular usage manager format.
 */

import { Color, Icon, Image } from "@raycast/api";
import { ProviderStats, TokenStats, CredentialStats, WindowStats, GroupUsage } from "./types";

/**
 * Format token count for display (e.g., 125000 -> "125k").
 */
export function formatTokens(count: number): string {
  if (count >= 1_000_000) {
    return `${(count / 1_000_000).toFixed(1)}M`;
  } else if (count >= 1_000) {
    return `${Math.round(count / 1_000)}k`;
  }
  return count.toString();
}

/**
 * Format token stats for display.
 */
export function formatTokenStats(tokens: TokenStats): string {
  const input = tokens.input_cached + tokens.input_uncached;
  const output = tokens.output;
  return `${formatTokens(input)} in / ${formatTokens(output)} out`;
}

/**
 * Format cost for display.
 */
export function formatCost(cost: number | null): string {
  if (cost === null || cost === 0) {
    return "-";
  }
  if (cost < 0.01) {
    return `$${cost.toFixed(4)}`;
  }
  return `$${cost.toFixed(2)}`;
}

/**
 * Format cooldown seconds as human-readable string.
 */
export function formatCooldown(seconds: number): string {
  if (seconds < 60) {
    return `${seconds}s`;
  } else if (seconds < 3600) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return secs > 0 ? `${mins}m ${secs}s` : `${mins}m`;
  } else {
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    return mins > 0 ? `${hours}h ${mins}m` : `${hours}h`;
  }
}

/**
 * Format Unix timestamp for display.
 * If the timestamp is in the past, returns "Available" since the window has reset.
 * Shows relative time for near-future resets, absolute time for far-future.
 */
export function formatResetTime(timestamp: number | null | undefined): string {
  if (!timestamp) {
    return "";
  }
  try {
    const now = Date.now();
    const resetMs = timestamp * 1000;

    // If reset time is in the past, the window has reset - quota is available
    if (resetMs <= now) {
      return "Available";
    }

    const diffMs = resetMs - now;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);

    // Show relative time for resets within 24 hours
    if (diffMins < 60) {
      return `in ${diffMins}m`;
    } else if (diffHours < 24) {
      const remainingMins = Math.floor((diffMs % 3600000) / 60000);
      return remainingMins > 0 ? `in ${diffHours}h ${remainingMins}m` : `in ${diffHours}h`;
    }

    // Show absolute time for resets more than 24 hours away
    const date = new Date(resetMs);
    return date.toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return "";
  }
}

/**
 * Get status icon for provider based on credential states.
 */
export function getProviderStatusIcon(stats: ProviderStats): Image.ImageLike {
  if (stats.exhausted_count > 0) {
    return { source: Icon.XMarkCircle, tintColor: Color.Red };
  } else if ((stats.on_cooldown_count ?? 0) > 0 || stats.active_count < stats.credential_count) {
    return { source: Icon.ExclamationMark, tintColor: Color.Yellow };
  } else {
    return { source: Icon.CheckCircle, tintColor: Color.Green };
  }
}

/**
 * Get status text for provider.
 */
export function getProviderStatusText(stats: ProviderStats): string {
  if (stats.exhausted_count > 0) {
    return `${stats.exhausted_count} exhausted`;
  } else if ((stats.on_cooldown_count ?? 0) > 0) {
    return `${stats.on_cooldown_count} on cooldown`;
  } else {
    return "All active";
  }
}

/**
 * Get status icon for credential.
 */
export function getCredentialStatusIcon(cred: CredentialStats): Image.ImageLike {
  if (cred.status === "exhausted") {
    return { source: Icon.XMarkCircle, tintColor: Color.Red };
  } else if (cred.status === "cooldown" || cred.status === "mixed") {
    return { source: Icon.Clock, tintColor: Color.Yellow };
  } else {
    return { source: Icon.CheckCircle, tintColor: Color.Green };
  }
}

/**
 * Get status text for credential.
 */
export function getCredentialStatusText(cred: CredentialStats): string {
  if (cred.status === "exhausted") {
    return "Exhausted";
  } else if (cred.status === "cooldown") {
    return "Cooldown";
  } else if (cred.status === "mixed") {
    return "Partial";
  } else {
    return "Active";
  }
}

/**
 * Get color based on remaining percentage.
 */
export function getProgressColor(remaining: number | null): Color {
  if (remaining === null) {
    return Color.SecondaryText;
  }
  if (remaining <= 10) {
    return Color.Red;
  } else if (remaining <= 30) {
    return Color.Yellow;
  } else {
    return Color.Green;
  }
}

/**
 * Get the appropriate progress circle icon based on percentage.
 */
export function getProgressIcon(remaining: number | null): Image.ImageLike {
  const color = getProgressColor(remaining);

  if (remaining === null) {
    return { source: Icon.Circle, tintColor: color };
  }

  if (remaining >= 88) {
    return { source: Icon.CircleProgress100, tintColor: color };
  } else if (remaining >= 63) {
    return { source: Icon.CircleProgress75, tintColor: color };
  } else if (remaining >= 38) {
    return { source: Icon.CircleProgress50, tintColor: color };
  } else if (remaining >= 13) {
    return { source: Icon.CircleProgress25, tintColor: color };
  } else {
    return { source: Icon.Circle, tintColor: color };
  }
}

/**
 * Get the primary window from a windows record.
 *
 * Priority order:
 * 1. 'daily' - standard daily window
 * 2. '5h' - Antigravity 5-hour rolling window
 * 3. '168h' (7 days) - Antigravity free tier window
 * 4. 'api_authoritative' - API-provided window
 * 5. First available window
 */
export function getPrimaryWindow(windows: Record<string, WindowStats>): WindowStats | null {
  return (
    windows["daily"] ||
    windows["5h"] ||
    windows["168h"] ||
    windows["api_authoritative"] ||
    Object.values(windows)[0] ||
    null
  );
}

/**
 * Calculate remaining percentage from a window's limit and request count.
 */
export function getWindowRemainingPct(window: WindowStats | null): number | null {
  if (!window) return null;

  // If remaining is explicitly set, calculate from it
  if (window.remaining !== null && window.limit !== null && window.limit > 0) {
    return Math.round((window.remaining / window.limit) * 100);
  }

  // Otherwise calculate from request count vs limit
  if (window.limit !== null && window.limit > 0) {
    return Math.max(0, Math.round(((window.limit - window.request_count) / window.limit) * 100));
  }

  return null;
}

/**
 * Get the best quota info from a credential's group_usage.
 * Returns the lowest remaining percentage across all groups.
 */
export function getCredentialQuotaInfo(cred: CredentialStats): {
  lowestPct: number | null;
  earliestReset: number | null;
  primaryGroup: string | null;
  primaryWindow: WindowStats | null;
} {
  let lowestPct: number | null = null;
  let earliestReset: number | null = null;
  let primaryGroup: string | null = null;
  let primaryWindow: WindowStats | null = null;

  // Handle missing group_usage
  if (!cred.group_usage) {
    return { lowestPct, earliestReset, primaryGroup, primaryWindow };
  }

  for (const [groupName, groupUsage] of Object.entries(cred.group_usage)) {
    if (!groupUsage || !groupUsage.windows) continue;

    const window = getPrimaryWindow(groupUsage.windows);
    if (!window) continue;

    const pct = getWindowRemainingPct(window);

    if (pct !== null) {
      if (lowestPct === null || pct < lowestPct) {
        lowestPct = pct;
        primaryGroup = groupName;
        primaryWindow = window;
      }
    }

    if (window.reset_at) {
      if (!earliestReset || window.reset_at < earliestReset) {
        earliestReset = window.reset_at;
      }
    }
  }

  return { lowestPct, earliestReset, primaryGroup, primaryWindow };
}

/**
 * Convert credentials dict to array for rendering.
 */
export function credentialsToArray(credentials: Record<string, CredentialStats> | CredentialStats[] | undefined | null): CredentialStats[] {
  if (!credentials) return [];
  // Handle if it's already an array (legacy format)
  if (Array.isArray(credentials)) return credentials;
  // Handle dict format
  return Object.values(credentials);
}

/**
 * Aggregate quota groups across all credentials for provider-level summary.
 */
export function aggregateProviderQuotaGroups(credentials: Record<string, CredentialStats> | CredentialStats[] | undefined | null): Record<string, {
  totalUsed: number;
  totalLimit: number;
  remainingPct: number | null;
  earliestReset: number | null;
  credentialsTotal: number;
  credentialsExhausted: number;
}> {
  const groups: Record<string, {
    totalUsed: number;
    totalLimit: number;
    remainingPcts: number[];
    earliestReset: number | null;
    credentialsTotal: number;
    credentialsExhausted: number;
  }> = {};

  // Handle empty/undefined credentials
  if (!credentials) return {};

  // Convert to array if needed
  const credArray = Array.isArray(credentials) ? credentials : Object.values(credentials);

  for (const cred of credArray) {
    // Skip if cred doesn't have group_usage
    if (!cred.group_usage) continue;

    for (const [groupName, groupUsage] of Object.entries(cred.group_usage)) {
      if (!groups[groupName]) {
        groups[groupName] = {
          totalUsed: 0,
          totalLimit: 0,
          remainingPcts: [],
          earliestReset: null,
          credentialsTotal: 0,
          credentialsExhausted: 0,
        };
      }

      const g = groups[groupName];
      g.credentialsTotal += 1;

      if (groupUsage.fair_cycle_exhausted) {
        g.credentialsExhausted += 1;
      }

      const window = getPrimaryWindow(groupUsage.windows);
      if (window) {
        g.totalUsed += window.request_count;
        if (window.limit) {
          g.totalLimit += window.limit;
        }

        const pct = getWindowRemainingPct(window);
        if (pct !== null) {
          g.remainingPcts.push(pct);
        }

        if (window.reset_at) {
          if (!g.earliestReset || window.reset_at < g.earliestReset) {
            g.earliestReset = window.reset_at;
          }
        }
      }
    }
  }

  // Calculate aggregated remaining percentages
  const result: Record<string, {
    totalUsed: number;
    totalLimit: number;
    remainingPct: number | null;
    earliestReset: number | null;
    credentialsTotal: number;
    credentialsExhausted: number;
  }> = {};

  for (const [groupName, g] of Object.entries(groups)) {
    let remainingPct: number | null = null;

    if (g.totalLimit > 0) {
      remainingPct = Math.max(0, Math.round(((g.totalLimit - g.totalUsed) / g.totalLimit) * 100));
    } else if (g.remainingPcts.length > 0) {
      remainingPct = Math.round(g.remainingPcts.reduce((a, b) => a + b, 0) / g.remainingPcts.length);
    }

    result[groupName] = {
      totalUsed: g.totalUsed,
      totalLimit: g.totalLimit,
      remainingPct,
      earliestReset: g.earliestReset,
      credentialsTotal: g.credentialsTotal,
      credentialsExhausted: g.credentialsExhausted,
    };
  }

  return result;
}
