/**
 * Utility functions for formatting quota data.
 * Ported from the Telegram bot implementation.
 */

import { Color, Icon, Image } from "@raycast/api";
import { ProviderStats, TokenStats, CredentialStats } from "./types";

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
 * Format ISO time string for display.
 */
export function formatResetTime(isoTime: string | undefined): string {
  if (!isoTime) {
    return "";
  }
  try {
    const date = new Date(isoTime);
    return date.toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return isoTime.slice(0, 16);
  }
}

/**
 * Get status icon for provider based on credential states.
 */
export function getProviderStatusIcon(stats: ProviderStats): Image.ImageLike {
  if (stats.exhausted_count > 0) {
    return { source: Icon.XMarkCircle, tintColor: Color.Red };
  } else if (stats.on_cooldown_count > 0 || stats.active_count < stats.credential_count) {
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
  } else if (stats.on_cooldown_count > 0) {
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
  } else if (cred.status === "cooldown" || cred.key_cooldown_remaining) {
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
  } else if (cred.status === "cooldown" || cred.key_cooldown_remaining) {
    const cd = cred.key_cooldown_remaining ? formatCooldown(cred.key_cooldown_remaining) : "";
    return `Cooldown ${cd}`.trim();
  } else {
    return "Active";
  }
}

/**
 * Calculate progress percentage for quota.
 */
export function getProgressPercentage(remaining: number | null): number {
  if (remaining === null) {
    return 0;
  }
  return Math.max(0, Math.min(100, remaining));
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
