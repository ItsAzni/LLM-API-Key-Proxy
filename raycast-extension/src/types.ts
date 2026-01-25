/**
 * TypeScript interfaces for the /v1/quota-stats API response.
 * Updated for modular usage manager format.
 */

export interface TokenStats {
  input_cached: number;
  input_uncached: number;
  input_cache_pct?: number;
  output: number;
}

export interface WindowStats {
  request_count: number;
  success_count: number;
  failure_count: number;
  prompt_tokens: number;
  completion_tokens: number;
  thinking_tokens: number;
  output_tokens: number;
  prompt_tokens_cache_read: number;
  prompt_tokens_cache_write: number;
  total_tokens: number;
  limit: number | null;
  remaining: number | null;
  max_recorded_requests: number;
  max_recorded_at: number | null;
  reset_at: number | null;
  approx_cost: number | null;
  first_used_at: number | null;
  last_used_at: number | null;
}

export interface ModelUsage {
  windows: Record<string, WindowStats>;
  totals: {
    request_count: number;
    success_count: number;
    failure_count: number;
    prompt_tokens: number;
    completion_tokens: number;
    thinking_tokens: number;
    output_tokens: number;
    prompt_tokens_cache_read: number;
    prompt_tokens_cache_write: number;
    total_tokens: number;
    approx_cost: number | null;
    first_used_at: number | null;
    last_used_at: number | null;
  };
}

export interface GroupUsage extends ModelUsage {
  fair_cycle_exhausted: boolean;
  fair_cycle_reason: string | null;
  cooldown_remaining: number | null;
  cooldown_source: string | null;
  custom_cap?: {
    limit: number;
    used: number;
    remaining: number;
    remaining_pct: number;
  };
}

export interface Totals {
  request_count: number;
  success_count: number;
  failure_count: number;
  prompt_tokens: number;
  completion_tokens: number;
  thinking_tokens: number;
  output_tokens: number;
  prompt_tokens_cache_read: number;
  prompt_tokens_cache_write: number;
  total_tokens: number;
  approx_cost: number | null;
  first_used_at: number | null;
  last_used_at: number | null;
}

export interface CredentialStats {
  stable_id: string;
  accessor_masked?: string;
  full_path?: string;
  identifier: string;
  email: string | null;
  tier: string | null;
  priority?: number;
  active_requests?: number;
  status: "active" | "cooldown" | "exhausted" | "mixed" | "unknown";
  totals?: Totals;
  model_usage?: Record<string, ModelUsage>;
  group_usage?: Record<string, GroupUsage>;
  cooldowns?: Record<string, unknown>;
  fair_cycle?: Record<string, unknown>;
}

export interface ProviderStats {
  provider?: string;
  credential_count: number;
  rotation_mode?: string;
  active_count: number;
  exhausted_count: number;
  on_cooldown_count?: number;
  total_requests: number;
  tokens: TokenStats;
  approx_cost: number | null;
  quota_groups?: Record<string, unknown>;
  // credentials can be dict (new format) or array (legacy format)
  credentials: Record<string, CredentialStats> | CredentialStats[];
}

export interface Summary {
  total_providers: number;
  total_credentials: number;
  active_credentials: number;
  exhausted_credentials: number;
  total_requests: number;
  tokens: TokenStats;
  approx_total_cost: number | null;
}

export interface QuotaStats {
  providers: Record<string, ProviderStats>;
  summary: Summary;
  timestamp: number;
  data_source: string;
}

export interface RefreshResult {
  action: string;
  scope: string;
  provider?: string;
  credential?: string;
  success: boolean;
  duration_ms?: number;
  credentials_refreshed?: number;
  failed_count?: number;
  message?: string;
}

export interface QuotaStatsWithRefresh extends QuotaStats {
  refresh_result?: RefreshResult;
}

export interface Preferences {
  proxyUrl: string;
  apiKey?: string;
}
