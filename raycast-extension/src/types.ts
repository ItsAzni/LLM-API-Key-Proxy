/**
 * TypeScript interfaces for the /v1/quota-stats API response.
 */

export interface TokenStats {
  input_cached: number;
  input_uncached: number;
  output: number;
}

export interface QuotaGroup {
  total_requests_used: number;
  total_requests_max: number;
  total_remaining_pct: number | null;
  next_reset_time_iso?: string;
}

export interface ModelGroup {
  requests_used: number;
  requests_max?: number;
  remaining_pct: number | null;
  is_exhausted: boolean;
  reset_time_iso?: string;
}

export interface CredentialStats {
  identifier: string;
  email?: string;
  tier?: string;
  status: "active" | "cooldown" | "exhausted" | "unknown";
  requests: number;
  tokens: TokenStats;
  approx_cost: number | null;
  model_groups?: Record<string, ModelGroup>;
  key_cooldown_remaining?: number;
}

export interface ProviderStats {
  credential_count: number;
  active_count: number;
  on_cooldown_count: number;
  exhausted_count: number;
  total_requests: number;
  tokens: TokenStats;
  approx_cost: number | null;
  quota_groups?: Record<string, QuotaGroup>;
  credentials: CredentialStats[];
}

export interface Summary {
  total_credentials: number;
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
