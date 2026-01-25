import {
  List,
  ActionPanel,
  Action,
  Icon,
  getPreferenceValues,
  showToast,
  Toast,
  openExtensionPreferences,
} from "@raycast/api";
import { useCachedPromise } from "@raycast/utils";
import React from "react";
import {
  QuotaStats,
  QuotaStatsWithRefresh,
  ProviderStats,
  CredentialStats,
  Preferences,
} from "./types";
import {
  formatTokenStats,
  formatCost,
  formatResetTime,
  getProviderStatusIcon,
  getProviderStatusText,
  getCredentialStatusIcon,
  getCredentialStatusText,
  getProgressColor,
  getProgressIcon,
  credentialsToArray,
  aggregateProviderQuotaGroups,
  getCredentialQuotaInfo,
  getPrimaryWindow,
  getWindowRemainingPct,
} from "./utils";

export default function QuotaCommand() {
  const { proxyUrl, apiKey } = getPreferenceValues<Preferences>();

  const headers: Record<string, string> = {};
  if (apiKey) {
    headers["Authorization"] = `Bearer ${apiKey}`;
  }

  // Fetch function - only called when explicitly triggered
  async function fetchQuotaStats(): Promise<QuotaStats> {
    const response = await fetch(`${proxyUrl}/v1/quota-stats`, { headers });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    return response.json();
  }

  const { data, isLoading, revalidate, error } = useCachedPromise(
    fetchQuotaStats,
    [],
    {
      keepPreviousData: true,
    }
  );

  // Handle force refresh
  async function forceRefresh(provider?: string) {
    try {
      const toast = await showToast({
        style: Toast.Style.Animated,
        title: "Refreshing quota...",
      });

      const response = await fetch(`${proxyUrl}/v1/quota-stats`, {
        method: "POST",
        headers: {
          ...headers,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          action: "force_refresh",
          scope: provider ? "provider" : "all",
          provider: provider,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const result: QuotaStatsWithRefresh = await response.json();

      if (result.refresh_result?.success) {
        toast.style = Toast.Style.Success;
        toast.title = "Quota refreshed";
        toast.message = result.refresh_result.message ||
          `${result.refresh_result.credentials_refreshed || 0} credentials in ${result.refresh_result.duration_ms || 0}ms`;
      } else {
        toast.style = Toast.Style.Failure;
        toast.title = "Refresh failed";
        toast.message = result.refresh_result?.message || "Unknown error";
      }

      // Revalidate to update the list
      revalidate();
    } catch (err) {
      showToast({
        style: Toast.Style.Failure,
        title: "Refresh failed",
        message: String(err),
      });
    }
  }

  // Error state - check for specific error types
  if (error) {
    const errorMessage = String(error);
    const isUnauthorized = errorMessage.includes("401") || errorMessage.toLowerCase().includes("unauthorized");
    const isForbidden = errorMessage.includes("403") || errorMessage.toLowerCase().includes("forbidden");

    if (isUnauthorized || isForbidden) {
      return (
        <List>
          <List.EmptyView
            icon={Icon.Key}
            title="Authentication Required"
            description={apiKey ? "Invalid API key. Please check your PROXY_API_KEY in preferences." : "Please configure your API key in preferences."}
            actions={
              <ActionPanel>
                <Action title="Open Preferences" icon={Icon.Gear} onAction={openExtensionPreferences} />
                <Action title="Retry" icon={Icon.ArrowClockwise} onAction={revalidate} />
              </ActionPanel>
            }
          />
        </List>
      );
    }

    return (
      <List>
        <List.EmptyView
          icon={Icon.ExclamationMark}
          title="Failed to connect"
          description={`Could not connect to ${proxyUrl}. Is the proxy running?`}
          actions={
            <ActionPanel>
              <Action title="Retry" icon={Icon.ArrowClockwise} onAction={revalidate} />
              <Action title="Open Preferences" icon={Icon.Gear} onAction={openExtensionPreferences} />
            </ActionPanel>
          }
        />
      </List>
    );
  }

  // Empty state
  if (data && Object.keys(data.providers).length === 0) {
    return (
      <List>
        <List.EmptyView
          icon={Icon.Tray}
          title="No providers configured"
          description="The proxy has no credentials configured."
        />
      </List>
    );
  }

  return (
    <List
      isLoading={isLoading}
      isShowingDetail
      navigationTitle="LLM Proxy Quota"
      searchBarPlaceholder="Filter providers..."
    >
      {data && (
        <>
          {/* Summary section */}
          <List.Section title="Summary">
            <List.Item
              title="All Providers"
              icon={Icon.BarChart}
              accessories={[
                { text: `${data.summary.total_credentials} creds` },
                { text: `${data.summary.total_requests} reqs` },
                { text: formatCost(data.summary.approx_total_cost) },
              ]}
              detail={
                <List.Item.Detail
                  metadata={
                    <List.Item.Detail.Metadata>
                      <List.Item.Detail.Metadata.Label title="Current Session" />
                      <List.Item.Detail.Metadata.Label
                        title="Providers"
                        text={data.summary.total_providers.toString()}
                      />
                      <List.Item.Detail.Metadata.Label
                        title="Credentials"
                        text={`${data.summary.active_credentials}/${data.summary.total_credentials} active`}
                      />
                      <List.Item.Detail.Metadata.Label
                        title="Requests"
                        text={data.summary.total_requests.toString()}
                      />
                      <List.Item.Detail.Metadata.Label
                        title="Tokens"
                        text={formatTokenStats(data.summary.tokens)}
                      />
                      <List.Item.Detail.Metadata.Label
                        title="Cost"
                        text={formatCost(data.summary.approx_total_cost)}
                      />
                      <List.Item.Detail.Metadata.Separator />

                      <List.Item.Detail.Metadata.Label
                        title="Data Source"
                        text={data.data_source}
                      />
                      <List.Item.Detail.Metadata.Label
                        title="Last Updated"
                        text={new Date(data.timestamp * 1000).toLocaleString()}
                      />
                    </List.Item.Detail.Metadata>
                  }
                />
              }
              actions={
                <ActionPanel>
                  <Action
                    title="Refresh"
                    icon={Icon.ArrowClockwise}
                    onAction={revalidate}
                    shortcut={{ modifiers: ["cmd"], key: "r" }}
                  />
                  <Action
                    title="Force Refresh All"
                    icon={Icon.ArrowClockwise}
                    onAction={() => forceRefresh()}
                    shortcut={{ modifiers: ["cmd", "shift"], key: "r" }}
                  />
                </ActionPanel>
              }
            />
          </List.Section>

          {/* Providers section */}
          <List.Section title="Providers">
            {Object.entries(data.providers).map(([providerName, stats]) => (
              <ProviderListItem
                key={providerName}
                name={providerName}
                stats={stats}
                onRefresh={revalidate}
                onForceRefresh={() => forceRefresh(providerName)}
              />
            ))}
          </List.Section>
        </>
      )}
    </List>
  );
}

function ProviderListItem({
  name,
  stats,
  onRefresh,
  onForceRefresh,
}: {
  name: string;
  stats: ProviderStats;
  onRefresh: () => void;
  onForceRefresh: () => void;
}) {
  // Aggregate quota groups from credentials for provider-level summary
  const quotaGroups = aggregateProviderQuotaGroups(stats.credentials);
  const quotaGroupEntries = Object.entries(quotaGroups);

  // Prioritize 'claude' group if it exists, otherwise use first available
  const claudeQuota = quotaGroups["claude"];
  const primaryQuota = claudeQuota || (quotaGroupEntries.length > 0 ? quotaGroupEntries[0][1] : null);
  const primaryGroupName = claudeQuota ? "claude" : (quotaGroupEntries.length > 0 ? quotaGroupEntries[0][0] : null);

  const remainingPct = primaryQuota?.remainingPct;
  const nextReset = primaryQuota?.earliestReset;

  // Build accessories with quota info - using tag for colored percentage
  const accessories: List.Item.Accessory[] = [
    { text: `${stats.active_count}/${stats.credential_count}` },
  ];

  // Add remaining quota percentage if available (as colored tag)
  if (remainingPct !== null && remainingPct !== undefined) {
    const label = primaryGroupName ? `${primaryGroupName}: ${remainingPct}%` : `${remainingPct}%`;
    accessories.push({
      tag: { value: label, color: getProgressColor(remainingPct) },
    });
  }

  // Add next reset time if available
  if (nextReset) {
    accessories.push({ text: formatResetTime(nextReset) });
  }

  return (
    <List.Item
      title={name.charAt(0).toUpperCase() + name.slice(1)}
      subtitle={getProviderStatusText(stats)}
      icon={getProviderStatusIcon(stats)}
      accessories={accessories}
      detail={<ProviderDetail name={name} stats={stats} quotaGroups={quotaGroups} />}
      actions={
        <ActionPanel>
          <Action
            title="Refresh"
            icon={Icon.ArrowClockwise}
            onAction={onRefresh}
            shortcut={{ modifiers: ["cmd"], key: "r" }}
          />
          <Action
            title="Force Refresh Provider"
            icon={Icon.ArrowClockwise}
            onAction={onForceRefresh}
            shortcut={{ modifiers: ["cmd", "shift"], key: "r" }}
          />
        </ActionPanel>
      }
    />
  );
}

function ProviderDetail({
  name,
  stats,
  quotaGroups
}: {
  name: string;
  stats: ProviderStats;
  quotaGroups: ReturnType<typeof aggregateProviderQuotaGroups>;
}) {
  const credentials = credentialsToArray(stats.credentials);

  return (
    <List.Item.Detail
      metadata={
        <List.Item.Detail.Metadata>
          {/* Overview */}
          <List.Item.Detail.Metadata.Label title="Overview" />
          <List.Item.Detail.Metadata.Label
            title="Status"
            text={getProviderStatusText(stats)}
            icon={getProviderStatusIcon(stats)}
          />
          <List.Item.Detail.Metadata.Label
            title="Credentials"
            text={`${stats.active_count}/${stats.credential_count} active`}
          />
          {stats.rotation_mode && (
            <List.Item.Detail.Metadata.Label
              title="Rotation Mode"
              text={stats.rotation_mode}
            />
          )}
          <List.Item.Detail.Metadata.Separator />

          {/* Usage */}
          <List.Item.Detail.Metadata.Label title="Usage" />
          <List.Item.Detail.Metadata.Label
            title="Requests"
            text={stats.total_requests.toString()}
          />
          <List.Item.Detail.Metadata.Label
            title="Tokens"
            text={formatTokenStats(stats.tokens)}
          />
          <List.Item.Detail.Metadata.Label
            title="Cost"
            text={formatCost(stats.approx_cost)}
          />
          <List.Item.Detail.Metadata.Separator />

          {/* Quota Groups */}
          {Object.keys(quotaGroups).length > 0 && (
            <>
              <List.Item.Detail.Metadata.Label title="Quota Groups" />
              {Object.entries(quotaGroups).map(([groupName, group]) => (
                <React.Fragment key={groupName}>
                  <List.Item.Detail.Metadata.Label
                    title={groupName}
                    text={
                      group.totalLimit > 0
                        ? `${group.totalUsed}/${group.totalLimit} (${group.remainingPct ?? "?"}% remaining)`
                        : `${group.remainingPct ?? "?"}% remaining`
                    }
                    icon={getProgressIcon(group.remainingPct)}
                  />
                  {group.earliestReset && (
                    <List.Item.Detail.Metadata.Label
                      title={`  ↳ Resets`}
                      text={formatResetTime(group.earliestReset)}
                      icon={Icon.Clock}
                    />
                  )}
                </React.Fragment>
              ))}
              <List.Item.Detail.Metadata.Separator />
            </>
          )}

          {/* Credentials */}
          <List.Item.Detail.Metadata.Label title="Credentials" />
          {credentials.map((cred) => {
            // Build status text
            const statusText = `${getCredentialStatusText(cred)}${cred.tier ? ` (${cred.tier})` : ""}`;
            const quotaInfo = getCredentialQuotaInfo(cred);
            const { lowestPct } = quotaInfo;

            // Get claude quota info
            let claudeText = "";
            if (cred.group_usage && cred.group_usage["claude"]) {
              const groupUsage = cred.group_usage["claude"];
              const window = getPrimaryWindow(groupUsage.windows);
              if (window) {
                const pct = getWindowRemainingPct(window);
                const requestsText = window.limit
                  ? `${window.request_count}/${window.limit}`
                  : `${window.request_count}`;
                const pctText = pct !== null ? ` (${pct}%)` : "";
                const resetText = window.reset_at ? ` → ${formatResetTime(window.reset_at)}` : "";
                claudeText = ` • ${requestsText}${pctText}${resetText}`;
              }
            }

            return (
              <List.Item.Detail.Metadata.Label
                key={cred.stable_id}
                title={(cred.email?.split("@")[0]) || cred.identifier}
                text={`${statusText}${claudeText}`}
                icon={lowestPct !== null
                  ? getProgressIcon(lowestPct)
                  : getCredentialStatusIcon(cred)
                }
              />
            );
          })}
        </List.Item.Detail.Metadata>
      }
    />
  );
}
