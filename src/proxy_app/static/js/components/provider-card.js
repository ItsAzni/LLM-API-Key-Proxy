/**
 * Provider Card Component - Renders a provider overview card
 */

import {
    formatTokens,
    formatCost,
    formatCooldown,
    getQuotaClass,
    capitalizeFirst,
    naturalSort,
    getProviderIcon,
    escapeHtml,
    getInputTokens
} from '../utils.js';
import { CONFIG } from '../config.js';

/**
 * Render a provider card
 * @param {string} providerName - Provider identifier
 * @param {Object} providerStats - Provider statistics from API
 * @param {string} viewMode - 'current' or 'global'
 * @returns {string} HTML string
 */
export function renderProviderCard(providerName, providerStats, viewMode) {
    const stats = viewMode === 'global' && providerStats.global
        ? providerStats.global
        : providerStats;

    const credentials = providerStats.credentials || [];
    const sortedCreds = [...credentials].sort(naturalSort);

    // Status counts
    const activeCount = providerStats.active_count || 0;
    const cooldownCount = providerStats.on_cooldown_count || 0;
    const exhaustedCount = providerStats.exhausted_count || 0;

    // Token stats
    const tokens = stats.tokens || {};
    const totalInput = getInputTokens(tokens);
    const totalOutput = tokens.output || 0;
    const cachePercent = tokens.input_cache_pct || 0;

    // Quota groups (for Antigravity)
    const quotaGroups = providerStats.quota_groups || {};
    const hasQuotaGroups = Object.keys(quotaGroups).length > 0;

    // Provider icon
    const icon = getProviderIcon(providerName);

    return `
        <article class="provider-card glass-card" data-provider="${escapeHtml(providerName)}">
            <header class="provider-header">
                <h2 class="provider-name">
                    <span class="provider-icon">${icon}</span>
                    ${escapeHtml(capitalizeFirst(providerName))}
                </h2>
                <div class="credential-count">
                    ${activeCount > 0 ? `
                        <span class="count-item" title="${activeCount} active">
                            <span class="count-dot active"></span>
                            ${activeCount}
                        </span>
                    ` : ''}
                    ${cooldownCount > 0 ? `
                        <span class="count-item" title="${cooldownCount} on cooldown">
                            <span class="count-dot cooldown"></span>
                            ${cooldownCount}
                        </span>
                    ` : ''}
                    ${exhaustedCount > 0 ? `
                        <span class="count-item" title="${exhaustedCount} exhausted">
                            <span class="count-dot exhausted"></span>
                            ${exhaustedCount}
                        </span>
                    ` : ''}
                </div>
            </header>

            ${hasQuotaGroups ? renderQuotaGroups(quotaGroups) : ''}

            <div class="stats-row">
                <div class="stat-item">
                    <span class="stat-value">${formatNumber(stats.total_requests || 0)}</span>
                    <span class="stat-label">Requests</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value">${formatTokens(totalInput)}/${formatTokens(totalOutput)}</span>
                    <span class="stat-label">In/Out</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value">${formatCost(stats.approx_cost)}</span>
                    <span class="stat-label">Cost</span>
                </div>
            </div>

            ${cachePercent > 0 ? `
                <div class="cache-stat">
                    <span class="cache-label">Cache Hit:</span>
                    <span class="cache-value">${cachePercent.toFixed(1)}%</span>
                </div>
            ` : ''}

            <div class="credentials-list">
                <h3 class="credentials-header">
                    Credentials (${credentials.length})
                </h3>
                ${sortedCreds.slice(0, CONFIG.MAX_VISIBLE_CREDENTIALS)
                    .map(cred => renderCredentialItem(providerName, cred))
                    .join('')}
                ${credentials.length > CONFIG.MAX_VISIBLE_CREDENTIALS ? `
                    <button class="show-more-btn glass-button"
                            data-provider="${escapeHtml(providerName)}"
                            data-action="show-all-credentials">
                        Show ${credentials.length - CONFIG.MAX_VISIBLE_CREDENTIALS} more...
                    </button>
                ` : ''}
            </div>
        </article>
    `;
}

/**
 * Format number with locale
 */
function formatNumber(num) {
    if (num === null || num === undefined) return '--';
    return num.toLocaleString();
}

/**
 * Render a credential item in the list
 */
function renderCredentialItem(providerName, credential) {
    const status = credential.status || 'unknown';
    const identifier = credential.identifier || 'Unknown';

    // Check for active cooldowns
    const cooldowns = credential.model_cooldowns || {};
    const activeCooldowns = Object.values(cooldowns)
        .filter(c => c.remaining_seconds > 0);
    const maxCooldown = activeCooldowns.length > 0
        ? Math.max(...activeCooldowns.map(c => c.remaining_seconds))
        : null;

    // Get quota info if available
    const modelGroups = credential.model_groups || {};
    const firstGroup = Object.values(modelGroups)[0];
    const quotaDisplay = firstGroup?.display;

    return `
        <div class="credential-item"
             data-provider="${escapeHtml(providerName)}"
             data-credential="${escapeHtml(credential.full_path || identifier)}"
             tabindex="0"
             role="button"
             aria-label="View details for ${escapeHtml(identifier)}">
            <span class="count-dot ${status}"></span>
            <span class="credential-identifier">${escapeHtml(identifier)}</span>
            ${credential.tier ? `
                <span class="tier-badge">${escapeHtml(credential.tier)}</span>
            ` : ''}
            ${quotaDisplay ? `
                <span class="quota-display">${escapeHtml(quotaDisplay)}</span>
            ` : ''}
            ${maxCooldown ? `
                <span class="cooldown-timer">
                    <span class="timer-icon">&#x23F1;</span>
                    ${formatCooldown(maxCooldown)}
                </span>
            ` : ''}
        </div>
    `;
}

/**
 * Render quota groups section
 */
function renderQuotaGroups(quotaGroups) {
    return `
        <div class="quota-groups">
            ${Object.entries(quotaGroups).map(([name, group]) => {
                const remaining = group.total_remaining_pct ?? group.avg_remaining_pct ?? 0;
                const used = group.total_requests_used || 0;
                const max = group.total_requests_max || '?';
                const colorClass = getQuotaClass(remaining);

                return `
                    <div class="quota-bar">
                        <div class="quota-bar-header">
                            <span class="quota-name">${escapeHtml(capitalizeFirst(name))}</span>
                            <span class="quota-value">${used}/${max}</span>
                        </div>
                        <div class="quota-bar-track">
                            <div class="quota-bar-fill ${colorClass}"
                                 style="width: ${Math.max(0, Math.min(100, remaining))}%"></div>
                        </div>
                    </div>
                `;
            }).join('')}
        </div>
    `;
}

/**
 * Render all credentials for a provider (expanded view)
 */
export function renderAllCredentials(providerName, credentials) {
    const sortedCreds = [...credentials].sort(naturalSort);
    return sortedCreds.map(cred => renderCredentialItem(providerName, cred)).join('');
}
