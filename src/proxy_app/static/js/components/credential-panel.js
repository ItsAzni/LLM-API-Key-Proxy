/**
 * Credential Panel Component - Renders credential detail drawer content
 */

import {
    formatTokens,
    formatCost,
    formatCooldown,
    formatTimeAgo,
    formatTimeUntil,
    getQuotaClass,
    getStatusClass,
    capitalizeFirst,
    escapeHtml,
    getInputTokens
} from '../utils.js';

/**
 * Render credential detail panel
 * @param {string} providerName - Provider identifier
 * @param {Object} credential - Credential data
 * @param {string} viewMode - 'current' or 'global'
 * @returns {string} HTML string
 */
export function renderCredentialPanel(providerName, credential, viewMode) {
    if (!credential) {
        return '<p class="empty-state">No credential selected</p>';
    }

    const stats = viewMode === 'global' && credential.global
        ? credential.global
        : credential;

    const status = credential.status || 'unknown';
    const identifier = credential.identifier || 'Unknown';
    const email = credential.email;
    const tier = credential.tier;
    const lastUsed = credential.last_used_ts;

    // Token stats
    const tokens = stats.tokens || {};
    const inputTokens = getInputTokens(tokens);
    const outputTokens = tokens.output || 0;
    const cachePercent = tokens.input_cache_pct || 0;

    // Cooldowns
    const modelCooldowns = credential.model_cooldowns || {};
    const activeCooldowns = Object.entries(modelCooldowns)
        .filter(([_, c]) => c.remaining_seconds > 0);

    // Model groups (quota)
    const modelGroups = credential.model_groups || {};

    // Per-model breakdown
    const models = credential.models || {};

    return `
        <div class="credential-detail">
            <!-- Header -->
            <div class="detail-header">
                <span class="status-badge ${getStatusClass(status)}">${status}</span>
                ${tier ? `<span class="tier-badge">${escapeHtml(tier)}</span>` : ''}
            </div>

            <!-- Basic Info -->
            <div class="detail-section">
                <h4 class="detail-section-title">Info</h4>
                <div class="detail-row">
                    <span class="detail-label">Identifier</span>
                    <span class="detail-value">${escapeHtml(identifier)}</span>
                </div>
                ${email ? `
                    <div class="detail-row">
                        <span class="detail-label">Email</span>
                        <span class="detail-value">${escapeHtml(email)}</span>
                    </div>
                ` : ''}
                <div class="detail-row">
                    <span class="detail-label">Last Used</span>
                    <span class="detail-value">${formatTimeAgo(lastUsed)}</span>
                </div>
            </div>

            <!-- Stats -->
            <div class="detail-section">
                <h4 class="detail-section-title">Statistics</h4>
                <div class="detail-row">
                    <span class="detail-label">Requests</span>
                    <span class="detail-value">${(stats.requests || 0).toLocaleString()}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Input Tokens</span>
                    <span class="detail-value">${formatTokens(inputTokens)}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Output Tokens</span>
                    <span class="detail-value">${formatTokens(outputTokens)}</span>
                </div>
                ${cachePercent > 0 ? `
                    <div class="detail-row">
                        <span class="detail-label">Cache Hit Rate</span>
                        <span class="detail-value">${cachePercent.toFixed(1)}%</span>
                    </div>
                ` : ''}
                <div class="detail-row">
                    <span class="detail-label">Approx. Cost</span>
                    <span class="detail-value">${formatCost(stats.approx_cost)}</span>
                </div>
            </div>

            <!-- Quota Groups -->
            ${Object.keys(modelGroups).length > 0 ? `
                <div class="detail-section">
                    <h4 class="detail-section-title">Quota</h4>
                    ${renderQuotaGroupsDetail(modelGroups)}
                </div>
            ` : ''}

            <!-- Active Cooldowns -->
            ${activeCooldowns.length > 0 ? `
                <div class="detail-section">
                    <h4 class="detail-section-title">Active Cooldowns</h4>
                    ${renderCooldowns(activeCooldowns)}
                </div>
            ` : ''}

            <!-- Per-Model Breakdown -->
            ${Object.keys(models).length > 0 ? `
                <div class="detail-section">
                    <h4 class="detail-section-title">Models</h4>
                    ${renderModels(models)}
                </div>
            ` : ''}
        </div>
    `;
}

/**
 * Render quota groups detail
 */
function renderQuotaGroupsDetail(modelGroups) {
    return Object.entries(modelGroups).map(([name, group]) => {
        const remaining = group.remaining_pct ?? 0;
        const used = group.requests_used || 0;
        const max = group.requests_max || '?';
        const colorClass = getQuotaClass(remaining);
        const resetTime = group.reset_time_iso;
        const confidence = group.confidence;

        return `
            <div class="quota-group-detail">
                <div class="quota-bar">
                    <div class="quota-bar-header">
                        <span class="quota-name">${escapeHtml(capitalizeFirst(name))}</span>
                        <span class="quota-value">${used}/${max} (${remaining}%)</span>
                    </div>
                    <div class="quota-bar-track">
                        <div class="quota-bar-fill ${colorClass}"
                             style="width: ${Math.max(0, Math.min(100, remaining))}%"></div>
                    </div>
                </div>
                ${resetTime ? `
                    <div class="detail-row">
                        <span class="detail-label">Resets</span>
                        <span class="detail-value">${formatResetTime(resetTime)}</span>
                    </div>
                ` : ''}
                ${confidence ? `
                    <div class="detail-row">
                        <span class="detail-label">Confidence</span>
                        <span class="detail-value">${escapeHtml(confidence)}</span>
                    </div>
                ` : ''}
                ${group.models?.length ? `
                    <div class="detail-row">
                        <span class="detail-label">Models</span>
                        <span class="detail-value">${group.models.length}</span>
                    </div>
                ` : ''}
            </div>
        `;
    }).join('');
}

/**
 * Format reset time from ISO string
 */
function formatResetTime(isoString) {
    try {
        const date = new Date(isoString);
        const now = new Date();
        const diff = date - now;

        if (diff <= 0) return 'Now';

        const hours = Math.floor(diff / 3600000);
        const minutes = Math.floor((diff % 3600000) / 60000);

        if (hours > 0) {
            return `in ${hours}h ${minutes}m`;
        }
        return `in ${minutes}m`;
    } catch {
        return isoString;
    }
}

/**
 * Render active cooldowns
 */
function renderCooldowns(cooldowns) {
    return `
        <div class="cooldown-list">
            ${cooldowns.map(([model, cooldown]) => `
                <div class="cooldown-item">
                    <span class="model-name">${escapeHtml(model)}</span>
                    <span class="cooldown-timer">
                        <span class="timer-icon">&#x23F1;</span>
                        ${formatCooldown(cooldown.remaining_seconds)}
                    </span>
                </div>
            `).join('')}
        </div>
    `;
}

/**
 * Render per-model breakdown
 */
function renderModels(models) {
    const modelEntries = Object.entries(models);
    if (modelEntries.length === 0) return '<p>No model data</p>';

    return `
        <div class="model-list">
            ${modelEntries.slice(0, 10).map(([modelName, modelData]) => {
                const requests = modelData.requests || modelData.request_count || 0;
                const successCount = modelData.success_count || 0;
                const failureCount = modelData.failure_count || 0;
                const promptTokens = modelData.prompt_tokens || 0;
                const completionTokens = modelData.completion_tokens || 0;
                const cost = modelData.approx_cost;
                const quotaDisplay = modelData.quota_display;

                return `
                    <div class="model-item">
                        <div class="model-name">${escapeHtml(modelName.split('/').pop())}</div>
                        <div class="model-stats">
                            <span>${requests} req</span>
                            ${failureCount > 0 ? `<span class="failure">${failureCount} fail</span>` : ''}
                            <span>${formatTokens(promptTokens)} in</span>
                            <span>${formatTokens(completionTokens)} out</span>
                            ${cost ? `<span>${formatCost(cost)}</span>` : ''}
                        </div>
                        ${quotaDisplay ? `
                            <div class="model-quota">${escapeHtml(quotaDisplay)}</div>
                        ` : ''}
                    </div>
                `;
            }).join('')}
            ${modelEntries.length > 10 ? `
                <p class="more-models">+${modelEntries.length - 10} more models</p>
            ` : ''}
        </div>
    `;
}
