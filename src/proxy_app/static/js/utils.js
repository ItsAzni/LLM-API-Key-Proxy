/**
 * Utility Functions - Formatting helpers
 */

/**
 * Format token count for display
 * Examples: 1234567 -> "1.2M", 12345 -> "12.3k", 123 -> "123"
 */
export function formatTokens(count) {
    if (count === null || count === undefined) return '--';
    if (count >= 1_000_000) {
        return `${(count / 1_000_000).toFixed(1)}M`;
    } else if (count >= 10_000) {
        return `${(count / 1_000).toFixed(1)}k`;
    } else if (count >= 1_000) {
        return `${(count / 1_000).toFixed(1)}k`;
    }
    return count.toLocaleString();
}

/**
 * Format cost for display
 * Examples: 12.50 -> "$12.50", 0.0012 -> "$0.0012", null -> "-"
 */
export function formatCost(cost) {
    if (cost === null || cost === undefined || cost === 0) {
        return '-';
    }
    if (cost < 0.01) {
        return `$${cost.toFixed(4)}`;
    }
    return `$${cost.toFixed(2)}`;
}

/**
 * Format time ago for display
 * Examples: 30s -> "30s ago", 120s -> "2m ago", 7200s -> "2h ago"
 */
export function formatTimeAgo(timestamp) {
    if (!timestamp) return 'Never';

    const delta = (Date.now() / 1000) - timestamp;

    if (delta < 0) {
        return 'Just now';
    } else if (delta < 60) {
        return `${Math.floor(delta)}s ago`;
    } else if (delta < 3600) {
        return `${Math.floor(delta / 60)}m ago`;
    } else if (delta < 86400) {
        return `${Math.floor(delta / 3600)}h ago`;
    } else {
        return `${Math.floor(delta / 86400)}d ago`;
    }
}

/**
 * Format seconds as duration
 * Examples: 45 -> "45s", 125 -> "2m 5s", 3700 -> "1h 1m"
 */
export function formatCooldown(seconds) {
    if (seconds === null || seconds === undefined || seconds <= 0) {
        return '-';
    }

    seconds = Math.floor(seconds);

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
 * Format seconds until a timestamp
 */
export function formatTimeUntil(timestamp) {
    if (!timestamp) return '-';
    const remaining = timestamp - (Date.now() / 1000);
    if (remaining <= 0) return 'Expired';
    return formatCooldown(remaining);
}

/**
 * Get quota bar CSS class based on remaining percentage
 */
export function getQuotaClass(percent) {
    if (percent === null || percent === undefined) return '';
    if (percent <= 10) return 'danger';
    if (percent <= 30) return 'warning';
    return '';
}

/**
 * Get status CSS class
 */
export function getStatusClass(status) {
    switch (status?.toLowerCase()) {
        case 'active':
            return 'active';
        case 'cooldown':
            return 'cooldown';
        case 'exhausted':
            return 'exhausted';
        default:
            return 'unknown';
    }
}

/**
 * Capitalize first letter
 */
export function capitalizeFirst(str) {
    if (!str) return '';
    return str.charAt(0).toUpperCase() + str.slice(1);
}

/**
 * Natural sort comparator for credential identifiers
 */
export function naturalSort(a, b) {
    const identA = a.identifier || '';
    const identB = b.identifier || '';
    return identA.localeCompare(identB, undefined, {
        numeric: true,
        sensitivity: 'base'
    });
}

/**
 * Generate unique ID
 */
export function generateId() {
    return Math.random().toString(36).substr(2, 9);
}

/**
 * Escape HTML to prevent XSS
 */
export function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

/**
 * Debounce function
 */
export function debounce(fn, delay) {
    let timeoutId;
    return function (...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => fn.apply(this, args), delay);
    };
}

/**
 * Format number with commas
 */
export function formatNumber(num) {
    if (num === null || num === undefined) return '--';
    return num.toLocaleString();
}

/**
 * Get provider icon
 */
export function getProviderIcon(provider) {
    const icons = {
        antigravity: '🌀',
        gemini: '✨',
        gemini_cli: '💎',
        openai: '🤖',
        anthropic: '🧠',
        openrouter: '🌐',
        qwen: '🔮',
        iflow: '🌊'
    };
    return icons[provider?.toLowerCase()] || '⚡';
}

/**
 * Truncate string with ellipsis
 */
export function truncate(str, maxLength = 30) {
    if (!str || str.length <= maxLength) return str;
    return str.substring(0, maxLength - 3) + '...';
}

/**
 * Calculate total tokens from token object
 */
export function getTotalTokens(tokens) {
    if (!tokens) return 0;
    return (tokens.input_cached || 0) +
           (tokens.input_uncached || 0) +
           (tokens.output || 0);
}

/**
 * Calculate input tokens from token object
 */
export function getInputTokens(tokens) {
    if (!tokens) return 0;
    return (tokens.input_cached || 0) + (tokens.input_uncached || 0);
}
