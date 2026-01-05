/**
 * Configuration constants for the LLM Proxy Dashboard
 */

export const CONFIG = {
    // API Configuration - Use dashboard-specific endpoints (no auth required)
    API_BASE: '',  // Empty = relative URLs (same origin)
    ENDPOINTS: {
        QUOTA_STATS: '/dashboard/api/quota-stats',
        PROVIDERS: '/v1/providers',
        MODELS: '/v1/models'
    },

    // Auto-refresh settings
    REFRESH_INTERVAL_MS: 5000,  // 5 seconds
    REFRESH_ON_FOCUS: true,
    STALE_THRESHOLD_MS: 15000,  // Consider data stale after 15s

    // UI settings
    ANIMATION_DURATION: 250,
    TOAST_DURATION: 4000,
    MAX_VISIBLE_CREDENTIALS: 5,

    // Storage keys for localStorage
    STORAGE_KEYS: {
        API_KEY: 'llm_proxy_dashboard_api_key',
        THEME: 'llm_proxy_dashboard_theme',
        AUTO_REFRESH: 'llm_proxy_dashboard_auto_refresh',
        VIEW_MODE: 'llm_proxy_dashboard_view_mode'
    },

    // Provider icons (emoji fallbacks)
    PROVIDER_ICONS: {
        antigravity: '🌀',
        gemini: '✨',
        gemini_cli: '💎',
        openai: '🤖',
        anthropic: '🧠',
        openrouter: '🌐',
        qwen: '🔮',
        iflow: '🌊',
        default: '⚡'
    }
};
