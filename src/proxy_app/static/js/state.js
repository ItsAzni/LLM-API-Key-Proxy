/**
 * State Management - Simple pub/sub state store
 */

import { CONFIG } from './config.js';

/**
 * Application state store with pub/sub pattern
 */
class AppState {
    constructor() {
        this.data = {
            // API data
            quotaStats: null,

            // UI state
            viewMode: 'current',  // 'current' or 'global'
            isLoading: false,
            lastUpdated: null,
            error: null,

            // Selection state
            selectedProvider: null,
            selectedCredential: null,

            // Settings
            autoRefresh: true,
            theme: 'light'  // Default to light theme
        };

        this.listeners = new Set();
    }

    /**
     * Subscribe to state changes
     * @returns {Function} Unsubscribe function
     */
    subscribe(listener) {
        this.listeners.add(listener);
        return () => this.listeners.delete(listener);
    }

    /**
     * Notify all listeners of state change
     */
    notify() {
        this.listeners.forEach(listener => {
            try {
                listener(this.data);
            } catch (error) {
                console.error('State listener error:', error);
            }
        });
    }

    /**
     * Update state with partial updates
     */
    update(updates) {
        this.data = { ...this.data, ...updates };
        this.notify();
    }

    /**
     * Set quota stats from API response
     */
    setQuotaStats(stats) {
        this.update({
            quotaStats: stats,
            lastUpdated: Date.now(),
            isLoading: false,
            error: null
        });
    }

    /**
     * Set loading state
     */
    setLoading(isLoading) {
        this.update({ isLoading });
    }

    /**
     * Set error state
     */
    setError(error) {
        this.update({
            error: error?.message || error,
            isLoading: false
        });
    }

    /**
     * Set view mode (current/global)
     */
    setViewMode(mode) {
        this.update({ viewMode: mode });
        localStorage.setItem(CONFIG.STORAGE_KEYS.VIEW_MODE, mode);
    }

    /**
     * Set theme
     */
    setTheme(theme) {
        this.update({ theme });
        localStorage.setItem(CONFIG.STORAGE_KEYS.THEME, theme);
        document.documentElement.dataset.theme = theme;
    }

    /**
     * Toggle theme
     */
    toggleTheme() {
        const newTheme = this.data.theme === 'dark' ? 'light' : 'dark';
        this.setTheme(newTheme);
    }

    /**
     * Set auto-refresh state
     */
    setAutoRefresh(enabled) {
        this.update({ autoRefresh: enabled });
        localStorage.setItem(CONFIG.STORAGE_KEYS.AUTO_REFRESH, enabled.toString());
    }

    /**
     * Select a credential for detail view
     */
    selectCredential(provider, credential) {
        this.update({
            selectedProvider: provider,
            selectedCredential: credential
        });
    }

    /**
     * Clear credential selection
     */
    clearSelection() {
        this.update({
            selectedProvider: null,
            selectedCredential: null
        });
    }

    /**
     * Load preferences from localStorage
     */
    loadPreferences() {
        // Theme - default to light
        const savedTheme = localStorage.getItem(CONFIG.STORAGE_KEYS.THEME) || 'light';
        document.documentElement.dataset.theme = savedTheme;

        // View mode
        const savedView = localStorage.getItem(CONFIG.STORAGE_KEYS.VIEW_MODE) || 'current';

        // Auto-refresh
        const savedAutoRefresh = localStorage.getItem(CONFIG.STORAGE_KEYS.AUTO_REFRESH);
        const autoRefresh = savedAutoRefresh !== 'false';

        this.update({
            theme: savedTheme,
            viewMode: savedView,
            autoRefresh
        });

        return { theme: savedTheme, viewMode: savedView, autoRefresh };
    }

    /**
     * Get current state (readonly)
     */
    getState() {
        return { ...this.data };
    }

    /**
     * Check if data is stale
     */
    isDataStale() {
        if (!this.data.lastUpdated) return true;
        return (Date.now() - this.data.lastUpdated) > CONFIG.STALE_THRESHOLD_MS;
    }

    /**
     * Get time since last update in seconds
     */
    getDataAge() {
        if (!this.data.lastUpdated) return null;
        return Math.floor((Date.now() - this.data.lastUpdated) / 1000);
    }
}

// Export singleton instance
export const appState = new AppState();
