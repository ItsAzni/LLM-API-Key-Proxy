/**
 * Main Application - Dashboard entry point (No auth required)
 */

import { CONFIG } from './config.js';
import { apiClient } from './api-client.js';
import { appState } from './state.js';
import { formatTokens, formatCost, getTotalTokens } from './utils.js';
import { renderProviderCard, renderAllCredentials } from './components/provider-card.js';
import { renderCredentialPanel } from './components/credential-panel.js';
import { showToast, showSuccess, showError } from './components/toast.js';

/**
 * Dashboard Application Class
 */
class Dashboard {
    constructor() {
        this.refreshInterval = null;
        this.dataAgeInterval = null;
        this.elements = {};
    }

    /**
     * Initialize the dashboard
     */
    async init() {
        this.cacheElements();
        this.bindEvents();
        this.loadPreferences();

        // No auth required - load data directly
        await this.loadData();
        this.startAutoRefresh();
        this.startDataAgeUpdater();
    }

    /**
     * Cache DOM elements for performance
     */
    cacheElements() {
        this.elements = {
            // Dashboard
            dashboard: document.getElementById('dashboard'),

            // Header
            refreshBtn: document.getElementById('refresh-btn'),
            themeToggle: document.getElementById('theme-toggle'),
            autoRefreshToggle: document.getElementById('auto-refresh-toggle'),

            // Summary stats
            totalRequests: document.getElementById('total-requests'),
            totalTokens: document.getElementById('total-tokens'),
            totalCost: document.getElementById('total-cost'),

            // View controls
            viewToggle: document.querySelectorAll('.view-toggle .toggle-option'),
            dataAge: document.getElementById('data-age'),
            dataSource: document.getElementById('data-source'),

            // Main content
            providersContainer: document.getElementById('providers-container'),

            // Drawer
            credentialDrawer: document.getElementById('credential-drawer'),
            drawerContent: document.getElementById('drawer-content'),
            drawerBackdrop: document.getElementById('drawer-backdrop'),
            closeDrawer: document.querySelector('.close-drawer'),

            // Toast
            toastContainer: document.getElementById('toast-container')
        };
    }

    /**
     * Bind event listeners
     */
    bindEvents() {
        // Theme toggle
        this.elements.themeToggle?.addEventListener('click', () => {
            appState.toggleTheme();
        });

        // Refresh button
        this.elements.refreshBtn?.addEventListener('click', () => this.handleRefresh());

        // Auto-refresh toggle
        this.elements.autoRefreshToggle?.addEventListener('change', (e) => {
            appState.setAutoRefresh(e.target.checked);
            if (e.target.checked) {
                this.startAutoRefresh();
            } else {
                this.stopAutoRefresh();
            }
        });

        // View mode toggle
        this.elements.viewToggle?.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const view = e.target.dataset.view;
                if (view) this.setViewMode(view);
            });
        });

        // Provider card clicks (event delegation)
        this.elements.providersContainer?.addEventListener('click', (e) => {
            const credItem = e.target.closest('.credential-item');
            const showMoreBtn = e.target.closest('.show-more-btn');

            if (credItem) {
                this.showCredentialDetail(
                    credItem.dataset.provider,
                    credItem.dataset.credential
                );
            } else if (showMoreBtn) {
                this.handleShowMoreCredentials(showMoreBtn.dataset.provider);
            }
        });

        // Keyboard navigation for credential items
        this.elements.providersContainer?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                const credItem = e.target.closest('.credential-item');
                if (credItem) {
                    e.preventDefault();
                    this.showCredentialDetail(
                        credItem.dataset.provider,
                        credItem.dataset.credential
                    );
                }
            }
        });

        // Drawer close
        this.elements.closeDrawer?.addEventListener('click', () => {
            this.hideCredentialDrawer();
        });

        this.elements.drawerBackdrop?.addEventListener('click', () => {
            this.hideCredentialDrawer();
        });

        // Escape key to close drawer
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && !this.elements.credentialDrawer?.hidden) {
                this.hideCredentialDrawer();
            }
        });

        // Subscribe to state changes
        appState.subscribe((state) => this.render(state));

        // Refresh on window focus
        if (CONFIG.REFRESH_ON_FOCUS) {
            window.addEventListener('focus', () => {
                if (appState.isDataStale()) {
                    this.loadData();
                }
            });
        }
    }

    /**
     * Load preferences from localStorage
     */
    loadPreferences() {
        const { theme, viewMode, autoRefresh } = appState.loadPreferences();

        // Update UI to match preferences
        if (this.elements.autoRefreshToggle) {
            this.elements.autoRefreshToggle.checked = autoRefresh;
        }

        // Update view toggle UI
        this.elements.viewToggle?.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.view === viewMode);
        });
    }

    /**
     * Load data from API
     */
    async loadData() {
        appState.setLoading(true);

        try {
            const stats = await apiClient.getQuotaStats();
            appState.setQuotaStats(stats);
        } catch (error) {
            appState.setError(error.message);
            showError('Failed to load data: ' + error.message);
        }
    }

    /**
     * Handle manual refresh
     */
    async handleRefresh() {
        this.elements.refreshBtn?.classList.add('refreshing');

        try {
            const stats = await apiClient.refreshQuotaStats('reload');
            appState.setQuotaStats(stats);
            showSuccess('Data refreshed');
        } catch (error) {
            showError('Refresh failed: ' + error.message);
        } finally {
            this.elements.refreshBtn?.classList.remove('refreshing');
        }
    }

    /**
     * Start auto-refresh interval
     */
    startAutoRefresh() {
        this.stopAutoRefresh();
        if (appState.data.autoRefresh) {
            this.refreshInterval = setInterval(() => {
                this.loadData();
            }, CONFIG.REFRESH_INTERVAL_MS);
        }
    }

    /**
     * Stop auto-refresh
     */
    stopAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
        }
    }

    /**
     * Start data age updater
     */
    startDataAgeUpdater() {
        this.stopDataAgeUpdater();
        this.dataAgeInterval = setInterval(() => {
            this.updateDataAge();
        }, 1000);
    }

    /**
     * Stop data age updater
     */
    stopDataAgeUpdater() {
        if (this.dataAgeInterval) {
            clearInterval(this.dataAgeInterval);
            this.dataAgeInterval = null;
        }
    }

    /**
     * Update data age display
     */
    updateDataAge() {
        const age = appState.getDataAge();
        if (age !== null && this.elements.dataAge) {
            this.elements.dataAge.textContent = `${age}s ago`;
            this.elements.dataAge.classList.toggle('stale', appState.isDataStale());
        }
    }

    /**
     * Set view mode
     */
    setViewMode(mode) {
        appState.setViewMode(mode);

        // Update toggle UI
        this.elements.viewToggle?.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.view === mode);
        });
    }

    /**
     * Handle show more credentials button
     */
    handleShowMoreCredentials(providerName) {
        const stats = appState.data.quotaStats;
        if (!stats?.providers?.[providerName]) return;

        const credentials = stats.providers[providerName].credentials || [];
        const card = this.elements.providersContainer.querySelector(
            `.provider-card[data-provider="${providerName}"]`
        );

        if (!card) return;

        const credList = card.querySelector('.credentials-list');
        if (!credList) return;

        // Replace credentials section with all credentials
        const header = credList.querySelector('.credentials-header');
        credList.innerHTML = '';
        if (header) {
            credList.appendChild(header.cloneNode(true));
        }
        credList.insertAdjacentHTML('beforeend', renderAllCredentials(providerName, credentials));
    }

    /**
     * Render the dashboard based on state
     */
    render(state) {
        this.renderSummary(state);
        this.renderProviders(state);
        this.renderDataInfo(state);
        this.renderDrawer(state);
    }

    /**
     * Render summary stats in header
     */
    renderSummary(state) {
        const stats = state.quotaStats;
        if (!stats) return;

        const summary = state.viewMode === 'global' && stats.global_summary
            ? stats.global_summary
            : stats.summary;

        if (!summary) return;

        if (this.elements.totalRequests) {
            this.elements.totalRequests.textContent = (summary.total_requests || 0).toLocaleString();
        }

        if (this.elements.totalTokens) {
            const tokens = summary.tokens || {};
            const total = getTotalTokens(tokens);
            this.elements.totalTokens.textContent = formatTokens(total);
        }

        if (this.elements.totalCost) {
            this.elements.totalCost.textContent = formatCost(summary.approx_total_cost);
        }
    }

    /**
     * Render provider cards
     */
    renderProviders(state) {
        const stats = state.quotaStats;

        if (!stats?.providers) {
            if (state.isLoading) {
                this.elements.providersContainer.innerHTML = `
                    <div class="loading-state glass-card">
                        <div class="spinner"></div>
                        <p>Loading providers...</p>
                    </div>
                `;
            }
            return;
        }

        const providers = Object.entries(stats.providers);

        if (providers.length === 0) {
            this.elements.providersContainer.innerHTML = `
                <div class="empty-state glass-card">
                    <p>No providers configured</p>
                    <p class="hint">Add API keys or OAuth credentials to get started.</p>
                </div>
            `;
            return;
        }

        this.elements.providersContainer.innerHTML = providers
            .map(([name, data]) => renderProviderCard(name, data, state.viewMode))
            .join('');
    }

    /**
     * Render data info (source and age)
     */
    renderDataInfo(state) {
        if (state.quotaStats && this.elements.dataSource) {
            this.elements.dataSource.textContent = state.quotaStats.data_source || 'cache';
        }
    }

    /**
     * Render credential drawer if a credential is selected
     */
    renderDrawer(state) {
        if (state.selectedCredential && state.selectedProvider) {
            this.elements.drawerContent.innerHTML = renderCredentialPanel(
                state.selectedProvider,
                state.selectedCredential,
                state.viewMode
            );
        }
    }

    /**
     * Show credential detail drawer
     */
    showCredentialDetail(providerName, credentialPath) {
        const stats = appState.data.quotaStats;
        if (!stats?.providers?.[providerName]) return;

        const credential = stats.providers[providerName].credentials
            ?.find(c => c.full_path === credentialPath || c.identifier === credentialPath);

        if (!credential) return;

        appState.selectCredential(providerName, credential);
        this.elements.credentialDrawer.hidden = false;
        this.elements.drawerBackdrop.hidden = false;

        // Focus management for accessibility
        this.elements.closeDrawer?.focus();
    }

    /**
     * Hide credential drawer
     */
    hideCredentialDrawer() {
        this.elements.credentialDrawer.hidden = true;
        this.elements.drawerBackdrop.hidden = true;
        appState.clearSelection();
    }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    const dashboard = new Dashboard();
    dashboard.init();
});
