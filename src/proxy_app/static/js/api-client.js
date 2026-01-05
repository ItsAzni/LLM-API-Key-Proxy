/**
 * API Client - Handles all communication with the proxy server
 */

import { CONFIG } from './config.js';

/**
 * Custom error for API failures
 */
export class ApiError extends Error {
    constructor(message, status) {
        super(message);
        this.name = 'ApiError';
        this.status = status;
    }
}

/**
 * API Client singleton
 */
class ApiClient {
    constructor() {
        this.apiKey = null;
        this.initialized = false;
    }

    /**
     * Initialize the client by fetching config from server
     */
    async init() {
        if (this.initialized) return;

        try {
            // Fetch API key from server config endpoint
            const response = await fetch('/dashboard/api/config');
            if (response.ok) {
                const config = await response.json();
                this.apiKey = config.apiKey;
            }
        } catch (error) {
            console.warn('Could not fetch dashboard config:', error);
        }

        this.initialized = true;
    }

    /**
     * Get headers for API requests
     */
    getHeaders() {
        const headers = {
            'Content-Type': 'application/json'
        };
        if (this.apiKey) {
            headers['Authorization'] = `Bearer ${this.apiKey}`;
        }
        return headers;
    }

    /**
     * Make an API request
     */
    async request(endpoint, options = {}) {
        // Ensure initialized
        await this.init();

        const url = `${CONFIG.API_BASE}${endpoint}`;

        try {
            const response = await fetch(url, {
                ...options,
                headers: {
                    ...this.getHeaders(),
                    ...options.headers
                }
            });

            if (!response.ok) {
                const errorBody = await response.text();
                throw new ApiError(
                    `HTTP ${response.status}: ${errorBody || response.statusText}`,
                    response.status
                );
            }

            return response.json();
        } catch (error) {
            if (error instanceof ApiError) {
                throw error;
            }
            // Network or other errors
            throw new ApiError(`Network error: ${error.message}`, 0);
        }
    }

    /**
     * Get quota stats (cached)
     */
    async getQuotaStats(provider = null) {
        const params = provider ? `?provider=${encodeURIComponent(provider)}` : '';
        return this.request(`${CONFIG.ENDPOINTS.QUOTA_STATS}${params}`);
    }

    /**
     * Refresh quota stats
     */
    async refreshQuotaStats(action = 'reload', scope = 'all', provider = null, credential = null) {
        return this.request(CONFIG.ENDPOINTS.QUOTA_STATS, {
            method: 'POST',
            body: JSON.stringify({
                action,
                scope,
                provider,
                credential
            })
        });
    }
}

// Export singleton instance
export const apiClient = new ApiClient();
