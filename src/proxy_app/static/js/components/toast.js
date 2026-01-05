/**
 * Toast Component - Notification system
 */

import { CONFIG } from '../config.js';
import { generateId } from '../utils.js';

/**
 * Show a toast notification
 * @param {string} message - Message to display
 * @param {string} type - 'success', 'error', 'warning', or 'info'
 * @param {number} duration - Duration in ms (default from config)
 */
export function showToast(message, type = 'info', duration = CONFIG.TOAST_DURATION) {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const id = generateId();
    const icons = {
        success: '&#x2714;',  // Checkmark
        error: '&#x2718;',    // X mark
        warning: '&#x26A0;',  // Warning triangle
        info: '&#x2139;'      // Info circle
    };

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.id = `toast-${id}`;
    toast.setAttribute('role', 'alert');
    toast.innerHTML = `
        <span class="toast-icon">${icons[type] || icons.info}</span>
        <span class="toast-message">${escapeHtml(message)}</span>
    `;

    container.appendChild(toast);

    // Auto-remove after duration
    setTimeout(() => {
        removeToast(id);
    }, duration);

    return id;
}

/**
 * Remove a toast by ID
 */
export function removeToast(id) {
    const toast = document.getElementById(`toast-${id}`);
    if (!toast) return;

    toast.classList.add('removing');
    setTimeout(() => {
        toast.remove();
    }, CONFIG.ANIMATION_DURATION);
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

/**
 * Show success toast
 */
export function showSuccess(message) {
    return showToast(message, 'success');
}

/**
 * Show error toast
 */
export function showError(message) {
    return showToast(message, 'error');
}

/**
 * Show warning toast
 */
export function showWarning(message) {
    return showToast(message, 'warning');
}

/**
 * Show info toast
 */
export function showInfo(message) {
    return showToast(message, 'info');
}
