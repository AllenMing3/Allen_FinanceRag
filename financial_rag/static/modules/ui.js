// FinRAG — ui.js | Shared UI components

const TOAST_ICONS = { success: '✓', error: '✗', warning: '⚠', info: 'ℹ' };
let _toastContainer = null;

function _ensureToastContainer() {
  if (!_toastContainer) {
    _toastContainer = document.createElement('div');
    _toastContainer.className = 'toast-container';
    document.body.appendChild(_toastContainer);
  }
  return _toastContainer;
}

/**
 * Show toast notification
 * @param {string} msg - Message
 * @param {'success'|'error'|'warning'|'info'} type
 * @param {number} duration - Auto dismiss ms (0 = manual)
 */
export function toast(msg, type = 'info', duration = 3500) {
  const container = _ensureToastContainer();
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.innerHTML = `
    <span class="toast-icon">${TOAST_ICONS[type] || 'ℹ'}</span>
    <span class="toast-msg">${msg}</span>
    <span class="toast-close">✕</span>
  `;
  el.querySelector('.toast-close').addEventListener('click', () => _removeToast(el));
  container.appendChild(el);
  if (duration > 0) {
    setTimeout(() => _removeToast(el), duration);
  }
}

function _removeToast(el) {
  if (!el.parentNode) return;
  el.classList.add('removing');
  setTimeout(() => el.remove(), 200);
}

/**
 * Confirm dialog (replaces native confirm)
 * @param {string} title
 * @param {string} message
 * @param {object} opts - { confirmText, cancelText, danger }
 * @returns {Promise<boolean>}
 */
export function confirmDialog(title, message, opts = {}) {
  return new Promise(resolve => {
    const overlay = document.createElement('div');
    overlay.className = 'modal-overlay';
    const confirmText = opts.confirmText || '确认';
    const cancelText = opts.cancelText || '取消';
    const btnClass = opts.danger ? 'btn-danger' : 'btn-primary';
    overlay.innerHTML = `
      <div class="modal-box">
        <h3>${title}</h3>
        <p>${message}</p>
        <div class="modal-actions">
          <button class="btn btn-secondary cancel-btn">${cancelText}</button>
          <button class="btn ${btnClass} confirm-btn">${confirmText}</button>
        </div>
      </div>
    `;
    document.body.appendChild(overlay);
    overlay.querySelector('.cancel-btn').addEventListener('click', () => { overlay.remove(); resolve(false); });
    overlay.querySelector('.confirm-btn').addEventListener('click', () => { overlay.remove(); resolve(true); });
    overlay.addEventListener('click', (e) => { if (e.target === overlay) { overlay.remove(); resolve(false); } });
  });
}

/**
 * Escape HTML
 */
export function escHtml(s) {
  return (typeof s === 'string' ? s : String(s ?? ''))
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

/**
 * Score color helper
 */
export function scoreColor(s) {
  return s >= 0.8 ? 'var(--ok)' : s >= 0.6 ? 'var(--warn)' : 'var(--danger)';
}

/**
 * Rank class helper
 */
export function rankClass(s) {
  return s >= 0.7 ? 'rank-high' : s >= 0.4 ? 'rank-mid' : 'rank-low';
}

/**
 * Render empty state
 */
export function emptyState(icon, text, hint = '') {
  return `<div class="empty-state">
    <div class="icon">${icon}</div>
    <p>${text}</p>
    ${hint ? `<div class="hint">${hint}</div>` : ''}
  </div>`;
}

/**
 * Progress bar HTML
 */
export function progressBar(pct, cls = '') {
  return `<div class="progress-bar">
    <div class="progress-fill ${cls}" style="width:${Math.min(100, Math.max(0, pct))}%"></div>
  </div>`;
}
