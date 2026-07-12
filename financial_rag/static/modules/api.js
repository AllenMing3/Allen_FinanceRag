// FinRAG — api.js | Unified API layer

/**
 * POST JSON request with error handling
 * @param {string} path - API endpoint
 * @param {object} body - Request body
 * @returns {Promise<object>}
 */
export async function api(path, body) {
  const resp = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const e = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(e.detail || resp.statusText);
  }
  return resp.json();
}

/**
 * GET JSON request
 * @param {string} path - API endpoint
 * @returns {Promise<object>}
 */
export async function apiGet(path) {
  const resp = await fetch(path);
  if (!resp.ok) {
    const e = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(e.detail || resp.statusText);
  }
  return resp.json();
}

/**
 * DELETE request
 */
export async function apiDelete(path) {
  const resp = await fetch(path, { method: 'DELETE' });
  return resp.json();
}

/**
 * Show loading indicator
 */
export function showLoading(id) {
  const el = document.getElementById(id);
  if (el) el.classList.add('show');
}

/**
 * Hide loading indicator
 */
export function hideLoading(id) {
  const el = document.getElementById(id);
  if (el) el.classList.remove('show');
}
