// FinRAG — render.js | Markdown rendering + shared badge components
import { escHtml } from './ui.js';

/**
 * Simple markdown → HTML
 */
export function renderMarkdown(text) {
  return escHtml(text)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/`(.+?)`/g, '<code>$1</code>')
    .replace(/^&gt; (.+)$/gm, '<div style="border-left:3px solid var(--accent);padding-left:10px;color:var(--text2);margin:6px 0">$1</div>')
    .replace(/^- (.+)$/gm, '<div style="padding:2px 0">• $1</div>')
    .replace(/\n/g, '<br>');
}

// ── Badge components ──

const VERDICT_MAP = {
  bullish:  ['利好', 'var(--ok)'],
  bearish:  ['利空', 'var(--danger)'],
  neutral:  ['中性', 'var(--text2)'],
  unknown:  ['分析失败', '#f59e0b'],
};

export function verdictBadge(v) {
  const [label, color] = VERDICT_MAP[v] || VERDICT_MAP.neutral;
  return `<span class="verdict-badge" style="background:${color}18;color:${color};border-color:${color}44">${label}</span>`;
}

export function confidenceBadge(c) {
  if (!c) return '';
  const map = { high: ['高置信', 'var(--ok)'], medium: ['中置信', '#f59e0b'], low: ['低置信', 'var(--danger)'] };
  const [label, color] = map[c] || [];
  if (!label) return '';
  return `<span class="confidence-badge" style="background:${color}18;color:${color};border-color:${color}44">${label}</span>`;
}

const DIR_MAP = {
  bullish: ['↑', 'var(--ok)'], bearish: ['↓', 'var(--danger)'], neutral: ['→', 'var(--text2)'],
  positive: ['↑', 'var(--ok)'], negative: ['↓', 'var(--danger)'],
  improving: ['↑', 'var(--ok)'], deteriorating: ['↓', 'var(--danger)'],
  stable: ['→', 'var(--text2)'], mixed: ['↔', '#f59e0b'],
};

export function directionArrow(dir) {
  const [arrow, color] = DIR_MAP[dir] || ['→', 'var(--text2)'];
  return `<span style="color:${color};font-weight:700">${arrow}</span>`;
}

export function severityBar(level) {
  const n = Math.min(5, Math.max(1, level || 1));
  let bars = '';
  for (let i = 1; i <= 5; i++) {
    const color = i <= n ? (n >= 4 ? 'var(--danger)' : n >= 3 ? '#f59e0b' : 'var(--ok)') : 'var(--border)';
    bars += `<span style="display:inline-block;width:3px;height:12px;background:${color};border-radius:2px;margin-right:2px"></span>`;
  }
  return bars;
}

export function sentimentPill(s) {
  const map = { positive: ['积极', 'var(--ok)'], negative: ['消极', 'var(--danger)'], neutral: ['中性', 'var(--text2)'] };
  const [label, color] = map[s] || map.neutral;
  return `<span style="font-size:10px;padding:1px 6px;border-radius:8px;background:${color}18;color:${color}">${label}</span>`;
}
