// FinRAG — overview.js | System overview panel
import { apiGet, api } from './api.js';
import { escHtml } from './ui.js';

/**
 * Update KB badge in topbar
 */
export function updateKBStatus(count) {
  const badge = document.getElementById('kbBadge');
  const status = document.getElementById('kbStatus');
  const docCount = document.getElementById('kbDocCount');
  const ingestCount = document.getElementById('ingestCount');
  if (ingestCount) ingestCount.textContent = count;
  if (count > 0) {
    badge.querySelector('.dot').className = 'dot dot-ok';
    status.textContent = `${count} 篇文档`;
    if (docCount) { docCount.textContent = `${count} 篇文档`; docCount.className = 'tag tag-ok'; }
  } else {
    badge.querySelector('.dot').className = 'dot dot-off';
    status.textContent = '未构建知识库';
    if (docCount) { docCount.textContent = '0 篇文档'; docCount.className = 'tag tag-info'; }
  }
}

export function updateMetaStatus(count) {
  const el = document.getElementById('metaStatus');
  if (el) el.textContent = `元数据: ${count}`;
}

/**
 * Update KB dashboard widget
 */
export function updateKBDashboard(data) {
  const docs = data.doc_count || 0;
  const built = data.kb_built || false;
  const size = data.file_size_kb || 0;
  const meta = data.meta_count || 0;
  const el = (id) => document.getElementById(id);
  if (el('kbDashDocs')) el('kbDashDocs').textContent = docs;
  if (el('kbDashIndex')) el('kbDashIndex').innerHTML = built
    ? '<span class="tag tag-ok">✓ 已构建</span>'
    : '<span class="tag tag-warn">未构建</span>';
  if (el('kbDashSize')) el('kbDashSize').textContent = size + ' KB';
  if (el('kbDashMeta')) el('kbDashMeta').textContent = meta;
}

/**
 * Render health diagnostic banners
 */
export function renderHealthBanners(kbStatus, initErrors) {
  const container = document.getElementById('healthBanners');
  if (!container) return;
  const banners = [];

  if (kbStatus) {
    if (kbStatus.state === 'failed') {
      banners.push({ cls: 'hb-error', icon: '🚨', msg: `知识库索引失败: ${kbStatus.reason}`, action: '前往数据管理', panel: 'data' });
    } else if (kbStatus.state === 'empty') {
      banners.push({ cls: 'hb-warn', icon: '📚', msg: '知识库为空 — 查询和分析功能将受限', action: '前往导入', panel: 'data' });
    }
  }

  for (const err of initErrors) {
    if (err.component === 'kb_build') continue;
    banners.push({
      cls: err.severity === 'critical' ? 'hb-error' : 'hb-warn',
      icon: err.severity === 'critical' ? '🚨' : '⚠️',
      msg: `${err.component}: ${err.error}`,
    });
  }

  if (banners.length === 0) {
    container.classList.remove('active');
    container.innerHTML = '';
    return;
  }

  container.innerHTML = banners.map((b, i) => `
    <div class="health-banner ${b.cls}">
      <span class="hb-icon">${b.icon}</span>
      <span class="hb-msg">${escHtml(b.msg)}</span>
      ${b.action ? `<button class="hb-action" data-panel="${b.panel}">${escHtml(b.action)}</button>` : ''}
    </div>
  `).join('');
  container.classList.add('active');

  container.querySelectorAll('.hb-action').forEach(btn => {
    btn.addEventListener('click', () => {
      const panel = btn.dataset.panel;
      if (panel) {
        const navItem = document.querySelector(`.nav-item[data-panel="${panel}"]`);
        if (navItem) navItem.click();
      }
    });
  });
}

/**
 * Load overview live stats
 */
export async function loadOverviewStats() {
  try {
    const setTxt = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
    const [cfg, kb] = await Promise.all([
      apiGet('/api/config').catch(() => null),
      apiGet('/api/kb/status').catch(() => null),
    ]);
    if (cfg) {
      setTxt('statModel', (cfg.llm_model || '-').replace('qwen-', 'QW-'));
      if (cfg.tool_count) setTxt('statTools', cfg.tool_count);
      if (cfg.agent_count) setTxt('statAgents', cfg.agent_count);
    }
    if (kb) {
      const docCount = kb.doc_count ?? 0;
      setTxt('statDocs', docCount);
      setTxt('statIndex', kb.kb_built ? 'Built' : 'Not built');
      setTxt('kblDocs', docCount + ' 篇');
      setTxt('kblIndex', kb.kb_built ? '✅ 已构建' : '⏳ 未构建');
      setTxt('kblChroma', kb.kb_built ? '✅ Active' : '⏳ Inactive');
      setTxt('kblMeta', (kb.meta_count ?? 0) + ' 条');
      const sources = kb.sources || {};
      const keys = Object.keys(sources);
      const srcEl = document.getElementById('kblSources');
      if (srcEl) {
        srcEl.innerHTML = keys.length > 0
          ? '<strong style="color:var(--text1)">来源分布：</strong> ' + keys.map(k => `${escHtml(k)}: ${sources[k]}`).join(' · ')
          : '<span style="color:var(--text3)">暂无文档来源</span>';
      }
    }
  } catch (e) {
    console.warn('Overview stats load failed:', e);
  }
}

/**
 * Refresh KB status (called globally)
 */
export async function refreshKBStatus() {
  try {
    const d = await apiGet('/api/kb/status');
    if (d.doc_count > 0) updateKBStatus(d.doc_count);
    if (d.meta_count > 0) updateMetaStatus(d.meta_count);
    updateKBDashboard(d);
  } catch (e) { /* silent */ }
}
