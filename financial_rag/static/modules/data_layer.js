// FinRAG — data_layer.js | 数据层可观测面板（存储归属 · 数据流 · 一致性体检）
import { apiGet } from './api.js';
import { escHtml } from './ui.js';

const LAYER_ICON = { source: '🔒', index: '⚙️', state: '📊', input: '📥' };

function fmtLive(live) {
  if (!live || !live.exists) return '<span class="dl-missing">未创建</span>';
  const count = live.count === null || live.count === undefined ? '—' : live.count;
  const err = live.error ? ` <span class="dl-err" title="${escHtml(live.error)}">读取失败</span>` : '';
  return `<span class="dl-count">${count}</span> 条 · ${live.size_kb} KB · ${escHtml(live.mtime || '')}${err}`;
}

function renderStores(m) {
  const el = document.getElementById('dlStores');
  if (!el) return;
  const byLayer = {};
  for (const s of m.stores) (byLayer[s.layer] = byLayer[s.layer] || []).push(s);
  el.innerHTML = m.layers.map(ly => {
    const stores = byLayer[ly.id] || [];
    const cards = stores.map(s => `
      <div class="dl-store ${s.live && s.live.exists ? '' : 'dl-store-missing'}">
        <div class="dl-store-head">
          <strong>${escHtml(s.name)}</strong>
          <span class="tag ${s.regenerable ? 'tag-info' : 'tag-warn'}">${s.regenerable ? '可再生' : '真相源'}</span>
        </div>
        <div class="dl-store-live">${fmtLive(s.live)}</div>
        <div class="dl-store-meta">
          <div>归属 <code>${escHtml(s.owner)}</code></div>
          <div>写入 <code>${escHtml((s.writers || []).join(' / ') || '（代码只读不写）')}</code></div>
          <div class="dl-path" title="${escHtml(s.path)}">${escHtml(s.path)}</div>
          ${s.rebuild_source ? `<div>重建自 ${escHtml(s.rebuild_source)}</div>` : ''}
        </div>
        <div class="dl-store-desc">${escHtml(s.desc)}</div>
      </div>`).join('');
    return `
      <div class="dl-layer">
        <div class="dl-layer-head">
          <span>${LAYER_ICON[ly.id] || '📁'} ${escHtml(ly.name)}</span>
          <span class="dl-layer-desc">${escHtml(ly.desc)}</span>
          <code class="dl-layer-dir">${escHtml(ly.dir)}</code>
        </div>
        <div class="dl-store-grid">${cards}</div>
      </div>`;
  }).join('');
}

function renderFlows(m) {
  const el = document.getElementById('dlFlows');
  if (!el) return;
  const nameOf = {};
  for (const s of m.stores) nameOf[s.id] = s.name;
  const flowsHtml = m.flows.map(f => `
    <div class="dl-flow">
      <div class="dl-flow-head">
        <strong>${escHtml(f.name)}</strong>
        <span class="dl-flow-trigger">${escHtml(f.trigger)}</span>
      </div>
      <div class="dl-flow-chain">
        ${f.writes_to.map((id, i) =>
          `${i > 0 ? '<span class="dl-flow-sep">·</span>' : ''}<span class="tag tag-info dl-flow-tag">${escHtml(nameOf[id] || id)}</span>`
        ).join('')}
      </div>
      <div class="dl-flow-note">${escHtml(f.note || '')}</div>
    </div>`).join('');
  const deps = m.dependencies || [];
  const depsHtml = deps.length ? `
    <div class="dl-dep-title">上游依赖（读，非写入）</div>
    ${deps.map(d => `
      <div class="dl-dep">
        <span class="dl-dep-edge"><code>${escHtml(nameOf[d.from] || d.from)}</code> → <code>${escHtml(nameOf[d.to] || d.to)}</code></span>
        <span class="dl-dep-note">${escHtml(d.note)}</span>
      </div>`).join('')}` : '';
  el.innerHTML = flowsHtml + depsHtml;
}

function renderMetaCoverage(m) {
  const el = document.getElementById('dlMeta');
  if (!el) return;
  const cov = m.metadata_coverage;
  if (!cov || !cov.total) {
    el.innerHTML = '<div class="dl-meta-empty">知识库为空，暂无元数据可统计</div>';
    return;
  }
  const rows = Object.entries(cov.fields).map(([f, v]) => {
    const cls = v.pct >= 80 ? 'ok' : v.pct >= 50 ? 'warn' : 'bad';
    return `
      <div class="dl-meta-row">
        <span class="dl-meta-field">${escHtml(f)}</span>
        <div class="dl-meta-bar"><div class="dl-meta-fill dl-meta-${cls}" style="width:${v.pct}%"></div></div>
        <span class="dl-meta-num">${v.filled}/${cov.total} · ${v.pct}%</span>
      </div>`;
  }).join('');
  el.innerHTML = `
    <div class="dl-meta-total">文档总数 <strong>${cov.total}</strong> — 各字段非空占比（入库时正则抽取，漏抽是静默的；偏低即词典/正则待补）</div>
    ${rows}`;
}

function renderHealth(m) {
  const el = document.getElementById('dlHealth');
  if (!el) return;
  const c = m.consistency;
  const chips = Object.entries(c.counts).map(([id, n]) =>
    `<span class="dl-chip">${escHtml(id)} <strong>${n === null || n === undefined ? '缺失' : n}</strong></span>`
  ).join('');
  el.innerHTML = c.ok
    ? `<div class="dl-health dl-health-ok">✅ 三存储一致性体检通过 — 真相源与派生索引计数对齐</div><div class="dl-chips">${chips}</div>`
    : `<div class="dl-health dl-health-bad">🚨 一致性体检发现 ${c.issues.length} 个问题</div>
       <div class="dl-chips">${chips}</div>
       <ul class="dl-issues">${c.issues.map(i => `<li>${escHtml(i)}</li>`).join('')}</ul>`;
}

/**
 * 加载数据层全景（GET /api/data-layer，纯只读，不触发任何重建）
 */
export async function loadDataLayer() {
  const loading = document.getElementById('dl-loading');
  try {
    if (loading) loading.classList.add('show');
    const m = await apiGet('/api/data-layer');
    renderHealth(m);
    renderMetaCoverage(m);
    renderStores(m);
    renderFlows(m);
  } catch (e) {
    const el = document.getElementById('dlHealth');
    if (el) el.innerHTML = `<div class="dl-health dl-health-bad">🚨 数据层加载失败: ${escHtml(e.message)} — 请确认服务已启动后点「刷新」</div>`;
  } finally {
    if (loading) loading.classList.remove('show');
  }
}