// FinRAG — kb.js | Knowledge base management
import { api, apiGet, apiDelete, showLoading, hideLoading } from './api.js';
import { toast, confirmDialog, escHtml, scoreColor, rankClass, emptyState } from './ui.js';
import { refreshKBStatus, updateKBStatus, updateMetaStatus, updateKBDashboard } from './overview.js';

// ── Document list ──

export function renderDocList(docs) {
  const el = document.getElementById('docList');
  if (!el) return;
  if (!docs || !docs.length) {
    el.innerHTML = emptyState('📭', '知识库为空', '先在「数据管理」导入数据');
    return;
  }
  let h = '';
  docs.forEach((d, i) => {
    const src = d.meta?.source || 'unknown';
    const preview = (d.text || '').slice(0, 120);
    h += `<div class="doc-item">
      <div class="doc-meta">#${i + 1}</div>
      <div class="doc-text"><span class="doc-source-tag">${escHtml(src)}</span>${escHtml(preview)}${d.text.length > 120 ? '...' : ''}</div>
    </div>`;
  });
  el.innerHTML = h;
}

// ── KB Manager ──

export async function refreshKBManager() {
  const container = document.getElementById('kbSourceList');
  if (!container) return;
  try {
    const d = await apiGet('/api/kb/status');
    const sources = d.sources || {};
    const keys = Object.keys(sources);
    if (!keys.length) {
      container.innerHTML = '<div style="padding:12px;text-align:center;color:var(--text3);font-size:12px">知识库为空</div>';
      return;
    }
    let h = `<div style="font-size:12px;color:var(--text2);margin-bottom:8px">📚 ${d.doc_count} 篇文档 · ${d.file_size_kb} KB · ${d.analyzed_count} 已分析</div>`;
    h += '<div style="display:flex;flex-direction:column;gap:4px">';
    keys.sort((a, b) => sources[b] - sources[a]);
    for (const src of keys) {
      const count = sources[src];
      const safeSrc = src.replace(/'/g, "\\'").replace(/"/g, '&quot;');
      h += `<div class="kb-source-item">
        <span class="src-name" title="${escHtml(src)}">${escHtml(src)}</span>
        <span class="tag tag-info" style="margin:0 8px">${count} 篇</span>
        <button class="btn btn-xs btn-danger" onclick="window._removeKBSource('${safeSrc}')">删除</button>
      </div>`;
    }
    h += '</div>';
    container.innerHTML = h;
  } catch (e) {
    container.innerHTML = `<div style="color:var(--danger);font-size:12px">加载失败: ${escHtml(e.message)}</div>`;
  }
}

// Expose to global for onclick handlers
window._removeKBSource = async function (source) {
  const ok = await confirmDialog('删除来源', `确认删除来源「${source}」的所有文档？`, { danger: true, confirmText: '删除' });
  if (!ok) return;
  try {
    await apiDelete(`/api/kb/source/${encodeURIComponent(source)}`);
    toast(`已删除来源: ${source}`, 'success');
    refreshKBManager();
    refreshKBStatus();
  } catch (e) { toast('删除失败: ' + e.message, 'error'); }
};

// ── Build ──

export async function buildKB() {
  showLoading('build-loading');
  try {
    const d = await api('/api/build', {});
    window._kbBuilt = true;
    const bcEl = document.getElementById('buildCount');
    if (bcEl) bcEl.textContent = d.doc_count;
    let h = `<div class="stats">
      <div class="stat">文档数 <strong>${d.doc_count}</strong></div>
      <div class="stat">BM25 <strong>${d.bm25_terms || '-'} terms</strong></div>
      <div class="stat">Embedding <strong>${d.embedding_dim || 'N/A'} 维</strong></div>
      <div class="stat">耗时 <strong>${d.elapsed_ms}ms</strong></div>
    </div>`;
    if (d.test_queries && d.test_queries.length) {
      h += '<div style="margin-top:12px;font-size:12px;color:var(--text2)">验证检索:</div>';
      d.test_queries.forEach(tq => {
        h += `<div style="margin-top:4px;font-size:12px"><strong>${escHtml(tq.query)}</strong>: `;
        tq.results.forEach(r => { h += `<span style="color:${scoreColor(r.score)}">[${r.score.toFixed(3)}]</span> `; });
        h += '</div>';
      });
    }
    document.getElementById('build-result').innerHTML = h;
    toast('索引构建完成', 'success');
    refreshKBStatus();
  } catch (e) {
    document.getElementById('build-result').innerHTML = `<div style="margin-top:8px"><span class="tag tag-fail">Error</span> ${escHtml(e.message)}</div>`;
    toast('构建失败: ' + e.message, 'error');
  }
  hideLoading('build-loading');
}

export async function clearKB() {
  const ok = await confirmDialog('清空知识库', '此操作将删除所有文档和索引，不可恢复。', { danger: true, confirmText: '清空' });
  if (!ok) return;
  try {
    await fetch('/api/kb/clear', { method: 'POST' });
    await fetch('/api/metadata/clear', { method: 'POST' });
  } catch (e) { /* silent */ }
  window._kbDocs = [];
  window._kbBuilt = false;
  updateKBStatus(0);
  updateMetaStatus(0);
  renderDocList([]);
  refreshKBManager();
  updateKBDashboard({ doc_count: 0, kb_built: false, file_size_kb: 0, meta_count: 0 });
  const bcEl = document.getElementById('buildCount');
  if (bcEl) bcEl.textContent = '0';
  const buildEl = document.getElementById('build-result');
  if (buildEl) buildEl.innerHTML = '';
  const queryEl = document.getElementById('query-result');
  if (queryEl) queryEl.innerHTML = '';
  toast('知识库已清空', 'info');
}

// ── Keyword search / delete ──

export async function searchKBKeyword() {
  const kw = document.getElementById('kbKeywordInput')?.value.trim();
  const box = document.getElementById('kbSearchResults');
  if (!kw || !box) { if (box) box.style.display = 'none'; return; }
  try {
    const d = await apiGet(`/api/kb/search?keyword=${encodeURIComponent(kw)}`);
    if (d.matched === 0) {
      box.innerHTML = `<div style="padding:8px;font-size:12px;color:var(--text2)">未找到包含「${escHtml(kw)}」的文档</div>`;
    } else {
      let html = `<div style="padding:8px;font-size:12px;color:var(--text2)">找到 <strong>${d.matched}</strong> 篇包含「${escHtml(kw)}」的文档：</div>`;
      html += '<div style="max-height:160px;overflow-y:auto;padding:0 8px">';
      for (const m of d.matches) {
        html += `<div style="font-size:11px;padding:4px 0;border-bottom:1px solid var(--border)">
          <span style="color:var(--text3)">[${escHtml(m.source)}]</span> ${escHtml((m.preview || '').slice(0, 80))}...
        </div>`;
      }
      html += '</div>';
      box.innerHTML = html;
    }
    box.style.display = 'block';
  } catch (e) {
    box.innerHTML = `<div style="padding:8px;color:var(--danger)">${e.message}</div>`;
    box.style.display = 'block';
  }
}

export async function deleteKBKeyword() {
  const kw = document.getElementById('kbKeywordInput')?.value.trim();
  if (!kw) return;
  const d = await apiGet(`/api/kb/search?keyword=${encodeURIComponent(kw)}`);
  if (d.matched === 0) { toast(`未找到包含「${kw}」的文档`, 'warning'); return; }
  const ok = await confirmDialog('删除匹配文档', `确认删除 ${d.matched} 篇包含「${kw}」的知识库文档？此操作不可恢复。`, { danger: true, confirmText: '删除' });
  if (!ok) return;
  try {
    const r = await apiDelete(`/api/kb/keyword/${encodeURIComponent(kw)}`);
    toast(`已删除 ${r.removed} 篇，剩余 ${r.remaining} 篇`, 'success');
    document.getElementById('kbSearchResults').style.display = 'none';
    document.getElementById('kbKeywordInput').value = '';
    refreshKBManager();
    refreshKBStatus();
  } catch (e) { toast('删除失败: ' + e.message, 'error'); }
}
