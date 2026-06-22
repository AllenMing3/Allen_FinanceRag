// Financial RAG — App JS

// ===== State =====
let kbDocs = [];
let kbBuilt = false;

// ===== Navigation =====
document.querySelectorAll('.flow-step').forEach(el => {
  el.addEventListener('click', () => {
    document.querySelectorAll('.flow-step').forEach(n => n.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    el.classList.add('active');
    document.getElementById('panel-' + el.dataset.panel).classList.add('active');
    // Auto-refresh learning history when switching to analyze tab
    if (el.dataset.panel === 'analyze') refreshLearningHistory();
  });
});

// ===== Helpers =====
async function api(path, body) {
  const resp = await fetch(path, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });
  if (!resp.ok) { const e = await resp.json().catch(() => ({detail: resp.statusText})); throw new Error(e.detail || resp.statusText); }
  return resp.json();
}
function showLoading(id) { document.getElementById(id).classList.add('show'); }
function hideLoading(id) { document.getElementById(id).classList.remove('show'); }
function scoreColor(s) { return s >= 0.8 ? 'var(--ok)' : s >= 0.6 ? 'var(--warn)' : 'var(--danger)'; }
function rankClass(s) { return s >= 0.7 ? 'rank-high' : s >= 0.4 ? 'rank-mid' : 'rank-low'; }
function escHtml(s) { return (typeof s === 'string' ? s : String(s ?? '')).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function updateKBStatus(count) {
  const badge = document.getElementById('kbBadge');
  const status = document.getElementById('kbStatus');
  const docCount = document.getElementById('kbDocCount');
  const ingestCount = document.getElementById('ingestCount');
  const buildCount = document.getElementById('buildCount');
  ingestCount.textContent = count;
  if (count > 0) {
    badge.querySelector('.dot').className = 'dot dot-ok';
    status.textContent = `${count} 篇文档`;
    docCount.textContent = `${count} 篇文档`;
    docCount.className = 'tag tag-ok';
  } else {
    badge.querySelector('.dot').className = 'dot dot-off';
    status.textContent = '未构建知识库';
    docCount.textContent = '0 篇文档';
    docCount.className = 'tag tag-info';
  }
}

function updateMetaStatus(count) {
  document.getElementById('metaStatus').textContent = `元数据: ${count}`;
}

function showKBStorage(path, size, built) {
  // Legacy compat — redirect to dashboard update
}

function updateKBDashboard(data) {
  const docs = data.doc_count || 0;
  const built = data.kb_built || false;
  const size = data.file_size_kb || 0;
  const meta = data.meta_count || 0;
  const el = (id) => document.getElementById(id);
  if (el('kbDashDocs')) el('kbDashDocs').textContent = docs;
  if (el('kbDashIndex')) el('kbDashIndex').innerHTML = built
    ? '<span class="tag tag-ok" style="font-size:11px">✓ 已构建</span>'
    : '<span class="tag tag-warn" style="font-size:11px">未构建</span>';
  if (el('kbDashSize')) el('kbDashSize').textContent = size + ' KB';
  if (el('kbDashMeta')) el('kbDashMeta').textContent = meta;
}

function renderDocList(docs) {
  const el = document.getElementById('docList');
  if (!docs || !docs.length) {
    el.innerHTML = '<div class="empty-state"><div class="icon">📭</div><p>知识库为空</p><div class="hint">先在「数据摄取」步骤导入数据</div></div>';
    return;
  }
  let h = '';
  docs.forEach((d, i) => {
    const src = d.meta?.source || 'unknown';
    const preview = (d.text || '').slice(0, 120);
    h += `<div class="doc-item">
      <div class="doc-meta">#${i+1}</div>
      <div class="doc-text"><span class="doc-source-tag">${escHtml(src)}</span>${escHtml(preview)}${d.text.length > 120 ? '...' : ''}</div>
    </div>`;
  });
  el.innerHTML = h;
}

// ===== STEP 1: Ingest =====
// ===== Ingest state =====
const _dirData = {};  // dirPath → files array
const _selectedFiles = {};  // dirPath → Set of selected filenames

async function loadDirBrowser() {
  try {
    const d = await fetch('/api/directories').then(r => r.json());
    d.directories.forEach(dir => {
      if (dir.exists) {
        _dirData[dir.path] = dir.files;
        _selectedFiles[dir.path] = new Set(dir.files.map(f => f.name));
      }
    });
    renderDirBrowser(d.directories);
  } catch(e) {
    document.getElementById('dirBrowser').innerHTML =
      `<div style="color:var(--danger)">加载目录失败: ${escHtml(e.message)}</div>`;
  }
}

function _getIngestMode() {
  const radio = document.querySelector('input[name="ingestMode"]:checked');
  return radio ? radio.value : 'analyze';
}

function renderDirBrowser(dirs) {
  const el = document.getElementById('dirBrowser');
  let h = '';
  dirs.forEach(dir => {
    const icon = dir.exists ? '📂' : '📭';
    const status = dir.exists ? `${dir.file_count} 个文件 · ${dir.total_size_kb} KB` : '目录不存在';
    const dirId = dir.path.replace(/[^a-zA-Z0-9]/g, '_');
    h += `<div class="dir-card" id="dirCard_${dirId}">`;
    // Header row: label + status + batch actions
    h += `<div class="dir-header"><div>`;
    h += `<span style="font-size:18px">${icon}</span> <strong>${escHtml(dir.label)}</strong>`;
    h += `<span style="font-size:12px;color:var(--text2);margin-left:8px">${escHtml(dir.path)}</span>`;
    h += `</div><div style="display:flex;align-items:center;gap:8px">`;
    h += `<span class="tag ${dir.file_count > 0 ? 'tag-info' : 'tag-off'}">${status}</span>`;
    h += `</div></div>`;
    // File list with checkboxes
    if (dir.exists && dir.files.length > 0) {
      h += `<div style="display:flex;align-items:center;gap:10px;padding:4px 0;margin-bottom:4px">`;
      h += `<label style="font-size:11px;cursor:pointer;display:flex;align-items:center;gap:4px">`;
      h += `<input type="checkbox" id="selAll_${dirId}" checked onchange="toggleSelectAll('${escHtml(dir.path)}', this.checked)"> 全选</label>`;
      h += `<span style="font-size:11px;color:var(--text2)" id="selCount_${dirId}">${dir.files.length}/${dir.files.length} 已选</span>`;
      h += `<button class="btn btn-primary btn-sm" style="margin-left:auto;font-size:11px" onclick="ingestSelected('${escHtml(dir.path)}', this)">📥 导入所选</button>`;
      h += `</div>`;
      h += '<div class="dir-files">';
      dir.files.forEach(f => {
        const typeIcon = f.ext === '.jsonl' ? '📝' : f.ext === '.json' ? '📦' : f.ext === '.txt' ? '📄' : '📎';
        const lineInfo = f.line_count > 0 ? ` · ${f.line_count} 条` : '';
        const previewId = `preview_${dirId}_${f.name.replace(/[^a-zA-Z0-9]/g, '_')}`;
        h += `<div class="dir-file" style="flex-wrap:wrap">`;
        h += `<label style="display:flex;align-items:center;gap:6px;flex:1;cursor:pointer;min-width:0">`;
        h += `<input type="checkbox" checked data-dir="${escHtml(dir.path)}" data-file="${escHtml(f.name)}" onchange="updateSelCount('${escHtml(dir.path)}', '${dirId}')">`;
        h += `<span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${typeIcon} ${escHtml(f.name)}</span>`;
        h += `</label>`;
        h += `<span style="color:var(--text2);flex-shrink:0">${f.size_kb} KB${lineInfo}</span>`;
        h += `<button class="btn btn-sm" style="padding:1px 6px;font-size:10px;color:var(--accent)" onclick="previewFile('${escHtml(dir.path)}', '${escHtml(f.name)}', '${previewId}')">👁️</button>`;
        h += `<div id="${previewId}" style="display:none;width:100%;padding:6px 8px;margin-top:4px;background:var(--bg);border-radius:6px;font-size:11px;font-family:monospace;white-space:pre-wrap;max-height:200px;overflow-y:auto;border:1px solid var(--border)"></div>`;
        h += `</div>`;
      });
      h += '</div>';
    }
    h += '</div>';
  });
  el.innerHTML = h;
}

function toggleSelectAll(dirPath, checked) {
  const checkboxes = document.querySelectorAll(`input[data-dir="${dirPath}"]`);
  checkboxes.forEach(cb => cb.checked = checked);
  if (checked) {
    _selectedFiles[dirPath] = new Set(_dirData[dirPath].map(f => f.name));
  } else {
    _selectedFiles[dirPath] = new Set();
  }
  const dirId = dirPath.replace(/[^a-zA-Z0-9]/g, '_');
  updateSelCount(dirPath, dirId);
}

function updateSelCount(dirPath, dirId) {
  const checkboxes = document.querySelectorAll(`input[data-dir="${dirPath}"]`);
  const selected = Array.from(checkboxes).filter(cb => cb.checked);
  const countEl = document.getElementById(`selCount_${dirId}`);
  if (countEl) countEl.textContent = `${selected.length}/${checkboxes.length} 已选`;
  const selAllEl = document.getElementById(`selAll_${dirId}`);
  if (selAllEl) selAllEl.checked = selected.length === checkboxes.length;
  // Update _selectedFiles
  _selectedFiles[dirPath] = new Set(selected.map(cb => cb.dataset.file));
}

async function previewFile(dirPath, fileName, previewId) {
  const el = document.getElementById(previewId);
  if (!el) return;
  if (el.style.display !== 'none' && el.innerHTML) {
    el.style.display = 'none';
    return;
  }
  el.innerHTML = '<span style="color:var(--text2)">加载中...</span>';
  el.style.display = 'block';
  try {
    const d = await fetch(`/api/file/preview?path=${encodeURIComponent(dirPath)}&file=${encodeURIComponent(fileName)}&lines=15`).then(r => r.json());
    if (d.lines && d.lines.length) {
      el.textContent = d.lines.join('\n');
      if (d.truncated) el.textContent += '\n... (更多内容省略)';
    } else {
      el.textContent = '(空文件)';
    }
  } catch(e) {
    el.innerHTML = `<span style="color:var(--danger)">预览失败: ${escHtml(e.message)}</span>`;
  }
}

async function ingestSelected(dirPath, btn) {
  const files = Array.from(_selectedFiles[dirPath] || []);
  if (files.length === 0) { alert('请至少选择一个文件'); return; }
  const analyze = _getIngestMode() === 'analyze';
  if (btn) { btn.disabled = true; btn.textContent = '导入中...'; }
  try {
    const d = await api('/api/ingest/files', { dir: dirPath, analyze, files });
    updateKBStatus(d.total || 0);
    if (btn) {
      const skipInfo = d.skipped_duplicates > 0 ? ` (跳过 ${d.skipped_duplicates} 重复)` : '';
      btn.textContent = `✓ ${d.loaded || 0} 篇已导入${skipInfo}`; btn.className = 'btn btn-success btn-sm';
    }
    if (d.status === 'analyzing_in_background') _pollIngestProgress(btn);
    refreshKBStatus();
  } catch(e) { if (btn) { btn.disabled = false; btn.textContent = '📥 导入所选'; } document.getElementById('ingest-file-result').innerHTML = `<div style="margin-top:8px"><span class="tag tag-fail">Error</span> ${escHtml(e.message)}</div>`; }
}

async function _pollIngestProgress(btn) {
  const resultEl = document.getElementById('ingest-file-result');
  const poll = setInterval(async () => {
    try {
      const p = await fetch('/api/ingest/progress').then(r => r.json());
      const pct = p.total > 0 ? Math.round(p.current / p.total * 100) : 0;
      if (resultEl) {
        resultEl.innerHTML = `<div style="margin-top:8px;font-size:13px">
          <span class="tag tag-info">分析中</span> ${p.current}/${p.total} (${pct}%) · 已分析 ${p.analyzed} · 失败 ${p.errors}
        </div>`;
      }
      if (btn) btn.textContent = `分析中 ${pct}%`;
      if (!p.running) {
        clearInterval(poll);
        if (btn) { btn.textContent = `✓ 完成: ${p.analyzed}/${p.total} 已分析`; btn.disabled = false; }
        if (resultEl) {
          resultEl.innerHTML = `<div style="margin-top:8px"><span class="tag tag-ok">完成</span> ${p.analyzed}/${p.total} 篇已分析，${p.errors} 篇失败</div>`;
        }
        refreshKBStatus(); refreshKBManager();
      }
    } catch(e) { clearInterval(poll); }
  }, 2000);
}

async function browseCustomDir() {
  const dir = document.getElementById('ingest-dir').value.trim();
  if (!dir) return;
  document.getElementById('ingest-file-result').innerHTML =
    `<div style="margin-top:8px;font-size:12px;color:var(--text2)">验证目录: ${escHtml(dir)}...</div>`;
  try {
    const resp = await fetch('/api/directories');
    const d = await resp.json();
    // Check if custom dir matches any known directory or just validate
    document.getElementById('ingest-file-result').innerHTML =
      `<div style="margin-top:8px"><span class="tag tag-info">提示</span> 输入路径后点击“导入”按钮导入文件</div>`;
  } catch(e) {
    document.getElementById('ingest-file-result').innerHTML =
      `<div style="margin-top:8px;font-size:12px;color:var(--text2)">输入路径后点击“导入”</div>`;
  }
}

async function ingestCustomDir() {
  const dir = document.getElementById('ingest-dir').value.trim();
  if (!dir) { alert('请输入目录路径'); return; }
  const analyze = _getIngestMode() === 'analyze';
  try {
    const d = await api('/api/ingest/files', { dir, analyze });
    updateKBStatus(d.total || 0);
    const skipInfo = d.skipped_duplicates > 0 ? ` · 跳过 ${d.skipped_duplicates} 篇重复` : '';
    document.getElementById('ingest-file-result').innerHTML =
      `<div style="margin-top:8px"><span class="tag tag-ok">OK</span> 新增 ${d.loaded || 0} 篇${skipInfo} (共 ${d.total} 篇)</div>`;
    if (d.status === 'analyzing_in_background') {
      const fakeBtn = null; // Progress shown in ingest-file-result
      _pollIngestProgress(fakeBtn);
    }
    refreshKBStatus(); refreshKBManager();
  } catch(e) {
    document.getElementById('ingest-file-result').innerHTML = `<div style="margin-top:8px"><span class="tag tag-fail">Error</span> ${escHtml(e.message)}</div>`;
  }
}

async function ingestNews() {
  const q = document.getElementById('ingest-news-q').value.trim();
  if (!q) return;
  const maxNews = parseInt(document.getElementById('ingest-news-count').value) || 30;
  try {
    const d = await api('/api/ingest/news', { query: q, max_news: maxNews });
    let h = `<div style="margin-top:8px"><span class="tag tag-ok">OK</span> 抓取 ${d.fetched} 条新闻 → 元数据`;
    if (d.has_summary) h += ' + AI摘要';
    h += ` (累计: ${d.meta_total} 条)</div>`;
    updateMetaStatus(d.meta_total);
    if (d.headlines && d.headlines.length) {
      h += '<div style="margin-top:8px">';
      d.headlines.slice(0, 8).forEach(item => {
        h += `<div class="news-item"><h4>${escHtml(item.title)}</h4><div class="meta">${escHtml(item.source)} · ${escHtml(item.publish_time)}</div></div>`;
      });
      if (d.headlines.length > 8) h += `<div style="font-size:12px;color:var(--text2)">... 还有 ${d.headlines.length - 8} 条</div>`;
      h += '</div>';
    }
    document.getElementById('ingest-news-result').innerHTML = h;
  } catch(e) {
    document.getElementById('ingest-news-result').innerHTML = `<div style="margin-top:8px"><span class="tag tag-fail">Error</span> ${escHtml(e.message)}</div>`;
  }
}

// ===== STEP 2: Build =====
async function buildKB() {
  showLoading('build-loading');
  try {
    const d = await api('/api/build', {});
    kbBuilt = true;
    document.getElementById('buildCount').textContent = d.doc_count;
    let h = `<div class="stats">
      <div class="stat">文档数 <strong>${d.doc_count}</strong></div>
      <div class="stat">BM25 <strong>${d.bm25_terms || '-'} terms</strong></div>
      <div class="stat">Embedding <strong>${d.embedding_dim || 'N/A'} 维</strong></div>
      <div class="stat">耗时 <strong>${d.elapsed_ms}ms</strong></div>
    </div>`;
    if (d.test_queries && d.test_queries.length) {
      h += '<div style="margin-top:12px;font-size:13px;color:var(--text2)">验证检索:</div>';
      d.test_queries.forEach(tq => {
        h += `<div style="margin-top:6px;font-size:13px"><strong>${escHtml(tq.query)}</strong>: `;
        tq.results.forEach(r => { h += `<span style="color:${scoreColor(r.score)}">[${r.score.toFixed(3)}]</span> `; });
        h += '</div>';
      });
    }
    document.getElementById('build-result').innerHTML = h;
    refreshKBStatus();
  } catch(e) {
    document.getElementById('build-result').innerHTML = `<div style="margin-top:8px"><span class="tag tag-fail">Error</span> ${escHtml(e.message)}</div>`;
  }
  hideLoading('build-loading');
}

async function clearKB() {
  try { await fetch('/api/kb/clear', {method:'POST'}); await fetch('/api/metadata/clear', {method:'POST'}); } catch(e) {}
  kbDocs = []; kbBuilt = false;
  updateKBStatus(0); updateMetaStatus(0); renderDocList([]); refreshKBManager();
  updateKBDashboard({doc_count:0, kb_built:false, file_size_kb:0, meta_count:0});
  document.getElementById('buildCount').textContent = '0';
  document.getElementById('build-result').innerHTML = '';
  document.getElementById('query-result').innerHTML = '';
}

// ===== STEP 3: Query =====
async function runKBQuery() {
  const q = document.getElementById('query-input').value.trim();
  if (!q) return;
  if (!kbBuilt && !kbDocs.length) { document.getElementById('query-result').innerHTML = '<div class="card"><span class="tag tag-warn">Warn</span> 请先构建知识库</div>'; return; }
  showLoading('query-loading');
  document.getElementById('query-result').innerHTML = '';
  try {
    const topK = parseInt(document.getElementById('query-topk').value);
    const d = await api('/api/kb-query', { query: q, top_k: topK });
    let h = '';
    if (d.scorecard && d.scorecard.stages) {
      h += '<div class="pipeline-stages">';
      d.scorecard.stages.forEach(s => {
        const pct = (s.score * 100).toFixed(0);
        h += `<div class="stage-item"><div class="stage-name">${s.name}</div><div class="stage-time" style="color:${scoreColor(s.score)}">${pct}%</div><div class="stage-bar"><div class="stage-bar-fill" style="width:${pct}%;background:${scoreColor(s.score)}"></div></div></div>`;
      });
      h += '</div>';
    }
    if (d.retrieval && d.retrieval.length) {
      h += `<div class="sources-section"><h3>📄 知识库来源 (${d.retrieval.length})</h3>`;
      d.retrieval.forEach((r, i) => {
        h += `<div class="source-result"><div class="source-rank ${rankClass(r.score)}">${i+1}</div><div class="source-body">
          <div class="text">${escHtml(r.text)}</div><div class="meta">
          <span>RRF: <strong style="color:${scoreColor(r.score)}">${r.score.toFixed(4)}</strong></span>
          ${r.bm25_rank ? `<span>BM25: #${r.bm25_rank} (${r.bm25_score.toFixed(4)})</span>` : '<span style="color:#999">BM25: -</span>'}
          ${r.vector_rank ? `<span>Vec: #${r.vector_rank} (${r.vector_score.toFixed(4)})</span>` : '<span style="color:#999">Vec: -</span>'}
          ${r.source ? `<span>来源: ${escHtml(r.source)}</span>` : ''}</div></div></div>`;
      });
      h += '</div>';
    }
    if (d.answer) h += `<div class="answer-section"><h3>💡 回答</h3><div class="answer-text">${escHtml(d.answer)}</div></div>`;
    if (d.news_context && d.news_context.length) {
      h += `<div class="card" style="margin-top:12px"><h3>📰 相关新闻 (${d.news_context.length})</h3>`;
      d.news_context.forEach(n => { h += `<div class="news-item"><h4>${escHtml(n.title)}</h4><div class="meta">${escHtml(n.source)} · ${escHtml(n.publish_time)}</div></div>`; });
      h += '</div>';
    }
    if (d.fill_stats) {
      const fs = d.fill_stats;
      h += `<div class="stats"><div class="stat">槽位 <strong>${fs.filled_slots}/${fs.total_slots}</strong></div><div class="stat">TTFT <strong>${fs.avg_ttft_ms}ms</strong></div><div class="stat">并行 <strong>${fs.parallel_gain}%</strong></div><div class="stat">耗时 <strong>${fs.elapsed_ms}ms</strong></div></div>`;
    }
    document.getElementById('query-result').innerHTML = h;
    document.getElementById('queryCount').textContent = (parseInt(document.getElementById('queryCount').textContent) || 0) + 1;
  } catch(e) { document.getElementById('query-result').innerHTML = `<div class="card"><span class="tag tag-fail">Error</span> ${escHtml(e.message)}</div>`; }
  hideLoading('query-loading');
}

// ===== STEP 4: Tools =====
async function runNewsTool() {
  const q = document.getElementById('tool-news-q').value.trim();
  if (!q) return;
  try {
    const d = await api('/api/news', { query: q, summarize: true });
    let h = `<div class="stats"><div class="stat">关键词 <strong>${escHtml(d.keyword)}</strong></div><div class="stat">数量 <strong>${d.total_found}</strong></div>`;
    if (d.meta_stored > 0) h += `<div class="stat"><span class="tag tag-ok">✓ ${d.meta_stored} 条元数据</span> (累计 ${d.meta_total})</div>`;
    h += '</div>';
    updateMetaStatus(d.meta_total);
    if (d.summary) h += `<div class="card" style="margin-top:8px"><h3>AI 摘要</h3><div style="font-size:13px;line-height:1.6">${escHtml(d.summary)}</div></div>`;
    if (d.headlines && d.headlines.length) {
      d.headlines.slice(0, 8).forEach(item => { h += `<div class="news-item"><h4>${escHtml(item.title)}</h4><div class="meta">${escHtml(item.source)} · ${escHtml(item.publish_time)}</div></div>`; });
    }
    document.getElementById('tool-news-result').innerHTML = h;
  } catch(e) { document.getElementById('tool-news-result').innerHTML = `<span class="tag tag-fail">${escHtml(e.message)}</span>`; }
}

async function runKlineTool() {
  const q = document.getElementById('tool-kline-q').value.trim();
  if (!q) return;
  const resultDiv = document.getElementById('tool-kline-result');
  resultDiv.innerHTML = '<span style="color:var(--text2)">分析中...</span>';
  try {
    const d = await api('/api/kline', { query: q, days: parseInt(document.getElementById('tool-kline-days').value), period: document.getElementById('tool-kline-period').value });
    if (d.error) { resultDiv.innerHTML = `<span class="tag tag-fail">${escHtml(d.error)}</span>`; return; }
    const s = d.stats || {}, ind = d.indicators || {};
    let h = `<div class="stats"><div class="stat"><strong>${escHtml(d.name)}</strong> (${escHtml(d.ts_code)})</div><div class="stat">收盘 <strong>${s.latest_close||'-'}</strong></div><div class="stat">涨跌 <strong style="color:${(s.period_change_pct||0)>=0?'var(--ok)':'var(--danger)'}">${s.period_change_pct||'-'}%</strong></div></div>`;
    if (ind.macd || ind.rsi || ind.bollinger || ind.kdj) {
      h += '<div style="margin-top:8px;font-size:12px;color:var(--text2)">技术指标</div><div class="stats">';
      if (ind.macd) h += `<div class="stat">MACD <strong>${ind.macd.signal}</strong></div>`;
      if (ind.rsi) h += `<div class="stat">RSI <strong>${ind.rsi.value}</strong> ${ind.rsi.signal}</div>`;
      if (ind.kdj) h += `<div class="stat">KDJ <strong>${ind.kdj.signal}</strong></div>`;
      if (ind.bollinger) h += `<div class="stat">布林 ${ind.bollinger.position}</div>`;
      h += '</div>';
    }
    if (d.analysis) h += `<div style="margin-top:12px;padding:10px;background:var(--bg);border-radius:6px;font-size:13px;line-height:1.7;white-space:pre-wrap">${escHtml(d.analysis)}</div>`;
    resultDiv.innerHTML = h;
  } catch(e) { resultDiv.innerHTML = `<span class="tag tag-fail">${escHtml(e.message)}</span>`; }
}

async function runSlotTool() {
  const q = document.getElementById('tool-slot-q').value.trim();
  if (!q) return;
  try {
    const d = await api('/api/slot', { query: q, template: document.getElementById('tool-slot-tpl').value });
    const sf = d.slot_fill;
    document.getElementById('tool-slot-result').innerHTML = `<div class="stats"><div class="stat">槽位 <strong>${sf.filled_slots}/${sf.total_slots}</strong></div><div class="stat">耗时 <strong>${sf.elapsed_ms}ms</strong></div></div><div class="result" style="margin-top:8px;font-size:13px">${escHtml(sf.rendered)}</div>`;
  } catch(e) { document.getElementById('tool-slot-result').innerHTML = `<span class="tag tag-fail">${escHtml(e.message)}</span>`; }
}

async function runScoreTool() {
  const q = document.getElementById('tool-score-q').value.trim();
  if (!q) return;
  try {
    const d = await api('/api/score', { query: q, top_k: 5 });
    let h = '';
    if (d.scorecard && d.scorecard.stages) {
      h += '<div class="pipeline-stages" style="padding:8px">';
      d.scorecard.stages.forEach(s => { const pct = (s.score*100).toFixed(0); h += `<div class="stage-item"><div class="stage-name" style="font-size:10px">${s.name}</div><div class="stage-time" style="font-size:12px;color:${scoreColor(s.score)}">${pct}%</div></div>`; });
      h += '</div>';
    }
    if (d.results) d.results.forEach(r => { h += `<div style="font-size:12px;margin-top:4px;color:${scoreColor(r.score)}">[${r.score.toFixed(3)}] ${escHtml(r.text.slice(0,80))}</div>`; });
    document.getElementById('tool-score-result').innerHTML = h;
  } catch(e) { document.getElementById('tool-score-result').innerHTML = `<span class="tag tag-fail">${escHtml(e.message)}</span>`; }
}

// ===== STEP 5: Smart Analysis =====
function verdictBadge(v) {
  const map = {
    bullish: ['利好','var(--ok)'], bearish: ['利空','var(--danger)'],
    neutral: ['中性','var(--text2)'], unknown: ['分析失败','#f59e0b'],
  };
  const [label, color] = map[v] || map.neutral;
  return `<span style="display:inline-block;padding:4px 14px;border-radius:20px;font-size:14px;font-weight:700;background:${color}22;color:${color};border:1px solid ${color}44">${label}</span>`;
}

function confidenceBadge(c) {
  if (!c) return '';
  const map = { high: ['高置信','var(--ok)'], medium: ['中置信','#f59e0b'], low: ['低置信','var(--danger)'] };
  const [label, color] = map[c] || [];
  if (!label) return '';
  return ` <span style="display:inline-block;padding:2px 8px;border-radius:12px;font-size:11px;background:${color}22;color:${color};border:1px solid ${color}44;margin-left:6px">${label}</span>`;
}

function directionArrow(dir) {
  const map = { bullish: ['↑','var(--ok)'], bearish: ['↓','var(--danger)'], neutral: ['→','var(--text2)'],
    positive: ['↑','var(--ok)'], negative: ['↓','var(--danger)'], improving: ['↑','var(--ok)'], deteriorating: ['↓','var(--danger)'],
    stable: ['→','var(--text2)'], mixed: ['↔','#f59e0b'] };
  const [arrow, color] = map[dir] || ['→','var(--text2)'];
  return `<span style="color:${color};font-weight:700">${arrow}</span>`;
}

function severityBar(level) {
  const n = Math.min(5, Math.max(1, level || 1));
  let bars = '';
  for (let i = 1; i <= 5; i++) {
    const color = i <= n ? (n >= 4 ? 'var(--danger)' : n >= 3 ? '#f59e0b' : 'var(--ok)') : 'var(--border)';
    bars += `<span style="display:inline-block;width:4px;height:12px;background:${color};border-radius:2px;margin-right:2px"></span>`;
  }
  return bars;
}

function sentimentPill(s) {
  const map = { positive: ['积极','var(--ok)'], negative: ['消极','var(--danger)'], neutral: ['中性','var(--text2)'] };
  const [label, color] = map[s] || map.neutral;
  return `<span style="font-size:10px;padding:1px 6px;border-radius:8px;background:${color}22;color:${color}">${label}</span>`;
}

async function analyzeNews() {
  const text = document.getElementById('analyze-news-text').value.trim();
  if (!text) return;
  const query = document.getElementById('analyze-news-q').value.trim();
  showLoading('analyze-news-loading');
  document.getElementById('analyze-news-result').innerHTML = '';
  try {
    const d = await api('/api/analyze/news', { text, query });
    const s = d.structured || {};
    let h = '<div style="margin-top:12px">';
    // Verdict + confidence
    h += `<div style="margin-bottom:14px;text-align:center">${verdictBadge(d.assessment)}${confidenceBadge(d.confidence)}</div>`;
    if (d.saved_to_kb) h += `<div style="text-align:center;font-size:11px;color:var(--text2);margin-bottom:8px">💾 分析结论已存入知识库</div>`;

    // Multi-dimensional impact
    if (s.impact) {
      const imp = s.impact;
      const dims = [
        {key:'industry',icon:'🏢',label:'行业'}, {key:'company',icon:'🏭',label:'公司'},
        {key:'tech',icon:'⚡',label:'技术'}, {key:'market',icon:'📈',label:'市场'}
      ];
      h += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:14px">';
      dims.forEach(dim => {
        const v = imp[dim.key] || {};
        const dir = v.direction || 'neutral';
        h += `<div style="padding:8px 10px;background:var(--bg2);border-radius:8px;font-size:12px">`;
        h += `<div style="font-weight:600;margin-bottom:4px">${dim.icon} ${dim.label} ${directionArrow(dir)}</div>`;
        h += `<div style="color:var(--text2);font-size:11px">${escHtml(v.summary || '')}</div></div>`;
      });
      h += '</div>';
    }

    // Key signals
    if (s.key_signals && s.key_signals.length) {
      h += '<div style="margin-bottom:14px"><div style="font-size:13px;font-weight:600;margin-bottom:6px">🔔 关键信号</div>';
      s.key_signals.forEach(sig => {
        const typeColor = sig.type === 'positive' ? 'var(--ok)' : sig.type === 'negative' ? 'var(--danger)' : 'var(--text2)';
        h += `<div style="display:flex;align-items:center;gap:8px;padding:5px 0;font-size:12px;border-bottom:1px solid var(--border)">`;
        h += `<span style="flex-shrink:0">${severityBar(sig.severity)}</span>`;
        h += `<span style="flex:1;color:var(--text)">${escHtml(sig.signal || '')}</span>`;
        h += `<span style="font-size:10px;color:${typeColor};font-weight:600">${sig.type === 'positive' ? '▲' : sig.type === 'negative' ? '▼' : '●'}</span>`;
        h += '</div>';
      });
      h += '</div>';
    }

    // Analysis text
    if (d.analysis) h += `<div class="answer-section" style="margin-bottom:12px"><h3>💡 综合分析</h3><div class="answer-text">${escHtml(d.analysis)}</div></div>`;

    // Extraction stats
    if (d.doc_type || d.entities || d.metrics) {
      h += '<div class="stats">';
      if (d.doc_type) h += `<div class="stat">文档类型 <strong>${escHtml(d.doc_type)}</strong></div>`;
      const ent = d.entities || {}, met = d.metrics || {};
      if (ent.companies && ent.companies.length) h += `<div class="stat">公司 <strong>${ent.companies.map(c=>escHtml(typeof c === 'string' ? c : c.name || c)).join(', ')}</strong></div>`;
      if (ent.persons && ent.persons.length) h += `<div class="stat">人物 <strong>${ent.persons.map(p=>escHtml(typeof p === 'string' ? p : p.name || p)).slice(0,3).join(', ')}</strong></div>`;
      if (ent.ai_models && ent.ai_models.length) h += `<div class="stat">AI模型 <strong>${ent.ai_models.map(m=>escHtml(typeof m === 'string' ? m : m.name || m)).slice(0,3).join(', ')}</strong></div>`;
      if (met.revenue) h += `<div class="stat">营收 <strong>${escHtml(String(met.revenue))}</strong></div>`;
      if (met.net_profit) h += `<div class="stat">净利润 <strong>${escHtml(String(met.net_profit))}</strong></div>`;
      h += '</div>';
    }

    // Risks + Watch Next (side by side)
    if ((s.risks && s.risks.length) || (s.watch_next && s.watch_next.length)) {
      h += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:12px">';
      if (s.risks && s.risks.length) {
        h += '<div style="padding:10px;background:rgba(217,48,37,.06);border-radius:8px;border:1px solid rgba(217,48,37,.15)">';
        h += '<div style="font-size:12px;font-weight:600;color:var(--danger);margin-bottom:6px">⚠️ 风险提示</div>';
        s.risks.forEach(r => { h += `<div style="font-size:11px;color:var(--text);padding:2px 0">• ${escHtml(r)}</div>`; });
        h += '</div>';
      }
      if (s.watch_next && s.watch_next.length) {
        h += '<div style="padding:10px;background:rgba(78,205,196,.06);border-radius:8px;border:1px solid rgba(78,205,196,.15)">';
        h += '<div style="font-size:12px;font-weight:600;color:var(--accent);margin-bottom:6px">👁️ 后续关注</div>';
        s.watch_next.forEach(w => { h += `<div style="font-size:11px;color:var(--text);padding:2px 0">• ${escHtml(w)}</div>`; });
        h += '</div>';
      }
      h += '</div>';
    }

    // KB sources
    if (d.kb_sources && d.kb_sources.length) {
      h += `<div style="margin-top:10px;font-size:13px;color:var(--text2)">📚 KB来源 (${d.kb_sources.length})</div>`;
      d.kb_sources.slice(0, 3).forEach((s, i) => {
        h += `<div class="source-result" style="margin-top:6px"><div class="source-rank rank-high">${i+1}</div><div class="source-body"><div class="text">${escHtml((s.text||'').slice(0,150))}</div><div class="meta"><span>相关度: <strong style="color:${scoreColor(s.score)}">${(s.score||0).toFixed(3)}</strong></span></div></div></div>`;
      });
    }
    h += '</div>';
    document.getElementById('analyze-news-result').innerHTML = h;
    if (d.saved_to_kb) refreshLearningHistory();
  } catch(e) { document.getElementById('analyze-news-result').innerHTML = `<div style="margin-top:8px"><span class="tag tag-fail">Error</span> ${escHtml(e.message)}</div>`; }
  hideLoading('analyze-news-loading');
}

async function analyzeTopic() {
  const topic = document.getElementById('analyze-topic-input').value.trim();
  if (!topic) return;
  const maxNews = parseInt(document.getElementById('analyze-topic-count').value);
  showLoading('analyze-topic-loading');
  document.getElementById('analyze-topic-result').innerHTML = '';
  const steps = ['① 提取关键词...', '② 抓取新闻...', '③ 查询知识库...', '④ LLM 综合研判...'];
  let stepIdx = 0;
  const loadingEl = document.querySelector('#analyze-topic-loading p');
  const progressTimer = setInterval(() => { stepIdx = Math.min(stepIdx + 1, steps.length - 1); if (loadingEl) loadingEl.textContent = steps[stepIdx]; }, 5000);
  try {
    const d = await api('/api/analyze/topic', { topic, max_news: maxNews });
    const s = d.structured || {};
    let h = '<div style="margin-top:12px">';
    // Verdict + confidence
    h += `<div style="margin-bottom:14px;text-align:center">${verdictBadge(d.assessment)}${confidenceBadge(d.confidence)}</div>`;
    if (d.saved_to_kb) h += `<div style="text-align:center;font-size:11px;color:var(--text2);margin-bottom:8px">💾 研判结论已存入知识库</div>`;

    // Stats bar
    const trendDir = s.sentiment_trend || 'mixed';
    h += `<div class="stats" style="margin-bottom:14px">`;
    h += `<div class="stat">话题 <strong>${escHtml(d.topic)}</strong></div>`;
    h += `<div class="stat">新闻 <strong>${d.news_count}</strong></div>`;
    h += `<div class="stat">情绪趋势 ${directionArrow(trendDir)} <strong style="font-size:11px">${escHtml({improving:'回暖',deteriorating:'恶化',stable:'稳定',mixed:'分化'}[trendDir] || trendDir)}</strong></div>`;
    h += '</div>';

    // Sub-topics
    if (s.sub_topics && s.sub_topics.length) {
      h += '<div style="margin-bottom:14px"><div style="font-size:13px;font-weight:600;margin-bottom:6px">🧩 子话题聚类</div>';
      h += '<div style="display:flex;flex-wrap:wrap;gap:8px">';
      s.sub_topics.forEach(st => {
        const sentColor = st.sentiment === 'positive' ? 'var(--ok)' : st.sentiment === 'negative' ? 'var(--danger)' : 'var(--text2)';
        h += `<div style="padding:8px 12px;background:var(--bg2);border-radius:8px;border-left:3px solid ${sentColor};font-size:12px;min-width:140px">`;
        h += `<div style="font-weight:600;margin-bottom:3px">${escHtml(st.name || '')} ${sentimentPill(st.sentiment)}</div>`;
        h += `<div style="color:var(--text2);font-size:11px">${escHtml(st.summary || '')}</div></div>`;
      });
      h += '</div></div>';
    }

    // Key players
    if (s.key_players && s.key_players.length) {
      h += '<div style="margin-bottom:14px"><div style="font-size:13px;font-weight:600;margin-bottom:6px">👤 关键玩家</div>';
      h += '<div style="display:flex;flex-direction:column;gap:4px">';
      s.key_players.forEach(p => {
        h += `<div style="display:flex;align-items:center;gap:8px;padding:5px 10px;background:var(--bg2);border-radius:6px;font-size:12px">`;
        h += `<span style="font-weight:600;min-width:80px">${escHtml(p.name || '')}</span>`;
        h += `<span style="flex:1;color:var(--text2);font-size:11px">${escHtml(p.role || '')}</span>`;
        if (p.mentions) h += `<span class="tag tag-info" style="font-size:10px">${p.mentions}次提及</span>`;
        h += '</div>';
      });
      h += '</div></div>';
    }

    // Analysis text
    if (d.analysis) h += `<div class="answer-section" style="margin-bottom:12px"><h3>💡 综合分析</h3><div class="answer-text">${escHtml(d.analysis)}</div></div>`;

    // Investment implication
    if (s.investment_implication) {
      h += `<div style="padding:10px;background:rgba(78,205,196,.06);border-radius:8px;border:1px solid rgba(78,205,196,.15);margin-bottom:12px">`;
      h += `<div style="font-size:12px;font-weight:600;color:var(--accent);margin-bottom:4px">💰 投资启示</div>`;
      h += `<div style="font-size:12px">${escHtml(s.investment_implication)}</div></div>`;
    }

    // Contrarian signals
    if (s.contrarian_signals && s.contrarian_signals.length) {
      h += `<div style="padding:10px;background:rgba(245,158,11,.06);border-radius:8px;border:1px solid rgba(245,158,11,.15);margin-bottom:12px">`;
      h += `<div style="font-size:12px;font-weight:600;color:#f59e0b;margin-bottom:4px">🔄 逆向信号</div>`;
      s.contrarian_signals.forEach(c => { h += `<div style="font-size:11px;padding:2px 0">• ${escHtml(c)}</div>`; });
      h += '</div>';
    }

    // Risks
    if (s.risks && s.risks.length) {
      h += `<div style="padding:10px;background:rgba(217,48,37,.06);border-radius:8px;border:1px solid rgba(217,48,37,.15);margin-bottom:12px">`;
      h += `<div style="font-size:12px;font-weight:600;color:var(--danger);margin-bottom:4px">⚠️ 风险提示</div>`;
      s.risks.forEach(r => { h += `<div style="font-size:11px;padding:2px 0">• ${escHtml(r)}</div>`; });
      h += '</div>';
    }

    // News list
    if (d.news && d.news.length) {
      h += '<div style="margin-top:10px;font-size:13px;color:var(--text2)">📰 相关新闻</div>';
      d.news.forEach(n => { h += `<div class="news-item"><h4>${escHtml(n.title)}</h4><div class="meta">${escHtml(n.source)} · ${escHtml(n.publish_time)}</div></div>`; });
    }
    // KB sources
    if (d.kb_sources && d.kb_sources.length) {
      h += `<div style="margin-top:10px;font-size:13px;color:var(--text2)">📚 KB (${d.kb_sources.length})</div>`;
      d.kb_sources.slice(0, 3).forEach((s, i) => {
        h += `<div class="source-result" style="margin-top:6px"><div class="source-rank rank-high">${i+1}</div><div class="source-body"><div class="text">${escHtml((s.text||'').slice(0,150))}</div><div class="meta"><span>相关度: <strong style="color:${scoreColor(s.score)}">${(s.score||0).toFixed(3)}</strong></span></div></div></div>`;
      });
    }
    h += '</div>';
    document.getElementById('analyze-topic-result').innerHTML = h;
    if (d.saved_to_kb) refreshLearningHistory();
  } catch(e) { document.getElementById('analyze-topic-result').innerHTML = `<div style="margin-top:8px"><span class="tag tag-fail">Error</span> ${escHtml(e.message)}</div>`; }
  clearInterval(progressTimer);
  hideLoading('analyze-topic-loading');
}

// ===== Learning History =====
async function refreshLearningHistory() {
  const container = document.getElementById('learningHistoryList');
  if (!container) return;
  try {
    const d = await fetch('/api/kb/history').then(r => r.json());
    if (d.count === 0) {
      container.innerHTML = '<div class="empty-state" style="padding:16px"><div class="icon">🧠</div><p>尚未积累学习记录</p><div class="hint">每次新闻解读或话题调研的结论会自动存入知识库，供未来分析参考</div></div>';
      return;
    }
    let h = `<div style="font-size:12px;color:var(--text2);margin-bottom:10px">已积累 <strong>${d.count}</strong> 条分析结论</div>`;
    h += '<div style="display:flex;flex-direction:column;gap:8px">';
    for (const item of d.history) {
      const type = item.source.startsWith('analysis:news') ? '📰 新闻解读' : '🔍 话题调研';
      const verdictColor = item.assessment.includes('利好') ? '#0d8a3e' : item.assessment.includes('利空') ? '#d93025' : '#666';
      const topic = item.source.replace(/^analysis:(news|topic):/, '');
      h += `<div style="padding:10px 12px;background:var(--bg2);border-radius:8px;font-size:13px">`;
      h += `<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px">`;
      h += `<span style="font-weight:600">${type}：${escHtml(topic)}</span>`;
      h += `<span style="font-size:11px;color:${verdictColor};font-weight:600">${escHtml(item.assessment)}</span>`;
      h += `</div>`;
      h += `<div style="font-size:11px;color:var(--text2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${escHtml(item.preview.slice(0, 120))}</div>`;
      h += `<div style="font-size:10px;color:var(--text2);margin-top:4px">⏱️ ${escHtml(item.timestamp)}</div>`;
      h += `</div>`;
    }
    h += '</div>';
    container.innerHTML = h;
  } catch(e) {
    container.innerHTML = `<div style="color:var(--danger);font-size:12px">加载失败: ${escHtml(e.message)}</div>`;
  }
}

// ===== KB Manager =====
async function refreshKBManager() {
  const container = document.getElementById('kbSourceList');
  try {
    const d = await fetch('/api/kb/status').then(r => r.json());
    const sources = d.sources || {};
    const keys = Object.keys(sources);
    if (!keys.length) {
      container.innerHTML = '<div class="empty-state" style="padding:12px"><p style="margin:0">知识库为空</p></div>';
      return;
    }
    let h = `<div style="font-size:12px;color:var(--text2);margin-bottom:8px">📚 ${d.doc_count} 篇文档 · ${d.file_size_kb} KB · ${d.analyzed_count} 已分析</div>`;
    h += '<div style="display:flex;flex-direction:column;gap:6px">';
    keys.sort((a, b) => sources[b] - sources[a]);
    for (const src of keys) {
      const count = sources[src];
      h += `<div style="display:flex;align-items:center;justify-content:space-between;padding:6px 10px;background:var(--bg2);border-radius:6px;font-size:13px">`;
      h += `<span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${escHtml(src)}">${escHtml(src)}</span>`;
      h += `<span class="tag tag-info" style="margin:0 8px;font-size:11px">${count} 篇</span>`;
      const safeSrc = src.replace(/'/g, "\\'").replace(/"/g, '&quot;');
      h += `<button class="btn btn-sm" style="padding:2px 8px;font-size:11px;color:var(--danger)" onclick="removeKBSource('${safeSrc}')">删除</button>`;
      h += `</div>`;
    }
    h += '</div>';
    container.innerHTML = h;
  } catch(e) {
    container.innerHTML = `<div style="color:var(--danger);font-size:12px">加载失败: ${escHtml(e.message)}</div>`;
  }
}

async function removeKBSource(source) {
  if (!confirm(`确认删除来源「${source}」的所有文档？`)) return;
  try {
    const resp = await fetch(`/api/kb/source/${encodeURIComponent(source)}`, { method: 'DELETE' });
    const d = await resp.json();
    refreshKBManager();
    refreshKBStatus();
  } catch(e) { alert('删除失败: ' + e.message); }
}

async function searchKBKeyword() {
  const kw = document.getElementById('kbKeywordInput').value.trim();
  const box = document.getElementById('kbSearchResults');
  if (!kw) { box.style.display = 'none'; return; }
  console.log('[KB Search] searching for:', kw);
  try {
    const resp = await fetch(`/api/kb/search?keyword=${encodeURIComponent(kw)}`);
    console.log('[KB Search] response status:', resp.status);
    const d = await resp.json();
    console.log('[KB Search] result:', d);
    if (d.matched === 0) {
      box.innerHTML = `<div style="padding:8px;font-size:12px;color:var(--text2)">未找到包含「${escHtml(kw)}」的文档</div>`;
    } else {
      let html = `<div style="padding:8px;font-size:12px;color:var(--text2)">找到 <strong>${d.matched}</strong> 篇包含「${escHtml(kw)}」的文档：</div>`;
      html += '<div style="max-height:160px;overflow-y:auto;padding:0 8px">';
      for (const m of d.matches) {
        html += `<div style="font-size:11px;padding:4px 0;border-bottom:1px solid var(--border)">
          <span style="color:var(--text2)">[${escHtml(m.source)}]</span> ${escHtml((m.preview||'').slice(0,80))}...
        </div>`;
      }
      html += '</div>';
      box.innerHTML = html;
    }
    box.style.display = 'block';
  } catch(e) { console.error('[KB Search] error:', e); box.innerHTML = `<div style="padding:8px;color:red">${e.message}</div>`; box.style.display = 'block'; }
}

async function deleteKBKeyword() {
  const kw = document.getElementById('kbKeywordInput').value.trim();
  if (!kw) return;
  console.log('[KB Delete] keyword:', kw);
  const d = await fetch(`/api/kb/search?keyword=${encodeURIComponent(kw)}`).then(r => r.json());
  console.log('[KB Delete] search result:', d);
  if (d.matched === 0) { alert(`未找到包含「${kw}」的文档`); return; }
  if (!confirm(`确认删除 ${d.matched} 篇包含「${kw}」的知识库文档？\n\n此操作不可恢复。`)) return;
  try {
    const resp = await fetch(`/api/kb/keyword/${encodeURIComponent(kw)}`, { method: 'DELETE' });
    console.log('[KB Delete] response status:', resp.status);
    const r = await resp.json();
    console.log('[KB Delete] result:', r);
    alert(`已删除 ${r.removed} 篇，剩余 ${r.remaining} 篇`);
    document.getElementById('kbSearchResults').style.display = 'none';
    document.getElementById('kbKeywordInput').value = '';
    refreshKBManager();
    refreshKBStatus();
  } catch(e) { console.error('[KB Delete] error:', e); alert('删除失败: ' + e.message); }
}

// ===== Init =====
async function refreshKBStatus() {
  try {
    const d = await fetch('/api/kb/status').then(r => r.json());
    if (d.doc_count > 0) { updateKBStatus(d.doc_count); }
    if (d.meta_count > 0) updateMetaStatus(d.meta_count);
    updateKBDashboard(d);
  } catch(e) {}
}

fetch('/api/config').then(r=>r.json()).then(d => {
  document.getElementById('modelInfo').textContent = `${d.llm_model} | ${d.embedding_model} | ${d.rerank_model}`;
  if (d.mock_mode) document.getElementById('mockBadge').classList.add('active');
  if (d.has_api_key) { document.getElementById('kbBadge').querySelector('.dot').className = 'dot dot-ok'; document.getElementById('kbStatus').textContent = 'API 已连接'; }
  refreshKBStatus();
  refreshKBManager();
  loadDirBrowser();
}).catch(() => {});
