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
  const card = document.getElementById('kbStorageCard');
  const stats = document.getElementById('kbStorageStats');
  card.style.display = 'block';
  stats.innerHTML = `
    <div class="stat">📁 路径 <strong style="font-size:11px;word-break:break-all">${escHtml(path)}</strong></div>
    <div class="stat">📦 大小 <strong>${size} KB</strong></div>
    <div class="stat">${built ? '<span class="tag tag-ok">✓ 索引已构建</span>' : '<span class="tag tag-warn">未构建索引</span>'}</div>
  `;
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
async function loadDirBrowser() {
  try {
    const d = await fetch('/api/directories').then(r => r.json());
    renderDirBrowser(d.directories);
  } catch(e) {
    document.getElementById('dirBrowser').innerHTML =
      `<div style="color:var(--danger)">加载目录失败: ${escHtml(e.message)}</div>`;
  }
}

function renderDirBrowser(dirs) {
  const el = document.getElementById('dirBrowser');
  let h = '';
  dirs.forEach(dir => {
    const icon = dir.exists ? '📂' : '📭';
    const status = dir.exists ? `${dir.file_count} 个文件 · ${dir.total_size_kb} KB` : '目录不存在';
    h += `<div class="dir-card"><div class="dir-header"><div>
      <span style="font-size:18px">${icon}</span> <strong>${escHtml(dir.label)}</strong>
      <span style="font-size:12px;color:var(--text2);margin-left:8px">${escHtml(dir.path)}</span>
    </div><div style="display:flex;align-items:center;gap:8px">
      <span class="tag ${dir.file_count > 0 ? 'tag-info' : 'tag-off'}">${status}</span>
      ${dir.exists && dir.file_count > 0 ? `<button class="btn btn-primary btn-sm" onclick="ingestDir('${escHtml(dir.path)}', this)">分析并导入</button>` : ''}
    </div></div>`;
    if (dir.exists && dir.files.length > 0) {
      h += '<div class="dir-files">';
      dir.files.forEach(f => {
        const typeIcon = f.ext === '.jsonl' ? '📝' : f.ext === '.json' ? '📦' : f.ext === '.txt' ? '📄' : '📎';
        const lineInfo = f.line_count > 0 ? ` · ${f.line_count} 条` : '';
        h += `<div class="dir-file"><span>${typeIcon} ${escHtml(f.name)}</span><span style="color:var(--text2)">${f.size_kb} KB${lineInfo}</span></div>`;
      });
      h += '</div>';
    }
    h += '</div>';
  });
  el.innerHTML = h;
}

async function ingestDir(dirPath, btn) {
  if (btn) { btn.disabled = true; btn.textContent = '导入中...'; }
  try {
    const d = await api('/api/ingest/files', { dir: dirPath, analyze: true });
    updateKBStatus(d.total || 0);
    if (btn) {
      btn.textContent = `✓ ${d.loaded} 篇已导入`; btn.className = 'btn btn-success btn-sm';
    }
    // If background analysis started, poll progress
    if (d.status === 'analyzing_in_background') {
      _pollIngestProgress(btn);
    }
    refreshKBStatus();
  } catch(e) { if (btn) { btn.disabled = false; btn.textContent = '失败'; } }
}

async function _pollIngestProgress(btn) {
  const resultEl = document.getElementById('ingest-dir-result') || document.getElementById('ingest-file-result');
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

async function ingestFiles() {
  const dir = document.getElementById('ingest-dir').value.trim();
  if (!dir) return;
  try {
    const d = await api('/api/ingest/files', { dir });
    const total = d.total || 0;
    updateKBStatus(total);
    document.getElementById('ingest-file-result').innerHTML =
      `<div style="margin-top:8px"><span class="tag tag-ok">OK</span> 已加载 ${d.loaded} 篇 (共 ${total} 篇)<br><span style="font-size:12px;color:var(--text2)">💾 ${escHtml(d.kb_path)}</span></div>`;
    refreshKBStatus(); refreshKBManager();
  } catch(e) {
    document.getElementById('ingest-file-result').innerHTML = `<div style="margin-top:8px"><span class="tag tag-fail">Error</span> ${escHtml(e.message)}</div>`;
  }
}

async function ingestNews() {
  const q = document.getElementById('ingest-news-q').value.trim();
  if (!q) return;
  try {
    const d = await api('/api/ingest/news', { query: q, max_news: 30 });
    let h = `<div style="margin-top:8px"><span class="tag tag-ok">OK</span> 抓取 ${d.fetched} 条新闻 → 元数据`;
    if (d.has_summary) h += ' + AI摘要';
    h += ` (累计: ${d.meta_total} 条)</div>`;
    updateMetaStatus(d.meta_total);
    if (d.headlines && d.headlines.length) {
      h += '<div style="margin-top:8px">';
      d.headlines.slice(0, 5).forEach(item => {
        h += `<div class="news-item"><h4>${escHtml(item.title)}</h4><div class="meta">${escHtml(item.source)} · ${escHtml(item.publish_time)}</div></div>`;
      });
      if (d.headlines.length > 5) h += `<div style="font-size:12px;color:var(--text2)">... 还有 ${d.headlines.length - 5} 条</div>`;
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
    if (d.kb_path) h += `<div style="font-size:12px;color:var(--text2)">💾 ${escHtml(d.kb_path)}</div>`;
    showKBStorage(d.kb_path || '', 0, true);
    if (d.test_queries && d.test_queries.length) {
      h += '<div style="margin-top:12px;font-size:13px;color:var(--text2)">验证检索:</div>';
      d.test_queries.forEach(tq => {
        h += `<div style="margin-top:6px;font-size:13px"><strong>${escHtml(tq.query)}</strong>: `;
        tq.results.forEach(r => { h += `<span style="color:${scoreColor(r.score)}">[${r.score.toFixed(3)}]</span> `; });
        h += '</div>';
      });
    }
    document.getElementById('build-result').innerHTML = h;
  } catch(e) {
    document.getElementById('build-result').innerHTML = `<div style="margin-top:8px"><span class="tag tag-fail">Error</span> ${escHtml(e.message)}</div>`;
  }
  hideLoading('build-loading');
}

async function clearKB() {
  try { await fetch('/api/kb/clear', {method:'POST'}); await fetch('/api/metadata/clear', {method:'POST'}); } catch(e) {}
  kbDocs = []; kbBuilt = false;
  updateKBStatus(0); updateMetaStatus(0); renderDocList([]); refreshKBManager();
  document.getElementById('buildCount').textContent = '0';
  document.getElementById('build-result').innerHTML = '';
  document.getElementById('query-result').innerHTML = '';
  document.getElementById('kbStorageCard').style.display = 'none';
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

async function analyzeNews() {
  const text = document.getElementById('analyze-news-text').value.trim();
  if (!text) return;
  const query = document.getElementById('analyze-news-q').value.trim();
  showLoading('analyze-news-loading');
  document.getElementById('analyze-news-result').innerHTML = '';
  try {
    const d = await api('/api/analyze/news', { text, query });
    let h = '<div style="margin-top:12px">';
    h += `<div style="margin-bottom:14px;text-align:center">${verdictBadge(d.assessment)}${confidenceBadge(d.confidence)}</div>`;
    if (d.saved_to_kb) h += `<div style="text-align:center;font-size:11px;color:var(--text2);margin-bottom:8px">💾 分析结论已存入知识库</div>`;
    if (d.analysis) h += `<div class="answer-section" style="margin-bottom:12px"><h3>💡 分析结论</h3><div class="answer-text">${escHtml(d.analysis)}</div></div>`;
    if (d.doc_type || d.entities || d.metrics) {
      h += '<div class="stats">';
      if (d.doc_type) h += `<div class="stat">文档类型 <strong>${escHtml(d.doc_type)}</strong></div>`;
      const ent = d.entities || {}, met = d.metrics || {};
      if (ent.companies && ent.companies.length) h += `<div class="stat">公司 <strong>${ent.companies.map(c=>escHtml(typeof c === 'string' ? c : c.name || c)).join(', ')}</strong></div>`;
      if (ent.persons && ent.persons.length) h += `<div class="stat">人物 <strong>${ent.persons.map(p=>escHtml(typeof p === 'string' ? p : p.name || p)).slice(0,3).join(', ')}</strong></div>`;
      if (ent.ai_models && ent.ai_models.length) h += `<div class="stat">AI模型 <strong>${ent.ai_models.map(m=>escHtml(typeof m === 'string' ? m : m.name || m)).slice(0,3).join(', ')}</strong></div>`;
      if (ent.tech_terms && ent.tech_terms.length) h += `<div class="stat">技术 <strong>${ent.tech_terms.map(t=>escHtml(typeof t === 'string' ? t : t)).slice(0,3).join(', ')}</strong></div>`;
      if (met.revenue) h += `<div class="stat">营收 <strong>${escHtml(String(met.revenue))}</strong></div>`;
      if (met.net_profit) h += `<div class="stat">净利润 <strong>${escHtml(String(met.net_profit))}</strong></div>`;
      h += '</div>';
    }
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
  // Cycling progress text
  const steps = ['① 提取关键词...', '② 抓取新闻...', '③ 查询知识库...', '④ LLM 综合研判...'];
  let stepIdx = 0;
  const loadingEl = document.querySelector('#analyze-topic-loading p');
  const progressTimer = setInterval(() => { stepIdx = Math.min(stepIdx + 1, steps.length - 1); if (loadingEl) loadingEl.textContent = steps[stepIdx]; }, 5000);
  try {
    const d = await api('/api/analyze/topic', { topic, max_news: maxNews });
    let h = '<div style="margin-top:12px">';
    h += `<div style="margin-bottom:14px;text-align:center">${verdictBadge(d.assessment)}${confidenceBadge(d.confidence)}</div>`;
    if (d.saved_to_kb) h += `<div style="text-align:center;font-size:11px;color:var(--text2);margin-bottom:8px">💾 研判结论已存入知识库</div>`;
    if (d.analysis) h += `<div class="answer-section" style="margin-bottom:12px"><h3>💡 综合研判</h3><div class="answer-text">${escHtml(d.analysis)}</div></div>`;
    h += `<div class="stats"><div class="stat">话题 <strong>${escHtml(d.topic)}</strong></div><div class="stat">新闻 <strong>${d.news_count}</strong></div></div>`;
    if (d.news && d.news.length) {
      h += '<div style="margin-top:10px;font-size:13px;color:var(--text2)">📰 相关新闻</div>';
      d.news.forEach(n => { h += `<div class="news-item"><h4>${escHtml(n.title)}</h4><div class="meta">${escHtml(n.source)} · ${escHtml(n.publish_time)}</div></div>`; });
    }
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
    if (d.doc_count > 0) { updateKBStatus(d.doc_count); showKBStorage(d.kb_path, d.file_size_kb, d.kb_built); }
    if (d.meta_count > 0) updateMetaStatus(d.meta_count);
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
