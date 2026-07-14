// FinRAG — query.js | RAG query + Pipeline + K-line analysis
import { api, showLoading, hideLoading } from './api.js';
import { toast, escHtml, scoreColor, rankClass } from './ui.js';

let _queryMode = 'kb';  // 'kb' | 'pipeline'

const GRADE_COLORS = { A: 'var(--ok)', B: '#2563eb', C: '#f59e0b', D: '#f97316', F: 'var(--danger)' };
const RISK_MAP = { low: ['低风险', 'var(--ok)'], medium: ['中风险', '#f59e0b'], high: ['高风险', 'var(--danger)'] };
const LAYER_NAMES = {
  L1_source_grounding: '来源锚定', L2_numerical_fidelity: '数值一致',
  L3_citation_integrity: '引用完整', L4_structure_compliance: '结构规范',
  L5_llm_critique: 'LLM质疑', L6_llm_assist: 'LLM协助',
};
const RETRIEVAL_STAGES = ['bm25_retrieval', 'vector_retrieval', 'rrf_fusion', 'rerank'];

// ── Toggle helper ──
function togglePanel(id) {
  const el = document.getElementById(id);
  if (el) { el.classList.toggle('open'); }
}

// ── Render: Retrieval scoring panel ──
function renderRetrievalPanel(stages) {
  const retStages = stages.filter(s => RETRIEVAL_STAGES.includes(s.stage));
  if (!retStages.length) return '';
  const avg = retStages.reduce((a, s) => a + s.score, 0) / retStages.length;
  const avgPct = (avg * 100).toFixed(0);
  const grade = avg >= 0.9 ? 'A' : avg >= 0.75 ? 'B' : avg >= 0.6 ? 'C' : avg >= 0.4 ? 'D' : 'F';
  const gc = GRADE_COLORS[grade] || 'var(--text2)';

  let h = `<div class="score-panel" id="ret-panel">
    <div class="score-panel-header" onclick="window.__togglePanel('ret-body')">
      <span class="score-panel-arrow">▸</span>
      <span class="score-panel-title">检索评分</span>
      <span class="score-panel-badge" style="color:${gc}">${grade} ${avgPct}%</span>
      <span class="score-panel-hint">${retStages.length}个阶段</span>
    </div>
    <div class="score-panel-body" id="ret-body">`;

  retStages.forEach((s, idx) => {
    const pct = (s.score * 100).toFixed(0);
    const color = scoreColor(s.score);
    const detailId = `ret-detail-${idx}`;
    h += `<div class="sp-stage" onclick="window.__togglePanel('${detailId}')">
      <div class="sp-stage-row">
        <span class="sp-stage-name">${s.name}</span>
        <span class="sp-stage-grade" style="color:${color}">${s.grade}</span>
        <span class="sp-stage-pct" style="color:${color}">${pct}%</span>
      </div>
      <div class="sp-stage-bar"><div class="sp-stage-bar-fill" style="width:${pct}%;background:${color}"></div></div>
      <div class="sp-stage-detail" id="${detailId}">
        ${s.diagnosis ? `<div class="spd-diag">⚠ ${escHtml(s.diagnosis)}</div>` : ''}
        ${s.warnings?.length ? `<div class="spd-warn">${s.warnings.map(w => escHtml(w)).join('；')}</div>` : ''}
        ${s.suggestions?.length ? `<div class="spd-sug">💡 ${s.suggestions.map(x => escHtml(x)).join('；')}</div>` : ''}
        ${renderDetailKv(s.details)}
        <div class="spd-meta">耗时 ${s.elapsed_ms}ms</div>
      </div>
    </div>`;
  });

  // Non-retrieval stages (LLM, hallucination_check, etc.)
  const otherStages = stages.filter(s => !RETRIEVAL_STAGES.includes(s.stage));
  if (otherStages.length) {
    h += `<div class="sp-divider"></div>`;
    otherStages.forEach((s, idx) => {
      const pct = (s.score * 100).toFixed(0);
      const color = scoreColor(s.score);
      const detailId = `ret-other-${idx}`;
      h += `<div class="sp-stage" onclick="window.__togglePanel('${detailId}')">
        <div class="sp-stage-row">
          <span class="sp-stage-name">${s.name}</span>
          <span class="sp-stage-grade" style="color:${color}">${s.grade}</span>
          <span class="sp-stage-pct" style="color:${color}">${pct}%</span>
        </div>
        <div class="sp-stage-bar"><div class="sp-stage-bar-fill" style="width:${pct}%;background:${color}"></div></div>
        <div class="sp-stage-detail" id="${detailId}">
          ${s.diagnosis ? `<div class="spd-diag">⚠ ${escHtml(s.diagnosis)}</div>` : ''}
          ${s.warnings?.length ? `<div class="spd-warn">${s.warnings.map(w => escHtml(w)).join('；')}</div>` : ''}
          ${s.suggestions?.length ? `<div class="spd-sug">💡 ${s.suggestions.map(x => escHtml(x)).join('；')}</div>` : ''}
          ${renderDetailKv(s.details)}
          <div class="spd-meta">耗时 ${s.elapsed_ms}ms</div>
        </div>
      </div>`;
    });
  }

  h += '</div></div>';
  return h;
}

// ── Render: Hallucination guard panel ──
function renderHallucinationPanel(hal) {
  if (!hal) return '';
  const [riskLabel, riskColor] = RISK_MAP[hal.risk] || ['—', 'var(--text3)'];
  const halPct = ((hal.overall_score || 0) * 100).toFixed(0);
  const layers = hal.layers || {};
  const layerKeys = Object.keys(layers);
  if (!layerKeys.length) return '';

  let h = `<div class="score-panel" id="hal-panel">
    <div class="score-panel-header" onclick="window.__togglePanel('hal-body')">
      <span class="score-panel-arrow">▸</span>
      <span class="score-panel-title">防幻觉校验</span>
      <span class="score-panel-badge" style="color:${riskColor}">${riskLabel} ${halPct}%</span>
      <span class="score-panel-hint">${layerKeys.length}层</span>
    </div>
    <div class="score-panel-body" id="hal-body">`;

  layerKeys.forEach((key, idx) => {
    const val = layers[key];
    const label = LAYER_NAMES[key] || key;
    const pct = ((val.score || 0) * 100).toFixed(0);
    const color = scoreColor(val.score || 0);
    const detailId = `hal-detail-${idx}`;
    const diag = val.diagnosis || '';
    h += `<div class="sp-stage" onclick="window.__togglePanel('${detailId}')">
      <div class="sp-stage-row">
        <span class="sp-stage-name">${label}</span>
        <span class="sp-stage-pct" style="color:${color}">${pct}%</span>
      </div>
      <div class="sp-stage-bar"><div class="sp-stage-bar-fill" style="width:${pct}%;background:${color}"></div></div>
      <div class="sp-stage-detail" id="${detailId}">
        ${diag ? `<div class="spd-diag">⚠ ${escHtml(diag)}</div>` : ''}
        ${val.details ? renderDetailKv(val.details) : ''}
      </div>
    </div>`;
  });

  h += '</div></div>';
  return h;
}

// ── Render: detail key-value pairs ──
function renderDetailKv(details) {
  if (!details || typeof details !== 'object') return '';
  const entries = Object.entries(details).filter(([k, v]) => v != null && v !== '' && v !== 0);
  if (!entries.length) return '';
  let h = '<div class="spd-kv">';
  entries.forEach(([k, v]) => {
    const display = typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(3)) : typeof v === 'object' ? JSON.stringify(v) : v;
    h += `<span class="spd-kv-item"><span class="spd-kv-key">${escHtml(k)}</span> <span class="spd-kv-val">${escHtml(String(display))}</span></span>`;
  });
  h += '</div>';
  return h;
}

// ── Render: Retrieval sources (expandable) ──
function renderSources(retrieval) {
  if (!retrieval?.length) return '';
  let h = `<div class="result-section">
    <div class="result-section-header"><span class="section-icon">📄</span> 知识库来源 <span class="section-count">${retrieval.length}</span></div>`;
  retrieval.forEach((r, i) => {
    const detailId = `src-detail-${i}`;
    h += `<div class="source-result" onclick="window.__togglePanel('${detailId}')">
      <div class="source-rank ${rankClass(r.score)}">${i + 1}</div>
      <div class="source-body">
        <div class="text">${escHtml(r.text)}</div>
        <div class="meta">
          <span>RRF: <strong style="color:${scoreColor(r.score)}">${r.score.toFixed(4)}</strong></span>
          ${r.source ? `<span>来源: ${escHtml(r.source)}</span>` : ''}
          <span class="source-expand-hint">点击展开 ▾</span>
        </div>
        <div class="source-detail" id="${detailId}">
          <div class="sd-scores">
            ${r.bm25_rank ? `<div class="sd-item"><span class="sd-label">BM25</span><span class="sd-rank">#${r.bm25_rank}</span><span class="sd-score" style="color:${scoreColor(r.bm25_score)}">${r.bm25_score.toFixed(4)}</span></div>` : '<div class="sd-item dim"><span class="sd-label">BM25</span><span>未命中</span></div>'}
            ${r.vector_rank ? `<div class="sd-item"><span class="sd-label">向量</span><span class="sd-rank">#${r.vector_rank}</span><span class="sd-score" style="color:${scoreColor(r.vector_score)}">${r.vector_score.toFixed(4)}</span></div>` : '<div class="sd-item dim"><span class="sd-label">向量</span><span>未命中</span></div>'}
          </div>
        </div>
      </div>
    </div>`;
  });
  h += '</div>';
  return h;
}

// ── KB Query ──

export async function runKBQuery() {
  const q = document.getElementById('query-input')?.value.trim();
  if (!q) return;
  if (!window._kbBuilt && !(window._kbDocs?.length)) {
    toast('请先构建知识库', 'warning');
    return;
  }
  showLoading('query-loading');
  document.getElementById('query-result').innerHTML = '';
  try {
    const topK = parseInt(document.getElementById('query-topk')?.value) || 5;
    const d = await api('/api/kb-query', { query: q, top_k: topK });
    let h = '';

    // ── 1. Answer (most important) ──
    if (d.answer) {
      h += `<div class="answer-section"><h3>💡 回答</h3><div class="answer-text">${escHtml(d.answer)}</div></div>`;
    }

    // ── 2. Retrieval sources ──
    h += renderSources(d.retrieval);

    // ── 3. News context ──
    if (d.news_context && d.news_context.length) {
      h += `<div class="result-section">
        <div class="result-section-header"><span class="section-icon">📰</span> 相关新闻 <span class="section-count">${d.news_context.length}</span></div>`;
      d.news_context.forEach(n => {
        h += `<div class="news-item"><h4>${escHtml(n.title)}</h4><div class="meta">${escHtml(n.source)} · ${escHtml(n.publish_time)}</div></div>`;
      });
      h += '</div>';
    }

    // ── 4. Scoring panels (auxiliary diagnostic, collapsed by default) ──
    const sc = d.scorecard || {};
    const stages = sc.stages || [];
    if (stages.length || d.hallucination) {
      h += '<div class="score-panels">';
      if (sc.overall_score != null) {
        const ovPct = ((sc.overall_score || 0) * 100).toFixed(0);
        const grade = sc.grade || '-';
        const gc = GRADE_COLORS[grade] || 'var(--text2)';
        const fs = d.fill_stats || {};
        h += `<div class="score-overview-strip">
          <span class="sos-item"><span class="sos-label">综合</span><span class="sos-val" style="color:${gc}">${grade} ${ovPct}%</span></span>`;
        if (d.hallucination) {
          const [rl, rc] = RISK_MAP[d.hallucination.risk] || ['—', 'var(--text3)'];
          const hp = ((d.hallucination.overall_score || 0) * 100).toFixed(0);
          h += `<span class="sos-item"><span class="sos-label">可信度</span><span class="sos-val" style="color:${rc}">${rl} ${hp}%</span></span>`;
        }
        if (fs.filled_slots != null) {
          const sp = (fs.filled_slots / Math.max(fs.total_slots || 1, 1) * 100).toFixed(0);
          h += `<span class="sos-item"><span class="sos-label">完成度</span><span class="sos-val">${fs.filled_slots}/${fs.total_slots} ${sp}%</span></span>`;
        }
        h += '</div>';
      }
      h += renderRetrievalPanel(stages);
      h += renderHallucinationPanel(d.hallucination);
      h += '</div>';
    }

    document.getElementById('query-result').innerHTML = h;
    const qcEl = document.getElementById('queryCount');
    if (qcEl) qcEl.textContent = (parseInt(qcEl.textContent) || 0) + 1;
  } catch (e) {
    document.getElementById('query-result').innerHTML = `<div class="card"><span class="tag tag-fail">Error</span> ${escHtml(e.message)}</div>`;
    toast('查询失败: ' + e.message, 'error');
  }
  hideLoading('query-loading');
}

// Expose toggle globally for onclick
window.__togglePanel = togglePanel;

// ── Mode switch ──

export function switchQueryMode(mode) {
  _queryMode = mode;
  const toggle = document.getElementById('query-mode-toggle');
  if (toggle) {
    toggle.querySelectorAll('label').forEach(l => {
      l.classList.toggle('active', l.dataset.mode === mode);
    });
  }
  // Show/hide pipeline info card
  const info = document.getElementById('pipeline-info');
  if (info) info.style.display = mode === 'pipeline' ? '' : 'none';
  // Show/hide mode-specific inputs
  document.querySelectorAll('.kb-option').forEach(el => {
    el.style.display = mode === 'kb' ? '' : 'none';
  });
  document.querySelectorAll('.pipeline-option').forEach(el => {
    el.style.display = mode === 'pipeline' ? '' : 'none';
  });
  // Update placeholder
  const input = document.getElementById('query-input');
  if (input) {
    input.placeholder = mode === 'pipeline'
      ? '输入调研话题，如：AI芯片行业最新动态'
      : '输入你的问题，如：商汤科技2024年营收多少？';
  }
  // Update flow hint
  const hint = document.getElementById('query-flow-hint');
  if (hint) {
    if (mode === 'pipeline') {
      hint.innerHTML =
        '<span class="fh-step">① 输入话题</span>' +
        '<span class="fh-arrow">→</span>' +
        '<span class="fh-step">② 系统自动抓新闻 + AI 深度分析</span>' +
        '<span class="fh-arrow">→</span>' +
        '<span class="fh-step">③ 生成结构化报告（带事实核查）</span>';
    } else {
      hint.innerHTML =
        '<span class="fh-step">① 输入问题</span>' +
        '<span class="fh-arrow">→</span>' +
        '<span class="fh-step">② 系统从知识库匹配答案</span>' +
        '<span class="fh-arrow">→</span>' +
        '<span class="fh-step">③ 点击查看来源原文</span>';
    }
  }
}

export function dispatchQuery() {
  if (_queryMode === 'pipeline') runPipelineQuery();
  else runKBQuery();
}

// ── Pipeline query ──

export async function runPipelineQuery() {
  const q = document.getElementById('query-input')?.value.trim();
  if (!q) return;
  showLoading('query-loading');
  document.getElementById('query-result').innerHTML = '';
  try {
    const template = document.getElementById('pipeline-template')?.value || 'quick';
    const maxFetch = parseInt(document.getElementById('pipeline-max-fetch')?.value) || 10;
    const d = await api('/api/pipeline', { query: q, template, max_fetch: maxFetch, verbose: false });
    let h = '';

    // ── 1. Timing strip ──
    h += `<div class="pipeline-timing">
      <div class="pt-item"><span class="pt-label">抓新闻</span><span class="pt-val">${d.fetch_ms}ms</span></div>
      <div class="pt-item"><span class="pt-label">入知识库</span><span class="pt-val">${d.index_ms}ms</span></div>
      <div class="pt-item"><span class="pt-label">AI 分析</span><span class="pt-val">${d.process_ms}ms</span></div>
      <div class="pt-item"><span class="pt-label">生成报告</span><span class="pt-val">${d.output_ms}ms</span></div>
      <div class="pt-item"><span class="pt-label">耗时</span><span class="pt-val total">${d.total_ms}ms</span></div>
    </div>`;

    // ── 2. Final output (most important) ──
    if (d.final_output) {
      h += `<div class="answer-section"><h3>💡 调研结果</h3><div class="answer-text">${escHtml(d.final_output)}</div></div>`;
    }

    // ── 3. Scorecard ──
    const sc = d.scorecard;
    if (sc) {
      const ovPct = ((sc.overall_score || 0) * 100).toFixed(0);
      const grade = sc.grade || '-';
      const gc = GRADE_COLORS[grade] || 'var(--text2)';
      h += '<div class="score-panels">';
      h += `<div class="score-overview-strip">
        <span class="sos-item"><span class="sos-label">综合</span><span class="sos-val" style="color:${gc}">${grade} ${ovPct}%</span></span>
      </div>`;
      // Render stages as retrieval panel (reuse existing renderer)
      const stages = (sc.stages || []).map(s => ({
        stage: s.name, name: s.name, score: s.score,
        grade: s.score >= 0.9 ? 'A' : s.score >= 0.75 ? 'B' : s.score >= 0.6 ? 'C' : s.score >= 0.4 ? 'D' : 'F',
        details: s.details ? (() => { try { return JSON.parse(s.details); } catch { return { info: s.details }; } })() : null,
        elapsed_ms: 0,
      }));
      h += renderRetrievalPanel(stages);
      h += '</div>';
    }

    // ── 4. Errors ──
    if (d.errors && d.errors.length) {
      h += `<div class="card" style="border-left:3px solid var(--danger)"><strong>⚠ 调研过程提示:</strong><ul style="margin:8px 0;padding-left:20px;font-size:12px">`;
      d.errors.forEach(e => { h += `<li>${escHtml(typeof e === 'string' ? e : JSON.stringify(e))}</li>`; });
      h += '</ul></div>';
    }

    document.getElementById('query-result').innerHTML = h;
    const qcEl = document.getElementById('queryCount');
    if (qcEl) qcEl.textContent = (parseInt(qcEl.textContent) || 0) + 1;
  } catch (e) {
    document.getElementById('query-result').innerHTML = `<div class="card"><span class="tag tag-fail">Error</span> ${escHtml(e.message)}</div>`;
    toast('深度调研失败: ' + e.message, 'error');
  }
  hideLoading('query-loading');
}

// ── K-line analysis ──

export async function runKlineTool() {
  const q = document.getElementById('tool-kline-q')?.value.trim();
  if (!q) return;
  const resultDiv = document.getElementById('tool-kline-result');
  resultDiv.innerHTML = '<span style="color:var(--text2)">分析中...</span>';
  try {
    const d = await api('/api/kline', {
      query: q,
      days: parseInt(document.getElementById('tool-kline-days')?.value) || 60,
      period: document.getElementById('tool-kline-period')?.value || 'daily',
    });
    if (d.error) { resultDiv.innerHTML = `<span class="tag tag-fail">${escHtml(d.error)}</span>`; return; }
    const s = d.stats || {}, ind = d.indicators || {};
    const chgColor = (s.period_change_pct || 0) >= 0 ? 'var(--ok)' : 'var(--danger)';
    let h = `<div class="kline-stats">
      <div class="kline-stat-card"><div class="label">名称</div><div class="value" style="font-size:13px">${escHtml(d.name)}</div></div>
      <div class="kline-stat-card"><div class="label">代码</div><div class="value" style="font-size:12px">${escHtml(d.ts_code)}</div></div>
      <div class="kline-stat-card"><div class="label">收盘</div><div class="value">${s.latest_close || '-'}</div></div>
      <div class="kline-stat-card"><div class="label">涨跌</div><div class="value" style="color:${chgColor}">${s.period_change_pct || '-'}%</div></div>
    </div>`;
    if (ind.macd || ind.rsi || ind.bollinger || ind.kdj) {
      h += '<div class="kline-stats">';
      if (ind.macd) h += `<div class="kline-stat-card"><div class="label">MACD</div><div class="value" style="font-size:12px">${ind.macd.signal}</div></div>`;
      if (ind.rsi) h += `<div class="kline-stat-card"><div class="label">RSI</div><div class="value">${ind.rsi.value}</div><div class="label">${ind.rsi.signal}</div></div>`;
      if (ind.kdj) h += `<div class="kline-stat-card"><div class="label">KDJ</div><div class="value" style="font-size:12px">${ind.kdj.signal}</div></div>`;
      if (ind.bollinger) h += `<div class="kline-stat-card"><div class="label">布林</div><div class="value" style="font-size:12px">${ind.bollinger.position}</div></div>`;
      h += '</div>';
    }
    if (d.analysis) h += `<div style="margin-top:12px;padding:12px;background:var(--surface2);border-radius:var(--radius-sm);font-size:13px;line-height:1.7;white-space:pre-wrap">${escHtml(d.analysis)}</div>`;
    resultDiv.innerHTML = h;
  } catch (e) {
    resultDiv.innerHTML = `<span class="tag tag-fail">${escHtml(e.message)}</span>`;
    toast('K线分析失败: ' + e.message, 'error');
  }
}
