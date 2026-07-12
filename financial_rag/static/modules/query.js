// FinRAG — query.js | RAG query + K-line analysis
import { api, showLoading, hideLoading } from './api.js';
import { toast, escHtml, scoreColor, rankClass } from './ui.js';

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

    // Pipeline stages
    if (d.scorecard && d.scorecard.stages) {
      h += '<div class="pipeline-stages">';
      d.scorecard.stages.forEach(s => {
        const pct = (s.score * 100).toFixed(0);
        h += `<div class="stage-item">
          <div class="stage-name">${s.name}</div>
          <div class="stage-score" style="color:${scoreColor(s.score)}">${pct}%</div>
          <div class="stage-bar"><div class="stage-bar-fill" style="width:${pct}%;background:${scoreColor(s.score)}"></div></div>
        </div>`;
      });
      h += '</div>';
    }

    // Retrieval sources
    if (d.retrieval && d.retrieval.length) {
      h += `<div class="result-section">
        <div class="result-section-header"><span class="section-icon">📄</span> 知识库来源 <span class="section-count">${d.retrieval.length}</span></div>`;
      d.retrieval.forEach((r, i) => {
        h += `<div class="source-result">
          <div class="source-rank ${rankClass(r.score)}">${i + 1}</div>
          <div class="source-body">
            <div class="text">${escHtml(r.text)}</div>
            <div class="meta">
              <span>RRF: <strong style="color:${scoreColor(r.score)}">${r.score.toFixed(4)}</strong></span>
              ${r.bm25_rank ? `<span>BM25: #${r.bm25_rank} (${r.bm25_score.toFixed(4)})</span>` : '<span style="color:var(--text3)">BM25: -</span>'}
              ${r.vector_rank ? `<span>Vec: #${r.vector_rank} (${r.vector_score.toFixed(4)})</span>` : '<span style="color:var(--text3)">Vec: -</span>'}
              ${r.source ? `<span>来源: ${escHtml(r.source)}</span>` : ''}
            </div>
          </div>
        </div>`;
      });
      h += '</div>';
    }

    // Answer
    if (d.answer) {
      h += `<div class="answer-section"><h3>💡 回答</h3><div class="answer-text">${escHtml(d.answer)}</div></div>`;
    }

    // News context
    if (d.news_context && d.news_context.length) {
      h += `<div class="result-section">
        <div class="result-section-header"><span class="section-icon">📰</span> 相关新闻 <span class="section-count">${d.news_context.length}</span></div>`;
      d.news_context.forEach(n => {
        h += `<div class="news-item"><h4>${escHtml(n.title)}</h4><div class="meta">${escHtml(n.source)} · ${escHtml(n.publish_time)}</div></div>`;
      });
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
