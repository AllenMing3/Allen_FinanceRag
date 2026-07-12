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

    // Quality scorecard: overall + hallucination + fill stats
    if (d.scorecard || d.hallucination || d.fill_stats) {
      const sc = d.scorecard || {};
      const hal = d.hallucination || {};
      const fs = d.fill_stats || {};
      const ovPct = ((sc.overall_score || 0) * 100).toFixed(0);
      const grade = sc.grade || '-';
      const gradeColors = { A: 'var(--ok)', B: '#2563eb', C: '#f59e0b', D: '#f97316', F: 'var(--danger)' };
      const gradeColor = gradeColors[grade] || 'var(--text2)';

      const riskMap = { low: ['低风险', 'var(--ok)'], medium: ['中风险', '#f59e0b'], high: ['高风险', 'var(--danger)'] };
      const [riskLabel, riskColor] = riskMap[hal.risk] || ['—', 'var(--text3)'];
      const halPct = ((hal.overall_score || 0) * 100).toFixed(0);

      h += '<div class="quality-scorecard">';
      // Top row: overall + hallucination + fill
      h += '<div class="qs-top">';
      h += `<div class="qs-cell"><div class="qs-label">综合评分</div><div class="qs-value" style="color:${gradeColor}">${grade} <span style="font-size:13px;color:var(--text2)">${ovPct}%</span></div></div>`;
      h += `<div class="qs-cell"><div class="qs-label">可信度</div><div class="qs-value" style="color:${riskColor}">${riskLabel} <span style="font-size:13px;color:var(--text2)">${halPct}%</span></div></div>`;
      if (fs.filled_slots != null) {
        const slotPct = (fs.filled_slots / Math.max(fs.total_slots || 1, 1) * 100).toFixed(0);
        h += `<div class="qs-cell"><div class="qs-label">槽位填充</div><div class="qs-value">${fs.filled_slots}/${fs.total_slots} <span style="font-size:13px;color:var(--text2)">${slotPct}%</span></div></div>`;
      }
      h += '</div>';

      // Layer scores (hallucination check detail)
      if (hal.layers && Object.keys(hal.layers).length) {
        const layerNames = {
          L1_source_grounding: '来源锚定', L2_numerical_fidelity: '数值一致',
          L3_citation_integrity: '引用完整', L4_structure_compliance: '结构规范',
          L5_llm_critique: 'LLM质疑', L6_llm_assist: 'LLM协助',
        };
        h += '<div class="qs-layers">';
        for (const [key, val] of Object.entries(hal.layers)) {
          const label = layerNames[key] || key;
          const pct = ((val.score || 0) * 100).toFixed(0);
          const color = scoreColor(val.score || 0);
          h += `<div class="qs-layer">
            <div class="qs-layer-name">${label}</div>
            <div class="qs-layer-bar"><div class="qs-layer-bar-fill" style="width:${pct}%;background:${color}"></div></div>
            <div class="qs-layer-score" style="color:${color}">${pct}%</div>
          </div>`;
        }
        h += '</div>';
      }
      h += '</div>';
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
