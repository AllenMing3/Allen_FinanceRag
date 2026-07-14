// FinRAG — analyze.js | News analysis + topic research + learning history
import { api, apiGet, showLoading, hideLoading } from './api.js';
import { toast, escHtml, scoreColor } from './ui.js';
import { verdictBadge, confidenceBadge, directionArrow, severityBar, sentimentPill } from './render.js';

// ── Shared: KB retrieval diagnostics ──

function renderKbSearchInfo(info, sources) {
  if (!info) return '';
  let h = '<div data-collapsible="true" style="margin-top:12px;border:1px solid var(--border);border-radius:var(--radius-sm);overflow:hidden">';
  h += '<div class="collapsible-header" style="padding:8px 12px;cursor:pointer;display:flex;justify-content:space-between;align-items:center;background:var(--surface2);font-size:12px;font-weight:600">';
  const srcCount = (sources || []).length;
  const rawCount = info.total_results || 0;
  const statusIcon = srcCount > 0 ? '✅' : (info.kb_built ? '⚠️' : '❌');
  h += `<span>${statusIcon} KB 检索: ${srcCount} 命中 / ${rawCount} 原始</span>`;
  h += '<span class="collapse-arrow">▶</span>';
  h += '</div>';
  h += '<div class="collapsible-body" style="padding:8px 12px;font-size:11px;color:var(--text2);display:none">';

  // Query used
  h += `<div style="margin-bottom:6px"><strong>检索词:</strong> ${escHtml((info.query || '').slice(0, 100))}${(info.query || '').length > 100 ? '...' : ''}</div>`;

  // KB status
  if (!info.kb_built) {
    h += `<div style="color:var(--danger)">❌ ${escHtml(info.reason || 'KB 未构建，请先导入数据并构建索引')}</div>`;
  } else if (info.reason) {
    h += `<div style="color:var(--danger)">❌ ${escHtml(info.reason)}</div>`;
  } else {
    // Scores
    h += `<div style="margin-bottom:4px"><strong>阈值:</strong> ${info.threshold || 0.4}</div>`;
    h += `<div style="margin-bottom:4px"><strong>原始结果数:</strong> ${rawCount} | <strong>通过阈值:</strong> ${info.above_threshold || 0}</div>`;
    if (info.top_scores && info.top_scores.length) {
      h += '<div style="margin-bottom:4px"><strong>Top 5 分数:</strong> ';
      h += info.top_scores.map(s => {
        const c = s >= (info.threshold || 0.4) ? 'var(--ok)' : 'var(--danger)';
        return `<span style="color:${c};font-weight:600">${s.toFixed(4)}</span>`;
      }).join(', ');
      h += '</div>';
    }
    if (rawCount === 0) {
      h += '<div style="color:var(--text3);margin-top:4px">💡 检索无结果。可能原因：KB 中没有相关内容，或检索词与 KB 文档重叠度低</div>';
    } else if ((info.above_threshold || 0) === 0) {
      h += `<div style="color:var(--danger);margin-top:4px">💡 ${rawCount} 条结果全部低于阈值 ${info.threshold}，被过滤。尝试降低阈值或丰富 KB 内容</div>`;
    }
  }
  h += '</div></div>';
  return h;
}

// ── Shared: hallucination scorecard ──

function renderHallucinationCard(hal) {
  if (!hal) return '';
  const pct = ((hal.overall_score || 0) * 100).toFixed(0);
  const riskMap = { low: ['低风险', 'var(--ok)'], medium: ['中风险', '#f59e0b'], high: ['高风险', 'var(--danger)'] };
  const [riskLabel, riskColor] = riskMap[hal.risk] || ['—', 'var(--text3)'];
  const layerNames = {
    L1_source_grounding: '来源锚定', L2_numerical_fidelity: '数值一致',
    L3_citation_integrity: '引用完整', L4_structure_compliance: '结构规范',
    L5_llm_critique: 'LLM质疑', L6_llm_assist: 'LLM协助',
  };
  let h = '<div class="quality-scorecard" style="margin-bottom:14px">';
  h += `<div class="qs-top" style="grid-template-columns:1fr 1fr">
    <div class="qs-cell"><div class="qs-label">可信度</div><div class="qs-value" style="color:${riskColor}">${riskLabel} <span style="font-size:13px;color:var(--text2)">${pct}%</span></div></div>
    <div class="qs-cell"><div class="qs-label">幻觉风险</div><div class="qs-value" style="font-size:14px;color:${riskColor}">${hal.passed ? '✅ 通过' : '❌ 未通过'}</div></div>
  </div>`;
  if (hal.layers && Object.keys(hal.layers).length) {
    h += '<div class="qs-layers">';
    for (const [key, val] of Object.entries(hal.layers)) {
      const label = layerNames[key] || key;
      const lp = ((val.score || 0) * 100).toFixed(0);
      const c = scoreColor(val.score || 0);
      h += `<div class="qs-layer"><div class="qs-layer-name">${label}</div><div class="qs-layer-bar"><div class="qs-layer-bar-fill" style="width:${lp}%;background:${c}"></div></div><div class="qs-layer-score" style="color:${c}">${lp}%</div></div>`;
    }
    h += '</div>';
  }
  h += '</div>';
  return h;
}

// ── News analysis ──

export async function analyzeNews() {
  const text = document.getElementById('analyze-news-text')?.value.trim();
  if (!text) return;
  const query = document.getElementById('analyze-news-q')?.value.trim() || '';
  showLoading('analyze-news-loading');
  document.getElementById('analyze-news-result').innerHTML = '';
  const steps = ['① 抽取实体与指标...', '② 查询知识库...', '③ LLM 综合研判...', '④ 防幻觉校验...'];
  let stepIdx = 0;
  const loadingEl = document.querySelector('#analyze-news-loading p');
  if (loadingEl) loadingEl.textContent = steps[0];
  const progressTimer = setInterval(() => { stepIdx = Math.min(stepIdx + 1, steps.length - 1); if (loadingEl) loadingEl.textContent = steps[stepIdx]; }, 4000);
  try {
    const d = await api('/api/analyze/news', { text, query });
    const s = d.structured || {};
    let h = '<div style="margin-top:12px">';

    // Verdict
    h += `<div class="verdict-center">${verdictBadge(d.assessment)}${confidenceBadge(d.confidence)}</div>`;
    if (d.saved_to_kb) h += `<div style="text-align:center;font-size:11px;color:var(--text3);margin-bottom:8px">💾 分析结论已存入知识库</div>`;
    h += renderHallucinationCard(d.hallucination);

    // Multi-dim impact
    if (s.impact) {
      const dims = [
        { key: 'industry', icon: '🏢', label: '行业' }, { key: 'company', icon: '🏭', label: '公司' },
        { key: 'tech', icon: '⚡', label: '技术' }, { key: 'market', icon: '📈', label: '市场' }
      ];
      h += '<div class="impact-grid">';
      dims.forEach(dim => {
        const v = s.impact[dim.key] || {};
        h += `<div class="impact-card">
          <div class="impact-label">${dim.icon} ${dim.label} ${directionArrow(v.direction || 'neutral')}</div>
          <div class="impact-desc">${escHtml(v.summary || '')}</div>
        </div>`;
      });
      h += '</div>';
    }

    // Key signals
    if (s.key_signals && s.key_signals.length) {
      h += '<div class="signal-list"><div class="signal-header">🔔 关键信号</div>';
      s.key_signals.forEach(sig => {
        const typeColor = sig.type === 'positive' ? 'var(--ok)' : sig.type === 'negative' ? 'var(--danger)' : 'var(--text2)';
        h += `<div class="signal-item">
          <span style="flex-shrink:0">${severityBar(sig.severity)}</span>
          <span class="signal-text">${escHtml(sig.signal || '')}</span>
          <span class="signal-type" style="color:${typeColor}">${sig.type === 'positive' ? '▲' : sig.type === 'negative' ? '▼' : '●'}</span>
        </div>`;
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
      if (ent.companies?.length) h += `<div class="stat">公司 <strong>${ent.companies.map(c => escHtml(typeof c === 'string' ? c : c.name || c)).join(', ')}</strong></div>`;
      if (ent.persons?.length) h += `<div class="stat">人物 <strong>${ent.persons.map(p => escHtml(typeof p === 'string' ? p : p.name || p)).slice(0, 3).join(', ')}</strong></div>`;
      if (ent.ai_models?.length) h += `<div class="stat">AI模型 <strong>${ent.ai_models.map(m => escHtml(typeof m === 'string' ? m : m.name || m)).slice(0, 3).join(', ')}</strong></div>`;
      if (met.revenue) h += `<div class="stat">营收 <strong>${escHtml(String(met.revenue))}</strong></div>`;
      if (met.net_profit) h += `<div class="stat">净利润 <strong>${escHtml(String(met.net_profit))}</strong></div>`;
      h += '</div>';
    }

    // Risks + Watch
    if ((s.risks && s.risks.length) || (s.watch_next && s.watch_next.length)) {
      h += '<div class="risk-watch-grid">';
      if (s.risks && s.risks.length) {
        h += '<div class="risk-box"><div class="box-title">⚠️ 风险提示</div>';
        s.risks.forEach(r => { h += `<div class="box-item">• ${escHtml(r)}</div>`; });
        h += '</div>';
      }
      if (s.watch_next && s.watch_next.length) {
        h += '<div class="watch-box"><div class="box-title">👁️ 后续关注</div>';
        s.watch_next.forEach(w => { h += `<div class="box-item">• ${escHtml(w)}</div>`; });
        h += '</div>';
      }
      h += '</div>';
    }

    // KB sources
    if (d.kb_sources && d.kb_sources.length) {
      h += `<div style="margin-top:12px;font-size:12px;color:var(--text3)">📚 KB来源 (${d.kb_sources.length})</div>`;
      d.kb_sources.slice(0, 3).forEach((s, i) => {
        h += `<div class="source-result" style="margin-top:6px"><div class="source-rank rank-high">${i + 1}</div><div class="source-body"><div class="text">${escHtml((s.text || '').slice(0, 150))}</div><div class="meta"><span>相关度: <strong style="color:${scoreColor(s.score)}">${(s.score || 0).toFixed(3)}</strong></span></div></div></div>`;
      });
    }

    // KB retrieval diagnostics (always show)
    h += renderKbSearchInfo(d.kb_search_info, d.kb_sources);

    h += '</div>';
    document.getElementById('analyze-news-result').innerHTML = h;
    if (d.saved_to_kb) refreshLearningHistory();
    if (d.session_id) window._openChat?.(d.session_id);
  } catch (e) {
    document.getElementById('analyze-news-result').innerHTML = `<div style="margin-top:8px"><span class="tag tag-fail">Error</span> ${escHtml(e.message)}</div>`;
    toast('分析失败: ' + e.message, 'error');
  }
  clearInterval(progressTimer);
  hideLoading('analyze-news-loading');
}

// ── Topic research ──

export async function analyzeTopic() {
  const topic = document.getElementById('analyze-topic-input')?.value.trim();
  if (!topic) return;
  const maxNews = parseInt(document.getElementById('analyze-topic-count')?.value) || 20;
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
    h += `<div class="verdict-center">${verdictBadge(d.assessment)}${confidenceBadge(d.confidence)}</div>`;
    if (d.saved_to_kb) h += `<div style="text-align:center;font-size:11px;color:var(--text3);margin-bottom:8px">💾 研判结论已存入知识库</div>`;
    h += renderHallucinationCard(d.hallucination);

    // Stats
    const trendDir = s.sentiment_trend || 'mixed';
    h += `<div class="stats" style="margin-bottom:14px">
      <div class="stat">话题 <strong>${escHtml(d.topic)}</strong></div>
      <div class="stat">新闻 <strong>${d.news_count}</strong></div>
      <div class="stat">情绪趋势 ${directionArrow(trendDir)} <strong style="font-size:11px">${escHtml({ improving: '回暖', deteriorating: '恶化', stable: '稳定', mixed: '分化' }[trendDir] || trendDir)}</strong></div>
    </div>`;

    // Sub-topics
    if (s.sub_topics && s.sub_topics.length) {
      h += '<div style="margin-bottom:14px"><div style="font-size:13px;font-weight:600;margin-bottom:6px">🧩 子话题聚类</div><div style="display:flex;flex-wrap:wrap;gap:8px">';
      s.sub_topics.forEach(st => {
        const sentColor = st.sentiment === 'positive' ? 'var(--ok)' : st.sentiment === 'negative' ? 'var(--danger)' : 'var(--text2)';
        h += `<div style="padding:8px 12px;background:var(--surface2);border-radius:var(--radius-sm);border-left:3px solid ${sentColor};font-size:12px;min-width:140px">
          <div style="font-weight:600;margin-bottom:3px">${escHtml(st.name || '')} ${sentimentPill(st.sentiment)}</div>
          <div style="color:var(--text2);font-size:11px">${escHtml(st.summary || '')}</div>
        </div>`;
      });
      h += '</div></div>';
    }

    // Key players
    if (s.key_players && s.key_players.length) {
      h += '<div style="margin-bottom:14px"><div style="font-size:13px;font-weight:600;margin-bottom:6px">👤 关键玩家</div>';
      s.key_players.forEach(p => {
        h += `<div style="display:flex;align-items:center;gap:8px;padding:6px 10px;background:var(--surface2);border-radius:var(--radius-sm);font-size:12px;margin-bottom:3px">
          <span style="font-weight:600;min-width:80px">${escHtml(p.name || '')}</span>
          <span style="flex:1;color:var(--text2);font-size:11px">${escHtml(p.role || '')}</span>
          ${p.mentions ? `<span class="tag tag-info">${p.mentions}次提及</span>` : ''}
        </div>`;
      });
      h += '</div>';
    }

    if (d.analysis) h += `<div class="answer-section" style="margin-bottom:12px"><h3>💡 综合分析</h3><div class="answer-text">${escHtml(d.analysis)}</div></div>`;

    if (s.investment_implication) {
      h += `<div style="padding:10px;background:rgba(78,205,196,.05);border-radius:var(--radius-sm);border:1px solid rgba(78,205,196,.15);margin-bottom:12px">
        <div style="font-size:12px;font-weight:600;color:var(--accent);margin-bottom:4px">💰 投资启示</div>
        <div style="font-size:12px">${escHtml(s.investment_implication)}</div>
      </div>`;
    }

    if (s.contrarian_signals && s.contrarian_signals.length) {
      h += `<div style="padding:10px;background:rgba(245,158,11,.05);border-radius:var(--radius-sm);border:1px solid rgba(245,158,11,.15);margin-bottom:12px">
        <div style="font-size:12px;font-weight:600;color:#f59e0b;margin-bottom:4px">🔄 逆向信号</div>`;
      s.contrarian_signals.forEach(c => { h += `<div style="font-size:11px;padding:2px 0">• ${escHtml(c)}</div>`; });
      h += '</div>';
    }

    if (s.risks && s.risks.length) {
      h += `<div style="padding:10px;background:rgba(217,48,37,.04);border-radius:var(--radius-sm);border:1px solid rgba(217,48,37,.12);margin-bottom:12px">
        <div style="font-size:12px;font-weight:600;color:var(--danger);margin-bottom:4px">⚠️ 风险提示</div>`;
      s.risks.forEach(r => { h += `<div style="font-size:11px;padding:2px 0">• ${escHtml(r)}</div>`; });
      h += '</div>';
    }

    if (d.news && d.news.length) {
      h += `<div style="margin-top:10px;font-size:12px;color:var(--text3)">📰 相关新闻</div>`;
      d.news.forEach(n => { h += `<div class="news-item"><h4>${escHtml(n.title)}</h4><div class="meta">${escHtml(n.source)} · ${escHtml(n.publish_time)}</div></div>`; });
    }

    if (d.kb_sources && d.kb_sources.length) {
      h += `<div style="margin-top:10px;font-size:12px;color:var(--text3)">📚 KB (${d.kb_sources.length})</div>`;
      d.kb_sources.slice(0, 3).forEach((s, i) => {
        h += `<div class="source-result" style="margin-top:6px"><div class="source-rank rank-high">${i + 1}</div><div class="source-body"><div class="text">${escHtml((s.text || '').slice(0, 150))}</div><div class="meta"><span>相关度: <strong style="color:${scoreColor(s.score)}">${(s.score || 0).toFixed(3)}</strong></span></div></div></div>`;
      });
    }

    // KB retrieval diagnostics (always show)
    h += renderKbSearchInfo(d.kb_search_info, d.kb_sources);

    h += '</div>';
    document.getElementById('analyze-topic-result').innerHTML = h;
    if (d.saved_to_kb) refreshLearningHistory();
    if (d.session_id) window._openChat?.(d.session_id);
  } catch (e) {
    document.getElementById('analyze-topic-result').innerHTML = `<div style="margin-top:8px"><span class="tag tag-fail">Error</span> ${escHtml(e.message)}</div>`;
    toast('调研失败: ' + e.message, 'error');
  }
  clearInterval(progressTimer);
  hideLoading('analyze-topic-loading');
}

// ── Learning history ──

export async function refreshLearningHistory() {
  const container = document.getElementById('learningHistoryList');
  if (!container) return;
  try {
    const d = await apiGet('/api/kb/history');
    if (d.count === 0) {
      container.innerHTML = `<div class="empty-state" style="padding:16px">
        <div class="icon">🧠</div><p>尚未积累学习记录</p>
        <div class="hint">每次新闻解读或话题调研的结论会自动存入知识库</div>
      </div>`;
      return;
    }
    let h = `<div style="font-size:12px;color:var(--text2);margin-bottom:10px">已积累 <strong>${d.count}</strong> 条分析结论</div>`;
    for (const item of d.history) {
      const type = (item.analysis_type || '').startsWith('news') ? '📰 新闻解读' : '🔍 话题调研';
      const verdictColor = (item.assessment || '').includes('利好') ? '#0d8a3e' : (item.assessment || '').includes('利空') ? '#d93025' : 'var(--text2)';
      h += `<div class="history-item">
        <div class="history-header">
          <span class="history-topic">${type}：${escHtml(item.topic || '未知话题')}</span>
          <span class="history-verdict" style="color:${verdictColor}">${escHtml(item.assessment || '')}</span>
        </div>
        <div class="history-time">⏱️ ${escHtml(item.timestamp || '')}</div>
      </div>`;
    }
    container.innerHTML = h;
  } catch (e) {
    container.innerHTML = `<div style="color:var(--danger);font-size:12px">加载失败: ${escHtml(e.message)}</div>`;
  }
}
