// FinRAG — quality.js | 质量与可观测性渲染（防幻觉评分卡 / KB 检索诊断 / 上下文组装 trace / 追问证据）
//
// 这些渲染器原先散落在 analyze.js 私有函数里，追问(chat.js)拿不到。
// 抽到共享模块后：新闻解读 / 话题调研 / 开卷追问 三处共用同一套质量可视化，
// 后端新增的 sources / hallucination / kb_search_info / context_trace 字段全部落地到 UI。
import { escHtml, scoreColor } from './ui.js';

const LAYER_NAMES = {
  L1_source_grounding: '来源锚定', L2_numerical_fidelity: '数值一致',
  L3_citation_integrity: '引用完整', L4_structure_compliance: '结构规范',
};

const RISK_MAP = {
  low: ['低风险', 'var(--ok)'], medium: ['中风险', 'var(--warn)'], high: ['高风险', 'var(--danger)'],
};

// ── Guard 层得分条（评分卡与追问详情共用）──

export function renderGuardLayers(hal) {
  if (!hal || !hal.layers || !Object.keys(hal.layers).length) return '';
  let h = '<div class="qs-layers">';
  for (const [key, val] of Object.entries(hal.layers)) {
    const label = LAYER_NAMES[key] || key;
    const lp = ((val.score || 0) * 100).toFixed(0);
    const c = scoreColor(val.score || 0);
    h += `<div class="qs-layer"><div class="qs-layer-name">${label}</div><div class="qs-layer-bar"><div class="qs-layer-bar-fill" style="width:${lp}%;background:${c}"></div></div><div class="qs-layer-score" style="color:${c}">${lp}%</div></div>`;
  }
  h += '</div>';
  return h;
}

// ── 完整防幻觉评分卡（新闻解读 / 话题调研）──

export function renderHallucinationCard(hal) {
  if (!hal) return '';
  const pct = ((hal.overall_score || 0) * 100).toFixed(0);
  const [riskLabel, riskColor] = RISK_MAP[hal.risk] || ['—', 'var(--text3)'];
  let h = '<div class="quality-scorecard" style="margin-bottom:14px">';
  h += `<div class="qs-top" style="grid-template-columns:1fr 1fr">
    <div class="qs-cell"><div class="qs-label">可信度</div><div class="qs-value" style="color:${riskColor}">${riskLabel} <span style="font-size:13px;color:var(--text2)">${pct}%</span></div></div>
    <div class="qs-cell"><div class="qs-label">幻觉风险</div><div class="qs-value" style="font-size:14px;color:${riskColor}">${hal.passed ? '✅ 通过' : '❌ 未通过'}</div></div>
  </div>`;
  h += renderGuardLayers(hal);
  h += '</div>';
  return h;
}

// ── 紧凑 Guard 徽章（追问质量条 / 历史消息）──

export function guardBadge(hal) {
  if (!hal) return '';
  const [label, color] = RISK_MAP[hal.risk] || ['—', 'var(--text3)'];
  const pct = ((hal.overall_score || 0) * 100).toFixed(0);
  return `<span class="fuq-badge" style="color:${color};border-color:${color}44;background:${color}14">🛡️ ${label} ${pct}%</span>`;
}

// 历史消息只存了 risk + score 摘要（无完整 layers），用轻量徽章还原
export function guardBadgeLite(risk, score) {
  if (!risk) return '';
  const [label, color] = RISK_MAP[risk] || ['—', 'var(--text3)'];
  const pct = ((score || 0) * 100).toFixed(0);
  return `<span class="fuq-badge fuq-badge-sm" style="color:${color};border-color:${color}44;background:${color}14">🛡️ ${label} ${pct}%</span>`;
}

// ── 回答引用的证据（1-based 编号，与回答中的 [N] 引用对齐）──

export function renderSources(sources) {
  if (!sources || !sources.length) {
    return '<div class="fuq-empty">本次回答未引用知识库证据（基于案件档案与模型知识生成）</div>';
  }
  let h = '<div class="fuq-sources">';
  sources.forEach((s, i) => {
    const text = s.text || '';
    h += `<div class="fuq-source">
      <span class="fuq-src-idx">[${i + 1}]</span>
      <div class="fuq-src-body">
        <div class="fuq-src-text">${escHtml(text.slice(0, 220))}${text.length > 220 ? '…' : ''}</div>
        <div class="fuq-src-meta"><span>📄 ${escHtml(s.source || '知识库')}</span><span style="color:${scoreColor(s.score || 0)}">相关度 ${(s.score || 0).toFixed(3)}</span></div>
      </div>
    </div>`;
  });
  h += '</div>';
  return h;
}

// ── 上下文组装 trace（context-builder 可观测性面板）──
// 把后端 ContextAssemblyTrace 画出来：保真率头号指标 + 新旧对比 + 逐项取舍 + 警告。
// 目的：让用户从 UI 直接"看懂" grounding 有没有被腰斩，而非黑盒。
export function renderContextTrace(trace) {
  if (!trace) return '';
  const items = trace.items || [];
  const avail = trace.available_chars || 0;
  const fid = trace.fidelity || 0;
  const fidPct = (fid * 100).toFixed(0);
  const fidColor = fid >= 0.8 ? 'var(--ok)' : fid >= 0.5 ? 'var(--warn)' : 'var(--danger)';
  // 数据驱动对比：同样召回下，旧 [:150]×3 规则会保留多少（非硬编码，随真实数据变化）
  const oldKept = items.slice(0, 3).reduce((a, it) => a + Math.min(it.available_chars || 0, 150), 0);
  const oldFid = avail > 0 ? oldKept / avail : 0;

  let h = '<div style="margin-top:10px;padding-top:8px;border-top:1px dashed var(--border)">';
  h += '<div style="font-weight:600;color:var(--text2);margin-bottom:6px">📐 喂给 AI 的上下文组装（context-builder）</div>';

  // 保真率主条
  h += '<div style="display:flex;align-items:center;gap:8px;margin-bottom:3px">';
  h += '<span style="font-size:11px;color:var(--text3);min-width:44px">保真率</span>';
  h += `<div style="flex:1;height:8px;background:var(--surface2);border-radius:4px;overflow:hidden"><div style="width:${fidPct}%;height:100%;background:${fidColor}"></div></div>`;
  h += `<strong style="color:${fidColor};font-size:12px;min-width:38px;text-align:right">${fidPct}%</strong>`;
  h += '</div>';

  // 新旧对比一句话——"用户看懂做了什么"的关键
  h += `<div style="font-size:11px;color:var(--text3);margin-bottom:6px">现组装 <strong style="color:${fidColor}">${trace.kept_chars}</strong>/${avail} 字完整喂入`;
  if (fid - oldFid > 0.01) {
    h += ` · 旧 [:150]×3 仅 <strong style="color:var(--danger)">${oldKept}</strong> 字 (${(oldFid * 100).toFixed(0)}%)`;
  }
  h += '</div>';

  // 预算 / 规模
  h += `<div style="font-size:11px;color:var(--text3);margin-bottom:8px">预算 ${trace.budget_chars} 字 · 利用率 ${((trace.budget_utilization || 0) * 100).toFixed(0)}% · ~${trace.estimated_tokens} tokens · 纳入 ${trace.included_items}/${trace.input_items} 项${trace.dropped_items ? ` · 丢弃 ${trace.dropped_items}` : ''}</div>`;

  // 逐项取舍（full/partial/dropped + 原因）
  const statusMeta = { full: ['完整', 'var(--ok)'], partial: ['部分', 'var(--warn)'], dropped: ['丢弃', 'var(--text3)'] };
  items.forEach(it => {
    const [label, color] = statusMeta[it.status] || [it.status || '?', 'var(--text3)'];
    h += '<div style="display:flex;align-items:center;gap:6px;font-size:11px;margin-bottom:2px">';
    h += `<span style="color:var(--text3);min-width:14px;text-align:right">${it.index}</span>`;
    h += `<span style="min-width:30px;color:${color};font-weight:600">${label}</span>`;
    h += `<span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--text2)">${escHtml(it.source || '—')}</span>`;
    h += `<span style="color:var(--text3);min-width:62px;text-align:right">${it.kept_chars}/${it.available_chars}字</span>`;
    h += `<span style="color:${scoreColor(it.score)};min-width:34px;text-align:right">${(it.score || 0).toFixed(2)}</span>`;
    h += '</div>';
    if (it.reason) h += `<div style="font-size:10px;color:var(--text3);margin:-1px 0 3px 52px">← ${escHtml(it.reason)}</div>`;
  });

  // 警告（静默腰斩 → 显式可见）
  (trace.warnings || []).forEach(w => {
    h += `<div style="font-size:11px;color:var(--warn);margin-top:5px">⚠️ ${escHtml(w)}</div>`;
  });

  h += '</div>';
  return h;
}

// ── KB 检索诊断（折叠面板，恒返回，即便 0 命中也展示过程）──

export function renderKbSearchInfo(info, sources) {
  if (!info) return '';
  let h = '<div data-collapsible="true" style="margin-top:12px;border:1px solid var(--border);border-radius:var(--radius-sm);overflow:hidden">';
  h += '<div class="collapsible-header" style="padding:8px 12px;cursor:pointer;display:flex;justify-content:space-between;align-items:center;background:var(--surface2);font-size:12px;font-weight:600">';
  const srcCount = (sources || []).length;
  const rawCount = info.total_results || 0;
  const statusIcon = srcCount > 0 ? '✅' : (info.kb_built ? '⚠️' : '❌');
  // 保真率进标题：即便折叠，用户也能一眼看到"喂给 AI 的内容完整度"
  const ct0 = info.context_trace;
  const fidBadge = ct0 ? ` · <span style="color:${ct0.fidelity >= 0.8 ? 'var(--ok)' : ct0.fidelity >= 0.5 ? 'var(--warn)' : 'var(--danger)'};font-weight:700">保真率 ${(ct0.fidelity * 100).toFixed(0)}%</span>` : '';
  h += `<span>${statusIcon} KB 检索: ${srcCount} 命中 / ${rawCount} 原始${fidBadge}</span>`;
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
  h += renderContextTrace(info.context_trace);
  h += '</div></div>';
  return h;
}

// ── 追问质量条（开卷追问专属）──
// 折叠态一眼看到：Guard 风险 / 引用来源数 / 保真率 / 耗时；展开看证据 + 层得分 + 检索诊断 + 组装 trace。
export function renderFollowupQuality(d) {
  if (!d || (!d.hallucination && !d.sources && !d.context_trace)) return '';
  const hal = d.hallucination;
  const srcCount = (d.sources || []).length;
  const trace = d.context_trace;
  const fid = trace ? (trace.fidelity || 0) : null;
  const fidColor = fid == null ? '' : fid >= 0.8 ? 'var(--ok)' : fid >= 0.5 ? 'var(--warn)' : 'var(--danger)';
  const secs = d.elapsed_ms ? (d.elapsed_ms / 1000).toFixed(1) : null;

  let h = '<div class="fu-quality" data-collapsible="true">';
  h += '<div class="collapsible-header fu-quality-header">';
  h += '<span class="fuq-badges">';
  h += guardBadge(hal);
  h += `<span class="fuq-badge fuq-badge-neutral">📚 ${srcCount} 来源</span>`;
  if (fid != null) h += `<span class="fuq-badge" style="color:${fidColor};border-color:${fidColor}44;background:${fidColor}14">📐 保真 ${(fid * 100).toFixed(0)}%</span>`;
  if (secs) h += `<span class="fuq-badge fuq-badge-neutral">⏱ ${secs}s</span>`;
  h += '</span>';
  h += '<span class="fuq-expand">证据与校验详情 <span class="collapse-arrow">▶</span></span>';
  h += '</div>';

  h += '<div class="collapsible-body fu-quality-body">';
  h += `<div class="fuq-section-title">📚 回答引用的证据 (${srcCount})</div>`;
  h += renderSources(d.sources);
  if (hal && hal.layers && Object.keys(hal.layers).length) {
    h += '<div class="fuq-section-title">🛡️ 防幻觉层得分 (L1-L4)</div>';
    h += renderGuardLayers(hal);
  }
  h += renderKbSearchInfo(d.kb_search_info, d.sources);
  h += renderContextTrace(trace);
  h += '</div></div>';
  return h;
}

// ── 通用折叠面板点击委托 ──
// 修复历史遗留：.collapsible-header / .collapsible-body 结构此前没有任何点击处理器，
// KB 检索诊断面板渲染后永远无法展开。这里用事件委托统一接管（含动态插入的追问质量条）。
export function initQualityCollapsibles() {
  document.addEventListener('click', (e) => {
    const header = e.target.closest('.collapsible-header');
    if (!header) return;
    const body = header.nextElementSibling;
    if (!body || !body.classList.contains('collapsible-body')) return;
    const open = body.style.display !== 'none';
    body.style.display = open ? 'none' : 'block';
    const arrow = header.querySelector('.collapse-arrow');
    if (arrow) arrow.textContent = open ? '▶' : '▼';
    header.classList.toggle('open', !open);
  });
}