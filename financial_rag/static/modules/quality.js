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

// 显式未校验（ReliabilityService skipped 契约）：未校验 ≠ 0 分，必须可区分
const SKIP_REASONS = { empty_answer: '无输出文本', no_sources: '无检索来源' };
const SKIPPED_BADGE = '<span class="fuq-badge" style="color:var(--text3);border-color:var(--border);background:var(--surface2)">⚪ 未校验</span>';

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
// ctx 可选：{kbSearchInfo, entities, topic}，传入后展开区附"低分缺什么知识"建议。

export function renderHallucinationCard(hal, ctx = {}) {
  if (!hal) return '';
  if (hal.skipped) {
    const reason = SKIP_REASONS[hal.skip_reason] || hal.skip_reason || '';
    return `<div class="quality-scorecard" style="margin-bottom:14px"><div class="qs-top" style="grid-template-columns:1fr">
      <div class="qs-cell"><div class="qs-label">幻觉风险</div><div class="qs-value" style="font-size:14px;color:var(--text3)">⚪ 未校验${reason ? `（${escHtml(reason)}）` : ''}</div></div>
    </div></div>`;
  }
  const pct = ((hal.overall_score || 0) * 100).toFixed(0);
  const [riskLabel, riskColor] = RISK_MAP[hal.risk] || ['—', 'var(--text3)'];
  let h = '<div class="quality-scorecard" style="margin-bottom:14px">';
  h += `<div class="qs-top" style="grid-template-columns:1fr 1fr">
    <div class="qs-cell"><div class="qs-label">可信度</div><div class="qs-value" style="color:${riskColor}">${riskLabel} <span style="font-size:13px;color:var(--text2)">${pct}%</span></div></div>
    <div class="qs-cell"><div class="qs-label">幻觉风险</div><div class="qs-value" style="font-size:14px;color:${riskColor}">${hal.passed ? '✅ 通过' : '❌ 未通过'}</div></div>
  </div>`;
  h += renderGuardLayers(hal);
  // 计分解释 + 知识缺口建议：分数不能是黑盒，明细后端已透出，这里如实讲给用户
  if (hal.layers && Object.keys(hal.layers).length) {
    h += '<div data-collapsible="true" style="margin-top:8px;border-top:1px dashed var(--border)">';
    h += '<div class="collapsible-header" style="padding:6px 0;cursor:pointer;display:flex;justify-content:space-between;align-items:center;font-size:11px;color:var(--text2);font-weight:600"><span>🧮 这个分数怎么算的 · 低分缺什么</span><span class="collapse-arrow">▶</span></div>';
    h += '<div class="collapsible-body" style="display:none;font-size:11px;color:var(--text2)">';
    h += renderScoreExplain(hal);
    h += renderScoreSuggestions(hal, ctx);
    h += '</div></div>';
  }
  h += '</div>';
  return h;
}

// ── 计分解释（确定性规则，与 guard/reflector.py 的权重/阈值保持一致）──

const LAYER_WEIGHTS_PCT = {
  L1_source_grounding: '41.7', L2_numerical_fidelity: '25',
  L3_citation_integrity: '16.7', L4_structure_compliance: '16.7',
};

function renderScoreExplain(hal) {
  const layers = hal.layers || {};
  let h = `<div style="margin-bottom:8px;line-height:1.6">总分 = 四层加权平均（来源锚定 ${LAYER_WEIGHTS_PCT.L1_source_grounding}% · 数值一致 ${LAYER_WEIGHTS_PCT.L2_numerical_fidelity}% · 引用完整 ${LAYER_WEIGHTS_PCT.L3_citation_integrity}% · 结构规范 ${LAYER_WEIGHTS_PCT.L4_structure_compliance}%）。<strong>≥60% 判"通过"</strong>。风险分级：≥80% 且无未锚定句 = 低；≥60% 且未锚定≤2句 = 中；其余 = 高。</div>`;

  const l1 = layers.L1_source_grounding;
  if (l1) {
    h += `<div style="margin-bottom:6px"><strong>来源锚定 ${((l1.score || 0) * 100).toFixed(0)}%</strong> = 有来源支撑的句子 ÷ 总句数（${l1.anchored || 0}/${l1.total || 0}）。每句与新闻/知识库做分词重叠比对，重叠率≥30% 记为有支撑。`;
    if (l1.unanchored && l1.unanchored.length) {
      h += '<div style="color:var(--warn);margin-top:3px">无直接支撑的句子（多为模型推理）：</div>';
      l1.unanchored.forEach(s => { h += `<div style="color:var(--text3);padding-left:8px">· ${escHtml(s)}</div>`; });
    }
    h += '</div>';
  }

  const l2 = layers.L2_numerical_fidelity;
  if (l2) {
    h += `<div style="margin-bottom:6px"><strong>数值一致 ${((l2.score || 0) * 100).toFixed(0)}%</strong> = 能在来源中核实的数字 ÷ 数字总数（${l2.verified || 0}/${l2.total || 0}）。`;
    if (l2.unmatched && l2.unmatched.length) {
      h += `<div style="color:var(--warn);margin-top:3px">对不上的数字：<strong>${l2.unmatched.map(escHtml).join('、')}</strong></div>`;
    }
    h += '</div>';
  }

  const l3 = layers.L3_citation_integrity;
  if (l3) {
    h += `<div style="margin-bottom:6px"><strong>引用完整 ${((l3.score || 0) * 100).toFixed(0)}%</strong>：`;
    if (l3.citations_found > 0) {
      h += `有效引用 ÷ 引用总数（${l3.valid || 0}/${l3.citations_found}）。`;
    } else {
      h += `深度分析不强制 [N] 引用标记，${l3.has_text_reference ? '检测到文字引用表述，计 80%' : '未检测到引用表述，计 60%'}。`;
    }
    h += '</div>';
  }

  const l4 = layers.L4_structure_compliance;
  if (l4) {
    h += `<div style="margin-bottom:6px"><strong>结构规范 ${((l4.score || 0) * 100).toFixed(0)}%</strong> = 命中的期望段落 ÷ 4（关键信号/影响分析/风险提示/后续关注）。`;
    if (l4.missing_sections && l4.missing_sections.length) {
      h += `<div style="color:var(--warn);margin-top:3px">缺失段落：${l4.missing_sections.map(escHtml).join('、')}</div>`;
    }
    h += '</div>';
  }
  return h;
}

// ── 低分归因与知识补充建议（纯确定性推导，不调 LLM）──

export function renderScoreSuggestions(hal, ctx = {}) {
  const tips = [];
  const info = ctx.kbSearchInfo;
  const companies = (ctx.entities || {}).companies || [];
  const firstCompany = companies.length ? (typeof companies[0] === 'string' ? companies[0] : companies[0].name || '') : '';
  const subject = ctx.topic || firstCompany || '本话题';

  if (info) {
    if (!info.kb_built) {
      tips.push('知识库尚未构建，本次分析完全没有背景资料支撑。请在「数据管理」导入文档并构建索引。');
    } else if ((info.total_results || 0) === 0) {
      tips.push(`知识库检索不到与「${subject}」相关的背景资料，结论只能依赖输入内容本身。建议导入相关公司/行业文档。`);
    } else if ((info.above_threshold || 0) === 0) {
      tips.push(`检索到 ${info.total_results} 条资料但相关度全部低于阈值 ${info.threshold || 0.4}，未被采纳。可补充与「${subject}」更直接相关的文档。`);
    }
  }

  const l2 = (hal.layers || {}).L2_numerical_fidelity;
  if (l2 && l2.unmatched && l2.unmatched.length) {
    tips.push(`以下数字在新闻与知识库中均无法核实：${l2.unmatched.join('、')}。可能是模型推算，补充含这些数据的原始资料可提升可信度。`);
  }
  const l1 = (hal.layers || {}).L1_source_grounding;
  if (l1 && l1.unanchored && l1.unanchored.length) {
    tips.push(`有 ${l1.unanchored.length} 句分析找不到直接来源支撑（明细见上），属于模型推理，建议人工复核或补充佐证资料。`);
  }

  if (!tips.length) return '';
  let h = '<div style="margin-top:8px;padding-top:6px;border-top:1px dashed var(--border)"><strong>🧩 低分原因与可补充的知识</strong>';
  tips.forEach(t => { h += `<div style="margin-top:3px;line-height:1.5">💡 ${escHtml(t)}</div>`; });
  h += '</div>';
  return h;
}

// ── 结论出处（模型 / 生成路径 / KB 参考 / 时间）──

const PATH_LABELS = {
  llm_json: 'LLM 结构化分析', llm_freetext: 'LLM 降级分析（自由文本解析）',
  llm_failed: 'LLM 分析失败', rule: '规则研判（无 LLM）', no_llm: '基础摘要（无 LLM）',
};

export function renderProvenance(prov) {
  if (!prov || !Object.keys(prov).length) return '';
  const pathLabel = PATH_LABELS[prov.path] || '未知路径';
  let h = '<div style="margin-bottom:14px;padding:8px 12px;background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius-sm);font-size:11px;color:var(--text2)">';
  h += '<div style="font-weight:600;margin-bottom:4px">🔎 结论出处</div>';
  h += `<div>模型：<strong>${escHtml(prov.model || '未知')}</strong>${prov.provider ? `（${escHtml(prov.provider)}）` : ''} · 生成方式：${escHtml(pathLabel)}</div>`;
  if ((prov.kb_hits || 0) > 0) {
    const docs = (prov.kb_documents || []).map(escHtml).join('、');
    h += `<div>知识库参考：${prov.kb_hits} 条证据${docs ? `（${docs}）` : ''}</div>`;
  } else {
    h += '<div style="color:var(--warn)">知识库参考：0 条 — 本结论未参考知识库背景，仅基于输入内容</div>';
  }
  if (prov.generated_at) h += `<div style="color:var(--text3)">生成时间：${escHtml(prov.generated_at)}</div>`;
  h += '</div>';
  return h;
}

// ── 紧凑 Guard 徽章（追问质量条 / 历史消息）──

export function guardBadge(hal) {
  if (!hal) return '';
  if (hal.skipped) return SKIPPED_BADGE;
  const [label, color] = RISK_MAP[hal.risk] || ['—', 'var(--text3)'];
  const pct = ((hal.overall_score || 0) * 100).toFixed(0);
  return `<span class="fuq-badge" style="color:${color};border-color:${color}44;background:${color}14">🛡️ ${label} ${pct}%</span>`;
}

// 历史消息只存了 risk + score 摘要（无完整 layers），用轻量徽章还原
export function guardBadgeLite(risk, score) {
  if (!risk) return '';
  if (risk === 'skipped') return SKIPPED_BADGE;
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
  // 自主检索徽标：折叠态也能看到“LLM 自主决策了几轮”
  const at0 = info.agentic_trace;
  const atBadge = (at0 && at0.enabled) ? ` · <span style="font-weight:700">🤖 自主 ${(at0.rounds || []).length} 轮</span>` : '';
  h += `<span>${statusIcon} KB 检索: ${srcCount} 命中 / ${rawCount} 原始${fidBadge}${atBadge}</span>`;
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
  // 自主检索决策轨迹：逐轮检索词/命中/分数/收工原因（0 命中也完整展示过程）
  const at = info.agentic_trace;
  if (at && at.enabled) {
    const stopLabel = { llm_stop: 'LLM 自主收工', budget: '轮数预算耗尽', fallback: '异常降级单发' }[at.stop_reason] || at.stop_reason;
    h += `<div style="margin-top:6px;padding-top:6px;border-top:1px dashed var(--border)"><strong>🤖 自主检索:</strong> ${(at.rounds || []).length} 轮 / ${at.tool_calls || 0} 次搜索 · ${stopLabel} · ${Math.round(at.elapsed_ms || 0)}ms</div>`;
    (at.rounds || []).forEach(rd => {
      const qs = (rd.queries || []).map(q => escHtml(String(q).slice(0, 40))).join(' / ') || '—';
      h += `<div style="margin-top:3px">第${rd.round}轮: ${qs} → ${rd.total_found} 条原始${rd.gate_blocked ? ' · ⛔ 门控拦截' : ''}`;
      if (rd.top_scores && rd.top_scores.length) h += ` · top分 ${rd.top_scores.map(s => s.toFixed(3)).join(', ')}`;
      h += '</div>';
    });
    if (at.error) h += `<div style="color:var(--warn);margin-top:3px">⚠️ ${escHtml(at.error)}</div>`;
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