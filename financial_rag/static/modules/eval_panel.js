// FinRAG — eval_panel.js | 质量评估页签：离线体检 + AI 预分析
//
// 当前为 mock 阶段：数据源是内置示例（取自真实评估跑出的数值形态），
// 渲染器与数据源解耦——后续接真实后端时只换 loadEvalInsights 的取数逻辑，
// renderEvalCard / renderEvalPanel 不动。
import { escHtml } from './ui.js';

// ── 示例数据（mock）──
// 数值形态与真实评估一致：L2 区分度偏弱 + 编造数字短板；L1 无回归但长尾低召回。
const MOCK_INSIGHTS = [
  {
    kind: 'l2_output',
    title: '输出质量',
    icon: '📝',
    desc: 'AI 生成的简报/回答是否忠实于来源 · 22 条样本（9 正例 / 13 缺陷例）',
    timestamp: '2026-11-19 00:42',
    overall: 'watch',
    summary: '整体输出质量尚可但存在明显短板：缺陷样本占比高（13/22），编造数字与无据声明两类缺陷的得分显著偏低，且好坏样本区分度（gap=0.0878）偏小，质量尺子对好坏样本的区分能力较弱。',
    findings: [
      { severity: 'high', finding: '「编造数字」类样本数值一致率严重偏低（0.7881），为所有缺陷类型中最低，远低于全局均值 0.9324', evidence: 'fabricated_number.numerical_fidelity = 0.7881' },
      { severity: 'high', finding: '「无据声明」类样本来源锚定极低（0.7292），显著拖累整体来源锚定 0.9508', evidence: 'ungrounded_claim.source_grounding = 0.7292' },
      { severity: 'medium', finding: '低质量样本 nb_bad_fabnum_02 数值一致率仅 0.6667，存在严重数字失真，未溯源数字含「49.9%」「50亿」等关键数值', evidence: 'nb_bad_fabnum_02.numerical_fidelity = 0.6667' },
      { severity: 'low', finding: '「字段缺失」类样本完成度仅 0.8155，提示字段完整性机制在该场景失效', evidence: 'missing_field.field_completeness = 0.8155' },
    ],
    possible_causes: [
      '模型对百分比、大额货币单位（如「50亿」「400港元」）的数值解析或约束生成能力不足',
      '无据声明多为预测性/主观性陈述（如「拟派发」「预计登陆」），生成时未有效抑制无依据推断',
      '好坏样本得分分布过于集中（bad 均值 0.9122 vs good 1.0），区分度鲁棒性存疑',
    ],
    next_steps: [
      '重点分析编造数字与无据声明两类样本的生成提示词与数值校验逻辑是否缺失或失效',
      '抽查 nb_bad_fabnum_02 等低分样本的原始输入、检索片段与流水线日志，确认是否存在来源缺失或幻觉放大',
      '评估是否在后处理中增加数值一致性校验与情态动词（「预计」「拟」「将」）未锚定检测',
    ],
  },
  {
    kind: 'l1_retrieval',
    title: '检索质量',
    icon: '🔍',
    desc: '知识库能否找全、找准相关内容 · 30 条查询 × 3 检索通道',
    timestamp: '2026-11-19 00:50',
    overall: 'watch',
    summary: '整体指标稳定无回退，但存在显著低召回查询（q30/q31 召回率仅 0.3333），research 类别召回率 0.6667 低于其他类别约 0.2，需关注长尾场景覆盖能力。',
    findings: [
      { severity: 'high', finding: '存在 5 个低召回查询（召回率 ≤ 0.5），其中 q30 和 q31 仅 0.3333，远低于整体 0.8667', evidence: 'q30 / q31 / q13 / q33 / q35' },
      { severity: 'high', finding: 'research 类别召回率 0.6667、map 0.6，为所有类别中最低，分别比整体低 0.2 和 0.2226', evidence: 'research.recall = 0.6667, research.map = 0.6' },
      { severity: 'medium', finding: 'financial_report 与 non_ai 类别精确率分别仅 0.2286 和 0.2，显著低于整体 0.3163', evidence: 'financial_report.precision = 0.2286' },
      { severity: 'low', finding: '过滤模块召回率 0.8125 略低于主通道，但无泄漏、无陷阱命中', evidence: 'filter.leakage_rate = 0.0, filter.trap_hits = 0.0' },
    ],
    possible_causes: [
      '低召回查询（q30/q31）可能涉及罕见术语、歧义表达或文档覆盖不足',
      'research 类别文档 chunk 粒度粗、语义稀疏或向量表征能力弱',
      '低精确率类别可能因相关文档噪声多、词频匹配失效或向量检索引入不相关高分片段',
    ],
    next_steps: [
      '抽样分析 q30/q31 的查询文本、真实相关文档与 top5 检索结果，定位漏检原因',
      '检查 research 类别文档的切分策略与 embedding 质量，对比其向量相似度分布',
      '对低精确率类别做坏例分析，验证是否由过滤规则过松或融合权重失衡导致',
    ],
  },
];

// ── 数据源（mock 阶段返回内置示例；接真实后端时仅改这里）──
async function loadEvalInsights() {
  return MOCK_INSIGHTS;
}

// ── 渲染 ──

const OVERALL_META = {
  ok: ['✅ 正常', 'var(--ok)'],
  watch: ['⚠️ 需关注', 'var(--warn)'],
  regressed: ['🚨 存在回归', 'var(--danger)'],
};
const SEVERITY_META = {
  high: ['高', 'var(--danger)'],
  medium: ['中', 'var(--warn)'],
  low: ['低', 'var(--text3)'],
};

function renderFindings(findings) {
  if (!findings || !findings.length) return '';
  let h = '<div class="fuq-section-title">🔎 关键发现</div>';
  findings.forEach(f => {
    const [label, color] = SEVERITY_META[f.severity] || ['?', 'var(--text3)'];
    h += `<div style="margin-bottom:8px">
      <div style="font-size:12px;color:var(--text1)">
        <span class="fuq-badge fuq-badge-sm" style="color:${color};border-color:${color}44;background:${color}14;margin-right:6px">${label}</span>${escHtml(f.finding)}
      </div>
      ${f.evidence ? `<div style="font-size:11px;color:var(--text3);margin:2px 0 0 4px">证据: <code>${escHtml(f.evidence)}</code></div>` : ''}
    </div>`;
  });
  return h;
}

function renderList(title, items) {
  if (!items || !items.length) return '';
  let h = `<div class="fuq-section-title">${title}</div><ul style="margin:0 0 4px 18px;padding:0">`;
  items.forEach(it => { h += `<li style="font-size:12px;color:var(--text2);margin-bottom:4px">${escHtml(it)}</li>`; });
  return h + '</ul>';
}

// 单份 AI 预分析卡片：折叠态一眼看到总体判断 + 摘要首句，展开看发现/根因/下一步。
// 复用全局 .collapsible-header 点击委托（quality.js initQualityCollapsibles）。
export function renderEvalCard(insight) {
  const [label, color] = OVERALL_META[insight.overall] || ['—', 'var(--text3)'];
  const brief = (insight.summary || '').split('。')[0];
  let h = '<div class="card" data-collapsible="true" style="margin-bottom:14px;padding:0;overflow:hidden">';
  h += '<div class="collapsible-header" style="padding:12px 16px;cursor:pointer;display:flex;justify-content:space-between;align-items:center;gap:10px;background:var(--surface2)">';
  h += `<span style="font-size:13px;font-weight:600">${insight.icon || '📊'} ${escHtml(insight.title)}
    <span class="fuq-badge" style="color:${color};border-color:${color}44;background:${color}14;margin-left:8px">${label}</span>
    <span style="font-size:11px;color:var(--text3);font-weight:400;margin-left:8px">${escHtml(insight.timestamp || '')}</span>
    <div style="font-size:11px;color:var(--text3);font-weight:400;margin-top:3px">${escHtml(brief)}…</div>
  </span>`;
  h += '<span class="collapse-arrow">▶</span></div>';

  h += '<div class="collapsible-body" style="display:none;padding:12px 16px">';
  h += `<div style="font-size:12px;color:var(--text2);margin-bottom:6px">${escHtml(insight.desc || '')}</div>`;
  h += `<div style="font-size:12px;color:var(--text1);background:var(--surface2);border-left:3px solid ${color};padding:8px 10px;border-radius:4px;margin-bottom:10px">${escHtml(insight.summary || '')}</div>`;
  h += renderFindings(insight.findings);
  h += renderList('🧭 可能根因（AI 推测，需人工确认）', insight.possible_causes);
  h += renderList('🛠️ 建议下一步', insight.next_steps);
  h += '</div></div>';
  return h;
}

export function renderEvalPanel(insights) {
  const el = document.getElementById('evalCards');
  if (!el) return;
  el.innerHTML = insights.map(renderEvalCard).join('');
}

/** 加载质量评估面板（mock 阶段：内置示例数据，不发请求） */
export async function loadEvalPanel() {
  const insights = await loadEvalInsights();
  renderEvalPanel(insights);
}