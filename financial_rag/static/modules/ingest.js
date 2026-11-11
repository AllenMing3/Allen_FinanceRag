// FinRAG — ingest.js | Data import (files + news + progress)
import { api, apiGet } from './api.js';
import { toast, escHtml, progressBar } from './ui.js';
import { updateKBStatus, updateMetaStatus, refreshKBStatus } from './overview.js';
import { refreshKBManager } from './kb.js';

// ── State ──
const _dirData = {};
const _selectedFiles = {};
let _lastIngestSummary = '';

const _MODE_HINTS = {
  analyze: '🔍 深度分析：导入后后台 AI 提炼财务指标 / 实体（需数分钟、消耗 API 额度），结果可在知识库查看',
  quick: '⚡ 快速导入：仅清洗 + 去重 + 入库，立即可检索，不调用 AI 分析',
};

const _FR_STATUS = {
  imported: ['✅', 'var(--ok)'],
  duplicate: ['🔁', 'var(--text2)'],
  rejected: ['🚫', 'var(--danger)'],
  empty: ['📭', 'var(--text3)'],
};

function _getIngestMode() {
  const radio = document.querySelector('input[name="ingestMode"]:checked');
  return radio ? radio.value : 'analyze';
}

window._updateIngestModeHint = function () {
  const el = document.getElementById('ingestModeHint');
  if (el) el.textContent = _MODE_HINTS[_getIngestMode()] || '';
};

// ── Directory browser ──

export async function loadDirBrowser() {
  try {
    const d = await apiGet('/api/directories');
    d.directories.forEach(dir => {
      if (dir.exists) {
        _dirData[dir.path] = dir.files;
        // 默认只勾未导入的文件——已导入的重复导入只会跳过，不打扰用户
        _selectedFiles[dir.path] = new Set(
          dir.files.filter(f => !(f.ingest && f.ingest.imported)).map(f => f.name));
      }
    });
    renderDirBrowser(d.directories);
    window._updateIngestModeHint();
  } catch (e) {
    document.getElementById('dirBrowser').innerHTML =
      `<div style="color:var(--danger)">加载目录失败: ${escHtml(e.message)}</div>`;
  }
}

function renderDirBrowser(dirs) {
  const el = document.getElementById('dirBrowser');
  let h = '';
  dirs.forEach(dir => {
    const icon = dir.exists ? '📂' : '📭';
    const dirId = dir.path.replace(/[^a-zA-Z0-9]/g, '_');
    const status = !dir.exists ? '目录不存在'
      : `${dir.imported_files || 0}/${dir.file_count} 已导入 · ${dir.total_size_kb} KB`;
    h += `<div class="dir-card" id="dirCard_${dirId}">`;
    h += `<div class="dir-header">`;
    h += `<div class="dir-label"><span class="dir-icon">${icon}</span> ${escHtml(dir.label)}<span class="dir-path">${escHtml(dir.path)}</span></div>`;
    h += `<span class="tag ${dir.file_count > 0 ? 'tag-info' : 'tag-off'}">${status}</span>`;
    h += `</div>`;
    if (dir.exists && dir.files.length > 0) {
      const newCount = dir.files.filter(f => !(f.ingest && f.ingest.imported)).length;
      h += `<div class="dir-files">`;
      h += `<div class="dir-toolbar">`;
      h += `<label class="dir-sel-all"><input type="checkbox" id="selAll_${dirId}" onchange="window._toggleSelectAll('${escHtml(dir.path)}', this.checked)"> 全选</label>`;
      h += `<button class="btn btn-secondary btn-xs" onclick="window._selectUnimported('${escHtml(dir.path)}')">只选未导入${newCount ? ` (${newCount})` : ''}</button>`;
      h += `<span class="dir-sel-count" id="selCount_${dirId}"></span>`;
      h += `<button class="btn btn-primary btn-sm" id="ingestBtn_${dirId}" style="margin-left:auto" onclick="window._ingestSelected('${escHtml(dir.path)}', this)">📥 导入所选</button>`;
      h += `</div>`;
      dir.files.forEach(f => {
        const typeIcon = f.ext === '.jsonl' ? '📝' : f.ext === '.json' ? '📦' : f.ext === '.txt' ? '📄' : '📎';
        const lineInfo = f.line_count > 0 ? ` · ${f.line_count} 条` : '';
        const previewId = `preview_${dirId}_${f.name.replace(/[^a-zA-Z0-9]/g, '_')}`;
        const ing = f.ingest || {};
        const checked = _selectedFiles[dir.path] && _selectedFiles[dir.path].has(f.name);
        const badge = ing.imported
          ? (ing.analyzed_count >= ing.doc_count
            ? `<span class="file-status fs-ok" title="已导入知识库，AI 分析完成">✅ 已导入 ${ing.doc_count} 篇</span>`
            : `<span class="file-status fs-warn" title="已导入知识库，AI 分析未完成">🕓 已导入 ${ing.doc_count} 篇 · 待分析</span>`)
          : '<span class="file-status fs-new">○ 未导入</span>';
        h += `<div class="dir-file${ing.imported ? ' is-imported' : ''}">`;
        h += `<label style="display:flex;align-items:center;gap:6px;flex:1;cursor:pointer;min-width:0">`;
        h += `<input type="checkbox" ${checked ? 'checked' : ''} data-dir="${escHtml(dir.path)}" data-file="${escHtml(f.name)}" onchange="window._updateSelCount('${escHtml(dir.path)}','${dirId}')">`;
        h += `<span class="file-name">${typeIcon} ${escHtml(f.name)}</span>`;
        h += `</label>`;
        h += `<span class="file-size">${f.size_kb} KB${lineInfo}</span>`;
        h += badge;
        h += `<button class="file-preview-btn" onclick="window._previewFile('${escHtml(dir.path)}','${escHtml(f.name)}','${previewId}')">👁</button>`;
        h += `<div id="${previewId}" class="file-preview"></div>`;
        h += `</div>`;
      });
      h += '</div>';
    }
    h += '</div>';
  });
  el.innerHTML = h;
  // 同步选中计数 / 导入按钮状态
  dirs.forEach(dir => {
    if (dir.exists && dir.files.length > 0) {
      window._updateSelCount(dir.path, dir.path.replace(/[^a-zA-Z0-9]/g, '_'));
    }
  });
}

// ── Global handlers ──

window._toggleSelectAll = function (dirPath, checked) {
  const checkboxes = document.querySelectorAll(`input[data-dir="${dirPath}"]`);
  checkboxes.forEach(cb => cb.checked = checked);
  _selectedFiles[dirPath] = checked ? new Set(_dirData[dirPath].map(f => f.name)) : new Set();
  const dirId = dirPath.replace(/[^a-zA-Z0-9]/g, '_');
  window._updateSelCount(dirPath, dirId);
};

window._updateSelCount = function (dirPath, dirId) {
  const checkboxes = document.querySelectorAll(`input[data-dir="${dirPath}"]`);
  const selected = Array.from(checkboxes).filter(cb => cb.checked);
  const countEl = document.getElementById(`selCount_${dirId}`);
  if (countEl) countEl.textContent = `${selected.length}/${checkboxes.length} 已选`;
  const selAllEl = document.getElementById(`selAll_${dirId}`);
  if (selAllEl) selAllEl.checked = checkboxes.length > 0 && selected.length === checkboxes.length;
  const btn = document.getElementById(`ingestBtn_${dirId}`);
  if (btn && !btn.dataset.busy) {
    btn.disabled = selected.length === 0;
    btn.textContent = selected.length > 0 ? `📥 导入所选 (${selected.length})` : '📥 导入所选';
  }
  _selectedFiles[dirPath] = new Set(selected.map(cb => cb.dataset.file));
};

window._selectUnimported = function (dirPath) {
  const files = _dirData[dirPath] || [];
  const sel = new Set(files.filter(f => !(f.ingest && f.ingest.imported)).map(f => f.name));
  document.querySelectorAll(`input[data-dir="${dirPath}"]`).forEach(cb => { cb.checked = sel.has(cb.dataset.file); });
  window._updateSelCount(dirPath, dirPath.replace(/[^a-zA-Z0-9]/g, '_'));
};

window._previewFile = async function (dirPath, fileName, previewId) {
  const el = document.getElementById(previewId);
  if (!el) return;
  if (el.style.display !== 'none' && el.innerHTML) { el.style.display = 'none'; return; }
  el.innerHTML = '<span style="color:var(--text3)">加载中...</span>';
  el.style.display = 'block';
  try {
    const d = await apiGet(`/api/file/preview?path=${encodeURIComponent(dirPath)}&file=${encodeURIComponent(fileName)}&lines=15`);
    if (d.lines && d.lines.length) {
      el.textContent = d.lines.join('\n');
      if (d.truncated) el.textContent += '\n... (更多内容省略)';
    } else {
      el.textContent = '(空文件)';
    }
  } catch (e) { el.innerHTML = `<span style="color:var(--danger)">预览失败: ${escHtml(e.message)}</span>`; }
};

window._ingestSelected = async function (dirPath, btn) {
  const files = Array.from(_selectedFiles[dirPath] || []);
  if (files.length === 0) { toast('请至少选择一个文件', 'warning'); return; }
  const analyze = _getIngestMode() === 'analyze';
  if (btn) { btn.disabled = true; btn.dataset.busy = '1'; btn.textContent = '导入中...'; }
  try {
    const d = await api('/api/ingest/files', { dir: dirPath, analyze, files });
    updateKBStatus(d.total || 0);
    renderIngestResult(d);
    const skipInfo = d.skipped_duplicates > 0 ? ` · 跳过 ${d.skipped_duplicates} 重复` : '';
    toast(`已导入 ${d.loaded || 0} 篇${skipInfo}`, 'success');
    if (d.status === 'analyzing_in_background') _pollIngestProgress(btn);
    refreshKBStatus();
    refreshKBManager();
    loadDirBrowser();  // 刷新文件状态徽章，默认勾选剩余未导入文件
  } catch (e) {
    if (btn) { btn.disabled = false; delete btn.dataset.busy; btn.textContent = '📥 导入所选'; }
    toast('导入失败: ' + e.message, 'error');
  }
};

// ── 逐文件导入结果面板（新增 / 重复 / 质量拒绝 / 解析为空）──

function _frRow(r) {
  const [icon, color] = _FR_STATUS[r.status] || ['•', 'var(--text2)'];
  const nums = [];
  if (r.loaded > 0) nums.push(`新增 ${r.loaded}`);
  if (r.duplicates > 0) nums.push(`重复 ${r.duplicates}`);
  if (r.rejected > 0) nums.push(`拒绝 ${r.rejected}`);
  if (r.status === 'empty') nums.push('解析为空');
  return `<div class="ir-file"><span>${icon}</span><span class="ir-name" style="color:${color}">${escHtml(r.file)}</span><span class="ir-nums">读 ${r.read} 篇 → ${nums.join(' · ') || '无变化'}</span></div>`;
}

function renderIngestResult(d) {
  const el = document.getElementById('ingest-file-result');
  if (!el) return;
  const fr = d.file_results || [];
  const tot = fr.reduce((a, r) => ({ dup: a.dup + r.duplicates, rej: a.rej + r.rejected }), { dup: 0, rej: 0 });
  let summary = `<span style="color:var(--ok)">✅ 新增 ${d.loaded || 0} 篇</span>`;
  if (tot.dup > 0) summary += ` · <span style="color:var(--text2)">🔁 重复跳过 ${tot.dup}</span>`;
  if (tot.rej > 0) summary += ` · <span style="color:var(--danger)">🚫 质量拒绝 ${tot.rej}</span>`;
  summary += ` <span style="color:var(--text3)">（知识库共 ${d.total || 0} 篇）</span>`;
  let h = '<div class="ingest-result">';
  h += `<div class="ir-summary">${summary}</div>`;
  if (d.message && (d.status === 'all_rejected' || d.status === 'no_new_docs')) {
    h += `<div class="ir-note">${escHtml(d.message)}</div>`;
  }
  fr.forEach(r => { h += _frRow(r); });
  const pp = d.preprocessing;
  if (pp && (pp.rejected_short || pp.rejected_irrelevant || pp.rejected_empty)) {
    const reasons = [];
    if (pp.rejected_short) reasons.push(`文本过短 ${pp.rejected_short}`);
    if (pp.rejected_irrelevant) reasons.push(`领域无关 ${pp.rejected_irrelevant}`);
    if (pp.rejected_empty) reasons.push(`空内容 ${pp.rejected_empty}`);
    h += `<div class="ir-note">🚫 拒绝原因：${reasons.join(' · ')}（清洗 + 领域相关性门控 + 最小长度）</div>`;
  }
  if (d.status === 'analyzing_in_background') {
    h += '<div class="ir-note">🔍 深度分析后台进行中，完成后徽章自动变为 ✅</div>';
  }
  h += '</div>';
  _lastIngestSummary = h;
  el.innerHTML = h;
}

function _pollIngestProgress(btn) {
  const resultEl = document.getElementById('ingest-file-result');
  const poll = setInterval(async () => {
    try {
      const p = await apiGet('/api/ingest/progress');
      const pct = p.total > 0 ? Math.round(p.current / p.total * 100) : 0;
      if (resultEl) {
        resultEl.innerHTML = _lastIngestSummary + `<div class="ingest-progress">
          <div class="progress-text"><span>AI 分析中 <strong>${p.current}/${p.total}</strong></span><span>${pct}% · 失败 ${p.errors}</span></div>
          ${progressBar(pct)}
        </div>`;
      }
      if (btn && btn.isConnected) btn.textContent = `分析中 ${pct}%`;
      if (!p.running) {
        clearInterval(poll);
        if (btn && btn.isConnected) { btn.textContent = `✓ 完成: ${p.analyzed}/${p.total}`; btn.disabled = false; delete btn.dataset.busy; }
        if (resultEl) {
          resultEl.innerHTML = _lastIngestSummary + `<div style="margin-top:8px"><span class="tag tag-ok">分析完成</span> ${p.analyzed}/${p.total} 篇已分析，${p.errors} 篇失败</div>`;
        }
        toast(`分析完成: ${p.analyzed}/${p.total}`, p.errors > 0 ? 'warning' : 'success');
        refreshKBStatus();
        refreshKBManager();
        loadDirBrowser();  // 待分析 → 已导入，刷新徽章
      }
    } catch (e) { clearInterval(poll); }
  }, 2000);
}

// ── Custom directory ──

export async function browseCustomDir() {
  const dir = document.getElementById('ingest-dir')?.value.trim();
  if (!dir) return;
  const resultEl = document.getElementById('ingest-file-result');
  if (resultEl) resultEl.innerHTML = `<div style="margin-top:8px;font-size:12px;color:var(--text2)">输入路径后点击"导入"按钮</div>`;
}

export async function ingestCustomDir() {
  const dir = document.getElementById('ingest-dir')?.value.trim();
  if (!dir) { toast('请输入目录路径', 'warning'); return; }
  const analyze = _getIngestMode() === 'analyze';
  try {
    const d = await api('/api/ingest/files', { dir, analyze });
    updateKBStatus(d.total || 0);
    renderIngestResult(d);
    const skipInfo = d.skipped_duplicates > 0 ? ` · 跳过 ${d.skipped_duplicates} 篇重复` : '';
    toast(`新增 ${d.loaded || 0} 篇${skipInfo}`, 'success');
    if (d.status === 'analyzing_in_background') _pollIngestProgress(null);
    refreshKBStatus();
    refreshKBManager();
    loadDirBrowser();
  } catch (e) { toast('导入失败: ' + e.message, 'error'); }
}

// ── File upload ──
const _uploadFiles = [];

export function initUploadZone() {
  const zone = document.getElementById('uploadZone');
  const input = document.getElementById('uploadInput');
  if (!zone || !input) return;

  zone.addEventListener('click', () => input.click());
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('dragover');
    _addUploadFiles(e.dataTransfer.files);
  });
  input.addEventListener('change', () => {
    if (input.files.length) _addUploadFiles(input.files);
    input.value = '';
  });
}

function _addUploadFiles(fileList) {
  const ALLOWED = ['.pdf', '.png', '.jpg', '.jpeg', '.webp'];
  for (const f of fileList) {
    const ext = '.' + f.name.split('.').pop().toLowerCase();
    if (!ALLOWED.includes(ext)) { toast(`跳过 ${f.name}：不支持的类型`, 'warning'); continue; }
    if (_uploadFiles.some(x => x.name === f.name)) continue;
    _uploadFiles.push(f);
  }
  _renderUploadList();
}

function _renderUploadList() {
  const el = document.getElementById('upload-file-list');
  if (!el) return;
  if (_uploadFiles.length === 0) { el.innerHTML = ''; return; }
  let h = '';
  _uploadFiles.forEach((f, i) => {
    const sizeKB = (f.size / 1024).toFixed(1);
    h += `<div class="upload-file">
      <span class="uf-name">${escHtml(f.name)}</span>
      <span class="uf-size">${sizeKB} KB</span>
      <span class="uf-remove" onclick="window._removeUploadFile(${i})">✕</span>
    </div>`;
  });
  h += `<div style="margin-top:8px;display:flex;gap:8px;align-items:center">
    <button class="btn btn-primary btn-sm" onclick="window._doUpload()">📥 上传并导入</button>
    <span style="font-size:11px;color:var(--text3)">${_uploadFiles.length} 个文件待上传</span>
  </div>`;
  el.innerHTML = h;
}

window._removeUploadFile = function (idx) {
  _uploadFiles.splice(idx, 1);
  _renderUploadList();
};

window._doUpload = async function () {
  if (_uploadFiles.length === 0) { toast('请先选择文件', 'warning'); return; }
  const resultEl = document.getElementById('upload-result');
  if (resultEl) resultEl.innerHTML = '<div style="margin-top:8px;color:var(--text3)">解析中，请稍候...</div>';
  const formData = new FormData();
  _uploadFiles.forEach(f => formData.append('files', f));
  try {
    const resp = await fetch('/api/ingest/upload', { method: 'POST', body: formData });
    if (!resp.ok) {
      const e = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(e.detail || resp.statusText);
    }
    const d = await resp.json();
    let h = `<div class="ingest-result"><div class="ir-summary">✅ ${escHtml(d.message || `已导入 ${d.loaded || 0} 篇`)}</div>`;
    (d.file_results || []).forEach(r => { h += _frRow(r); });
    h += '</div>';
    if (resultEl) resultEl.innerHTML = h;
    updateKBStatus(d.total || 0);
    refreshKBStatus();
    refreshKBManager();
    _uploadFiles.length = 0;
    _renderUploadList();
    toast(d.message, 'success');
  } catch (e) {
    if (resultEl) resultEl.innerHTML = `<div style="margin-top:8px"><span class="tag tag-fail">Error</span> ${escHtml(e.message)}</div>`;
    toast('上传失败: ' + e.message, 'error');
  }
};

// ── News fetch ──

export async function ingestNews() {
  const q = document.getElementById('ingest-news-q')?.value.trim();
  if (!q) { toast('请输入主题', 'warning'); return; }
  const maxNews = parseInt(document.getElementById('ingest-news-count')?.value) || 30;
  try {
    const d = await api('/api/ingest/news', { query: q, max_news: maxNews });
    let h = `<div style="margin-top:8px"><span class="tag tag-ok">OK</span> 抓取 ${d.fetched} 条新闻 → 元数据`;
    if (d.has_summary) h += ' + AI摘要';
    h += ` (累计: ${d.meta_total} 条)</div>`;
    if ((d.kb_added || 0) > 0 || (d.kb_skipped_duplicates || 0) > 0) {
      h += `<div style="margin-top:6px;font-size:12px;color:var(--text2)">📚 优质正文自动入知识库：<strong style="color:var(--ok)">+${d.kb_added || 0} 篇</strong>${d.kb_skipped_duplicates ? ` · 跳过重复 ${d.kb_skipped_duplicates}` : ''}（KB 共 ${d.kb_total} 篇）</div>`;
      updateKBStatus(d.kb_total || 0);
      refreshKBStatus();
      refreshKBManager();
    }
    updateMetaStatus(d.meta_total);
    if (d.headlines && d.headlines.length) {
      h += '<div style="margin-top:8px">';
      d.headlines.slice(0, 8).forEach(item => {
        h += `<div class="news-item"><h4>${escHtml(item.title)}</h4><div class="meta">${escHtml(item.source)} · ${escHtml(item.publish_time)}</div></div>`;
      });
      if (d.headlines.length > 8) h += `<div style="font-size:11px;color:var(--text3)">... 还有 ${d.headlines.length - 8} 条</div>`;
      h += '</div>';
    }
    document.getElementById('ingest-news-result').innerHTML = h;
    toast(`抓取 ${d.fetched} 条新闻`, 'success');
  } catch (e) {
    document.getElementById('ingest-news-result').innerHTML = `<div style="margin-top:8px"><span class="tag tag-fail">Error</span> ${escHtml(e.message)}</div>`;
    toast('抓取失败: ' + e.message, 'error');
  }
}
