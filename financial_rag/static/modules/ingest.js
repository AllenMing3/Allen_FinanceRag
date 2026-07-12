// FinRAG — ingest.js | Data import (files + news + progress)
import { api, apiGet } from './api.js';
import { toast, escHtml, progressBar } from './ui.js';
import { updateKBStatus, updateMetaStatus, refreshKBStatus } from './overview.js';
import { refreshKBManager } from './kb.js';

// ── State ──
const _dirData = {};
const _selectedFiles = {};

function _getIngestMode() {
  const radio = document.querySelector('input[name="ingestMode"]:checked');
  return radio ? radio.value : 'analyze';
}

// ── Directory browser ──

export async function loadDirBrowser() {
  try {
    const d = await apiGet('/api/directories');
    d.directories.forEach(dir => {
      if (dir.exists) {
        _dirData[dir.path] = dir.files;
        _selectedFiles[dir.path] = new Set(dir.files.map(f => f.name));
      }
    });
    renderDirBrowser(d.directories);
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
    const status = dir.exists ? `${dir.file_count} 个文件 · ${dir.total_size_kb} KB` : '目录不存在';
    const dirId = dir.path.replace(/[^a-zA-Z0-9]/g, '_');
    h += `<div class="dir-card" id="dirCard_${dirId}">`;
    h += `<div class="dir-header">`;
    h += `<div class="dir-label"><span class="dir-icon">${icon}</span> ${escHtml(dir.label)}<span class="dir-path">${escHtml(dir.path)}</span></div>`;
    h += `<span class="tag ${dir.file_count > 0 ? 'tag-info' : 'tag-off'}">${status}</span>`;
    h += `</div>`;
    if (dir.exists && dir.files.length > 0) {
      h += `<div class="dir-files">`;
      h += `<div style="display:flex;align-items:center;gap:10px;padding:4px 0;margin-bottom:6px">`;
      h += `<label style="font-size:11px;cursor:pointer;display:flex;align-items:center;gap:4px">`;
      h += `<input type="checkbox" id="selAll_${dirId}" checked onchange="window._toggleSelectAll('${escHtml(dir.path)}', this.checked)"> 全选</label>`;
      h += `<span style="font-size:11px;color:var(--text3)" id="selCount_${dirId}">${dir.files.length}/${dir.files.length} 已选</span>`;
      h += `<button class="btn btn-primary btn-sm" style="margin-left:auto" onclick="window._ingestSelected('${escHtml(dir.path)}', this)">📥 导入所选</button>`;
      h += `</div>`;
      dir.files.forEach(f => {
        const typeIcon = f.ext === '.jsonl' ? '📝' : f.ext === '.json' ? '📦' : f.ext === '.txt' ? '📄' : '📎';
        const lineInfo = f.line_count > 0 ? ` · ${f.line_count} 条` : '';
        const previewId = `preview_${dirId}_${f.name.replace(/[^a-zA-Z0-9]/g, '_')}`;
        h += `<div class="dir-file">`;
        h += `<label style="display:flex;align-items:center;gap:6px;flex:1;cursor:pointer;min-width:0">`;
        h += `<input type="checkbox" checked data-dir="${escHtml(dir.path)}" data-file="${escHtml(f.name)}" onchange="window._updateSelCount('${escHtml(dir.path)}','${dirId}')">`;
        h += `<span class="file-name">${typeIcon} ${escHtml(f.name)}</span>`;
        h += `</label>`;
        h += `<span class="file-size">${f.size_kb} KB${lineInfo}</span>`;
        h += `<button class="file-preview-btn" onclick="window._previewFile('${escHtml(dir.path)}','${escHtml(f.name)}','${previewId}')">👁</button>`;
        h += `<div id="${previewId}" class="file-preview"></div>`;
        h += `</div>`;
      });
      h += '</div>';
    }
    h += '</div>';
  });
  el.innerHTML = h;
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
  if (selAllEl) selAllEl.checked = selected.length === checkboxes.length;
  _selectedFiles[dirPath] = new Set(selected.map(cb => cb.dataset.file));
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
  if (btn) { btn.disabled = true; btn.textContent = '导入中...'; }
  try {
    const d = await api('/api/ingest/files', { dir: dirPath, analyze, files });
    updateKBStatus(d.total || 0);
    const skipInfo = d.skipped_duplicates > 0 ? ` · 跳过 ${d.skipped_duplicates} 重复` : '';
    toast(`已导入 ${d.loaded || 0} 篇${skipInfo}`, 'success');
    if (btn) { btn.textContent = `✓ ${d.loaded || 0} 篇已导入`; btn.className = 'btn btn-success btn-sm'; }
    if (d.status === 'analyzing_in_background') _pollIngestProgress(btn);
    refreshKBStatus();
    refreshKBManager();
  } catch (e) {
    if (btn) { btn.disabled = false; btn.textContent = '📥 导入所选'; }
    toast('导入失败: ' + e.message, 'error');
  }
};

function _pollIngestProgress(btn) {
  const resultEl = document.getElementById('ingest-file-result');
  const poll = setInterval(async () => {
    try {
      const p = await apiGet('/api/ingest/progress');
      const pct = p.total > 0 ? Math.round(p.current / p.total * 100) : 0;
      if (resultEl) {
        resultEl.innerHTML = `<div class="ingest-progress">
          <div class="progress-text"><span>分析中 <strong>${p.current}/${p.total}</strong></span><span>${pct}% · 失败 ${p.errors}</span></div>
          ${progressBar(pct)}
        </div>`;
      }
      if (btn) btn.textContent = `分析中 ${pct}%`;
      if (!p.running) {
        clearInterval(poll);
        if (btn) { btn.textContent = `✓ 完成: ${p.analyzed}/${p.total}`; btn.disabled = false; }
        if (resultEl) {
          resultEl.innerHTML = `<div style="margin-top:8px"><span class="tag tag-ok">完成</span> ${p.analyzed}/${p.total} 篇已分析，${p.errors} 篇失败</div>`;
        }
        toast(`分析完成: ${p.analyzed}/${p.total}`, p.errors > 0 ? 'warning' : 'success');
        refreshKBStatus();
        refreshKBManager();
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
    const skipInfo = d.skipped_duplicates > 0 ? ` · 跳过 ${d.skipped_duplicates} 篇重复` : '';
    toast(`新增 ${d.loaded || 0} 篇${skipInfo}`, 'success');
    if (d.status === 'analyzing_in_background') _pollIngestProgress(null);
    refreshKBStatus();
    refreshKBManager();
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
    if (resultEl) resultEl.innerHTML = `<div style="margin-top:8px"><span class="tag tag-ok">OK</span> ${escHtml(d.message)}</div>`;
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
