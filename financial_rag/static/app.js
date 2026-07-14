// FinRAG — app.js | Entry point

import { apiGet } from './modules/api.js';
import { initCollapsibleCards } from './modules/ui.js';
import { renderHealthBanners, loadOverviewStats, refreshKBStatus } from './modules/overview.js';
import { refreshKBManager, buildKB, clearKB, searchKBKeyword, deleteKBKeyword, renderDocList } from './modules/kb.js';
import { loadDirBrowser, browseCustomDir, ingestCustomDir, ingestNews, initUploadZone } from './modules/ingest.js';
import { runKBQuery, runKlineTool, switchQueryMode, dispatchQuery, runPipelineQuery } from './modules/query.js';
import { analyzeNews, analyzeTopic, refreshLearningHistory } from './modules/analyze.js';
import { loadChatSessions, sendFollowup } from './modules/chat.js';

// ── Expose to global (for onclick handlers in HTML) ──
window._loadOverviewStats = loadOverviewStats;
window._refreshKBStatus = refreshKBStatus;
window._refreshKBManager = refreshKBManager;
window._buildKB = buildKB;
window._clearKB = clearKB;
window._searchKBKeyword = searchKBKeyword;
window._deleteKBKeyword = deleteKBKeyword;
window._browseCustomDir = browseCustomDir;
window._ingestCustomDir = ingestCustomDir;
window._ingestNews = ingestNews;
window._runKBQuery = runKBQuery;
window._runKlineTool = runKlineTool;
window._switchQueryMode = switchQueryMode;
window._dispatchQuery = dispatchQuery;
window._runPipelineQuery = runPipelineQuery;
window._analyzeNews = analyzeNews;
window._analyzeTopic = analyzeTopic;
window._refreshLearningHistory = refreshLearningHistory;
window._loadChatSessions = loadChatSessions;
window._sendFollowup = sendFollowup;

// ── Global state ──
window._kbDocs = [];
window._kbBuilt = false;

// ── Navigation ──
document.querySelectorAll('.nav-item').forEach(el => {
  el.addEventListener('click', () => {
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    el.classList.add('active');
    const panel = document.getElementById('panel-' + el.dataset.panel);
    if (panel) panel.classList.add('active');
    // Auto-refresh on panel switch
    if (el.dataset.panel === 'analyze') refreshLearningHistory();
    if (el.dataset.panel === 'overview') loadOverviewStats();
  });
});

// ── Init ──
apiGet('/api/config').then(d => {
  document.getElementById('modelInfo').textContent = `${d.llm_model} | ${d.embedding_model} | ${d.rerank_model}`;
  if (d.mock_mode) document.getElementById('mockBadge').classList.add('active');
  const kbs = d.kb_status || {};
  if (kbs.state === 'ready') {
    document.getElementById('kbBadge').querySelector('.dot').className = 'dot dot-ok';
    document.getElementById('kbStatus').textContent = `${kbs.doc_count} 篇文档`;
  } else if (kbs.state === 'failed') {
    document.getElementById('kbBadge').querySelector('.dot').className = 'dot dot-off';
    document.getElementById('kbStatus').textContent = '索引失败';
  }
  renderHealthBanners(d.kb_status, d.init_errors || []);
  refreshKBStatus();
  refreshKBManager();
  loadDirBrowser();
  initUploadZone();
  renderDocList([]);
}).catch(e => {
  console.error('[Init] Config load failed:', e);
  renderHealthBanners(null, [{ component: 'config', error: e.message, severity: 'critical' }]);
});

loadOverviewStats();
initCollapsibleCards();
