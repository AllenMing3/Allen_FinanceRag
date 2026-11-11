// FinRAG — chat.js | Conversation management
import { api, apiGet, apiDelete } from './api.js';
import { toast, escHtml } from './ui.js';
import { renderMarkdown } from './render.js';
import { renderFollowupQuality, guardBadgeLite } from './quality.js';

let chatSessionId = null;

export function getChatSessionId() { return chatSessionId; }

export async function openChat(sessionId) {
  chatSessionId = sessionId;
  window._openChat = openChat; // expose globally for analyze.js
  const section = document.getElementById('chatSection');
  if (section) section.style.display = 'block';
  try {
    const d = await apiGet(`/api/chat/sessions/${sessionId}`);
    renderChatMessages(d.messages || []);
    loadChatSessions();
  } catch (e) {
    console.error('Failed to load chat session:', e);
  }
  document.getElementById('chatFollowupInput')?.focus();
}

// Expose globally
window._openChat = openChat;

function renderChatMessages(messages) {
  const el = document.getElementById('chatMessages');
  if (!el) return;
  if (!messages.length) {
    el.innerHTML = '<div style="text-align:center;color:var(--text3);padding:40px;font-size:13px">开始追问吧 💬</div>';
    return;
  }
  let h = '';
  messages.forEach(msg => {
    const cls = msg.role === 'user' ? 'user' : 'assistant';
    const content = cls === 'assistant' ? renderMarkdown(msg.content) : escHtml(msg.content);
    h += `<div class="chat-msg ${cls}">${content}`;
    // 助手历史消息：从落盘 metadata 还原 Guard 徽章 / 来源数 / 耗时
    const meta = msg.metadata || {};
    if (cls === 'assistant' && (meta.guard_risk || meta.elapsed_ms || meta.source_count != null)) {
      h += '<div class="msg-meta-row">';
      h += guardBadgeLite(meta.guard_risk, meta.guard_score);
      if (meta.source_count != null) h += `<span class="fuq-badge fuq-badge-neutral fuq-badge-sm">📚 ${meta.source_count} 来源</span>`;
      if (meta.elapsed_ms) h += `<span class="msg-meta-time">⏱ ${(meta.elapsed_ms / 1000).toFixed(1)}s</span>`;
      h += '</div>';
    }
    if (msg.timestamp) h += `<div class="msg-time">${msg.timestamp}</div>`;
    h += '</div>';
  });
  el.innerHTML = h;
  el.scrollTop = el.scrollHeight;
}

export async function loadChatSessions() {
  try {
    const d = await apiGet('/api/chat/sessions');
    const list = d.sessions || [];
    const el = document.getElementById('chatSessionList');
    if (!el) return;
    if (!list.length) {
      el.innerHTML = '<div style="text-align:center;color:var(--text3);padding:16px;font-size:11px">尚无会话<br>分析新闻或话题后自动创建</div>';
      return;
    }
    let h = '';
    list.forEach(s => {
      const active = s.id === chatSessionId ? 'active' : '';
      const typeClass = s.type === 'news' ? 'news' : 'topic';
      const typeLabel = s.type === 'news' ? '新闻' : '话题';
      h += `<div class="chat-session-item ${active}" onclick="window._openChat('${s.id}')">
        <span class="chat-session-type ${typeClass}">${typeLabel}</span>
        <span class="session-title">${escHtml(s.title)}</span>
        <span class="session-del" onclick="event.stopPropagation();window._deleteChatSession('${s.id}')">×</span>
      </div>`;
    });
    el.innerHTML = h;
  } catch (e) {
    console.error('Failed to load sessions:', e);
  }
}

window._deleteChatSession = async function (id) {
  const ok = await import('./ui.js').then(m => m.confirmDialog('删除会话', '确定删除这个会话？', { danger: true, confirmText: '删除' }));
  if (!ok) return;
  try {
    await apiDelete(`/api/chat/sessions/${id}`);
    toast('会话已删除', 'info');
    if (chatSessionId === id) {
      chatSessionId = null;
      // Try to auto-switch to the first remaining session
      try {
        const d = await apiGet('/api/chat/sessions');
        const remaining = d.sessions || [];
        if (remaining.length > 0) {
          openChat(remaining[0].id);
          return; // openChat handles loadChatSessions
        }
      } catch (_) { /* silent */ }
      // No sessions left — clear and hide
      const msgEl = document.getElementById('chatMessages');
      if (msgEl) msgEl.innerHTML = '';
      const section = document.getElementById('chatSection');
      if (section) section.style.display = 'none';
    }
    loadChatSessions();
  } catch (e) {
    console.error('Delete failed:', e);
    toast('删除失败: ' + e.message, 'error');
  }
};

export async function sendFollowup() {
  if (!chatSessionId) return;
  const input = document.getElementById('chatFollowupInput');
  const msg = input?.value.trim();
  if (!msg) return;
  input.value = '';
  const messagesEl = document.getElementById('chatMessages');
  if (!messagesEl) return;
  if (messagesEl.querySelector('div[style*="text-align:center"]')) messagesEl.innerHTML = '';
  messagesEl.innerHTML += `<div class="chat-msg user">${escHtml(msg)}</div>`;
  messagesEl.innerHTML += `<div class="chat-msg loading" id="chatLoading"><span class="spinner"></span> 分析中...</div>`;
  messagesEl.scrollTop = messagesEl.scrollHeight;
  input.disabled = true;
  const sendBtn = document.getElementById('chatSendBtn');
  if (sendBtn) sendBtn.disabled = true;
  try {
    const d = await api('/api/chat/followup', { session_id: chatSessionId, message: msg });
    const loading = document.getElementById('chatLoading');
    if (loading) loading.remove();
    if (d.error) {
      messagesEl.innerHTML += `<div class="chat-msg assistant" style="color:var(--danger)">❌ ${escHtml(d.error)}</div>`;
      toast(d.error, 'error');
    } else {
      const now = new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
      // 开卷追问质量闭环：回答 + 证据/Guard/检索诊断/组装 trace（后端新增字段全部落地）
      messagesEl.innerHTML += `<div class="chat-msg assistant">${renderMarkdown(d.answer)}${renderFollowupQuality(d)}<div class="msg-time">${now}${d.elapsed_ms ? ` · ${(d.elapsed_ms / 1000).toFixed(1)}s` : ''}</div></div>`;
    }
    messagesEl.scrollTop = messagesEl.scrollHeight;
  } catch (e) {
    const loading = document.getElementById('chatLoading');
    if (loading) loading.remove();
    messagesEl.innerHTML += `<div class="chat-msg assistant" style="color:var(--danger)">❌ ${escHtml(e.message)}</div>`;
    toast('发送失败: ' + e.message, 'error');
  }
  input.disabled = false;
  if (sendBtn) sendBtn.disabled = false;
  input.focus();
}
