/**
 * DocuSense Chat Interface Controller
 */
(function() {
  'use strict';

  // ── Auth guard ──
  // Every /api call here needs a token; bounce to sign-in before the UI paints
  // rather than letting each request fail on its own.
  if (!API.isAuthenticated()) {
    window.location.replace('/static/auth.html');
    return;
  }

  // ── State ──
  let currentConversationId = null;
  let isProcessing = false;

  // ── DOM refs ──
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  const sidebar     = $('#sidebar');
  const chatList    = $('#chat-list');
  const docList     = $('#doc-list');
  const docCount    = $('#doc-count');
  const messages    = $('#messages');
  const welcome     = $('#welcome');
  const queryInput  = $('#query-input');
  const sendBtn     = $('#send-btn');
  const uploadZone  = $('#upload-zone');
  const fileInput   = $('#file-input');
  const uploadProg  = $('#upload-progress');
  const uploadFill  = $('#upload-fill');
  const uploadText  = $('#upload-text');
  const chatTitle   = $('#chat-title');

  // ── Init ──
  renderAccount();
  loadChats();
  loadDocuments();
  setupEventListeners();

  /** Show who is signed in, from the session cached at login. */
  function renderAccount() {
    const user = API.getUser();
    if (!user) return;

    const nameEl = $('#user-name');
    const emailEl = $('#user-email');
    const avatarEl = $('#user-avatar');

    const displayName = user.name || user.email.split('@')[0];
    if (nameEl) nameEl.textContent = displayName;
    if (emailEl) emailEl.textContent = user.email;
    if (avatarEl) avatarEl.textContent = displayName.charAt(0).toUpperCase();
  }

  function setupEventListeners() {
    // Send message
    sendBtn.addEventListener('click', handleSend);
    queryInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    });

    // Auto-resize textarea
    queryInput.addEventListener('input', () => {
      queryInput.style.height = 'auto';
      queryInput.style.height = Math.min(queryInput.scrollHeight, 150) + 'px';
      sendBtn.disabled = !queryInput.value.trim();
    });

    // New chat
    $('#new-chat-btn').addEventListener('click', startNewChat);
    $('#logout-btn')?.addEventListener('click', () => API.logout());

    // File upload
    uploadZone.addEventListener('click', () => fileInput.click());
    uploadZone.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') fileInput.click();
    });
    fileInput.addEventListener('change', (e) => {
      if (e.target.files[0]) handleFileUpload(e.target.files[0]);
    });

    // Attach button in chat
    $('#attach-btn').addEventListener('click', () => fileInput.click());

    // Drag & drop
    uploadZone.addEventListener('dragover', (e) => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
    uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
    uploadZone.addEventListener('drop', (e) => {
      e.preventDefault();
      uploadZone.classList.remove('drag-over');
      if (e.dataTransfer.files[0]) handleFileUpload(e.dataTransfer.files[0]);
    });

    // Sidebar toggle (mobile)
    $('#sidebar-toggle')?.addEventListener('click', openSidebar);
    $('#sidebar-close')?.addEventListener('click', closeSidebar);

    // Suggestions
    $$('.chat__suggestion').forEach(btn => {
      btn.addEventListener('click', () => {
        const q = btn.dataset.query;
        queryInput.value = q;
        sendBtn.disabled = false;
        handleSend();
      });
    });
  }

  // ── Sidebar ──
  function openSidebar() {
    sidebar.classList.add('open');
    let overlay = $('.sidebar-overlay');
    if (!overlay) {
      overlay = document.createElement('div');
      overlay.className = 'sidebar-overlay';
      document.body.appendChild(overlay);
      overlay.addEventListener('click', closeSidebar);
    }
    overlay.classList.add('visible');
  }

  function closeSidebar() {
    sidebar.classList.remove('open');
    const overlay = $('.sidebar-overlay');
    if (overlay) overlay.classList.remove('visible');
  }

  // ── Conversations ──
  async function loadChats() {
    try {
      const chats = await API.listChats();
      renderChatList(chats);
    } catch (e) {
      chatList.innerHTML = '<div class="sidebar__empty">No conversations yet</div>';
    }
  }

  function renderChatList(chats) {
    if (!chats.length) {
      chatList.innerHTML = '<div class="sidebar__empty">No conversations yet</div>';
      return;
    }
    chatList.innerHTML = chats.map(c => `
      <div class="sidebar__item ${c.conversation_id === currentConversationId ? 'active' : ''}"
           data-id="${c.conversation_id}" role="listitem" tabindex="0">
        <div class="sidebar__item-icon">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>
        </div>
        <span class="sidebar__item-text">${escHtml(c.title || 'Untitled')}</span>
        <span class="sidebar__item-meta">${timeAgo(c.updated_at || c.created_at)}</span>
      </div>
    `).join('');

    chatList.querySelectorAll('.sidebar__item').forEach(el => {
      el.addEventListener('click', () => switchConversation(el.dataset.id));
      el.addEventListener('keydown', (e) => { if(e.key === 'Enter') switchConversation(el.dataset.id); });
    });
  }

  async function switchConversation(id) {
    currentConversationId = id;
    chatTitle.textContent = 'Loading...';
    closeSidebar();

    // Mark active
    chatList.querySelectorAll('.sidebar__item').forEach(el => {
      el.classList.toggle('active', el.dataset.id === id);
    });

    try {
      const history = await API.getChatHistory(id);
      clearMessages();
      if (history.messages && history.messages.length) {
        history.messages.forEach(m => {
          if (m.role === 'user') addUserMessage(m.content);
          else addBotMessage({ answer: m.content, sources: m.sources || [] });
        });
      }
      // Try to get title from sidebar
      const item = chatList.querySelector(`[data-id="${id}"]`);
      chatTitle.textContent = item?.querySelector('.sidebar__item-text')?.textContent || 'Chat';
    } catch (e) {
      chatTitle.textContent = 'Chat';
      addSystemMessage('Failed to load conversation history.');
    }
  }

  async function startNewChat() {
    currentConversationId = null;
    chatTitle.textContent = 'New Conversation';
    clearMessages();
    showWelcome();
    closeSidebar();
  }

  // ── Messages ──
  function clearMessages() {
    messages.innerHTML = '';
  }

  function showWelcome() {
    const w = document.createElement('div');
    w.className = 'chat__welcome';
    w.id = 'welcome';
    w.innerHTML = `
      <div class="chat__welcome-icon">
        <svg width="56" height="56" viewBox="0 0 28 28" fill="none">
          <rect width="28" height="28" rx="8" fill="url(#wlg2)"/>
          <path d="M8 9h8a3 3 0 013 3v0a3 3 0 01-3 3H8V9z" fill="white" opacity="0.9"/>
          <path d="M8 15h10a3 3 0 013 3v0a3 3 0 01-3 3H8v-6z" fill="white" opacity="0.6"/>
          <defs><linearGradient id="wlg2" x1="0" y1="0" x2="28" y2="28"><stop stop-color="#6c5ce7"/><stop offset="1" stop-color="#a855f7"/></linearGradient></defs>
        </svg>
      </div>
      <h3>How can I help with your research?</h3>
      <p>Upload a paper and ask me anything about it.</p>
    `;
    messages.appendChild(w);
  }

  function hideWelcome() {
    const w = messages.querySelector('.chat__welcome');
    if (w) w.remove();
  }

  function addUserMessage(text) {
    hideWelcome();
    const el = document.createElement('div');
    el.className = 'message message--user';
    el.innerHTML = `
      <div class="message__avatar">U</div>
      <div class="message__bubble">${escHtml(text)}</div>
    `;
    messages.appendChild(el);
    scrollToBottom();
  }

  function addBotMessage(data) {
    const el = document.createElement('div');
    el.className = 'message message--bot';

    let answerHtml = formatAnswer(data.answer || '');

    // References
    let refsHtml = '';
    if (data.reference_list && data.reference_list.length) {
      refsHtml = `<div class="message__refs">${
        data.reference_list.map((ref, i) =>
          `<div class="message__ref"><span class="message__ref-num">[${i+1}]</span><span>${escHtml(ref)}</span></div>`
        ).join('')
      }</div>`;
    }

    // Confidence
    let confHtml = '';
    if (data.confidence != null) {
      const level = data.confidence > 0.7 ? 'high' : data.confidence > 0.4 ? 'medium' : 'low';
      confHtml = `<div class="message__confidence message__confidence--${level}">
        Confidence: ${Math.round(data.confidence * 100)}%
      </div>`;
    }

    el.innerHTML = `
      <div class="message__avatar">
        <svg width="18" height="18" viewBox="0 0 28 28" fill="none"><rect width="28" height="28" rx="8" fill="url(#blg)"/>
        <path d="M8 9h8a3 3 0 013 3v0a3 3 0 01-3 3H8V9z" fill="white" opacity="0.9"/>
        <path d="M8 15h10a3 3 0 013 3v0a3 3 0 01-3 3H8v-6z" fill="white" opacity="0.6"/>
        <defs><linearGradient id="blg" x1="0" y1="0" x2="28" y2="28"><stop stop-color="#6c5ce7"/><stop offset="1" stop-color="#a855f7"/></linearGradient></defs></svg>
      </div>
      <div class="message__bubble">
        ${answerHtml}
        ${refsHtml}
        ${confHtml}
      </div>
    `;
    messages.appendChild(el);
    scrollToBottom();
  }

  function addFileMessage(filename, size) {
    hideWelcome();
    const el = document.createElement('div');
    el.className = 'message message--user';
    const sizeStr = size > 1e6 ? (size/1e6).toFixed(1) + ' MB' : (size/1e3).toFixed(0) + ' KB';
    el.innerHTML = `
      <div class="message__avatar">U</div>
      <div class="message__bubble">
        <div class="message__file">
          <div class="message__file-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
          </div>
          <div>
            <div class="message__file-name">${escHtml(filename)}</div>
            <div class="message__file-size">${sizeStr}</div>
          </div>
        </div>
      </div>
    `;
    messages.appendChild(el);
    scrollToBottom();
  }

  function addSystemMessage(text) {
    const el = document.createElement('div');
    el.className = 'message message--bot';
    el.innerHTML = `
      <div class="message__avatar">!</div>
      <div class="message__bubble" style="border-color:var(--error);opacity:0.8">${escHtml(text)}</div>
    `;
    messages.appendChild(el);
    scrollToBottom();
  }

  function showTyping() {
    const el = document.createElement('div');
    el.className = 'typing-indicator';
    el.id = 'typing';
    el.innerHTML = `
      <div class="message__avatar" style="background:var(--bg-tertiary);border:1px solid var(--border-medium)">
        <svg width="18" height="18" viewBox="0 0 28 28" fill="none"><rect width="28" height="28" rx="8" fill="url(#tlg)"/>
        <path d="M8 9h8a3 3 0 013 3v0a3 3 0 01-3 3H8V9z" fill="white" opacity="0.9"/>
        <path d="M8 15h10a3 3 0 013 3v0a3 3 0 01-3 3H8v-6z" fill="white" opacity="0.6"/>
        <defs><linearGradient id="tlg" x1="0" y1="0" x2="28" y2="28"><stop stop-color="#6c5ce7"/><stop offset="1" stop-color="#a855f7"/></linearGradient></defs></svg>
      </div>
      <div class="typing-dots"><span></span><span></span><span></span></div>
    `;
    messages.appendChild(el);
    scrollToBottom();
  }

  function hideTyping() {
    const el = $('#typing');
    if (el) el.remove();
  }

  // ── Send ──
  async function handleSend() {
    const query = queryInput.value.trim();
    if (!query || isProcessing) return;

    isProcessing = true;
    sendBtn.disabled = true;
    queryInput.value = '';
    queryInput.style.height = 'auto';

    addUserMessage(query);
    showTyping();

    try {
      let data;
      if (currentConversationId) {
        // Continue existing chat
        data = await API.chat(currentConversationId, query);
      } else {
        // Start new chat for first message
        const chat = await API.startChat(query.substring(0, 60));
        currentConversationId = chat.conversation_id;
        chatTitle.textContent = query.substring(0, 60);
        data = await API.chat(currentConversationId, query);
        loadChats(); // Refresh sidebar
      }

      hideTyping();
      addBotMessage(data);
    } catch (e) {
      hideTyping();
      addSystemMessage('Error: ' + e.message);
    } finally {
      isProcessing = false;
      sendBtn.disabled = false;
      queryInput.focus();
    }
  }

  // ── File Upload ──
  async function handleFileUpload(file) {
    const validTypes = ['.pdf', '.docx', '.txt', '.md'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!validTypes.includes(ext)) {
      alert('Unsupported file type. Please upload PDF, DOCX, TXT, or MD files.');
      return;
    }

    // Show progress
    uploadZone.style.display = 'none';
    uploadProg.style.display = 'flex';
    uploadFill.style.width = '0%';
    uploadText.textContent = 'Ingesting...';

    // Animate progress
    let progress = 0;
    const interval = setInterval(() => {
      progress = Math.min(progress + Math.random() * 15, 90);
      uploadFill.style.width = progress + '%';
    }, 500);

    // Show file in chat
    addFileMessage(file.name, file.size);

    try {
      const result = await API.ingest(file);
      clearInterval(interval);
      uploadFill.style.width = '100%';
      uploadText.textContent = '✓ Ingested!';

      // Bot confirmation
      addBotMessage({
        answer: `Successfully ingested **${escHtml(result.filename)}**!\n\n` +
          `• ${result.num_chunks} chunks created\n` +
          `• ${result.num_embeddings} embeddings generated\n` +
          (result.is_research_paper ? `• Research paper detected: *${escHtml(result.paper_title || 'Untitled')}*\n` : '') +
          `• Processing time: ${result.processing_time?.toFixed(1)}s\n\n` +
          `You can now ask questions about this paper!`,
        confidence: 1.0
      });

      // Refresh documents
      loadDocuments();

      setTimeout(() => {
        uploadProg.style.display = 'none';
        uploadZone.style.display = 'flex';
        fileInput.value = '';
      }, 2000);
    } catch (e) {
      clearInterval(interval);
      uploadFill.style.width = '0%';
      uploadText.textContent = 'Failed!';
      addSystemMessage('Upload failed: ' + e.message);
      setTimeout(() => {
        uploadProg.style.display = 'none';
        uploadZone.style.display = 'flex';
        fileInput.value = '';
      }, 2000);
    }
  }

  // ── Documents ──
  async function loadDocuments() {
    try {
      const response = await API.listDocuments();
      const docs = response.documents || [];
      docCount.textContent = docs.length;
      renderDocList(docs);
    } catch (e) {
      docList.innerHTML = '<div class="sidebar__empty">No documents</div>';
    }
  }

  function renderDocList(docs) {
    if (!docs.length) {
      docList.innerHTML = '<div class="sidebar__empty">Upload your first paper</div>';
      return;
    }
    docList.innerHTML = docs.map(d => `
      <div class="sidebar__item" data-doc-id="${d.document_id}" role="listitem" tabindex="0">
        <div class="sidebar__item-icon">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
        </div>
        <span class="sidebar__item-text">${escHtml(d.filename || d.document_id)}</span>
        <button class="sidebar__item-delete" title="Delete" aria-label="Delete ${escHtml(d.filename)}" data-del-id="${d.document_id}">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/></svg>
        </button>
      </div>
    `).join('');

    docList.querySelectorAll('.sidebar__item-delete').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        if (!confirm('Delete this document?')) return;
        try {
          await API.deleteDocument(btn.dataset.delId);
          loadDocuments();
        } catch (err) {
          alert('Delete failed: ' + err.message);
        }
      });
    });
  }

  // ── Helpers ──
  function scrollToBottom() {
    requestAnimationFrame(() => {
      messages.scrollTop = messages.scrollHeight;
    });
  }

  function escHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  function formatAnswer(text) {
    // Bold: **text**
    text = escHtml(text);
    text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    // Italic: *text*
    text = text.replace(/\*(.+?)\*/g, '<em>$1</em>');
    // Citations: [1], [2], etc.
    text = text.replace(/\[(\d+)\]/g, '<span class="message__cite">[$1]</span>');
    // Bullet points
    text = text.replace(/^• (.+)$/gm, '<div style="padding-left:1em">• $1</div>');
    // Line breaks
    text = text.replace(/\n/g, '<br>');
    return text;
  }

  function timeAgo(dateStr) {
    if (!dateStr) return '';
    const d = new Date(dateStr);
    const now = new Date();
    const diff = (now - d) / 1000;
    if (diff < 60) return 'now';
    if (diff < 3600) return Math.floor(diff/60) + 'm';
    if (diff < 86400) return Math.floor(diff/3600) + 'h';
    return Math.floor(diff/86400) + 'd';
  }
})();
