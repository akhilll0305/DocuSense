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
  // Flagged statements, by id, for the "was this flag right?" buttons; the
  // text is kept here rather than round-tripped through HTML attributes.
  const flaggedClaims = new Map();
  let flaggedSeq = 0;

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

  // Composer placeholder. Below this width the full text wraps to a second
  // line that a one-row textarea clips. Declared here, with the other lookups,
  // because `const` is not hoisted: declared next to fitComposer it sat in the
  // temporal dead zone when setupEventListeners ran, and the ReferenceError
  // took the send button and the Enter-to-send handler down with it.
  const NARROW_COMPOSER = window.matchMedia('(max-width: 560px)');
  const WIDE_PLACEHOLDER = queryInput.placeholder;
  const NARROW_PLACEHOLDER = queryInput.dataset.placeholderNarrow || WIDE_PLACEHOLDER;

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

  /**
   * The account dialog: change password, and sign out everywhere.
   *
   * Both endpoints have existed since sessions became revocable; neither had
   * a way in from the interface, which made "sessions can be ended
   * server-side" a claim only reachable with curl.
   */
  function setupAccountDialog() {
    const modal = $('#account-modal');
    if (!modal || typeof modal.showModal !== 'function') return;

    const form = $('#password-form');
    const status = $('#account-status');
    const submit = $('#password-submit');
    const current = $('#current-password');
    const next = $('#new-password');
    const confirm = $('#confirm-password');

    function setStatus(message, tone) {
      if (!message) {
        status.hidden = true;
        status.textContent = '';
        return;
      }
      status.textContent = message;
      status.dataset.tone = tone;
      status.hidden = false;
    }

    function open() {
      form.reset();
      setStatus(null);
      const user = API.getUser();
      $('#account-modal-email').textContent = user ? user.email : '';
      modal.showModal();
      current.focus();
    }

    $('#account-btn')?.addEventListener('click', open);
    $('#account-close')?.addEventListener('click', () => modal.close());

    // A click on the backdrop lands on the dialog element itself, since the
    // content sits in children. Anything inside stops here.
    modal.addEventListener('click', (e) => {
      if (e.target === modal) modal.close();
    });

    // Escape as well. A modal <dialog> is supposed to close on Escape by
    // itself, but that is the browser's behaviour rather than this page's,
    // and it could not be confirmed under automation — so the page owns it,
    // and closing an already-closing dialog is a no-op.
    modal.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') modal.close();
    });

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      setStatus(null);

      // Checked here as well as by the server: the mismatch and the length
      // are knowable without a round trip, and the server's 422 for a short
      // password would otherwise be the first sign of a typo.
      if (next.value !== confirm.value) {
        setStatus('The two new passwords do not match.', 'error');
        confirm.focus();
        return;
      }
      if (next.value.length < 8) {
        setStatus('The new password must be at least 8 characters.', 'error');
        next.focus();
        return;
      }

      submit.disabled = true;
      try {
        await API.changePassword(current.value, next.value);
        form.reset();
        setStatus(
          'Password changed. Every other device has been signed out.',
          'ok'
        );
      } catch (err) {
        setStatus(err.message || 'Could not change the password.', 'error');
      } finally {
        submit.disabled = false;
      }
    });

    $('#logout-all-btn')?.addEventListener('click', () => {
      if (!window.confirm(
        'Sign out on every device, including this one?'
      )) return;
      API.logout({ everywhere: true });
    });
  }

  /** Grow the composer to fit its content, up to a scrollable ceiling. */
  function autosizeInput() {
    queryInput.style.height = 'auto';
    queryInput.style.height = Math.min(queryInput.scrollHeight, 150) + 'px';
  }

  /**
   * Keep the composer usable at any width.
   *
   * Two things went wrong on a phone. The placeholder is long enough to wrap
   * at 430px, and a one-row textarea clipped the second line behind its own
   * scrollbar; and the initial height was whatever the browser picked,
   * because autosize only ever ran on `input`.
   *
   * The placeholder is shortened rather than relying on the textarea growing
   * to fit it: Chrome counts the placeholder in scrollHeight and Firefox does
   * not, so autosize alone would fix this in one browser only.
   */
  function fitComposer() {
    queryInput.placeholder = NARROW_COMPOSER.matches
      ? NARROW_PLACEHOLDER
      : WIDE_PLACEHOLDER;
    autosizeInput();
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
      autosizeInput();
      sendBtn.disabled = !queryInput.value.trim();
    });

    // The composer has to fit its contents before anyone types, too: an empty
    // textarea is sized to its placeholder, and at narrow widths the full
    // placeholder wraps to a second line that was being cut off.
    fitComposer();
    // The media query is what actually decides the placeholder, so listen to
    // it directly; `resize` covers rewrapping of a draft already typed.
    NARROW_COMPOSER.addEventListener('change', fitComposer);
    window.addEventListener('resize', fitComposer);

    // New chat
    $('#new-chat-btn').addEventListener('click', startNewChat);
    $('#logout-btn')?.addEventListener('click', () => API.logout());
    setupAccountDialog();

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
        <span class="sidebar__item-text" title="${escHtml(c.title || 'Untitled')}">${escHtml(c.title || 'Untitled')}</span>
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
    chatTitle.textContent = 'Loading…';
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
          else addBotMessage({
            answer: m.content,
            sources: m.sources || [],
            verification: m.verification,
          });
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
    chatTitle.textContent = 'New enquiry';
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
      <p class="eyebrow">The reading room</p>
      <h2 class="chat__welcome-title">What would you like to know?</h2>
      <p class="chat__welcome-lede">
        Add a paper to your shelf, then ask in your own words. Answers come
        back with the section and citation they came from.
      </p>
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

  /** The bot avatar markup, shared by streamed and buffered messages. */
  function botAvatarHtml() {
    return `
      <div class="message__avatar">
        <svg width="16" height="16" viewBox="0 0 26 26" fill="none" aria-hidden="true">
          <path d="M6 7.5h9M6 11h14M6 14.5h14M6 18h7" stroke="currentColor" stroke-width="1.6" stroke-linecap="square"/>
        </svg>
      </div>`;
  }

  /**
   * Create an empty bot message and return its bubble, so streamed tokens can
   * be written into it as they arrive.
   */
  function beginBotMessage() {
    hideWelcome();
    const el = document.createElement('div');
    el.className = 'message message--bot';
    el.innerHTML = `${botAvatarHtml()}<div class="message__bubble message__bubble--streaming"></div>`;
    messages.appendChild(el);
    scrollToBottom();
    return el.querySelector('.message__bubble');
  }

  /** Replace the streamed text with the final rendering and its footer. */
  function finishBotMessage(bubble, data) {
    bubble.classList.remove('message__bubble--streaming');
    bubble.innerHTML = renderMarkdown(data.answer || bubble.textContent || '');
    bubble.insertAdjacentHTML('beforeend', answerFooterHtml(data));
    markUnverified(bubble, data);
    scrollToBottom();
  }

  /**
   * Everything under an answer, on one line: whether its statements were
   * found in the papers, and its sources, collapsed until asked for. Both
   * used to be printed in full under every answer, which made a short answer
   * look like a form.
   */
  function answerFooterHtml(data) {
    const check = verificationHtml(data);
    const refs = referencesHtml(data);
    if (!check && !refs) return '';
    return `<div class="message__footer">${check}${refs}</div>`;
  }

  /**
   * The claim check. Every sentence of the answer is scored against the
   * retrieved passages by an entailment model.
   *
   * When everything is found it is one quiet mark, not a sentence to read:
   * the useful signal is the exception. When something is not found, the
   * statement is marked in the answer itself and listed here with the
   * closest passage, so the reader can judge it rather than take the flag
   * on trust. The wording is "not found", never "false": about a third of
   * flags are statements that are right but paraphrased past what the model
   * recognises (docs/BENCHMARKS.md). Nothing is shown when the check did not
   * run.
   */
  function verificationHtml(data) {
    const v = data && data.verification;
    if (!v || !v.num_claims) return '';

    if (v.num_unsupported === 0) {
      return `<span class="message__check message__check--ok"
                    title="Each statement in this answer was matched to a passage from your papers by an entailment model.">
                <span class="message__check-mark" aria-hidden="true">✓</span>
                Checked against your papers
              </span>`;
    }

    const flagged = (v.claims || []).filter(c => !c.supported);
    const n = flagged.length || v.num_unsupported;
    return `<details class="message__check message__check--warn">
              <summary>
                <span class="message__check-mark" aria-hidden="true">!</span>
                ${n === 1 ? '1 statement' : `${n} statements`} not found in your papers
              </summary>
              <div class="message__check-body">
                <p class="message__check-note">
                  No retrieved passage supports ${n === 1 ? 'this statement' : 'these statements'}.
                  ${n === 1 ? 'It' : 'They'} may still be right, phrased differently
                  from the paper, so check before relying on ${n === 1 ? 'it' : 'them'}.
                </p>
                ${flagged.map(c => {
                  const id = `fc${++flaggedSeq}`;
                  flaggedClaims.set(id, { claim: c, conversationId: currentConversationId });
                  return `
                  <div class="message__check-claim">
                    <p class="message__check-text">${escHtml(c.claim)}</p>
                    ${c.evidence ? `<p class="message__check-evidence"><span>Closest passage</span>${escHtml(c.evidence)}</p>` : ''}
                    <div class="message__check-vote" data-claim-id="${id}">
                      <span>Is it in your paper?</span>
                      <button type="button" data-verdict="agree">No, it isn't</button>
                      <button type="button" data-verdict="disagree">Yes, it is</button>
                    </div>
                  </div>`;
                }).join('')}
              </div>
            </details>`;
  }

  /**
   * Mark the paragraphs and list items that hold an unverified statement,
   * so the reader sees *where* in the answer the flag applies. Matched on a
   * normalised prefix of the claim, since the checker reads plain text and
   * the answer is rendered markdown.
   */
  function markUnverified(bubble, data) {
    const v = data && data.verification;
    if (!v || !v.num_unsupported) return;

    const norm = (t) => t.toLowerCase().replace(/[*_`#]/g, '').replace(/\s+/g, ' ').trim();
    const blocks = [...bubble.querySelectorAll('p, li, td')]
      .filter(b => !b.closest('.message__footer'));

    (v.claims || []).filter(c => !c.supported).forEach(c => {
      const probe = norm(c.claim).slice(0, 48);
      if (probe.length < 12) return;
      const hit = blocks.find(b => norm(b.textContent).includes(probe));
      if (!hit) return;
      // The sentence itself when it can be found; the whole paragraph
      // otherwise. Underlining a paragraph also marks the sentences in it
      // that the checker did find support for.
      const target = wrapSentence(hit, c.claim) || hit;
      target.classList.add('is-unverified');
      target.title = 'Not found in your papers. See the note under this answer.';
    });
  }

  /**
   * Wrap the stretch of `block` that holds `claim` in a span and return it,
   * or null if it cannot be located. Located by the claim's opening and
   * closing words rather than its whole text, because the checker reads the
   * answer with citation markers removed and the page shows them.
   */
  function wrapSentence(block, claim) {
    const clean = (t) => t.replace(/[*_`#]/g, '').replace(/\s+/g, ' ').trim();
    const words = clean(claim).split(' ');
    if (words.length < 4) return null;
    const head = words.slice(0, 3).join(' ').toLowerCase();
    const tail = words.slice(-2).join(' ').toLowerCase();

    // Text nodes with their offsets in the block's (whitespace-collapsed)
    // text, so a character position maps back to a node.
    const nodes = [];
    let text = '';
    const walker = document.createTreeWalker(block, NodeFilter.SHOW_TEXT);
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      nodes.push({ node: n, start: text.length });
      text += n.nodeValue;
    }
    const lower = text.toLowerCase();
    const start = lower.indexOf(head);
    if (start < 0) return null;
    const tailAt = lower.indexOf(tail, start + head.length - 1);
    if (tailAt < 0 || tailAt - start > clean(claim).length * 2) return null;
    const end = tailAt + tail.length;

    const locate = (pos) => {
      for (let i = nodes.length - 1; i >= 0; i--) {
        if (nodes[i].start <= pos) return { node: nodes[i].node, offset: pos - nodes[i].start };
      }
      return null;
    };
    const a = locate(start), b = locate(end);
    if (!a || !b) return null;
    try {
      const range = document.createRange();
      range.setStart(a.node, a.offset);
      range.setEnd(b.node, Math.min(b.offset, b.node.nodeValue.length));
      const span = document.createElement('span');
      span.appendChild(range.extractContents());
      range.insertNode(span);
      return span;
    } catch {
      return null;
    }
  }

  function referencesHtml(data) {
    const raw = data.reference_list;
    if (!raw || (Array.isArray(raw) && !raw.length)) return '';

    const lines = Array.isArray(raw) ? raw : String(raw).split(/\r?\n/);

    const refs = lines
      .map(l => String(l).trim())
      .filter(Boolean)
      // Drop the block's own heading, which would otherwise be numbered as [1].
      .filter(l => !/^references:?$/i.test(l))
      // A source with no bibliographic data yields a placeholder that tells the
      // reader nothing; the inline text already says where the answer came from.
      .filter(l => !/unknown document/i.test(l))
      .map(l => {
        // Entries usually arrive already numbered — keep that number rather
        // than stacking a second one in front of it.
        const m = l.match(/^\[(\d+)\]\s*(.*)$/);
        return m ? { num: m[1], text: m[2] } : { num: null, text: l };
      });

    if (!refs.length) return '';

    return `<details class="message__sources">
              <summary>${refs.length === 1 ? '1 source' : `${refs.length} sources`}</summary>
              <div class="message__refs">${
                refs.map((r, i) =>
                  `<div class="message__ref">
                     <span class="message__ref-num">[${r.num || i + 1}]</span>
                     <span>${escHtml(r.text)}</span>
                   </div>`
                ).join('')
              }</div>
            </details>`;
  }

  function addBotMessage(data) {
    hideWelcome();
    const el = document.createElement('div');
    el.className = 'message message--bot';
    el.innerHTML = `${botAvatarHtml()}<div class="message__bubble"></div>`;
    const bubble = el.querySelector('.message__bubble');
    bubble.innerHTML = renderMarkdown(data.answer || '') + answerFooterHtml(data);
    markUnverified(bubble, data);
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
    el.className = 'chat__system';
    el.setAttribute('role', 'status');
    el.textContent = text;
    messages.appendChild(el);
    scrollToBottom();
  }

  function showTyping() {
    const el = document.createElement('div');
    el.className = 'typing-indicator';
    el.id = 'typing';
    el.innerHTML = `
      ${botAvatarHtml()}
      <div class="typing-dots"><span></span><span></span><span></span></div>
      <span class="typing-label" id="typing-label"></span>
    `;
    messages.appendChild(el);
    scrollToBottom();
  }

  /** Show what the backend is doing while the user waits for the first token. */
  function setTypingLabel(text) {
    const label = $('#typing-label');
    if (label) label.textContent = text || '';
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

    let isNewChat = false;
    try {
      if (!currentConversationId) {
        const chat = await API.startChat(query.substring(0, 60));
        currentConversationId = chat.conversation_id;
        chatTitle.textContent = query.substring(0, 60);
        isNewChat = true;
      }

      // Generation takes tens of seconds on a local model, so render the
      // answer as it streams rather than leaving the user staring at a dot.
      let bubble = null;
      let answer = '';
      let failed = false;

      await API.chatStream(currentConversationId, query, {}, {
        onStatus: (msg) => setTypingLabel(msg),
        onToken: (text) => {
          if (!bubble) {
            hideTyping();
            bubble = beginBotMessage();
          }
          answer += text;
          bubble.innerHTML = renderMarkdown(answer);
          scrollToBottom();
        },
        onDone: (data) => {
          hideTyping();
          if (bubble) finishBotMessage(bubble, data);
          // No tokens streamed (e.g. nothing retrieved). After an error the
          // stream still closes with an empty "done", which is not an answer.
          else if (!failed && data.answer) addBotMessage(data);
        },
        onError: (err) => {
          failed = true;
          hideTyping();
          addSystemMessage(friendlyError(err));
        }
      });
    } catch (e) {
      hideTyping();
      addSystemMessage(friendlyError(e));
    } finally {
      // The conversation exists once started, whether or not the answer did.
      if (isNewChat) loadChats();
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
      addSystemMessage(`${file.name} is not a supported type. Add a PDF, DOCX, TXT or Markdown file.`);
      return;
    }

    // Show progress
    uploadZone.style.display = 'none';
    uploadProg.style.display = 'flex';
    uploadFill.style.width = '0%';
    uploadText.textContent = `Reading ${file.name}…`;

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
      uploadText.textContent = 'Added to your shelf';

      // renderMarkdown escapes, so the filename and title go in raw.
      const title = result.is_research_paper && result.paper_title
        ? `*${result.paper_title}*, ` : '';
      addBotMessage({
        answer: `**${result.filename}** is on your shelf: ${title}` +
          `${result.num_chunks} passages indexed in ${result.processing_time?.toFixed(1)}s.\n\n` +
          `Ask anything about it below.`
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
      uploadText.textContent = 'Could not add it';
      addSystemMessage(`Could not add ${file.name}: ${friendlyError(e)}`);
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
        <span class="sidebar__item-text" title="${escHtml(d.filename || d.document_id)}">${escHtml(d.filename || d.document_id)}</span>
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
          addSystemMessage('Could not delete the document: ' + friendlyError(err));
        }
      });
    });
  }

  // ── Claim-check feedback ──
  // One listener for every answer, present and future. A reader who has the
  // paper open is the only judge of whether a flag was right; each vote is
  // stored with the checker's score so its precision can be measured on real
  // answers (scripts/export_claim_feedback.py).
  messages.addEventListener('click', async (e) => {
    const btn = e.target.closest('.message__check-vote button');
    if (!btn) return;
    const row = btn.closest('.message__check-vote');
    const entry = flaggedClaims.get(row.dataset.claimId);
    if (!entry || row.classList.contains('is-sent')) return;

    row.querySelectorAll('button').forEach(b => { b.disabled = true; });
    try {
      await API.claimFeedback({
        claim: entry.claim.claim,
        flagged: true,
        verdict: btn.dataset.verdict,
        evidence: entry.claim.evidence || '',
        entailment: typeof entry.claim.entailment === 'number' ? entry.claim.entailment : null,
        conversation_id: entry.conversationId || null,
      });
      row.classList.add('is-sent');
      row.innerHTML = btn.dataset.verdict === 'agree'
        ? '<span>Thanks. Noted that it is not in your paper.</span>'
        : '<span>Thanks. Noted that the paper does say this; that helps tune the checker.</span>';
    } catch (err) {
      row.querySelectorAll('button').forEach(b => { b.disabled = false; });
      row.querySelector('span').textContent = 'Could not save that: ' + friendlyError(err);
    }
  });

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

  /**
   * Inline markdown on already-escaped text: code, bold, italic, and
   * citations set in the accent colour. Answers carry author-year forms in
   * two renderings, parenthetical "(Saadi et al., 2025, methodology)" and
   * narrative "Saadi et al. (2025)"; bare [1] markers are handled too.
   */
  function inline(text) {
    text = escHtml(text);
    text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
    text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/(^|[^*])\*(?!\s)(.+?)\*(?!\*)/g, '$1<em>$2</em>');
    text = text.replace(/\[(\d+)\]/g, '<span class="message__cite">[$1]</span>');
    text = text.replace(
      /\((?:[A-Z][^()]{0,120}?)(?:\d{4}|n\.d\.)[^()]{0,60}?\)/g,
      m => `<span class="message__cite">${m}</span>`
    );
    text = text.replace(
      /([A-Z][A-Za-z'-]+(?:\s+(?:et al\.?|and\s+[A-Z][A-Za-z'-]+))?\s*\((?:\d{4}|n\.d\.)[^()]{0,60}?\))/g,
      '<span class="message__cite">$1</span>'
    );
    return text;
  }

  /**
   * The markdown a model actually writes: paragraphs, headings, bulleted and
   * numbered lists, and pipe tables. Answers used to be shown with their
   * markup as literal "###" and "-" and every line break kept, which is most
   * of why a long answer read as a wall. Escaping happens per inline span,
   * so nothing in the source becomes HTML. Safe to run on a partial stream.
   */
  function renderMarkdown(src) {
    const lines = String(src || '').replace(/\r\n?/g, '\n').split('\n');
    const out = [];
    let para = [];
    let list = null;
    let table = null;

    const flushPara = () => {
      if (para.length) out.push(`<p>${inline(para.join(' '))}</p>`);
      para = [];
    };
    const flushList = () => {
      if (list) {
        out.push(`<${list.type}>${list.items.map(i => `<li>${inline(i)}</li>`).join('')}</${list.type}>`);
      }
      list = null;
    };
    const flushTable = () => {
      if (table && table.length) {
        const [head, ...body] = table;
        out.push(
          `<div class="message__table"><table><thead><tr>${head.map(c => `<th>${inline(c)}</th>`).join('')}</tr></thead>` +
          `<tbody>${body.map(r => `<tr>${r.map(c => `<td>${inline(c)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`
        );
      }
      table = null;
    };
    const flushAll = () => { flushPara(); flushList(); flushTable(); };

    for (const raw of lines) {
      const line = raw.trim();
      let m;

      if (!line) { flushAll(); continue; }

      if (line.startsWith('|')) {
        flushPara(); flushList();
        // The |---|---| row separates the header and carries nothing.
        if (/^\|?[\s:|-]+\|?$/.test(line) && line.includes('-')) continue;
        const cells = line.replace(/^\||\|$/g, '').split('|').map(c => c.trim());
        (table = table || []).push(cells);
        continue;
      }
      flushTable();

      if ((m = line.match(/^#{1,6}\s+(.*)$/))) {
        flushPara(); flushList();
        out.push(`<h4 class="message__h">${inline(m[1].replace(/#+$/, '').trim())}</h4>`);
        continue;
      }
      if (/^(-{3,}|\*{3,}|_{3,})$/.test(line)) { flushPara(); flushList(); continue; }

      if ((m = line.match(/^[-*•]\s+(.*)$/)) || (m = line.match(/^(\d+)[.)]\s+(.*)$/))) {
        const type = /^\d/.test(line) ? 'ol' : 'ul';
        const item = type === 'ol' ? m[2] : m[1];
        flushPara();
        if (!list || list.type !== type) { flushList(); list = { type, items: [] }; }
        list.items.push(item);
        continue;
      }

      // A wrapped line continues whatever block it sits in.
      if (list) list.items[list.items.length - 1] += ' ' + line;
      else para.push(line);
    }
    flushAll();
    return out.join('');
  }

  /**
   * Turn a failure into something a reader can act on. The raw text is a
   * provider's error with its URL in it, which is noise to anyone but the
   * operator, who has it in the server log.
   */
  function friendlyError(err) {
    const msg = String((err && err.message) || err || '');
    if (/429|too many requests|rate limit/i.test(msg)) {
      return 'The answer model is busy right now. Give it a few seconds and ask again.';
    }
    if (/timed? ?out|timeout/i.test(msg)) {
      return 'That took too long to answer. Try again, or ask something narrower.';
    }
    if (/failed to fetch|networkerror|load failed/i.test(msg)) {
      return 'Could not reach the server. Check your connection and try again.';
    }
    const clean = msg
      .replace(/https?:\/\/\S+/g, '')
      .replace(/For more information check:?/gi, '')
      .replace(/\s+/g, ' ')
      .trim();
    return clean || 'Something went wrong. Try again.';
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
