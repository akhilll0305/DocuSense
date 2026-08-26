/**
 * DocuSense API Client
 * Thin wrapper over all /api/* endpoints, with bearer-token auth.
 */
const API = (() => {
  const BASE = '/api';
  const TOKEN_KEY = 'docusense_token';
  const USER_KEY = 'docusense_user';

  // ---- token storage -------------------------------------------------
  // Wrapped because storage throws in private mode and when site data is blocked.
  function getToken() {
    try { return localStorage.getItem(TOKEN_KEY); } catch { return null; }
  }

  function getUser() {
    try { return JSON.parse(localStorage.getItem(USER_KEY) || 'null'); } catch { return null; }
  }

  function setSession(token, user) {
    try {
      localStorage.setItem(TOKEN_KEY, token);
      localStorage.setItem(USER_KEY, JSON.stringify(user));
    } catch { /* session lasts for this page only */ }
  }

  function clearSession() {
    try {
      localStorage.removeItem(TOKEN_KEY);
      localStorage.removeItem(USER_KEY);
    } catch { /* nothing to clear */ }
  }

  // ---- transport -----------------------------------------------------
  class ApiError extends Error {
    constructor(message, status) {
      super(message);
      this.name = 'ApiError';
      this.status = status;
    }
  }

  async function request(method, path, body, isFormData = false) {
    const opts = { method, headers: {} };

    const token = getToken();
    if (token) opts.headers['Authorization'] = `Bearer ${token}`;

    if (body) {
      if (isFormData) {
        opts.body = body;
      } else {
        opts.headers['Content-Type'] = 'application/json';
        opts.body = JSON.stringify(body);
      }
    }

    const res = await fetch(`${BASE}${path}`, opts);

    // An expired or revoked token should send the user back to sign in
    // rather than surfacing a confusing error on every action.
    if (res.status === 401 && !path.startsWith('/auth/')) {
      clearSession();
      window.location.href = '/static/auth.html';
      throw new ApiError('Session expired', 401);
    }

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      let detail = err.detail;
      // FastAPI validation errors arrive as an array of issue objects.
      if (Array.isArray(detail)) detail = detail.map(d => d.msg).join(', ');
      throw new ApiError(detail || `HTTP ${res.status}`, res.status);
    }

    return res.status === 204 ? null : res.json();
  }

  return {
    ApiError,
    getToken,
    getUser,
    clearSession,
    isAuthenticated: () => Boolean(getToken()),

    // Auth
    async register(email, password, name) {
      const data = await request('POST', '/auth/register', { email, password, name });
      setSession(data.access_token, data.user);
      return data;
    },

    async login(email, password) {
      const data = await request('POST', '/auth/login', { email, password });
      setSession(data.access_token, data.user);
      return data;
    },

    async me() {
      return request('GET', '/auth/me');
    },

    logout() {
      clearSession();
      window.location.href = '/static/auth.html';
    },

    // Ingestion
    async ingest(file) {
      const form = new FormData();
      form.append('file', file);
      return request('POST', '/ingest', form, true);
    },

    // Q&A
    async ask(query, { top_k = 5, mode = 'answer', filters = null } = {}) {
      return request('POST', '/ask', { query, top_k, mode, filters });
    },

    // Chat
    async startChat(title = 'New Chat') {
      return request('POST', '/chat/start', { title });
    },

    async chat(conversationId, query, { mode = 'answer', top_k = 5 } = {}) {
      return request('POST', `/chat/${conversationId}`, { query, mode, top_k });
    },

    async getChatHistory(conversationId) {
      return request('GET', `/chat/${conversationId}`);
    },

    async listChats() {
      return request('GET', '/chats');
    },

    // Documents
    async listDocuments() {
      return request('GET', '/documents');
    },

    async deleteDocument(documentId) {
      return request('DELETE', `/documents/${documentId}`);
    },

    // System
    async health() {
      return request('GET', '/health');
    },

    async stats() {
      return request('GET', '/stats');
    }
  };
})();
