/**
 * DocuSense API Client
 * Thin wrapper over all /api/* endpoints
 */
const API = (() => {
  const BASE = '/api';

  async function request(method, path, body, isFormData = false) {
    const opts = { method };
    if (body) {
      if (isFormData) {
        opts.body = body;
      } else {
        opts.headers = { 'Content-Type': 'application/json' };
        opts.body = JSON.stringify(body);
      }
    }
    const res = await fetch(`${BASE}${path}`, opts);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res.json();
  }

  return {
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
