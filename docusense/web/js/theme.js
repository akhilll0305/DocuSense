/**
 * Light and dark reading, shared by every page.
 *
 * Loaded in <head>, not at the end of <body>: the stored choice has to be on
 * <html> before the first paint, or a reader who picked light on a dark OS
 * sees a dark flash on every navigation. With no stored choice nothing is set
 * and the CSS follows the system setting on its own.
 *
 * Any element with [data-theme-toggle] becomes a toggle. The toggle used to
 * exist on the chat page only, so it appeared and disappeared as you moved
 * between pages.
 */
(function () {
  'use strict';

  const KEY = 'docusense_theme';
  const root = document.documentElement;

  // Storage throws in private mode and when site data is blocked.
  function stored() {
    try { return localStorage.getItem(KEY); } catch { return null; }
  }

  function store(value) {
    try { localStorage.setItem(KEY, value); } catch { /* per-page only */ }
  }

  /** The theme in effect, whether chosen explicitly or inherited from the OS. */
  function current() {
    const explicit = root.getAttribute('data-theme');
    if (explicit === 'light' || explicit === 'dark') return explicit;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  const initial = stored();
  if (initial === 'light' || initial === 'dark') root.setAttribute('data-theme', initial);

  function label(toggle) {
    const next = current() === 'dark' ? 'light' : 'dark';
    toggle.setAttribute('aria-label', `Switch to ${next} mode`);
    toggle.setAttribute('title', `Switch to ${next} mode`);
  }

  function wire() {
    document.querySelectorAll('[data-theme-toggle]').forEach((toggle) => {
      label(toggle);
      toggle.addEventListener('click', () => {
        const next = current() === 'dark' ? 'light' : 'dark';
        root.setAttribute('data-theme', next);
        store(next);
        document.querySelectorAll('[data-theme-toggle]').forEach(label);
      });
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', wire);
  } else {
    wire();
  }
})();
