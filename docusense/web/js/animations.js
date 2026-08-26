/**
 * DocuSense — motion.
 *
 * No animation library: IntersectionObserver for reveals, CSS transitions for
 * everything else. Motion here is meant to carry meaning (a page being marked
 * up, a rule appearing under a sticky header), not to decorate.
 *
 * Every effect is skipped when the user prefers reduced motion; the CSS
 * already renders the finished state in that case.
 */
(function () {
  'use strict';

  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');

  /* ── Scroll reveal ──────────────────────────────────────────────
     Elements marked [data-reveal] settle into place once. */
  function initReveal() {
    const targets = document.querySelectorAll('[data-reveal]');
    if (!targets.length) return;

    if (reduced.matches || !('IntersectionObserver' in window)) {
      targets.forEach(el => el.classList.add('is-in'));
      return;
    }

    const io = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add('is-in');
        io.unobserve(entry.target);   // reveal is one-way
      });
    }, {
      // Trigger a little before the element reaches the fold so it is
      // already settled by the time it is comfortably in view.
      rootMargin: '0px 0px -12% 0px',
      threshold: 0.1
    });

    targets.forEach(el => io.observe(el));

    // Watchdog. A background or throttled tab can delay observer callbacks
    // indefinitely, which would leave the page blank. After a few seconds,
    // show everything regardless — a missed animation is a far smaller
    // failure than unreadable content.
    setTimeout(() => {
      targets.forEach(el => el.classList.add('is-in'));
      io.disconnect();
    }, 4000);
  }

  /* ── Highlight sweep ────────────────────────────────────────────
     The .mark spans in headlines draw their highlighter when scrolled to. */
  function initMarks() {
    const marks = document.querySelectorAll('[data-mark]');
    if (!marks.length) return;

    if (reduced.matches || !('IntersectionObserver' in window)) {
      marks.forEach(el => el.classList.add('is-drawn'));
      return;
    }

    const io = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (!entry.isIntersecting) return;
        // Let the surrounding text settle first, then draw.
        setTimeout(() => entry.target.classList.add('is-drawn'), 260);
        io.unobserve(entry.target);
      });
    }, { threshold: 0.6 });

    marks.forEach(el => io.observe(el));
  }

  /* ── Specimen choreography ──────────────────────────────────────
     The hero sheet annotates itself in reading order: highlight the method
     sentence, note the matched section in the margin, rule under the
     citation, then show the answer that falls out of it. */
  function initSpecimen() {
    const sheet = document.getElementById('sheet');
    if (!sheet) return;

    const steps = [
      { el: sheet.querySelector('[data-seq="1"]'), at: 700 },
      { el: sheet.querySelector('[data-seq="2"]'), at: 1350 },
      { el: sheet.querySelector('[data-seq="3"]'), at: 1850 },
      { el: document.querySelector('[data-seq="4"]'), at: 2300 }
    ];

    if (reduced.matches) {
      steps.forEach(s => s.el && s.el.classList.add('is-on'));
      return;
    }

    const timers = [];
    const run = () => steps.forEach(s => {
      if (s.el) timers.push(setTimeout(() => s.el.classList.add('is-on'), s.at));
    });

    // Only start once the figure is actually on screen.
    if (!('IntersectionObserver' in window)) { run(); return; }

    const io = new IntersectionObserver((entries) => {
      if (!entries[0].isIntersecting) return;
      run();
      io.disconnect();
    }, { threshold: 0.25 });

    io.observe(sheet);

    // Same watchdog as the reveals: the annotations and the answer they
    // produce are content, so they must appear even if the observer never
    // fires (background tab, throttled renderer).
    timers.push(setTimeout(() => {
      io.disconnect();
      steps.forEach(s => s.el && s.el.classList.add('is-on'));
    }, 5000));

    // Abandon pending steps if the page is being left.
    window.addEventListener('pagehide', () => timers.forEach(clearTimeout), { once: true });
  }

  /* ── Masthead rule ──────────────────────────────────────────────
     The hairline under the header appears only once content sits beneath it. */
  function initMasthead() {
    const masthead = document.getElementById('masthead');
    if (!masthead) return;

    let ticking = false;
    const update = () => {
      masthead.classList.toggle('is-stuck', window.scrollY > 12);
      ticking = false;
    };

    window.addEventListener('scroll', () => {
      // Coalesce scroll events into one write per frame.
      if (!ticking) { ticking = true; requestAnimationFrame(update); }
    }, { passive: true });

    update();
  }

  /* ── Mobile navigation ──────────────────────────────────────────── */
  function initNav() {
    const toggle = document.getElementById('nav-toggle');
    const menu = document.getElementById('mobile-nav');
    if (!toggle || !menu) return;

    const setOpen = (open) => {
      menu.hidden = !open;
      toggle.setAttribute('aria-expanded', String(open));
      toggle.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
    };

    toggle.addEventListener('click', () => setOpen(menu.hidden));

    // Any destination closes the menu behind you.
    menu.querySelectorAll('a').forEach(a =>
      a.addEventListener('click', () => setOpen(false))
    );

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && !menu.hidden) { setOpen(false); toggle.focus(); }
    });
  }

  function init() {
    initReveal();
    initMarks();
    initSpecimen();
    initMasthead();
    initNav();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
