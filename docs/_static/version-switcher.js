(function () {
  'use strict';

  // ── 2-mode theme toggle (light ↔ dark, no "auto") ──────────────────────────
  // Normalize any stored "auto" (Furo default) to "light" before first paint.
  try {
    var _stored = localStorage.getItem('theme');
    if (!_stored || _stored === 'auto') {
      localStorage.setItem('theme', 'light');
      if (document.body) document.body.dataset.theme = 'light';
    }
  } catch (_e) {}

  // ───────────────────────────────────────────────────────────────────────────

  function fetchVersions() {
    const candidates = [
      'versions.json',
      '../versions.json',
      '../../versions.json',
      '../../../versions.json',
      '../../../../versions.json',
      '/versions.json'
    ];

    return candidates.reduce((promise, url) => {
      return promise.catch(() =>
        fetch(url, { cache: 'no-store' }).then((response) => {
          if (!response.ok) {
            throw new Error('Not found');
          }
          return response.json();
        })
      );
    }, Promise.reject(new Error('No versions manifest found')));
  }

  function normalizePath(pathname) {
    return pathname.endsWith('/') ? pathname + 'index.html' : pathname;
  }

  function buildTargetUrl(versionName, versions) {
    const pathname = normalizePath(window.location.pathname);
    const parts = pathname.split('/').filter(Boolean);
    const versionSet = new Set(versions.map((v) => v.name));

    const versionIndex = parts.findIndex((part) => versionSet.has(part));

    if (versionIndex === -1) {
      return '/' + [versionName, 'index.html'].join('/');
    }

    parts[versionIndex] = versionName;
    return '/' + parts.join('/');
  }

  // The topnav is built by wl-topnav.js inside its own DOMContentLoaded
  // handler, which is registered AFTER this file's (see html_js_files order in
  // conf.py). In practice the versions.json fetch resolves late enough that the
  // slot already exists -- but "in practice" is doing too much work there: a
  // cached manifest can resolve in the same tick, and the switcher would then
  // silently fall back to the in-page location. Wait for the slot instead.
  function whenSlotReady(selector, timeoutMs) {
    return new Promise((resolve) => {
      const existing = document.querySelector(selector);
      if (existing) {
        resolve(existing);
        return;
      }
      let settled = false;
      const finish = (node) => {
        if (settled) return;
        settled = true;
        observer.disconnect();
        clearTimeout(timer);
        resolve(node);
      };
      const observer = new MutationObserver(() => {
        const found = document.querySelector(selector);
        if (found) finish(found);
      });
      const timer = setTimeout(() => finish(document.querySelector(selector)), timeoutMs);
      observer.observe(document.documentElement, { childList: true, subtree: true });
    });
  }

  function mountSelector(versions) {
    if (!Array.isArray(versions) || versions.length === 0) {
      return;
    }

    const pathname = window.location.pathname;
    const current = versions.find((v) => pathname.indexOf('/' + v.name + '/') !== -1) || versions[0];

    const container = document.createElement('div');
    container.className = 'wl-version-switcher';

    const label = document.createElement('label');
    label.setAttribute('for', 'wl-version-select');
    label.textContent = 'Docs version';

    const select = document.createElement('select');
    select.id = 'wl-version-select';

    versions.forEach((v) => {
      const option = document.createElement('option');
      option.value = v.name;
      option.textContent = v.label;
      option.selected = v.name === current.name;
      select.appendChild(option);
    });

    select.addEventListener('change', function () {
      const target = buildTargetUrl(select.value, versions);
      window.location.href = target;
    });

    container.appendChild(label);
    container.appendChild(select);

    whenSlotReady('.wl-topnav-version', 5000).then((slot) => {
      if (slot) {
        container.classList.add('wl-version-switcher--topnav');
        slot.appendChild(container);
        return;
      }
      // No topnav on this page -- fall back to where the switcher used to live
      // rather than dropping the control entirely.
      const articleHeader =
        document.querySelector('.content .article-header') || document.querySelector('.content');
      if (articleHeader) {
        articleHeader.prepend(container);
      } else {
        document.body.prepend(container);
      }
    });
  }

  document.addEventListener('DOMContentLoaded', function () {
    fetchVersions().then(mountSelector).catch(function () {
      return null;
    });
  });
})();
