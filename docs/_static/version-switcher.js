(function () {
  'use strict';

  function ensureDefaultTheme() {
    try {
      const storedTheme = window.localStorage.getItem('theme');
      if (!storedTheme || storedTheme === 'auto') {
        window.localStorage.setItem('theme', 'light');
      }
    } catch (error) {
      return;
    }
  }

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

    const articleHeader = document.querySelector('.content .article-header') || document.querySelector('.content');
    if (articleHeader) {
      articleHeader.prepend(container);
    } else {
      document.body.prepend(container);
    }
  }

  ensureDefaultTheme();

  document.addEventListener('DOMContentLoaded', function () {
    fetchVersions().then(mountSelector).catch(function () {
      return null;
    });
  });
})();
