(function () {
  'use strict';

  document.addEventListener('DOMContentLoaded', function () {
    var root = (document.documentElement.dataset.content_root || './').replace(/\/$/, '');

    var nav = document.createElement('nav');
    nav.className = 'wl-topnav';
    nav.setAttribute('role', 'navigation');
    nav.setAttribute('aria-label', 'Site navigation');

    // Sun / moon SVG paths (inline, no sprite dependency)
    var svgSun =
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">' +
        '<circle cx="12" cy="12" r="4"/>' +
        '<path d="M12 2v2M12 20v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42' +
          'M2 12h2M20 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>' +
      '</svg>';
    var svgMoon =
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">' +
        '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>' +
      '</svg>';

    var ghIcon =
      '<svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">' +
        '<path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38' +
          ' 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13' +
          '-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66' +
          '.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15' +
          '-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0' +
          ' 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56' +
          '.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0' +
          ' 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8' +
          'c0-4.42-3.58-8-8-8z"/>' +
      '</svg>';

    nav.innerHTML =
      '<div class="wl-topnav-inner">' +
        '<div class="wl-topnav-menu">' +
          '<a class="wl-topnav-btn" href="' + root + '/examples/index.html">Examples</a>' +
          '<a class="wl-topnav-btn" href="' + root + '/quickstart.html">Quickstart</a>' +
          '<a class="wl-topnav-btn wl-topnav-btn--outline" href="https://www.graybx.com/"' +
            ' target="_blank" rel="noopener noreferrer">' +
            'Site' +
            '<svg class="wl-topnav-ext-icon" viewBox="0 0 12 12" fill="none"' +
              ' stroke="currentColor" stroke-width="1.6">' +
              '<path d="M5 2H2v8h8V7M7 1h4v4M11 1 5.5 6.5"/>' +
            '</svg>' +
          '</a>' +
          '<a class="wl-topnav-btn wl-topnav-btn--outline wl-topnav-github"' +
            ' href="https://github.com/GrayboxTech/weightslab"' +
            ' target="_blank" rel="noopener noreferrer"' +
            ' aria-label="GitHub repository">' +
            ghIcon +
            '<span class="wl-topnav-stars">★ <span class="wl-stars-count">–</span></span>' +
          '</a>' +
          '<button class="wl-topnav-btn wl-topnav-btn--icon wl-topnav-theme"' +
            ' aria-label="Toggle Light / Dark color theme" title="Toggle theme">' +
            '<span class="wl-ti-light">'  + svgSun  + '</span>' +
            '<span class="wl-ti-dark">'   + svgMoon + '</span>' +
          '</button>' +
        '</div>' +
      '</div>';

    var page = document.querySelector('.page');
    if (page) {
      page.parentNode.insertBefore(nav, page);
    } else {
      document.body.prepend(nav);
    }

    // Wire 2-mode toggle
    nav.querySelector('.wl-topnav-theme').addEventListener('click', function () {
      var next = (localStorage.getItem('theme') || 'light') === 'light' ? 'dark' : 'light';
      document.body.dataset.theme = next;
      localStorage.setItem('theme', next);
    });
  });
})();
