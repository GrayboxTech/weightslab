(function () {
  'use strict';

  var TIPS = [
    'Edit <code>hyperparameters.yaml</code> while training — changes apply within 1 second, no restart needed.',
    'Click any sample in the studio to deny it from future batches. The deny-aware sampler persists tags across runs.',
    'Call <code>wl.keep_serving()</code> after your training loop to keep the studio live for post-training analysis.',
    'Add <code>per_sample=True</code> to a <code>@wl.signal</code> decorator to store one value per sample per step.',
    'Set <code>is_training=True</code> on your DataLoader kwargs to activate the deny-aware sampler.',
    'The studio streams signals in real-time — no need to wait for an epoch to end to see results.',
    '<code>weightslab start example --cls</code> launches a full MNIST classification demo in one command.',
    'Use <code>subscribe_to=</code> on a signal to build reactive per-sample analytics derived from other signals.',
    'Run <code>weightslab ui launch --certs</code> to enable HTTPS + mTLS for secure remote studio access.',
    'Set <code>preload_labels=False</code> for large datasets to speed up startup; labels are loaded lazily.',
    'Use <code>array_return_proxies=True</code> (default) to avoid loading the full dataset array into RAM.',
    'Set <code>WEIGHTSLAB_LOG_LEVEL=DEBUG</code> to see full gRPC logs when debugging connectivity issues.',
  ];

  var INTERVAL = 12000; // ms between rotations
  var FADE     = 280;   // ms fade duration

  function mount() {
    if (sessionStorage.getItem('wl_ribbon_dismissed')) return;

    var ribbon = document.createElement('div');
    ribbon.className = 'wl-ribbon';
    ribbon.setAttribute('role', 'status');

    ribbon.innerHTML =
      '<span class="wl-ribbon-icon" aria-hidden="true">' +
        '<svg width="13" height="13" viewBox="0 0 20 20" fill="currentColor">' +
          '<path d="M10 1a7 7 0 0 0-3.46 13.07A1 1 0 0 0 7 15v1a1 1 0 0 0 1 1h4a1 1 0 0 0 1-1v-1c0-.17.06-.34.15-.47A7 7 0 0 0 10 1zm-1 15v-1h2v1H9zm5.29-4.29A5 5 0 1 1 5 10a5 5 0 0 1 9.29 2.71z"/>' +
        '</svg>' +
      '</span>' +
      '<span class="wl-ribbon-label">Tip</span>' +
      '<span class="wl-ribbon-sep">—</span>' +
      '<span class="wl-ribbon-text" id="wl-ribbon-text"></span>' +
      '<button class="wl-ribbon-close" aria-label="Dismiss tips ribbon" title="Dismiss">' +
        '<svg width="11" height="11" viewBox="0 0 12 12" stroke="currentColor" stroke-width="2" fill="none">' +
          '<line x1="1" y1="1" x2="11" y2="11"/><line x1="11" y1="1" x2="1" y2="11"/>' +
        '</svg>' +
      '</button>';

    // Insert after topnav if present, otherwise prepend to body
    var topnav = document.querySelector('.wl-topnav');
    if (topnav && topnav.nextSibling) {
      topnav.parentNode.insertBefore(ribbon, topnav.nextSibling);
    } else {
      document.body.insertBefore(ribbon, document.body.firstChild);
    }
    document.body.classList.add('wl-ribbon-visible');

    var idx     = Math.floor(Math.random() * TIPS.length);
    var textEl  = document.getElementById('wl-ribbon-text');

    function showTip(i) {
      textEl.style.opacity = '0';
      setTimeout(function () {
        textEl.innerHTML = TIPS[i];
        textEl.style.opacity = '1';
      }, FADE);
    }

    showTip(idx);

    var timer = setInterval(function () {
      idx = (idx + 1) % TIPS.length;
      showTip(idx);
    }, INTERVAL);

    ribbon.querySelector('.wl-ribbon-close').addEventListener('click', function () {
      clearInterval(timer);
      ribbon.remove();
      document.body.classList.remove('wl-ribbon-visible');
      sessionStorage.setItem('wl_ribbon_dismissed', '1');
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mount);
  } else {
    mount();
  }
})();
