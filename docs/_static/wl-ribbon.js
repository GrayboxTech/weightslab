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
<<<<<<< HEAD
    'Run <code>weightslab launch --certs</code> to enable HTTPS + mTLS for secure remote studio access.',
=======
    'Run <code>weightslab start --certs</code> to enable HTTPS + mTLS for secure remote studio access.',
>>>>>>> a6f0c1acbac49221358a21b0bde87b348c09204b
    'Set <code>preload_labels=False</code> for large datasets to speed up startup; labels are loaded lazily.',
    'Use <code>array_return_proxies=True</code> (default) to avoid loading the full dataset array into RAM.',
    'Set <code>WEIGHTSLAB_LOG_LEVEL=DEBUG</code> to see full gRPC logs when debugging connectivity issues.',
  ];

  var INTERVAL = 12000; // ms between rotations
  var FADE     = 280;   // ms fade duration

  function mount() {
    var el = document.getElementById('wl-topnav-tip');
    if (!el) return;

    var idx = Math.floor(Math.random() * TIPS.length);

    function showTip(i) {
      el.style.opacity = '0';
      setTimeout(function () {
        el.innerHTML = TIPS[i];
        el.style.opacity = '1';
      }, FADE);
    }

    showTip(idx);

    setInterval(function () {
      idx = (idx + 1) % TIPS.length;
      showTip(idx);
    }, INTERVAL);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mount);
  } else {
    mount();
  }
})();
