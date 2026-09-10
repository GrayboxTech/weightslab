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
    'Run <code>weightslab start --certs</code> to enable HTTPS + mTLS for secure remote studio access.',
    'Set <code>preload_labels=False</code> for large datasets to speed up startup; labels are loaded lazily.',
    'Use <code>array_return_proxies=True</code> (default) to avoid loading the full dataset array into RAM.',
    'Set <code>WEIGHTSLAB_LOG_LEVEL=DEBUG</code> to see full gRPC logs when debugging connectivity issues.',
    'Call <code>wl.ai_report_generation()</code> — or run <code>report</code> in <code>weightslab cli</code> — for a full HTML report with plots, dataset analysis, and agent-written insights.',
    'Type <code>/init</code> in the experiment agent bar to bring the integrated OpenCode agent online for a running experiment — no separate server to start.',
    'Type <code>/loop 30m &lt;prompt&gt;</code> in the experiment agent bar to have the agent check in on your training on a recurring interval.',
    'Export tagged samples straight to CVAT, Label Studio, or V7 with <code>wl.export_annotations("cvat", tags=["ToReview"])</code> — no custom relabeling script needed.',
    'Right-click any curve to add a step note, hide it, load weights from that step, or change its color — no separate panel needed.',
    'Curves now render error bands and flag outlier steps automatically, so anomalies stand out without manual smoothing.',
    'Filter the plots panel with a regex in the search bar to isolate exactly the curves you want.',
    'From a live Jupyter or Colab notebook, ask the agent to generate analysis code against your on-training experiment — no need to stop training first.',
    'WeightsLab now tracks GPU, CPU, and RAM usage automatically during training and agent runs — check the resource panel, no separate monitoring setup needed.',
  ];

  var INTERVAL = 5000; // ms between rotations
  var FADE     = 280;   // ms fade duration

  function mount() {
    var el = document.getElementById('wl-topnav-tip');
    if (!el) return;

    var wrap = el.closest('.wl-topnav-center');

    // CSS (.wl-topnav-menu: flex 0 0 auto, .wl-topnav-center: the only
    // flexible item) already makes overlap structurally impossible -- this
    // just hides the tip once the space it's been squeezed into is too
    // narrow to show anything but a sliver + ellipsis. Re-checked on
    // rotation and resize since the available width changes with both.
    var MIN_USABLE_WIDTH = 60; // px
    function hideIfTooNarrow() {
      if (!wrap) return;
      wrap.classList.toggle(
        'wl-topnav-center--collides',
        wrap.getBoundingClientRect().width < MIN_USABLE_WIDTH
      );
    }

    var idx = Math.floor(Math.random() * TIPS.length);

    function showTip(i) {
      el.style.opacity = '0';
      setTimeout(function () {
        el.innerHTML = TIPS[i];
        el.style.opacity = '1';
        hideIfTooNarrow();
      }, FADE);
    }

    showTip(idx);
    window.addEventListener('resize', avoidCollision);

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
