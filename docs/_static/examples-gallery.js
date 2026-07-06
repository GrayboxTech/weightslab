(function () {
  'use strict';

  // Sections in display order
  var SECTIONS = [
    { key: 'pytorch',     label: 'PyTorch'     },
    { key: 'lightning',   label: 'Lightning'   },
    { key: 'ultralytics', label: 'Ultralytics' },
    { key: 'usecase',     label: 'Usecases'    }
  ];

  // URLs are root-relative (from the doc root), prefixed with content_root at render time
  var EXAMPLES = [
    {
      badge: 'PyTorch', color: 'pytorch',
      title: 'Classification — MNIST',
      desc: 'CNN digit classifier on MNIST. Register hyperparameters, monitor per-sample loss, and use the deny-aware sampler to focus on hard examples.',
      tags: ['classification', 'supervised', 'mnist', 'cnn'],
      url: 'examples/pytorch/classification.html'
    },
    {
      badge: 'PyTorch', color: 'pytorch',
      title: 'Segmentation — BDD100k',
      desc: 'Per-pixel semantic segmentation with a UNet. Track per-sample IoU and visualise mask overlays directly in the studio.',
      tags: ['segmentation', 'semantic', 'bdd100k', 'masks', 'dense prediction'],
      url: 'examples/pytorch/segmentation.html'
    },
    {
      badge: 'PyTorch', color: 'pytorch',
      title: 'Detection — Penn-Fudan',
      desc: 'Bounding-box detection on Penn-Fudan pedestrians. Per-instance multi-index dataframe with (sample_id, annotation_id) keys.',
      tags: ['detection', 'object detection', 'bounding boxes', 'penn-fudan'],
      url: 'examples/pytorch/detection.html'
    },
    {
      badge: 'PyTorch', color: 'pytorch',
      title: 'Clustering — Face Recognition',
      desc: 'Metric learning with triplet loss on face datasets. Store and explore high-dimensional embeddings per sample in the studio.',
      tags: ['clustering', 'unsupervised', 'embeddings', 'face recognition', 'metric learning'],
      url: 'examples/pytorch/clustering.html'
    },
    {
      badge: 'PyTorch', color: 'pytorch',
      title: 'Generation / Anomaly Detection',
      desc: 'Unsupervised anomaly detection on MVTec with a multi-task UNet. Monitor reconstruction quality and per-sample anomaly scores.',
      tags: ['anomaly detection', 'generation', 'unsupervised', 'mvtec', 'reconstruction'],
      url: 'examples/pytorch/generation.html'
    },
    {
      badge: 'Lightning', color: 'lightning',
      title: 'Classification — MNIST (Lightning)',
      desc: 'Same MNIST classification wrapped in a LightningModule. WeightsLab hooks replace only the guard functions — the rest is unchanged.',
      tags: ['classification', 'supervised', 'mnist', 'pytorch lightning'],
      url: 'examples/lightning/classification.html'
    },
    {
      badge: 'Ultralytics', color: 'ultralytics',
      title: 'Detection — YOLO',
      desc: 'Drop-in WLAwareTrainer for YOLO training. Track mAP, per-image loss, and discard low-quality samples without touching the model.',
      tags: ['detection', 'yolo', 'object detection', 'mAP'],
      url: 'examples/ultralytics/detection.html'
    },
    {
      badge: 'Usecase', color: 'usecase',
      title: 'LiDAR Detection — 2D and 3D',
      desc: 'Point-cloud BEV previews, dual 2D/3D bounding box signals, streaming GetPointCloud RPC, and an interactive three.js 3D viewer.',
      tags: ['lidar', 'point cloud', '3d detection', 'bev', 'streaming'],
      url: 'examples/usecases/lidar_detection.html'
    },
    {
      badge: 'Usecase', color: 'usecase',
      title: 'Loss-Shape Classification',
      desc: 'Dynamic subscribed signal that classifies each sample\'s loss trajectory (monotonic, U-shape, spiked, …) and auto-tags it.',
      tags: ['loss analysis', 'signal', 'categorical tag', 'per-sample', 'trajectory'],
      url: 'examples/usecases/loss_shape_classification.html'
    }
  ];

  var FILTERS = [
    { label: 'All',         color: null          },
    { label: 'PyTorch',     color: 'pytorch'     },
    { label: 'Lightning',   color: 'lightning'   },
    { label: 'Ultralytics', color: 'ultralytics' },
    { label: 'Usecase',     color: 'usecase'     }
  ];

  function esc(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  function buildCard(ex, baseUrl) {
    var tagsHtml = ex.tags.map(function (t) {
      return '<span class="wl-eg-tag">' + esc(t) + '</span>';
    }).join('');
    var search = (ex.title + ' ' + ex.tags.join(' ')).toLowerCase();
    var href = baseUrl + '/' + ex.url;
    return (
      '<a class="wl-eg-card" href="' + esc(href) + '"' +
        ' data-search="' + esc(search) + '"' +
        ' data-color="' + esc(ex.color) + '">' +
        '<span class="wl-eg-badge wl-eg-badge--' + ex.color + '">' + esc(ex.badge) + '</span>' +
        '<p class="wl-eg-title">' + esc(ex.title) + '</p>' +
        '<p class="wl-eg-desc">' + esc(ex.desc) + '</p>' +
        '<div class="wl-eg-tags">' + tagsHtml + '</div>' +
      '</a>'
    );
  }

  function render() {
    var root = document.getElementById('wl-examples-gallery');
    if (!root) return;

    // Pre-filter: category pages set data-filter="pytorch" etc. on the div
    var preFilter = root.dataset.filter || null;

    // Base URL: resolve example links from any page depth using content_root
    var contentRoot = (document.documentElement.dataset.content_root || './').replace(/\/+$/, '');

    // ── Filter pills ─────────────────────────────────────────────────────────
    var filterBtns = FILTERS.map(function (f) {
      var colorAttr = f.color ? ' data-color="' + esc(f.color) + '"' : '';
      // Pre-select: "All" when no preFilter, or the matching framework button
      var isActive = preFilter ? f.color === preFilter : f.color === null;
      var activeClass = isActive ? ' wl-eg-filter--active' : '';
      return '<button class="wl-eg-filter-btn' + activeClass + '"' + colorAttr + '>' + esc(f.label) + '</button>';
    }).join('');

    // ── Sections ─────────────────────────────────────────────────────────────
    var sectionsHtml = SECTIONS.map(function (sec) {
      var cards = EXAMPLES.filter(function (ex) { return ex.color === sec.key; });
      if (!cards.length) return '';
      return (
        '<div class="wl-eg-section" data-section="' + esc(sec.key) + '">' +
          '<h2 class="wl-eg-section-title">' +
            '<span class="wl-eg-badge wl-eg-badge--' + esc(sec.key) + '">' + esc(sec.label) + '</span>' +
          '</h2>' +
          '<div class="wl-eg-grid">' +
            cards.map(function (ex) { return buildCard(ex, contentRoot); }).join('') +
          '</div>' +
        '</div>'
      );
    }).join('');

    root.innerHTML =
      '<div class="wl-eg-toolbar">' +
        '<div class="wl-eg-search-wrap">' +
          '<svg class="wl-eg-search-icon" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.8">' +
            '<circle cx="8.5" cy="8.5" r="5.5"/>' +
            '<line x1="13" y1="13" x2="18" y2="18"/>' +
          '</svg>' +
          '<input id="wl-eg-input" class="wl-eg-search-input" type="search" autocomplete="off"' +
            ' placeholder="Search by task, framework or keyword…"/>' +
        '</div>' +
        '<div class="wl-eg-filters" role="group" aria-label="Filter by framework">' +
          filterBtns +
        '</div>' +
      '</div>' +
      sectionsHtml +
      '<p class="wl-eg-empty" id="wl-eg-empty" style="display:none">No examples match your search.</p>';

    // ── Filter logic ─────────────────────────────────────────────────────────
    var activeColor = preFilter;

    function applyFilters() {
      var q = document.getElementById('wl-eg-input').value.toLowerCase().trim();
      var totalVisible = 0;

      document.querySelectorAll('.wl-eg-section').forEach(function (section) {
        var cards = section.querySelectorAll('.wl-eg-card');
        var sectionVisible = 0;
        cards.forEach(function (c) {
          var matchText  = !q || c.dataset.search.indexOf(q) !== -1;
          var matchColor = !activeColor || c.dataset.color === activeColor;
          var show = matchText && matchColor;
          c.style.display = show ? '' : 'none';
          if (show) sectionVisible++;
        });
        // Hide the whole section block when none of its cards match
        section.style.display = sectionVisible === 0 ? 'none' : '';
        totalVisible += sectionVisible;
      });

      document.getElementById('wl-eg-empty').style.display = totalVisible === 0 ? '' : 'none';
    }

    document.getElementById('wl-eg-input').addEventListener('input', applyFilters);

    // Apply pre-filter immediately on category pages
    if (preFilter) applyFilters();

    document.querySelectorAll('.wl-eg-filter-btn').forEach(function (btn) {
      btn.addEventListener('click', function () {
        document.querySelectorAll('.wl-eg-filter-btn').forEach(function (b) {
          b.classList.remove('wl-eg-filter--active');
        });
        btn.classList.add('wl-eg-filter--active');
        activeColor = btn.dataset.color || null;
        applyFilters();
      });
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', render);
  } else {
    render();
  }
})();
