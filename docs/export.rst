Annotation Export
==================

WeightsLab can export bounding-box/segmentation annotations to a
relabeling-tool format, so a dataset (or a slice of one) can be handed off
for an outsourced relabeling pass. Three ways to trigger it, all backed by
the same code path:

- **Weights Studio UI** — an "Export" button next to Save/Grid settings, with
  a format picker (CVAT / Label Studio / V7). Triggers a browser download.
- **CLI** — ``weightslab export`` connects over gRPC to a running experiment,
  same as ``weightslab cli``.
- **Python** — :func:`wl.export_annotations`, called in-process (no gRPC
  round-trip needed since it already runs alongside the registered dataframe).

Supported formats
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Format
     - Output shape
     - Schema reference
   * - ``cvat``
     - A single CVAT XML 1.1 file (one ``<image>`` element per sample, with
       ``<box>``/``<polygon>`` children).
     - `CVAT XML format <https://opencv.github.io/cvat/docs/manual/advanced/xml_format/>`_
   * - ``label_studio``
     - A single JSON file — a list of "tasks", each with a ``result`` list of
       ``rectanglelabels``/``polygonlabels`` entries. Coordinates are
       percentages (0-100) of the image's width/height, per Label Studio's
       convention.
     - `Label Studio export format <https://labelstud.io/guide/export.html#JSON>`_
   * - ``v7``
     - A zip of one Darwin JSON 2.0 file per image (V7 matches annotations to
       images by filename on import).
     - `Darwin JSON reference <https://docs.v7labs.com/reference/darwin-json>`_

Bounding boxes are exported for every format. Segmentation masks are
converted to polygons via OpenCV contour extraction — this needs the
optional ``export`` extra:

.. code-block:: bash

   pip install weightslab[export]

Bounding-box-only export needs no extra dependency; if OpenCV isn't
installed, segmentation samples still export their boxes and a warning is
logged once, rather than failing the whole export.

Usage
------

**Python**

.. code-block:: python

   import weightslab as wl

   wl.export_annotations("cvat")                              # everything, under root_log_dir
   wl.export_annotations("cvat", tags=["ToReview"])           # only samples tagged ToReview
   wl.export_annotations("label_studio", "val.json", origin="val_loader")
   wl.export_annotations("v7", "out/", class_names=["bg", "cat", "dog"])

See :doc:`user_functions` for the full :func:`wl.export_annotations` reference.

**CLI**

.. code-block:: bash

   weightslab export --format cvat                      # everything, CVAT XML, into "."
   weightslab export -f v7 out/ --origin val_loader      # V7/Darwin, val split only
   weightslab export -f cvat --tag ToReview              # only samples tagged ToReview

Connects over gRPC to a running experiment (``127.0.0.1:50051`` by default),
same as ``weightslab cli``. See :doc:`user_commands` for every flag.

**Weights Studio UI**

The "Export" button sits next to the Save and Grid settings controls in the
Details panel. Clicking it opens a small format picker (CVAT / Label
Studio / V7) with an optional tag selector; the chosen format (and tags, if
any) trigger an ``ExportAnnotations`` gRPC call and the response downloads
as a file in your browser.

**In-app chat agent**

Because the chat agent (see :doc:`agent`) has general tool access to the
live experiment process, you can also just ask for this in plain language --
e.g. "export the samples tagged ToReview to CVAT format for relabeling" --
and it calls :func:`wl.export_annotations` with the matching ``tags=``
argument itself. No special wiring is needed beyond the API existing.

Filtering by tag
------------------

All three entry points accept a tag filter (``tags=`` in Python, ``--tag`` on
the CLI, repeatable; the tag picker in the UI) that restricts the export to
samples carrying **any** of the given tags -- boolean tags set via
:func:`wl.tag_samples` or categorical values set via
:func:`wl.set_categorical_tag` both work, since they share the same
``tag:<name>`` column. Omit it to export every sample. This is the mechanism
for a "send only what needs another look" relabeling handoff, e.g. tagging
uncertain samples as ``ToReview`` during data exploration and exporting just
that subset.

How annotations are resolved
------------------------------

Every export path collects annotations from the same registered dataframe
that backs the rest of WeightsLab (`get_dataframe()`), grouping the
``(sample_id, annotation_id)`` multi-index rows by sample:

- **Boxes** — read from the ``target`` (or ``prediction``, with
  ``use_predictions=True``) column when it holds coordinate-shaped data
  (``(x1, y1, x2, y2[, conf][, cls])``), whether that's a single box per
  sample or several boxes exploded across annotation rows.
- **Masks -> polygons** — read from the same column when it holds a dense
  ``(H, W)`` array (pixel value = class id); one polygon per connected
  region per class id.

Two real gaps in the current data model drive the "best effort" behavior
below — call these out explicitly if an export looks wrong:

- **No dedicated class-id -> name registry.** Labels are resolved, in order:
  an explicit ``class_names`` argument; else a ``class_names`` attribute on
  the dataset object backing the relevant split; else ``"class_<id>"``.
- **No per-sample stored image path or dimensions.** A real image path is
  best-effort resolved from a few common dataset attribute names
  (``image_paths``, ``img_files``, ``images``, ``imgs``, ``files``,
  ``samples``); dimensions come from that file (via Pillow) or, for
  segmentation samples, directly from the mask's own shape. When no path
  resolves, the exported filename is synthetic (``sample_<id>.jpg``) — **no
  image file is copied or embedded**, so you must ensure the filenames you
  upload to CVAT/Label Studio/V7 match the ones in the export.
