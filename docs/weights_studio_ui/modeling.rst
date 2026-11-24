.. _studio-modeling:

Modeling
========

.. note:: Beta

   The Modeling tab is new in v3 and marked **BETA** in the studio. What it
   shows is read from your live model; the edits it offers change that model,
   so look at the result before you keep training.

The **Modeling** tab, next to **Main** under the agent bar, shows the model
itself: its architecture as a graph, what each layer does on samples you
pick, the weights and neurons of any layer, and the tools to change them while
training is paused. It also compares how the model sees two sets of samples,
and starts :doc:`why_fail`.

It works on any model wrapped with ``wl.watch_or_edit(model, flag="model")``;
there is nothing else to call. Neuron-level prune and add need one more
argument, see `Editing the model`_. To remove the tab, set
``ENABLE_MODELING=0`` (see :doc:`../weights_studio/configuration`).

The architecture graph
----------------------

The graph is traced from one real forward pass, on the first sample of a
loader (else the model's ``dummy_input``, else its ``input_shape``). No
``torch.fx`` is involved, so any ``nn.Module`` can be drawn.

- **Layers** are boxes; **groups** (an encoder, a block, a head) are frames
  around them, taken from the module tree and labelled with the role their
  name suggests (backbone, neck, head, attention, MLP ...).
- **Edges** follow the tensors from one layer to the next. A residual add of
  two branches is its own small **+** node, so a residual stream reads as a
  chain.
- A module that did not run in the forward pass is drawn **dashed**.
- A module shared by several parents gets one node per caller.

Controls, top of the tab:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Control
     - What it does
   * - **Vertical / Horizontal**
     - The direction of the flow.
   * - **Fit**
     - Shows the whole graph.
   * - **Expand all / Collapse all**
     - Unfolds or folds every group.
   * - **Training overview**
     - Keeps a live card on every visible layer (see `Weights & neurons cards`_).
   * - **Clear**
     - Forgets the samples last run through the model.

In the graph:

- **Wheel** zooms around the cursor; **drag** the background to pan.
- **Click** a layer or a group to inspect it; **Ctrl/Shift-click** adds it to
  the selection.
- **Double-click** a group, or click its chevron, to fold or unfold it.
- **Drag** a layer to move it, with the layers that follow it; Ctrl- or
  Shift-drag moves that layer only. **Right-click the background → Reset node
  positions** puts them back.

The inspector on the right describes the selected layer: its type, shapes,
parameters, which operations it supports, and once samples have been run, its
activations.

Running samples through the model
---------------------------------

In **Main**, select samples in the data grid (or right-click a tag) and choose
**Model Representation**. The model runs on them in the background, under the
training lock, and keeps per-layer statistics and previews.

- Each layer's **border** becomes its health: green all round when every
  neuron fired on these samples; an orange or red share of the border for the
  share of neurons that never did. A weighted layer is judged at the ReLU (or
  activation) it feeds, because its own output is signed.
- An inner **blue** ring shows the share of the layer that is frozen.
- The inspector's **Activations** section shows the maps of the selected
  layer for each sample; **double-click** a map to zoom on it (←/→ moves
  between layers, ↑/↓ between samples; the pixel values show on hover).
- **Layers with silent neurons** lists the layers where the most neurons
  never fired.

Weights & neurons cards
-----------------------

**Double-click** a layer, or select layers and choose **Right-click → Weights
& neurons**, to open a card beside each. A card has two lanes:

- **Neurons**: one cell per neuron. The colour is its health, the brightness
  its weight norm; a **blue** border means frozen, an **orange** border that
  it changed since the previous refresh. A group's card shows one grid per
  weighted layer inside.
- **Weights**: the weight histogram, with the previous refresh outlined, and
  the weight norm per neuron when the card is large enough.

Working with cards:

- **Click** a card to inspect its layer.
- **Drag** a card: the cards of the layers that follow it come along;
  Ctrl- or Shift-drag moves only that card.
- **Resize** from a bottom corner; selected cards take the same size.
- **Double-click** a chart for the neurons & weights zoom.
- **Right-click** a neuron for the neuron actions; elsewhere on the card for
  the card menu (inspect, zoom, full or compact, close). The **×** left of the
  title closes the card.
- **Arrange cards** (top right of the graph, or right-click the background)
  puts the dragged layers back in place and lays every card out again beside
  its layer, clear of the graph and of each other.

**Training overview** keeps a compact card on every visible weighted layer and
refreshes them while training runs. Right-click the button to set how often,
and whether each refresh also re-runs the model on the samples.

Editing the model
-----------------

Edits apply to the live model. Pause training first; a structural edit is
written to the checkpoint when training resumes.

On a layer or a group (**right-click** in the graph, or the inspector's
buttons):

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Action
     - What it does
   * - **Freeze / Unfreeze**
     - Stops (or restarts) training of the weights.
   * - **Reset**
     - Re-initialises the weights.
   * - **Perturb**
     - Adds relative random noise to the weights (×0.1 from the menu).

On neurons (**right-click** a neuron in a card, the zoom or the inspector's
neuron chart; Ctrl/Shift-click selects several): the same four actions, plus
**Prune** (remove the neurons and the inputs that read them downstream) and
**Add** (new neurons, which read nothing yet, so the model's output does not
change).

Prune and add need the neuron graph:

.. code-block:: python

   model = wl.watch_or_edit(model, flag="model", compute_dependencies=True)

It follows every index of a neuron through the model, the next layer's
inputs, batch norm statistics, both branches of a residual, the same position
in every head, tied weights, and re-runs the model after each edit: if the
output shape changes or the forward fails, the edit is rolled back and the
reason is shown. The model's outputs (the classes, the vocabulary) and an
embedding's vocabulary rows are never edited. Freezing single neurons works on
any layer, with or without the neuron graph.

Compare A vs B
--------------

How differently does the model see two sets of samples? In **Main**:

1. Right-click samples or a tag → **Compare: Set as A**.
2. Right-click other samples or a tag → **Compare with A**.

With two samples selected, **Compare These 2 Samples** does both at once.

The Modeling tab then tints each layer by how differently it sees A
(violet) and B (aqua):

- **Two samples**: the normalized squared distance of the two activation maps,
  from 0 (identical) to 2.
- **Two groups**: for each neuron, the difference of the group averages
  relative to how much the samples vary within each group (Cohen's d); the
  layer's score is the RMS of its neurons' d. About 0 means the layer sees
  both alike, 0.8 or more is a large difference.

Click a layer for the neurons that make the difference and the neurons that
fire only on A or only on B; the compare zoom shows A, B and A − B side by
side. Samples in both sets are left out of both, and each side is capped at
256 samples (an even, seeded subset).

Transformers
------------

On a transformer (for example ``NanoGPT`` and ``MicroGPT`` from the baseline
zoo, or :doc:`../examples/pytorch/shakespeare_char`):

- Attention and MLP blocks get those roles; the residual stream is the chain
  of **+** nodes.
- Attention weights computed inside an attention module (a ``softmax`` over
  ``(batch, heads, T, S)``, or ``scaled_dot_product_attention``) are shown per
  head.
- The neurons of a q / k / v projection are grouped by head, so a whole head
  can be selected and frozen, reset or perturbed.
- Token-id inputs show as tokens, and as text when the model carries its
  alphabet as ``model.itos`` or has a ``decode_tokens`` method.

Unusual inputs
--------------

The tab calls the model with the first element of a batch (WeightsLab's
``(inputs, ids, labels, ...)`` convention), or with a dict batch as keyword
arguments minus its label and id keys. When your loader yields something
else, tell it how: the function returns a tensor or list (``model(x)``), a
dict (``model(**x)``), or an ``(args, kwargs)`` pair.

.. code-block:: python

   from weightslab.components.model_representation import set_modeling_inputs_fn

   set_modeling_inputs_fn(lambda batch: {"pixel_values": batch["image"]})

A model can also define ``wl_modeling_inputs(self, batch)`` itself.

**"Connections are approximate", or prune and add greyed out.** No input could
be built, or the forward pass failed (the message says which): register
``set_modeling_inputs_fn``. Prune and add need ``compute_dependencies=True``.
When they are missing on one layer only, its neurons cannot change (a model
output, an embedding's vocabulary, an operation the neuron graph does not
follow); hovering the greyed button gives the reason.
