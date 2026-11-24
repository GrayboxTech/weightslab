Language Model, tiny Shakespeare (PyTorch)
==========================================

.. raw:: html

   <div class="wl-eg-page-tags">
     <span class="wl-eg-badge wl-eg-badge--pytorch">PyTorch</span>
     <span class="wl-eg-tag">language model</span>
     <span class="wl-eg-tag">llm</span>
     <span class="wl-eg-tag">transformer</span>
     <span class="wl-eg-tag">gpt</span>
     <span class="wl-eg-tag">text</span>
     <span class="wl-eg-tag">shakespeare</span>
   </div>

**Example:** ``weightslab/examples/PyTorch/wl-shakespeare-char/main.py``

**Task:** Character-level language modelling on Karpathy's tiny Shakespeare
(the 1.1 MB excerpt from char-rnn and nanoGPT), with a GPT from WeightsLab's
baseline zoo: ``NanoGPT`` by default, or ``MicroGPT``.

Run it::

    cd weightslab/examples/PyTorch/wl-shakespeare-char
    python main.py

The text is downloaded on the first run. Open the Weights Studio URL the script
prints and press **Train**.

What you get in the studio
--------------------------

- **Data grid**: each sample is a window of 64 characters, shown as its text,
  with where it starts in the corpus. Sort it by the per-window loss or by
  ``test_metric/char_accuracy`` (the share of next characters predicted right)
  to find the passages the model struggles with.
- **Modeling tab**: the GPT's blocks, the residual stream, attention per head
  and the input as characters (see :doc:`../../weights_studio_ui/modeling`).
  Heads can be frozen, reset or perturbed while training is paused.
- **Why do these fail?** on any windows: the check step shows each window's
  share of characters predicted right and its most confident miss (see
  :doc:`../../weights_studio_ui/why_fail`).
- **Generated text**: every ``sample_every`` steps the model writes a short
  passage to the console and to ``<root_log_dir>/samples.txt``. After about
  500 steps on a CPU it already has the shape of a play:

  .. code-block:: text

     GRIZARD ES:
     Yeare has 's my maghat o sple withh bown is wear:

Integration walkthrough
-----------------------

1. Windows of text as samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first 90% of the text gives the training windows (one every 32
characters, about 31k), the last 10% the test windows (one every 64, about
1.7k). ``get_items`` tells the studio what to show for each sample, here the
window as text and its offset in the corpus:

.. code-block:: python

   class CharWindows(Dataset):
       def __getitem__(self, i):
           p = self.starts[i]
           return self.ids[p:p + self.block], self.ids[p + 1:p + self.block + 1]

       def get_items(self, i, include_metadata=False, include_labels=False, include_images=False):
           x, y = self[i]
           text = "".join(self.itos[int(t)] for t in x) if include_images else None
           meta = {"char_offset": self.offset + self.starts[i]} if include_metadata else {}
           return text, str(i), (y.numpy() if include_labels else None), meta

2. The model knows its alphabet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   _model = zoo.NanoGPT(vocab_size=len(chars), block_size=64, n_embd=128, n_head=4, n_layer=4)
   _model.itos = chars    # the Modeling tab shows tokens as characters
   model = wl.watch_or_edit(_model, flag="model", device=device, compute_dependencies=True)

``compute_dependencies=True`` builds the neuron graph, so neurons and heads can
also be pruned or added from the Modeling tab.

3. One loss per window
~~~~~~~~~~~~~~~~~~~~~~

WeightsLab records one value per sample, so the cross-entropy is averaged over
each window's positions rather than over the batch:

.. code-block:: python

   class SequenceCE(nn.Module):
       def forward(self, logits, targets):
           return F.cross_entropy(logits.transpose(1, 2), targets, reduction="none").mean(dim=1)

   train_criterion = wl.watch_or_edit(SequenceCE(), flag="loss", signal_name="train-loss-CE", log=True)

The test pass adds each window's character accuracy with
``wl.save_signals(batch_ids=ids, signals={"test_metric/char_accuracy": acc})``.

Configuration
-------------

``config.yaml`` next to ``main.py``:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Key
     - Meaning
   * - ``model.class``
     - ``NanoGPT`` (fused attention, GELU, LayerNorm, tied embeddings) or
       ``MicroGPT`` (Karpathy's microgpt: RMSNorm, ReLU, no biases).
   * - ``model.block_size``, ``n_embd``, ``n_head``, ``n_layer``, ``dropout``
     - The model's size. The defaults (about 0.8M parameters) train on a CPU;
       Karpathy's "baby GPT" for a GPU is ``block_size: 256``, ``n_embd: 384``,
       ``n_head: 6``, ``n_layer: 6``, ``dropout: 0.2``.
   * - ``data.train_share``
     - The share of the text used for training (0.9).
   * - ``data.*_loader.stride``
     - One window every *stride* characters.
   * - ``sample_every``
     - Steps between generated samples.
   * - ``data_root``
     - Where the text is downloaded (default: ``<root_log_dir>/data``).
