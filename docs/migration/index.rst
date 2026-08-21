:orphan:

.. _migration:

Migration guides
================

.. attention:: Draft — not yet linked from the main navigation

   These guides are written but not published: the entry is commented out of
   the site's index while the mappings are reviewed against each tool's
   current API. Reach them by direct link until that entry is uncommented.

Moving an existing experiment onto WeightsLab, from whichever tool you are
using now. Each guide follows the same four sections:

**Migration notes**
   What changes conceptually — the part worth reading before you touch code.

**Replaced parts**
   A call-for-call mapping from the tool's API to WeightsLab's.

**Updated examples**
   A complete before/after training script.

**Expanded UI documentation**
   Where the tool's UI concepts land in Weights Studio.

.. toctree::
   :maxdepth: 1

   from_wandb
   from_tensorboard
   from_voxel51
   from_3lc

The one idea behind all four
-----------------------------

Every tool below is, in the end, **write-only**. Your training loop reports
outward — scalars, images, tables, dataset revisions — and a UI reads what was
reported. Changing anything means stopping the run, editing code or data, and
starting again.

WeightsLab keeps that reporting, and adds a return path. The UI holds live
references to the same objects your loop is using, so an action taken there —
discarding samples, retagging them, editing a learning rate, freezing a layer —
lands on the **running** experiment:

.. code-block:: python

   import weightslab as wl

   # These wrappers are the return path: the objects stay yours, and the UI
   # gets a handle on them.
   parameters  = wl.watch_or_edit(parameters, flag="hyperparameters")
   model       = wl.watch_or_edit(model, flag="model", device=device)
   train_loader = wl.watch_or_edit(train_dataset, flag="data",
                                   loader_name="train_loader", is_training=True)

   wl.serve(serving_grpc=True)   # the UI can now reach all of the above

That difference is why the mappings below are rarely one-to-one, and why the
migration notes matter more than the tables.

.. note::

   These guides describe each tool's typical usage at the time of writing.
   They are a starting point for a port, not a specification of the other
   tool's API — check against its current documentation as you go.
