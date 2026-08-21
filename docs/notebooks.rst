External Integrations
=====================

This page groups integration modes outside the core training-loop docs.

Notebook integrations
---------------------

Google Colab (cloud)
~~~~~~~~~~~~~~~~~~~~

Typical behavior:

1. Start the backend in Colab.
2. Open a raw TCP bore tunnel from Colab.
3. Start Studio locally and bridge to Colab with ``weightslab tunnel``.
4. Open Studio locally and control the remote run in real time.

Colab-side startup (inside notebook):

.. code-block:: python

   import weightslab as wl

   # Starts gRPC backend and auto-deploys a bore tunnel endpoint.
   # The notebook output prints: weightslab tunnel bore.pub:<port>
   wl.serve(serving_grpc=True, serving_bore=True)

   # Keep backend available while you interact from local Studio
   wl.keep_serving()

Alternative Colab flow (manual tunnel command):

.. code-block:: bash

   bore local 50051 --to bore.pub

Local machine flow:

.. code-block:: bash

   # Terminal 1: Studio UI
   weightslab start

   # Terminal 2: bridge to Colab endpoint printed by the notebook
   weightslab tunnel bore.pub:12345

See :doc:`weights_studio` and :doc:`user_commands` for tunnel details.

Local Jupyter Notebook
~~~~~~~~~~~~~~~~~~~~~~

From the Studio landing page (no backend connected), you can bootstrap or reopen
a local notebook workflow. This runs as a standalone local Jupyter server
process.

In-training notebook (UI embedded)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a backend is connected, Studio can open its embedded notebook panel. This
notebook runs inside the training process and sees live objects directly
(``df``, model, optimizer, checkpoints).

Python code execution in cells:

.. code-block:: python

   import matplotlib.pyplot as plt

   # Explore available signal columns
   signal_cols = [c for c in df.columns if c.startswith("signals")]
   print(signal_cols[:10])

   # Example: quick stats for loss-like columns
   loss_cols = [c for c in signal_cols if "loss" in c.lower()]
   print(df[loss_cols].describe().T[["mean", "std", "min", "max"]])

   # Example: plot recent values for the first available loss-like signal
   if loss_cols:
       c = loss_cols[0]
       recent = df[c].dropna().tail(200)
       plt.figure(figsize=(8, 3))
       plt.plot(recent.values)
       plt.title(f"Recent values for {c}")
       plt.xlabel("Recent samples")
       plt.ylabel("Signal value")
       plt.grid(alpha=0.25)
       plt.show()

Agent query cells (``>`` prompts):

.. code-block:: text

   > Generate Python code to plot training loss statistics over the last 200 samples
   > including mean, std, min, max, and one line chart.

The assistant writes the code into the cell; then you run that generated code as
a normal Python cell.

See the "Embedded Experiment Notebook" section in :doc:`weights_studio`.
