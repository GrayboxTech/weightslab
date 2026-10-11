# WeightsLab: YOLO boxes, one point per box (Ultralytics)

A COCO-pretrained YOLO11n is fine-tuned on **coco128** through `wl.WLAwareTrainer`.
At the end of every epoch the trainer projects **every box** of the train and
validation splits onto one 3-D map: each labelled box, and each confident
prediction with no label under it. On a detection set this is the question
you actually ask: are the boxes labelled `truck` sitting inside the `car`
cluster?

## Run

```bash
cd weightslab/examples/Ultralytics/wl-detection-instances
python main.py          # needs: pip install ultralytics
weightslab start        # in another terminal, then open http://localhost:8080
```

coco128 and `yolo11n.pt` are downloaded by Ultralytics on first use. Change
`data_root` in `config.yaml` to any Ultralytics dataset YAML (`coco8.yaml` for
an 8-image smoke test, or your own).

Open the Studio once training has started. The trainer registers its data
loaders when training begins, and a page opened before that shows an empty
grid until you reload it.

## What to look at

**Projection board**: press the Projection button in the Data header, then pick
`instances` in the board's picker. `instances_pca` is the same boxes laid out
by PCA, written by the explicit call at the end of `main.py`.

| Colour by (board cog) | Shows |
|---|---|
| Class (default) | the box's class: is a class one cluster, or several? |
| Match (label vs prediction) | `consistent`, `wrong class` (a truck predicted as a car), `no prediction` (a spurious or shifted label), `missing label` (a confident prediction with nothing labelled under it) |
| Predicted class / Confidence / IoU with prediction / Box area | the prediction on each box |
| Split | train and val boxes on the same map |

- **Lasso** a group of boxes. The Data board narrows to the **images** they are in.
- **Double-click** a box. Its image opens in the modal with that box singled out.

**List mode** (the list button in the Data header):

- the **+** on a row lists that image's boxes: a crop, the class, how the
  prediction on it agrees, the predicted class, confidence, IoU and any per-box
  signal;
- **Instances only** (the button beside the list toggle) lists every box of the
  page, one row each; click a header to sort, double-click a box to open it.

## How it is wired

Everything is in `config.yaml`:

```yaml
instance_projection:
  every_n_epochs: 1
  prefix: instances
  method: umap        # or pca
  conf: 0.25          # a prediction needs this confidence to count
  match_iou: 0.5      # IoU above which a prediction sits on a label
```

Each box becomes one vector, read from the maps YOLO's Detect head reads
(P3/P4/P5): the box is pooled out of every level with `roi_align`, with no crops
and no second model. Predictions are pooled the same way, so an unlabelled car
the model found lands among the labelled cars. The explicit form, usable at any
time (for instance from the Studio notebook):

```python
from weightslab.integrations.ultralytics import project_trainer_instances
project_trainer_instances(yolo.trainer, prefix="instances_pca", method="pca")
# or, for any model and loaders:
wl.project_instances(model, {"train": train_loader, "val": val_loader}, prefix="boxes")
```

See `docs/projection.rst`, "One point per box".
