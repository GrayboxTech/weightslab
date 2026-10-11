# WeightsLab: a plain-PyTorch detector, one point per box

The Penn-Fudan pedestrian detector of [`../wl-detection`](../wl-detection): a
MobileNetV3-Small backbone and a small grid head, no detection framework. It is
trained the same way, and every `instance_projection.every_n_steps` steps
`wl.project_instances` places **every box** of the train and test splits on one
map: each labelled pedestrian, and each confident prediction with no label
under it.

WeightsLab does not know this model, so [`instance_adapter.py`](instance_adapter.py)
describes it in about 60 lines. That adapter is what any other detector needs.

## Run

```bash
cd weightslab/examples/PyTorch/wl-detection-instances
python main.py
weightslab start        # in another terminal, then open http://localhost:8080
```

Press **Play** in the Studio. The data, model and criterions are imported from
`../wl-detection`, and Penn-Fudan (~50 MB) downloads into `../wl-detection/data`
on first run. The first box projection lands at step 150 (`every_n_steps` in
`config.yaml`), and again every 150 steps.

## What to look at

- **Projection board**: press the Projection button in the Data header and pick
  `instances`. Colour by **Match (label vs prediction)**:
  - `consistent`: a labelled pedestrian the model found;
  - `no prediction`: a labelled pedestrian it missed, or whose box is off;
  - `missing label`: a confident prediction with no label under it.

  Penn-Fudan labels only the main pedestrians of each photo. A `missing label`
  cluster is very often real people in the background that the annotators left
  out. Lasso it, and the Data board shows the images to relabel. Double-click a
  point to open its image with that box singled out.
- **List mode**: the **+** on a row lists the image's boxes, with the per-box
  IoU this example logs (`train_iou/instance`). **Instances only** lists every
  box of the page, one row each.

Penn-Fudan has a single class, so `wrong class` cannot happen here. See the
[Ultralytics example](../../Ultralytics/wl-detection-instances) for 80 COCO
classes, trucks among cars included.

## The adapter

```python
class SmallDetectorInstanceAdapter(InstanceAdapter):
    def batches(self, loader):          # images + labelled boxes (pixels) + annotation ids
        for images, ids, targets, _ in tracked_batches(loader, det_collate):
            ...
            yield InstanceBatch(inputs=images, sample_ids=ids, gt=gt, image_size=(h, w))

    def forward(self, model, batch, *, conf):   # feature map(s) + predictions after NMS
        ...                                     # a forward hook on model.neck
        return InstanceForward(feature_maps=[fmap], strides=[stride], predictions=preds)
```

`tracked_batches` reads the whole split in dataset order without moving the
training loop's iterator. The sweep runs outside the guard contexts and does
not age the model. RoI pooling, matching, the four categories, the UMAP/PCA
layout and storage for the Studio are all WeightsLab's:

```python
wl.project_instances(model, [train_loader, test_loader], adapter=adapter,
                     prefix="instances", method="umap", conf=0.3)
```

See `docs/projection.rst`, "One point per box".
