# WeightsLab Signals and Evaluation Functions Guide

This guide covers two essential decorators in WeightsLab: `@wl.signal()` for metric tracking and `@wl.eval_fn()` for custom evaluation functions.

## Table of Contents
1. [@wl.eval_fn Decorator](#wl-eval-fn-decorator)
2. [@wl.signal Decorator](#wl-signal-decorator)
3. [Complete Detection Use Case Example](#complete-detection-use-case-example)
4. [Best Practices](#best-practices)

---

## @wl.eval_fn Decorator

### Overview

The `@wl.eval_fn` decorator registers a custom evaluation function with WeightsLab's evaluation system. It enables independent evaluation workflows that automatically integrate with the UI, logging, and signal computation.

### Signature

```python
@wl.eval_fn
def my_eval_function(loader):
    """
    Args:
        loader: DataLoader or compatible object providing batches during evaluation
    """
    pass
```

### Key Features

- **Single Parameter**: Function accepts only one parameter: the data `loader`
- **Ledger-Based**: All model, device, and metric components are retrieved from `wl.ledger`
- **Guard Context**: Use `wl.guard_testing_context` to properly track evaluation metrics
- **Signal Integration**: All metrics and losses are automatically logged and available to signals
- **Cancellation Support**: Evaluation can be canceled mid-execution via the UI

### Requirements

Before registering an `@wl.eval_fn`:

1. **Model** must be registered:
   ```python
   model = wl.watch_or_edit(model, flag="model", device=device)
   ```

2. **Loss functions** must be registered:
   ```python
   loss_fn = wl.watch_or_edit(
       my_loss, 
       flag="loss", 
       name="val_my_loss",
       per_sample=True,  # For per-sample tracking
       log=True
   )
   ```

3. **Metric functions** must be registered:
   ```python
   metric_fn = wl.watch_or_edit(
       my_metric,
       flag="metric",
       name="val_my_metric",
       per_sample=True,
       log=True
   )
   ```

4. **Data loader** must be registered:
   ```python
   loader = wl.watch_or_edit(
       dataset,
       flag="data",
       loader_name="val_loader",
       batch_size=32
   )
   ```

### Simple Example

```python
import weightslab as wl
import torch

@wl.eval_fn
def simple_evaluate(loader):
    """Simple evaluation function for classification."""
    model = wl.ledger.get_model()
    device = wl.ledger.get_device()
    criterion = wl.ledger.get_loss(name="val_loss")
    metric = wl.ledger.get_metric(name="val_accuracy")
    
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with wl.guard_testing_context:
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            with torch.no_grad():
                logits = model(images)
            
            # Compute loss and metrics
            loss = criterion(logits, labels)
            acc = metric(logits, labels)
            
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            
            # Optional: print progress
            print(f"Batch {batch_idx}: loss={loss.item():.4f}, acc={acc:.4f}")
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    print(f"Evaluation complete: avg_loss={avg_loss:.4f}")
```

### Accessing Ledger Components

```python
# Get the model
model = wl.ledger.get_model()
model = wl.ledger.get_model(name="main")  # Optional: specific model name

# Get the device
device = wl.ledger.get_device()
device = wl.ledger.get_device(name="main")

# Get loss functions
loss = wl.ledger.get_loss(name="val_loss")
losses = wl.ledger.get_loss(name="val_loss")  # List if multiple registered

# Get metric functions
metric = wl.ledger.get_metric(name="val_accuracy")

# Get optimizer
optimizer = wl.ledger.get_optimizer()

# Get data loader
loader = wl.ledger.get_dataloader(split_name="val")
```

---

## @wl.signal Decorator

### Overview

The `@wl.signal` decorator creates metric computation functions that automatically subscribe to evaluation events. Signals are called at specific points during evaluation (e.g., after each validation batch or at regular intervals).

### Signature

```python
@wl.signal(
    name: str,
    subscribe_to: str = "val_metric",
    compute_every_n_steps: int = 1
)
def my_signal_fn(...):
    pass
```

### Parameters

- **`name`** (str): Unique identifier for the signal (e.g., `"detection/dice_score"`)
- **`subscribe_to`** (str): Event to subscribe to. Common values:
  - `"val_metric"` — Called with metric computation results
  - `"train_metric"` — Called during training metric computation
  - Custom event names defined by your application
- **`compute_every_n_steps`** (int): Only compute signal every N steps. Default: 1 (every step)

### Function Signature

Signal functions typically have flexible signatures:

```python
@wl.signal(name="my_signal", subscribe_to="val_metric")
def my_signal(metric_value, **kwargs):
    """
    Args:
        metric_value: The computed metric from the subscribed event
        **kwargs: Additional arguments passed by the framework
    
    Returns:
        float or dict: Computed signal value to be logged
    """
    return float(metric_value)
```

### Common Use Cases

#### 1. Dice Score from IoU (Detection)

Convert IoU values to Dice/F1 scores:

```python
@wl.signal(
    name="detection/dice_score", 
    subscribe_to="val_metric", 
    compute_every_n_steps=10
)
def detection_dice_signal(pred, batch, conf: float = 0.25, iou_thres: float = 0.5, **kwargs):
    """
    Compute Dice score (F1) for detection: Dice = 2*IoU / (1 + IoU)
    """
    from ultralytics.utils.nms import box_iou
    
    # Decode predictions to bounding boxes
    img_h, img_w = batch['img'].shape[-2:]
    preds_nms = _decode_predictions(pred, img_h, img_w, conf, iou_thres)
    
    # Compute IoU matrix
    iou_matrix = box_iou(gt_boxes, pred_boxes)
    max_ious = iou_matrix.max(dim=1).values
    
    # Convert to Dice: Dice = 2*IoU / (1 + IoU)
    dice = 2.0 * max_ious / (1.0 + max_ious)
    
    return dice.mean().item() if dice.numel() > 0 else 0.0
```

#### 2. Aggregate Multiple Metrics

```python
@wl.signal(name="aggregated/f1_score", subscribe_to="val_metric")
def f1_score_signal(precision, recall, **kwargs):
    """
    Compute F1 score from precision and recall.
    
    Args:
        precision: Precision metric from validation
        recall: Recall metric from validation
    """
    p = float(precision) if precision is not None else 0.0
    r = float(recall) if recall is not None else 0.0
    
    if p + r == 0:
        return 0.0
    
    f1 = 2 * (p * r) / (p + r)
    return f1
```

#### 3. Error Detection Signal

```python
@wl.signal(name="monitoring/loss_spike", subscribe_to="train_metric")
def loss_spike_detector(loss, baseline=1.0, threshold=2.0, **kwargs):
    """Detect when loss spikes above threshold * baseline."""
    loss_val = float(loss) if loss is not None else 0.0
    return 1.0 if loss_val > (baseline * threshold) else 0.0
```

---

## Complete Detection Use Case Example

This example shows a full detection workflow with both `@wl.eval_fn` and `@wl.signal` decorators working together.

### Setup: Register Components

```python
import weightslab as wl
from ultralytics import YOLO
import torch

# 1. Load model
model = YOLO("yolo11s.pt")

# 2. Register model and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = wl.watch_or_edit(
    model,
    flag="model",
    device=device,
    compute_dependencies=False
)

# 3. Register loss functions
from utils.criterions import PerSampleDetectionLoss

train_loss = wl.watch_or_edit(
    PerSampleDetectionLoss(model, loss_type=0),
    flag="loss",
    name="train_detection_loss/bboxes",
    per_sample=True,
    log=True
)

val_loss = wl.watch_or_edit(
    PerSampleDetectionLoss(model, loss_type=0),
    flag="loss",
    name="val_detection_loss/bboxes",
    per_sample=True,
    log=True
)

# 4. Register metric functions
from utils.criterions import PerSampleIoU

val_metric = wl.watch_or_edit(
    PerSampleIoU(conf=0.25, iou_thres=0.5),
    flag="metric",
    name="val_per_sample_iou",
    per_sample=True,
    log=True
)

# 5. Register data loaders
train_loader = wl.watch_or_edit(
    train_dataset,
    flag="data",
    loader_name="train",
    batch_size=16
)

val_loader = wl.watch_or_edit(
    val_dataset,
    flag="data",
    loader_name="val",
    batch_size=16
)
```

### Define Evaluation Function

```python
@wl.eval_fn
def validate(loader):
    """
    Standalone detection validation function using ledger components.
    
    This function:
    - Retrieves model and losses from WeightsLab ledger
    - Processes YOLO predictions and ground truth
    - Computes per-sample losses and metrics
    - Integrates with WeightsLab signal system
    """
    # Get components from ledger
    model = wl.ledger.get_model()
    device = wl.ledger.get_device()
    val_criterion_boxes = wl.ledger.get_loss(name="val_detection_loss/bboxes")
    val_metric = wl.ledger.get_metric(name="val_per_sample_iou")
    
    if model is None:
        raise RuntimeError("Model not found in ledger")
    
    model.eval()
    
    with wl.guard_testing_context:
        for batch_idx, batch in enumerate(loader):
            images = batch['img'].float().to(device)
            targets = batch['bboxes']
            
            # Forward pass
            with torch.no_grad():
                raw_preds = model(images)
            
            # Compute loss
            loss = val_criterion_boxes(
                raw_preds,
                batch,
                batch_ids=batch['batch_idx']
            )
            
            # Compute metric (IoU)
            ious = val_metric(
                raw_preds,
                batch,
                batch_ids=batch['batch_idx']
            )
            
            # Log progress
            loss_mean = loss.mean().item() if isinstance(loss, torch.Tensor) else loss
            iou_mean = ious.mean().item() if isinstance(ious, torch.Tensor) else ious
            
            print(f"Batch {batch_idx}: loss={loss_mean:.4f}, iou={iou_mean:.4f}")
```

### Define Dice Score Signal

```python
@wl.signal(
    name="detection/dice_score",
    subscribe_to="val_metric",
    compute_every_n_steps=10
)
def detection_dice_signal(pred, batch, conf: float = 0.25, iou_thres: float = 0.5, **kwargs):
    """
    Compute Dice score for detection predictions.
    
    Dice = 2*IoU / (1 + IoU)  (F1 score based on IoU)
    """
    from ultralytics.utils.nms import box_iou
    from ultralytics.utils.ops import xywh2xyxy
    
    # Decode predictions
    img_h, img_w = batch['img'].shape[-2:]
    preds_nms = _decode_predictions(pred, img_h, img_w, conf, iou_thres)
    
    # Convert GT boxes to xyxy format
    scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=batch['bboxes'].dtype)
    gt_xyxy = xywh2xyxy(batch['bboxes'].detach().cpu()) * scale
    
    batch_size = batch['img'].shape[0]
    dice_scores = torch.full((batch_size,), float("nan"))
    
    # Compute Dice for each image in batch
    for i in range(batch_size):
        batch_mask = batch['batch_idx'] == i
        if batch_mask.sum() > 0 and preds_nms[i][:, :4].numel() > 0:
            iou_matrix = box_iou(gt_xyxy[batch_mask], preds_nms[i][:, :4])
            max_ious = iou_matrix.max(dim=1).values
            
            # Dice = 2*IoU / (1 + IoU)
            dice = 2.0 * max_ious / (1.0 + max_ious)
            dice_scores[i] = dice.mean()
    
    dice_scores = torch.nan_to_num(dice_scores, nan=0.0)
    
    return dice_scores.mean().item() if dice_scores.numel() > 0 else 0.0
```

### Integration with Trainer

```python
class WLCompatileDetTrainer(DetectionTrainer):
    """Detection trainer with WeightsLab integration."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_experiment_modules()
    
    def _init_experiment_modules(self):
        """Initialize WL tracking for model, losses, and metrics."""
        # Setup model (already registered above)
        self.model = wl.watch_or_edit(self.model, flag="model")
        
        # Setup losses (already registered above)
        self.criterion = wl.ledger.get_loss(name="train_detection_loss/bboxes")
    
    def validate(self):
        """Use WL evaluation function."""
        validate(self.val_loader)  # Calls registered @wl.eval_fn
    
    def train(self):
        """Main training loop with evaluation."""
        for epoch in range(self.epochs):
            # Training step
            loss = self._train_step()
            
            # Evaluation (calls @wl.eval_fn)
            if epoch % self.val_frequency == 0:
                if wl.run_pending_evaluation():
                    continue  # Evaluation was triggered
```

---

## Best Practices

### 1. Error Handling in eval_fn

```python
@wl.eval_fn
def safe_evaluate(loader):
    """Include error handling in evaluation functions."""
    try:
        model = wl.ledger.get_model()
        device = wl.ledger.get_device()
        
        if model is None or device is None:
            raise RuntimeError(
                "Model or device not found in ledger. "
                "Register them with wl.watch_or_edit(model, flag='model')"
            )
        
        # Evaluation logic...
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise
```

### 2. Using Guard Contexts

Always use `wl.guard_testing_context` to properly track evaluation:

```python
@wl.eval_fn
def evaluate(loader):
    model = wl.ledger.get_model()
    
    with wl.guard_testing_context:  # ← Important!
        for batch in loader:
            # All metric logging happens within this context
            predictions = model(batch)
            loss = metric_fn(predictions, batch)
```

### 3. Per-Sample Tracking

For detailed analysis, use `per_sample=True` when registering losses/metrics:

```python
loss_fn = wl.watch_or_edit(
    my_loss,
    flag="loss",
    name="val_loss",
    per_sample=True,  # ← Track per-sample losses
    log=True
)
```

### 4. Signal Parameter Flexibility

Signals receive flexible parameters from the subscribed event. Use `**kwargs` to handle variations:

```python
@wl.signal(name="flexible/signal", subscribe_to="val_metric")
def flexible_signal(main_metric=None, **kwargs):
    """Handle optional parameters flexibly."""
    if main_metric is None:
        return 0.0
    return float(main_metric)
```

### 5. Cancellation Support

Evaluation can be canceled by the user. The framework handles this automatically, but you can check if needed:

```python
@wl.eval_fn
def evaluate(loader):
    model = wl.ledger.get_model()
    
    with wl.guard_testing_context:
        for batch_idx, batch in enumerate(loader):
            # Framework checks for cancellation automatically
            # No explicit cancellation check needed
            predictions = model(batch)
            loss = metric_fn(predictions, batch)
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}...")  # User can see progress
```

### 6. Documentation

Always document your signals:

```python
@wl.signal(
    name="domain/metric_name",
    subscribe_to="val_metric",
    compute_every_n_steps=5
)
def my_signal(raw_metric, **kwargs):
    """
    Compute derived metric from raw validation metric.
    
    Args:
        raw_metric: The raw metric computed by val function
        **kwargs: Additional framework arguments
    
    Returns:
        float: Computed signal value
    
    Example:
        >>> signal_value = my_signal(0.85)
        >>> print(signal_value)
        0.92
    """
    return process_metric(raw_metric)
```

---

## Summary

| Feature | @wl.eval_fn | @wl.signal |
|---------|-------------|-----------|
| **Purpose** | Custom evaluation logic | Metric computation |
| **Triggered** | On-demand or scheduled | On subscribed events |
| **Parameters** | Loader only | Event-dependent |
| **Logging** | Automatic within guard context | Automatic on return |
| **Common Use** | Validation/testing | F1, Dice, loss monitoring |
| **Ledger Access** | Yes | Via kwargs |

Both decorators work together to create a flexible, modular evaluation system integrated with WeightsLab's tracking and UI.
