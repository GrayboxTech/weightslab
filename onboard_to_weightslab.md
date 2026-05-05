# Onboarding on weightslab

```diff
--- train_baseline.py
+++ train_wl.py
@@ -1,11 +1,12 @@
 import argparse
 import torch
 import torch.nn as nn
-from torch.utils.data import DataLoader
 from torchvision import datasets, transforms, models
 from torchmetrics.classification import MulticlassAccuracy

-import wandb
+import weightslab as wl
+from weightslab.components.global_monitoring import (
+    guard_training_context, guard_testing_context)


+@wl.signal(name="image_bytes")
+def image_bytes(ctx): return ctx.image.nbytes  # static per-sample signal
+
+@wl.signal(name="byte_adjusted_loss", subscribe_to="loss/CE")
+def byte_adjusted_loss(ctx): return ctx.subscribed_value / ctx.image_bytes  # chains on image_bytes
+
 def main():
@@ -15,29 +16,38 @@

     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     parameters = {"batch_size": 128, "lr": 1e-3}

-    wandb.init(project="cifar10")
-
     transform = transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
     ])
     train_set = datasets.CIFAR10("./data", train=True,  download=True, transform=transform)
     test_set  = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
-    train_loader = DataLoader(train_set, batch_size=parameters["batch_size"], shuffle=True, num_workers=2)
-    test_loader  = DataLoader(test_set,  batch_size=256,                              num_workers=2)
+    wl.watch_or_edit(parameters, flag="hyperparameters")  # live-editable in UI
+
+    train_loader = wl.watch_or_edit(
+        train_set, flag="data", loader_name="train_loader",
+        batch_size=parameters["batch_size"], shuffle=True, is_training=True)
+    test_loader  = wl.watch_or_edit(
+        test_set,  flag="data", loader_name="test_loader",
+        batch_size=256, shuffle=False, is_training=False)

     model = models.resnet18(weights=None)
     model.fc = nn.Linear(model.fc.in_features, 10)
     model = model.to(device)
     optimizer = torch.optim.Adam(model.parameters(), lr=parameters["lr"])

-    criterion = nn.CrossEntropyLoss()
-    accuracy  = MulticlassAccuracy(num_classes=10).to(device)
+    criterion = wl.watch_or_edit(
+        nn.CrossEntropyLoss(), flag="loss", signal_name="loss/CE")
+    accuracy  = wl.watch_or_edit(
+        MulticlassAccuracy(num_classes=10).to(device),
+        flag="metric", signal_name="acc")
+
+    wl.serve(serving_grpc=True)

     for epoch in range(1, args.epochs + 1):
         model.train()
         accuracy.reset()
         for x, y in train_loader:
+            with guard_training_context:
                 x, y = x.to(device), y.to(device)
                 logits = model(x)
                 loss = criterion(logits, y)
                 optimizer.zero_grad()
                 loss.backward()
                 optimizer.step()
                 accuracy.update(logits, y)
-            wandb.log({"train/loss": loss.item()})
-        wandb.log({"train/acc": accuracy.compute().item(), "epoch": epoch})
+            wl.save_signals(preds_raw=logits, targets=y,
+                            signals={"metric/accuracy": accuracy.compute().item()})

         model.eval()
         accuracy.reset()
         with torch.no_grad():
             for x, y in test_loader:
+                with guard_testing_context:
                     x, y = x.to(device), y.to(device)
                     accuracy.update(model(x), y)
-        wandb.log({"test/acc": accuracy.compute().item(), "epoch": epoch})
+                wl.save_signals(preds_raw=logits, targets=y,
+                                signals={"metric/accuracy": accuracy.compute().item()})

-    wandb.finish()
+    wl.keep_serving()
```
