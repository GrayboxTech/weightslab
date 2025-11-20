import tqdm
import torch

from collections import defaultdict

from weightslab.components.global_monitoring import guard_training_context


# Fashion GLOBAL
def train(loader, model, optimizer, criterion):
    with guard_training_context:
        # INFERENCE
        optimizer.zero_grad()
        (input, in_id, label) = next(loader)
        input = input.to('cuda')
        label = label.to('cuda')
        preds = model(input)
        loss = criterion(preds.float(), label.long())
        loss.mean().backward()
        optimizer.step()
    print(
        f"| Train Loss: {loss.mean():.4f} "
        
    )

def test(loader, model, criterion_bin, criterion_mlt, metric_bin, metric_mlt, device):
    with torch.no_grad():
        losses = 0.0
        metric_totals = defaultdict(float)
        for test_step, (inputs, ids, labels) in enumerate(tqdm.tqdm(loader, desc='Testing..')):
            inputs = inputs.to(device)
            bin_labels = (labels == 0).float().to(device)
            mlt_labels = labels.to(device)
            preds = model(inputs)
            losses_batch_bin = criterion_bin(preds[:, 0], bin_labels)
            losses_batch_mlt = criterion_mlt(preds, mlt_labels)
            losses_batch = torch.cat([losses_batch_bin[..., None], losses_batch_mlt[..., None]], axis=1)
            test_loss = torch.mean(losses_batch_bin) + torch.mean(losses_batch_mlt)
            metric_bin.update(preds[:, 0], bin_labels)
            metric_mlt.update(preds, mlt_labels)
            test_acc_bin = metric_bin.compute() * 100
            test_acc_mlt = metric_mlt.compute() * 100
        losses = losses + test_loss
        metric_totals['bin'] += test_acc_bin
        metric_totals['mlt'] += test_acc_mlt
        print(
            f"| Test Loss: {test_loss:.4f} " +
            f"| Test Acc mlt: {metric_totals['mlt']:.2f}%"
            f"| Test Acc bin: {metric_totals['bin']:.2f}%"
        )