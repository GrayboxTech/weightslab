import os
import tqdm
import time
import torch
import tempfile
import torch.nn as nn
import weightslab as wl
import weightslab.cli as cli
import torch.optim as optim
import yaml

from torchvision import datasets, transforms

from torchmetrics.classification import Accuracy

from weightslab.tests.torch_models import FashionCNN as CNN
from weightslab.components.global_monitoring import guard_training_context


# Initialize WeightsLab CLI
cli.initialize(launch_client=True)

# --- Configuration Constants ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TMP_DIR = tempfile.mkdtemp()


# --- Define functions ---
def train(loader, model, optimizer, criterion_mlt):
    with guard_training_context:
        # Get next batch
        (input, ids, label) = next(loader)
        input = input.to('cuda')
        label = label.to('cuda')

        # Inference
        optimizer.zero_grad()
        preds = model(input)
        loss = criterion_mlt(preds.float(), label.long()).mean()

        # Propagate
        loss.backward()
        optimizer.step()
    
    # Returned signals detach from the computational graph
    return loss.detach().cpu().item()

def test(loader, model, criterion_mlt, metric_mlt, device):
    with torch.no_grad():
        losses = 0.0
        metric_total = 0
        for (inputs, ids, labels) in loader:
            # Process data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Inference
            preds = model(inputs)
            losses_batch_mlt = criterion_mlt(preds, labels)
            
            test_loss = torch.mean(losses_batch_mlt)
            metric_mlt.update(preds, labels)

            # Compute signals
            losses = losses + test_loss
        loss = losses / len(loader)    
        metric_total = metric_mlt.compute() * 100

    # Returned signals detach from the computational graph
    return loss.detach().cpu().item(), metric_total.detach().cpu().item()


# Note: hyperparameter file watching is handled centrally by the ledger.
# Use `wl.watch_or_edit(<yaml_path>, flag='hp', name='<ledger_name>')`
# or pass a dict to `wl.watch_or_edit(..., flag='hp')` to register parameters.


if __name__ == '__main__':
    print('Hello world')
    
    start_time = time.time()
    device = DEVICE


    # Load YAML hyperparameters (fallback to defaults if missing)
    parameters = {}
    config_path = os.path.join(os.path.dirname(__file__), 'mnist_training_config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as fh:
            parameters = yaml.safe_load(fh) or {}
    exp_name = parameters.get('name', 'Anonymous_Exp')

    # Normalize device entry
    if parameters.get('device', 'auto') == 'auto':
        parameters['device'] = device

    # Ensure root_log_dir default
    if not parameters.get('root_log_dir'):
        parameters['root_log_dir'] = os.path.join(TMP_DIR, 'logs')

    # Wire more parameters
    epochs = parameters.get('epochs', 10)
    log_dir = parameters.get('root_log_dir', os.path.join(TMP_DIR, 'logs'))
    tqdm_display = parameters.get('tqdm_display', True)

    # Hyper Parameters
    wl.watch_or_edit(parameters, flag='parameters', name=exp_name, defaults=parameters, poll_interval=1.0)

    # Model
    _model = CNN()
    model = wl.watch_or_edit(_model, flag='model', device=parameters.get('device', device))

    # Optimizer
    lr = parameters.get('optimizer', {}).get('Adam', {}).get('lr', 0.01)
    _optimizer = optim.Adam(_model.parameters(), lr=lr )
    optimizer = wl.watch_or_edit(_optimizer, flag='optimizer')

    # Data
    _train_dataset = datasets.MNIST(
        root=os.path.join(TMP_DIR, 'data'),
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    )
    _test_dataset = datasets.MNIST(
        root=os.path.join(TMP_DIR, 'data'),
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    )
    train_bs = parameters.get('data', {}).get('train_dataset', {}).get('batch_size', 16)
    test_bs = parameters.get('data', {}).get('test_dataset', {}).get('batch_size', 16)
    train_shuffle = parameters.get('data', {}).get('train_dataset', {}).get('train_shuffle', True)
    test_shuffle = parameters.get('data', {}).get('test_dataset', {}).get('test_shuffle', False)

    train_loader = wl.watch_or_edit(_train_dataset, flag='data', batch_size=train_bs, shuffle=train_shuffle)
    test_loader = wl.watch_or_edit(_test_dataset, flag='data', batch_size=test_bs, shuffle=test_shuffle)

    # Criterion
    criterion_bin = nn.BCELoss(reduction='none')
    criterion_mlt = nn.CrossEntropyLoss(reduction='none')

    # Metrics
    metric_bin = Accuracy(task='binary').to(device)
    metric_mlt = Accuracy(task='multiclass', num_classes=10).to(device)

    # ================
    # 6. Training Loop
    print("\nStarting Training...")
    max_steps = parameters.get('training_steps_to_do', 150)
    train_range = range(max_steps)
    if tqdm_display:
        train_range = tqdm.trange(max_steps)
    for train_step in train_range:
        # Train
        train_loss = train(train_loader, model, optimizer, criterion_mlt)

        # Test
        test_loss, test_metric = None, None
        if train_step > 0 and train_step % 125 == 0:
            test_loss, test_metric = test(test_loader, model, criterion_mlt, metric_mlt, device)

        # Verbose
        print(
            f"Step {train_step}/{parameters.get('training_steps_to_do', max_steps)}: " +
            f"| Train Loss: {train_loss:.4f} " +
            (f"| Test Loss: {test_loss:.4f} " if test_loss is not None else '') +
            (f"| Test Acc mlt: {test_metric:.2f}% " if test_metric is not None else '')
        )
    print(f"--- Training completed in {time.time() - start_time:.2f} seconds ---")
    print(f"Log directory: {log_dir}")
    print(f"Epochs: {epochs}")
