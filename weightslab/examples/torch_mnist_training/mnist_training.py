import os
import tqdm
import time
import torch
import tempfile
import torch.nn as nn
from weightslab.ledgers import list_dataloaders
import weightslab as wl
import weightslab.cli as cli
import torch.optim as optim
import yaml

from torchvision import datasets, transforms

from torchmetrics.classification import Accuracy

from weightslab.tests.torch_models import FashionCNN as CNN
from weightslab.components.global_monitoring import \
    guard_training_context, \
    guard_testing_context
import logging
logging.basicConfig(level=logging.INFO)
from weightslab.trainer.trainer_services import serve


# Initialize WeightsLab CLI
cli.initialize(launch_client=False)

# Initialize WeightsLab Serving
# serve(threading=True)  <-- Moved down to ensure dataloaders are ready

# --- Define functions ---
def train(loader, model, optimizer, criterion_mlt, device='cpu'):
    with guard_training_context:
        # Get next batch
        (input, ids, label) = next(loader)
        input = input.to(device)
        label = label.to(device)

        # Inference
        optimizer.zero_grad()
        output = model(input)
        loss_batch = criterion_mlt(output.float(), label.long())
        loss = loss_batch.mean()
        if output.ndim == 1:
            preds = (output > 0.0).long()
        else:
            preds = output.argmax(dim=1, keepdim=True)

        # Propagate
        loss.backward()
        optimizer.step()

        # Data update
        wl.update_train_test_data_statistics(
            model_age=model.get_age(),
            batch_ids=ids,
            losses_batch=loss_batch,
            preds=preds
        )
    
    # Returned signals detach from the computational graph
    return loss.detach().cpu().item()

def test(loader, model, criterion_mlt, metric_mlt, device):
    losses = 0.0
    metric_total = 0
    with guard_testing_context, torch.no_grad():
        for (inputs, ids, labels) in loader:
            # Process data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Inference
            output = model(inputs)
            losses_batch_mlt = criterion_mlt(output, labels)
            test_loss = torch.mean(losses_batch_mlt)
            metric_mlt.update(output, labels)
            if output.ndim == 1:
                preds = (output > 0.0).long()
            else:
                preds = output.argmax(dim=1, keepdim=True)

            # Compute signals
            losses = losses + test_loss

            # Data update
            wl.update_train_test_data_statistics(
                model_age=model.get_age(),
                batch_ids=ids,
                losses_batch=losses_batch_mlt,
                preds=preds
            )

    loss = losses / len(loader)    
    metric_total = metric_mlt.compute() * 100

    # Returned signals detach from the computational graph
    return loss.detach().cpu().item(), metric_total.detach().cpu().item()


if __name__ == '__main__':
    start_time = time.time()

    # Load YAML hyperparameters (fallback to defaults if missing)
    parameters = {}
    config_path = os.path.join(os.path.dirname(__file__), 'mnist_training_config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as fh:
            parameters = yaml.safe_load(fh) or {}
    exp_name = parameters.get('experiment_name')

    # Normalize device entry
    if parameters.get('device', 'auto') == 'auto':
        parameters['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = parameters['device']

    # Ensure root_log_dir default
    if not parameters.get('root_log_dir'):
        TMP_DIR = tempfile.mkdtemp()
        parameters['root_log_dir'] = os.path.join(TMP_DIR, 'logs')
    os.makedirs(parameters['root_log_dir'], exist_ok=True)

    # Wire more parameters
    epochs = parameters.get('epochs', 10)
    log_dir = parameters.get('root_log_dir')
    tqdm_display = parameters.get('tqdm_display', True)

    # Logger
    from weightslab.utils.board import Dash as Logger
    logger = Logger()
    wl.watch_or_edit(logger, flag='logger', name=exp_name, log_dir=log_dir)

    # Hyper Parameters
    wl.watch_or_edit(parameters, flag='parameters', name=exp_name, defaults=parameters, poll_interval=1.0)

    # Model
    _model = CNN()
    model = wl.watch_or_edit(_model, flag='model', name=exp_name, device=parameters.get('device', device))

    # Optimizer
    lr = parameters.get('optimizer', {}).get('lr', 0.01)
    _optimizer = optim.Adam(_model.parameters(), lr=lr )
    optimizer = wl.watch_or_edit(_optimizer, flag='optimizer', name=exp_name)

    # Data
    _train_dataset = datasets.MNIST(
        root=os.path.join(parameters.get('root_log_dir'), 'data'),
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    )
    _test_dataset = datasets.MNIST(
        root=os.path.join(parameters.get('root_log_dir'), 'data'),
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
    train_loader = wl.watch_or_edit(_train_dataset, flag='data', name='train_loader', batch_size=train_bs, shuffle=train_shuffle, is_training=True)
    test_loader = wl.watch_or_edit(_test_dataset, flag='data', name='test_loader', batch_size=test_bs, shuffle=test_shuffle)
    
    # Start serving NOW that dataloaders are registered
    serve(threading=True)
    print("=" * 60)
    print("✅ Server started successfully!")
    print("📊 Data is available at: http://localhost:3001")
    print("=" * 60)
    
    # Optional: Uncomment the next 2 lines to pause for data inspection
    # print("Pausing for data inspection... (Press Ctrl+C to stop)")
    # time.sleep(999999)

    # ====================
    # 4. Define criterions
    train_criterion_mlt = wl.watch_or_edit(
        nn.CrossEntropyLoss(reduction='none'),
        flag='train_loss/mlt_loss',
        log=True
    )
    test_criterion_mlt = wl.watch_or_edit(
        nn.CrossEntropyLoss(reduction='none'),
        flag='test_loss/mlt_loss',
        log=True
    )
    test_metric_mlt = wl.watch_or_edit(
        Accuracy(task='multiclass', num_classes=10).to(device),
        flag='test_metric/mlt_metric',
        log=True
    )

    # ================
    # 6. Training Loop
    max_steps = parameters.get('training_steps_to_do', 6666)
    
    print("\n" + "=" * 60)
    print("🚀 STARTING TRAINING")
    print(f"📈 Total steps: {max_steps}")
    print(f"🔄 Evaluation every {parameters.get('eval_full_to_train_steps_ratio', 50)} steps")
    print(f"💾 Logs will be saved to: {log_dir}")
    print("=" * 60 + "\n")
    
    # Resume training automatically (bypasses default paused state)
    try:
        model.pause_ctrl.resume()
        print("✅ Training resumed automatically")
    except Exception as e:
        print(f"⚠️  Could not auto-resume: {e}")
    
    train_range = range(max_steps)
    if tqdm_display:
        train_range = tqdm.trange(max_steps, dynamic_ncols=True)
    
    for train_step in train_range:
        # Train
        train_loss = train(train_loader, model, optimizer, train_criterion_mlt, device)

        # Test
        test_loss, test_metric = None, None
        if train_step % parameters.get('eval_full_to_train_steps_ratio', 50) == 0:
            test_loss, test_metric = test(test_loader, model, test_criterion_mlt, test_metric_mlt, device)

        # Verbose - print every 10 steps or when evaluating
        if train_step % 10 == 0 or test_loss is not None:
            status = f"[Step {train_step:5d}/{max_steps}] Train Loss: {train_loss:.4f}"
            if test_loss is not None:
                status += f" | Test Loss: {test_loss:.4f} | Test Acc: {test_metric:.2f}%"
            print(status)
    
    print("\n" + "=" * 60)
    print(f"✅ Training completed in {time.time() - start_time:.2f} seconds")
    print(f"💾 Logs saved to: {log_dir}")
    print("=" * 60)
