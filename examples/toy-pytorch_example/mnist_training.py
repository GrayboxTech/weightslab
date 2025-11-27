import os
import tqdm
import time
import warnings
import torch
import tempfile
import torch.nn as nn
import weightslab as wl
import torch.optim as optim
import yaml

from torchvision import datasets, transforms
from torchmetrics.classification import Accuracy

from weightslab.utils.board import Dash as Logger
from weightslab.components.global_monitoring import \
    guard_training_context, \
    guard_testing_context


warnings.filterwarnings("ignore")


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Set input shape
        self.input_shape = (1, 1, 28, 28)

        # Feature Blocks (Same as before)
        # Block 1
        self.c1 = nn.Conv2d(1, 4, 3, padding=1)
        self.b1 = nn.BatchNorm2d(4)
        self.r1 = nn.ReLU()
        self.m1 = nn.MaxPool2d(2)

        # Block 2
        self.c2 = nn.Conv2d(4, 4, 3)  # Default stride=1, no padding
        self.b2 = nn.BatchNorm2d(4)
        self.r2 = nn.ReLU()
        self.m2 = nn.MaxPool2d(2)

        # Classifier Block (Includes Flatten)
        # Automatically flattens the BxCxHxW tensor to Bx(C*H*W)
        self.f3 = nn.Flatten()
        self.fc3 = nn.Linear(in_features=4 * 6 * 6, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=10)
        self.s4 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.m1(self.r1(self.b1(self.c1(x))))
        x = self.m2(self.r2(self.b2(self.c2(x))))
        x = self.s4(self.fc4(self.fc3(self.f3(x))))
        return x


    
# --- Define functions ---
def train(loader, model, optimizer, criterion_mlt):
    print(f'\n{time.time()}: Training step...')
    with guard_training_context:
        print(f'\n{time.time()}: In training context')
        # Get next batch
        (input, ids, label) = next(loader)
        input = input.to('cuda')
        label = label.to('cuda')
        print(f'Input shape: {input.shape}, Label shape: {label.shape}')

        # Inference
        optimizer.zero_grad(set_to_none=True)
        output = model(input)

        # Compute loss
        loss_batch = criterion_mlt(
            output.float(),
            label.long(),
            model_age=model.get_age(),
            batch_ids=ids,
            preds=output.argmax(dim=1, keepdim=True)
        )
        loss = loss_batch.mean()

        # Propagate
        loss.backward()
        optimizer.step()
    print(f'{time.time()}: Out training context.\n')

    # Returned signals detach from the computational graph
    return loss.detach().cpu().item()


def test(loader, model, criterion_mlt, metric_mlt, device):
    losses = 0.0
    metric_total = 0
    for (inputs, ids, labels) in loader:
        with guard_testing_context, torch.no_grad():
            # Process data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Inference
            output = model(inputs)
            losses_batch_mlt = criterion_mlt(
                output,
                labels,
                model_age=model.get_age(),
                batch_ids=ids,
                preds=output.argmax(dim=1, keepdim=True)
            )
            metric_mlt.update(output, labels)

            # Compute signals
            losses = losses + torch.mean(losses_batch_mlt)

    loss = losses / len(loader)    
    metric_total = metric_mlt.compute() * 100

    # Returned signals detach from the computational graph
    return loss.detach().cpu().item(), metric_total.detach().cpu().item()


if __name__ == '__main__':
    print('Hello world')
    start_time = time.time()

    # Initialize WeightsLab CLI & gRPC
    wl.serve(
        serving_grpc=True,
        n_workers_grpc=6,
        port_grpc=50051,

        serving_cli=True,
        port_cli=0,
        host_cli='127.0.0.1',
        launch_cli=True
    )

    # ====================
    # Hyperparameters & Setup
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

    # Wire more optional parameters
    log_dir = parameters.get('root_log_dir')
    tqdm_display = parameters.get('tqdm_display', True)

    # ========================
    # Watch or Edit Components
    # Logger
    logger = Logger()
    wl.watch_or_edit(logger, flag='logger', name='log', log_dir=log_dir)

    # Hyper Parameters
    wl.watch_or_edit(parameters, flag='hyperparameters', name='hp', defaults=parameters, poll_interval=1.0)

    # Model
    _model = CNN()
    model = wl.watch_or_edit(_model, flag='model', name='model', device=parameters.get('device', device))

    # Optimizer
    lr = parameters.get('optimizer', {}).get('lr', 0.01)
    _optimizer = optim.Adam(model.parameters(), lr=lr )
    optimizer = wl.watch_or_edit(_optimizer, flag='optimizer', name='opt')

    # Data
    _train_dataset = datasets.MNIST(
        root=os.path.join(parameters.get('root_log_dir'), 'data'),
        train=False,
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

    # ====================
    # 4. Define criterions
    train_criterion_mlt = wl.watch_or_edit(
        nn.CrossEntropyLoss(reduction='none'),
        flag='loss',
        name='train_loss/mlt_loss',
        log=True
    )
    test_criterion_mlt = wl.watch_or_edit(
        nn.CrossEntropyLoss(reduction='none'),
        flag='loss',
        name='test_loss/mlt_loss',
        log=True
    )
    test_metric_mlt = wl.watch_or_edit(
        Accuracy(task='multiclass', num_classes=10).to(device),
        flag='metric',
        name='test_metric/mlt_metric',
        log=True
    )

    # ================
    # 6. Training Loop
    print("\nStarting Training...")
    max_steps = parameters.get('training_steps_to_do', 6666)
    train_range = range(max_steps)
    if tqdm_display:
        train_range = tqdm.trange(max_steps, dynamic_ncols=True)
    for train_step in train_range:
        # Train
        train_loss = train(train_loader, model, optimizer, train_criterion_mlt)

        # Test
        test_loss, test_metric = None, None
        if train_step > 0 and train_step % parameters.get('eval_full_to_train_steps_ratio', 50) == 0:
            test_loss, test_metric = test(test_loader, model, test_criterion_mlt, test_metric_mlt, device)

        # Verbose
        print(
            f"Step {train_step}/{max_steps}: " +
            f"| Train Loss: {train_loss:.4f} " +
            (f"| Test Loss: {test_loss:.4f} " if test_loss is not None else '') +
            (f"| Test Acc mlt: {test_metric:.2f}% " if test_metric is not None else '')
        )
    print(f"--- Training completed in {time.time() - start_time:.2f} seconds ---")
    print(f"Log directory: {log_dir}")
