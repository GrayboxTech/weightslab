import os
import tqdm
import time
import torch
import tempfile
import torch.nn as nn
import weightslab as wl
import torch.optim as optim

from collections import defaultdict
from weightslab.examples.training_tools import train, test    

from torchvision import datasets, transforms

from torchmetrics.classification import Accuracy

from weightslab.tests.torch_models import FashionCNN
from weightslab.ledgers import get_optimizer, get_model, get_dataloader, register_optimizer
from weightslab.components.global_monitoring import pause_controller, guard_training_context


# --- Configuration Constants ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TMP_DIR = tempfile.mkdtemp()
TMP_DIR = rf'C:\Users\GUILLA~1\Desktop\trash\1763569624.0058134'  # str(time.time())

if __name__ == '__main__':
    print('Hello world')
    start_time = time.time()
    device = DEVICE
    parameters = {
        'data': {
            'train_dataset': {
                'train_shuffle': True,
                'batch_size': 16
            },
            'test_dataset': {
                'test_shuffle': False,
                'batch_size': 16
            }
        },
        'optimizer': {
            'Adam': {
                'lr': 0.01
            }
        },
        "epochs": 10,
        "training_steps_to_do": 16,
        "name": "MT_FashionCNN_Test",
        "root_log_dir": os.path.join(TMP_DIR, 'logs'),
        "tqdm_display": True,
        "skip_loading": False,
        "device": device
    }

    # Model
    _model = FashionCNN()
    model = wl.watch_or_edit(_model, flag='model', device=device)

    # Optimizer
    _optimizer = optim.Adam(_model.parameters(), lr=0.01 )
    optimizer = wl.watch_or_edit(_optimizer, flag='optimizer')
    optimizer_updated = hasattr(optimizer, '_updated')
    registered_optimizer = hasattr(get_optimizer('Adam'), '_updated')

    # Data
    _train_dataset = datasets.FashionMNIST(
        root=os.path.join(TMP_DIR, 'data'),
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    )
    _test_dataset = datasets.FashionMNIST(
        root=os.path.join(TMP_DIR, 'data'),
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    )
    train_loader = wl.watch_or_edit(_train_dataset, flag='data', batch_size=16, shuffle=True)
    test_loader = wl.watch_or_edit(_test_dataset, flag='data', batch_size=16, shuffle=True)

    # Criterion
    criterion_bin = nn.BCELoss(reduction='none')
    criterion_mlt = nn.CrossEntropyLoss(reduction='none')

    # Metrics
    metric_bin = Accuracy(task='binary').to(device)
    metric_mlt = Accuracy(task='multiclass', num_classes=10).to(device)

    # ================
    # 6. Training Loop
    def control_loop():
        """
        Example: simple stdin loop:
        - type 'p' to pause
        - type 'r' to resume
        - type 'operate <op_type> <layer_id> <nb_neurons>' to perform an operation, e.g.:
            'operate 1 2 1' to add 1 neurons to layer with ID 2
            'operate 2 2 5' to prune indexed neuron 5 to layer with ID 2
        TODO (GP): implement a better CLI or GUI for this; not working wi. list of indexs now.
        """
        def extract_op_info(s: str):
            parts = s.strip().split()
            if len(parts) < 3:
                return None
            try:
                op_type = int(parts[1])
                layer_id = int(parts[2])
                nb_neurons = int(parts[3]) if len(parts) > 3 else None
                return (op_type, layer_id, nb_neurons)
            except Exception:
                return None
        while True:
            cmd = input("[control] enter p=pause, r=resume: ").strip().lower()
            if cmd.startswith("p"):
                print("[control] pausing…")
                pause_controller.pause()
            elif cmd.startswith("r"):
                print("[control] resuming…")
                pause_controller.resume()
            elif cmd.startswith("operate"):
                op_type, layer_id, nb_neurons = extract_op_info(cmd)
                print(f'[control] performing operation {op_type} on layer {layer_id} with {nb_neurons} neurons…')
                with model as m:
                    m.operate(layer_id, nb_neurons, op_type)
                    print(f'New model architecture {layer_id} info: {m}')
                print("[control] quitting control loop…")
                # Option 1: Manually clearing the gradient for each parameter
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad = None  # Reset gradient to None
            else:
                print("[control] unknown command.")

    # start control thread
    from threading import Thread
    t = Thread(target=control_loop, daemon=True)
    t.start()

    print("\nStarting Training...")
    for train_step in tqdm.trange(150):
        # Train
        train(train_loader, model, optimizer, criterion_mlt)

        # Test
        test(test_loader, model, criterion_bin, criterion_mlt, metric_bin, metric_mlt, device) if train_step > 0 and train_step % 125 == 0 else None