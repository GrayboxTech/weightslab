import os
import tqdm
import time
import torch
import tempfile
import torch.nn as nn
import weightslab as wl
import torch.optim as optim

from collections import defaultdict
from weightslab.examples.training_tools import train    

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
    criterion = nn.CrossEntropyLoss(reduction='none')

    # ================
    # 6. Training Loop
    def control_loop():
        """
        Example: simple stdin loop:
        - type 'p' to pause
        - type 'r' to resume
        """
        while True:
            cmd = input("[control] enter p=pause, r=resume: ").strip().lower()
            if cmd == "p":
                print("[control] pausing…")
                pause_controller.pause()
            elif cmd == "r":
                print("[control] resuming…")
                pause_controller.resume()
            elif cmd == "a":
                print("[control] Add Neurons…")
                pause_controller.pause()
                with model as m:
                    m.operate(1, 1, 1)
                print(f'New model architecture:\n{model}')
            elif cmd == "b":
                print("[control] Prune Neurons…")
                pause_controller.pause()
                with model as m:
                    m.operate(2, 1, 2)
                print(f'New model architecture:\n{model}')

    # start control thread
    from threading import Thread
    t = Thread(target=control_loop, daemon=True)
    t.start()

    print("\nStarting Training...")
    for train_step in tqdm.trange(150):
        time.sleep(0.2)
        # Pause outside
        while pause_controller.is_paused():
            pass
        train(train_loader, model, optimizer, criterion)

    # Old Not working approach
    # print("\nStarting Training...")
    # for train_step, (inputs, ids, labels) in enumerate(tqdm.tqdm(train_loader, desc='Training..')):
    #     # TO DEVICE
    #     inputs = inputs.to(device)
    #     labels = labels.to(device)

    #     # INFERENCE
    #     optimizer.zero_grad()
    #     preds = model(inputs)
    #     losses_batch = criterion(preds, labels)
    #     train_loss = torch.mean(losses_batch)
    #     train_loss.backward()
    #     optimizer.step()

    #     print(
    #         f"Step {train_step}/{len(train_loader)}: " +
    #         f"| Train Loss: {train_loss:.4f} "
    #     )

    #     # if train_step == 0 or train_step % 5 != 0:
    #     #     continue
    #     # with torch.no_grad():
    #     #     losses = 0.0
    #     #     metric_totals = defaultdict(float)
    #     #     for test_step, (inputs, ids, labels) in enumerate(tqdm.tqdm(test_loader, desc='Testing..')):
    #     #         inputs = inputs.to(device)
    #     #         bin_labels = (labels == 0).float().to(device)
    #     #         mlt_labels = labels.to(device)
    #     #         preds = model(inputs)
    #     #         losses_batch_bin = criterion_bin(preds[:, 0], bin_labels)
    #     #         losses_batch_mlt = criterion_mlt(preds, mlt_labels)
    #     #         losses_batch = torch.cat([losses_batch_bin[..., None], losses_batch_mlt[..., None]], axis=1)
    #     #         test_loss = torch.mean(losses_batch_bin) + torch.mean(losses_batch_mlt)
    #     #         metric_bin.update(preds[:, 0], bin_labels)
    #     #         metric_mlt.update(preds, mlt_labels)
    #     #         test_acc_bin = metric_bin.compute() * 100
    #     #         test_acc_mlt = metric_mlt.compute() * 100
    #     #     losses += test_loss
    #     #     metric_totals['bin'] += test_acc_bin
    #     #     metric_totals['mlt'] += test_acc_mlt
    #     #     print(
    #     #         f"Step {test_step}/{len(test_loader)}: " +
    #     #         f"| Train Loss: {train_loss:.4f} " +
    #     #         f"| Test Loss: {test_loss:.4f} " +
    #     #         f"| Test Acc mlt: {metric_totals['mlt']:.2f}%"
    #     #         f"| Test Acc bin: {metric_totals['bin']:.2f}%"
    #     #     )
    # print("\n--- Training Finished ---")

    # # Test: update optimizer globally and verify main loop uses new optimizer
    # print('Updating global optimizer...')
    # register_optimizer('_optimizer', get_optimizer('new_optimizer'))
    # print('Global optimizer updated! Main loop will use new optimizer automatically.')
