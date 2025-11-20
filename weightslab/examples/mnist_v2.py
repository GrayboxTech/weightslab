import torch as th
import os
import tqdm
import time
import torch
import tempfile
import traceback
import torch.nn as nn
import torch.optim as optim

from collections import defaultdict

from torchvision import datasets, transforms

from torchmetrics.classification import Accuracy

from weightslab.components.global_monitoring import pause_controller
from weightslab.tests.torch_models import FashionCNN as MNISTCNN
from weightslab.src import WeightsLab
# from weightslab.trainer.trainer_services import serve
from weightslab.components.tracking import TrackingMode


# --- Configuration Constants ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# TMP_DIR = tempfile.mkdtemp()
stamp = "1763569624.0058134"  # str(time.time())
TMP_DIR = rf'C:\Users\GUILLA~1\Desktop\trash\{stamp}'


if __name__ == '__main__':
    print('Hello world')

    # Run experiment
    start_time = time.time()
    device = DEVICE

    # -1. Define global hyperparameters for the experiment
    parameters = {
        # trainer worker parameters
        "name": "MT_MNISTCNN_Test",
        "training_steps_to_do": 1024,
        "learning_rate": 0.01,
        "batch_size": 256,
        "eval_full_to_train_steps_ratio": 20,  # TO DELETE ? 
        "experiment_dump_to_train_steps_ratio": 50,
        "num_classes": 10,
        
        'data': {
            'train_dataset': {
                'train_shuffle': True,
                'batch_size': 256
            },
            'test_dataset': {
                'test_shuffle': False,
                'batch_size': 256
            }
        },
        'optimizer': {
            'Adam': {
                'lr': 0.01
            }
        },

        "root_log_dir": os.path.join(TMP_DIR, 'logs'),
        "tqdm_display": True,
        "skip_loading": False,
        "device": device
    }

    # =======================================
    # 0. Initialize weightsLab and the model, move to device
    wl_exp = WeightsLab(parameters)
    model = wl_exp.watch_or_edit(MNISTCNN().to(device), flag='model')

    # =========================
    # 1. Initialize the dataset
    print(f'Using tmp dir {TMP_DIR}')
    train_dataset = datasets.MNIST(
        root=os.path.join(TMP_DIR, 'data'),
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ]
        )
    )  # Download and load the training data
    test_dataset = datasets.MNIST(
        root=os.path.join(TMP_DIR, 'data'),
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ]
        )
    )  # Download and load the test data
    train_loader = wl_exp.watch_or_edit(
        train_dataset,
        obj_name='train_dataset',
        flag='data'
    )
    test_loader = wl_exp.watch_or_edit(
        test_dataset,
        obj_name='test_dataset',
        flag='data'
    )

    # =======================
    # 3. Define the optimizer
    optimizer = wl_exp.watch_or_edit(
        optim.Adam,
        flag='optimizer'
    )  # Watch or Edit optimizer
    
    # ====================
    # 4. Define criterions
    _criterion_bin = nn.BCELoss(reduction='none')
    criterion_bin = wl_exp.watch(
        _criterion_bin,
        flag='train_loss/bin_loss',
        name='criterion_bin',
        log=True
    )
    _criterion_mlt = nn.CrossEntropyLoss(reduction='none')
    criterion_mlt = wl_exp.watch(
        _criterion_mlt,
        name='criterion_mlt',
        flag='train_loss/mlt_loss',
        log=True
    )

    # =================
    # 5. Define metrics
    metric_bin = wl_exp.watch(
        Accuracy(task="binary", num_classes=1),
        name='metric_bin',
        flag='train_metric/bin_metric',
        log=True
    ).to(device)
    metric_mlt = wl_exp.watch(
        Accuracy(task="multiclass", num_classes=10),
        name='metric_mlt',
        flag='train_metric/mlt_metric',
        log=True
    ).to(device)

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
                pause_controller.resume()

    # start control thread
    from threading import Thread
    t = Thread(target=control_loop, daemon=True)
    t.start()
    print("\nStarting Training...")
    # It works
    # |-|-|-|-
    """Train the model for one step."""
    from weightslab.examples.training_tools import train    
    for train_step in tqdm.trange(150):
        # wl_exp.train_one_step()
        """Train the model for one step."""
        while pause_controller.is_paused():
            pass
        train(wl_exp, wl_exp.optimizer, wl_exp.train_dataset_iterator, wl_exp.model, wl_exp.criterion_mlt)
