import os
import tqdm
import time
import torch
import tempfile
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from torchvision import datasets, transforms
from torchmetrics.classification import Accuracy
from weightslab.tests.torch_models import FashionCNN
from weightslab.ledgers import get_optimizer, get_model, get_dataloader, register_optimizer
import weightslab as wl


# --- Configuration Constants ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TMP_DIR = tempfile.mkdtemp()


if __name__ == '__main__':
    print('Hello world')
    start_time = time.time()
    device = DEVICE
    parameters = {
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
        "epochs": 10,
        "training_steps_to_do": 256,
        "name": "MT_FashionCNN_Test",
        "root_log_dir": os.path.join(TMP_DIR, 'logs'),
        "tqdm_display": True,
        "skip_loading": False,
        "device": device
    }

    # 1. Register model, dataloaders, optimizer in ledger
    _model = FashionCNN()
    model = wl.watch_or_edit(_model, flag='model', device=device)
    model.pause_ctrl.resume()

    _optimizer = optim.Adam(_model.parameters(), lr=0.01 )
    optimizer = wl.watch_or_edit(_optimizer, flag='optimizer')

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
    # register_optimizer('train_loader', train_dataset)
    # register_optimizer('test_loader', test_dataset)
    train_loader = wl.watch_or_edit(_train_dataset, flag='data', batch_size=256, shuffle=True)
    test_loader = wl.watch_or_edit(_test_dataset, flag='data', batch_size=256, shuffle=True)

    # optimizer = get_optimizer('_optimizer')
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    criterion_bin = nn.BCELoss(reduction='none')
    criterion_mlt = nn.CrossEntropyLoss(reduction='none')
    metric_bin = Accuracy(task="binary", num_classes=1).to(device)
    metric_mlt = Accuracy(task="multiclass", num_classes=10).to(device)


    def update_model(train_step, model):
        def new_opt():  #
            _optimizer = optim.Adam(_model.parameters(), lr=0.01 )
            return wl.watch_or_edit(_optimizer, flag='optimizer')
        optimizer = get_optimizer('Adam')
        if train_step == 0:
            pass
        elif train_step == 1:
            model.pause_ctrl.pause()
            model.operate(0, 1, 1)  # ADD 1 neurons
            # model.operate(4, 1, 1)  # ADD 1 neurons
            model.pause_ctrl.resume()
            optimizer = new_opt()
        elif train_step == 2:
            model.pause_ctrl.pause()
            model.operate(0, 1, 2)  # ADD 1 neurons
            model.pause_ctrl.resume()
            optimizer = new_opt()
        return model, optimizer

    print("\nStarting Training...")
    for train_step, (inputs, ids, labels) in enumerate(tqdm.tqdm(train_loader, desc='Training..')):
        model, optimizer = update_model(train_step, model)
        inputs = inputs.to(device)
        bin_labels = (labels == 0).float().to(device)
        mlt_labels = labels.to(device)
        optimizer.zero_grad()
        preds = model(inputs)
        losses_batch_bin = criterion_bin(preds[:, 0], bin_labels)
        losses_batch_mlt = criterion_mlt(preds, mlt_labels)
        train_loss = torch.mean(losses_batch_bin) + torch.mean(losses_batch_mlt)
        metric_bin.update(preds[:, 0], bin_labels)
        metric_mlt.update(preds, mlt_labels)
        train_acc_mlt = metric_mlt.compute() * 100
        train_acc_bin = metric_bin.compute() * 100
        train_loss.backward()
        """
            File "c:\Users\GuillaumePelluet\.vscode\extensions\ms-python.debugpy-2025.16.0-win32-x64\bundled\libs\debugpy\_vendored\pydevd\_pydevd_bundle\pydevd_runpy.py", line 310, in run_path
                return _run_module_code(code, init_globals, run_name, pkg_name=pkg_name, script_name=fname)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            File "c:\Users\GuillaumePelluet\.vscode\extensions\ms-python.debugpy-2025.16.0-win32-x64\bundled\libs\debugpy\_vendored\pydevd\_pydevd_bundle\pydevd_runpy.py", line 127, in _run_module_code
                _run_code(code, mod_globals, init_globals, mod_name, mod_spec, pkg_name, script_name)
            File "c:\Users\GuillaumePelluet\.vscode\extensions\ms-python.debugpy-2025.16.0-win32-x64\bundled\libs\debugpy\_vendored\pydevd\_pydevd_bundle\pydevd_runpy.py", line 118, in _run_code
                exec(code, run_globals)
            File "C:\Users\GuillaumePelluet\Documents\Codes\grayBox\weightslab\weightslab\examples\fashion_mnist_global.py", line 126, in <module>
                train_loss.backward()
            File "c:\Users\GuillaumePelluet\Documents\Codes\grayBox\python_env\weightslab\Lib\site-packages\torch\_tensor.py", line 492, in backward
                torch.autograd.backward(
            File "c:\Users\GuillaumePelluet\Documents\Codes\grayBox\python_env\weightslab\Lib\site-packages\torch\autograd\__init__.py", line 251, in backward
                Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
            RuntimeError: Function ConvolutionBackward0 returned an invalid gradient at index 1 - got [4, 5, 3, 3] but expected shape compatible with [4, 4, 3, 3]
        """
        optimizer.step()
        print(
            f"Step {train_step}/{len(train_loader)}: " +
            f"| Train Loss: {train_loss:.4f} " +
            f"| Train Acc mlt: {train_acc_mlt:.2f}%"
            f"| Train Acc bin: {train_acc_bin:.2f}%"
        )
        if train_step == 0 or train_step % 5 != 0:
            continue
        with torch.no_grad():
            losses = 0.0
            metric_totals = defaultdict(float)
            for test_step, (inputs, ids, labels) in enumerate(tqdm.tqdm(test_loader, desc='Testing..')):
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
            losses += test_loss
            metric_totals['bin'] += test_acc_bin
            metric_totals['mlt'] += test_acc_mlt
            print(
                f"Step {test_step}/{len(test_loader)}: " +
                f"| Train Loss: {train_loss:.4f} " +
                f"| Test Loss: {test_loss:.4f} " +
                f"| Test Acc mlt: {metric_totals['mlt']:.2f}%"
                f"| Test Acc bin: {metric_totals['bin']:.2f}%"
            )
    print("\n--- Training Finished ---")

    # Test: update optimizer globally and verify main loop uses new optimizer
    print('Updating global optimizer...')
    register_optimizer('_optimizer', get_optimizer('new_optimizer'))
    print('Global optimizer updated! Main loop will use new optimizer automatically.')
