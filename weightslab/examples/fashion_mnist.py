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
from weightslab.src import WeightsLab


# --- Configuration Constants ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TMP_DIR = tempfile.mkdtemp()


if __name__ == '__main__':
    print('Hello world')

    # Run experiment
    start_time = time.time()
    device = DEVICE

    # -1. Define global hyperparameters for the experiment
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

    # =======================================
    # 0. Initialize weightsLab and the model, move to device
    wl_exp = WeightsLab(parameters)
    model = wl_exp.watch_or_edit(FashionCNN().to(device), flag='model')

    # =========================
    # 1. Initialize the dataset
    TMP_DIR = r'C:\Users\GUILLA~1\Desktop\trash'
    print(f'Using tmp dir {TMP_DIR}')
    train_dataset = datasets.FashionMNIST(
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
    test_dataset = datasets.FashionMNIST(
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
    criterion_bin = wl_exp.watch(
        nn.BCELoss(reduction='none'),
        flag='train_loss/bin_loss',
        log=True
    )
    criterion_mlt = wl_exp.watch(
        nn.CrossEntropyLoss(reduction='none'),
        flag='train_loss/mlt_loss',
        log=True
    )

    # =================
    # 5. Define metrics
    metric_bin = wl_exp.watch(
        Accuracy(task="binary", num_classes=1),
        flag='train_metric/bin_metric',
        log=True
    ).to(device)
    metric_mlt = wl_exp.watch(
        Accuracy(task="multiclass", num_classes=10),
        flag='train_metric/mlt_metric',
        log=True
    ).to(device)

    # ================
    # 6. Training Loop
    print("\nStarting Training...")
    for train_step, (inputs, ids, labels) in enumerate(tqdm.tqdm(train_loader, desc='Training..')):
        if train_step == 0:
            model.operate(0, 1, 1)  # ADD 1 neurons
            model.pause_ctrl.resume()
        elif train_step == 1:
            model.pause_ctrl.pause()
            model.operate(0, 1, 1)  # ADD 1 neurons
            # model.operate(4, 1, 1)  # ADD 1 neurons
            model.pause_ctrl.resume()
        elif train_step == 2:
            model.pause_ctrl.pause()
            model.operate(0, 1, 2)  # ADD 1 neurons
            model.pause_ctrl.resume()

        # optimizer = wl_exp.optimizer  # This line should not exist with ledger approach!

        # Process data
        inputs = inputs.to(device)
        bin_labels = (labels == 0).float().to(device)
        mlt_labels = labels.to(device)

        # WL_EXP: Go for training
        with wl_exp.training_guard:
            # WL_EXP: Get model age
            model_age = wl_exp.model.get_age()

            # Train one step
            optimizer.zero_grad()  # Zero the parameter gradients
            preds = model(inputs)

            # Compute Losses
            # # Compute Binary Loss & Log
            losses_batch_bin = criterion_bin(preds[:, 0], bin_labels)
            # # Compute Mlt Loss & Log
            losses_batch_mlt = criterion_mlt(preds, mlt_labels)
            # # Compute Final Loss
            train_loss = torch.mean(losses_batch_bin) + torch.mean(losses_batch_mlt)

            # Update dataset statistics
            wl_exp.update_data_statistics(
                model_age,
                ids,
                {'bin_cls': losses_batch_bin, 'bin_mlt': losses_batch_mlt},
                preds
            )  # save as (input_id, losses_batch[i]: (lossA, lossB))

            # Update metrics
            metric_bin.update(preds[:, 0], bin_labels)
            metric_mlt.update(preds, mlt_labels)
            train_acc_mlt = metric_mlt.compute() * 100
            train_acc_bin = metric_bin.compute() * 100

            # Backward pass and optimization
            train_loss.backward()
            optimizer.step()

        print(
            f"Step {train_step}/{len(train_loader)}: " +
            f"| Train Loss: {train_loss:.4f} " +
            f"| Train Acc mlt: {train_acc_mlt:.2f}%"
            f"| Train Acc bin: {train_acc_bin:.2f}%"
        )
        if train_step == 0 or train_step % 5 != 0:
            continue

        # --------------------------------------------------------------------
        # --------------------------------------------------------------------
        # Test Loop
        with torch.no_grad():  # Disable gradient calculation for evaluation
            with wl_exp.testing_guard:
                losses = 0.0
                metric_totals = defaultdict(float)
                for test_step, (inputs, ids, labels) in enumerate(tqdm.tqdm(test_loader, desc='Testing..')):
                    # Process data
                    inputs = inputs.to(device)
                    bin_labels = (labels == 0).float().to(device)
                    mlt_labels = labels.to(device)

                    # Infer
                    preds = model(inputs)

                    # Compute Losses
                    # # Compute Binary Loss & Log - fwd
                    losses_batch_bin = criterion_bin(preds[:, 0], bin_labels, flag='test_loss/bin_loss')
                    # # Compute Mlt Loss & Log - fwd
                    losses_batch_mlt = criterion_mlt(preds, mlt_labels, flag='test_loss/mlt_loss')
                    # # Save Batch Statistics
                    losses_batch = torch.cat([losses_batch_bin[..., None], losses_batch_mlt[..., None]], axis=1)
                    # # Compute Final Loss
                    test_loss = torch.mean(losses_batch_bin) + torch.mean(losses_batch_mlt)

                    # Update dataset statistics
                    wl_exp.update_data_statistics(
                        model_age,
                        ids,
                        {'bin_cls': losses_batch_bin, 'bin_mlt': losses_batch_mlt},
                        preds,
                        is_training=False
                    )  # save as (input_id, losses_batch[i]: (lossA, lossB))

                    # Update metrics
                    metric_bin.update(preds[:, 0], bin_labels)
                    metric_mlt.update(preds, mlt_labels)
                    test_acc_bin = metric_bin.compute(flag='test_metric/bin_metric') * 100
                    test_acc_mlt = metric_mlt.compute(flag='test_metric/mlt_metric') * 100
            # step_loss, metric_results = self.eval_one_step()
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
