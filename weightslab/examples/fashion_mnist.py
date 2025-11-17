import os
import time
import torch
import tempfile
import torch.nn as nn
import torch.optim as optim

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
                'batch_size': 64
            },
            'test_dataset': {
                'test_shuffle': False,
                'batch_size': 1
            }
        },
        'optimizer': {
            'Adam': {
                'learning_rate': 0.01
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
    TMP_DIR = r'C:\Users\GUILLA~1\AppData\Local\Temp\tmp1eohr08t'
    train_dataset = datasets.FashionMNIST(
        root=os.path.join(TMP_DIR, 'data'),
        train=False,
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
    criterion_bin = wl_exp.watch(nn.CrossEntropyLoss(), flag='loss/bin_loss')
    criterion_mlt = wl_exp.watch(nn.CrossEntropyLoss(), flag='loss/mlt_loss')

    # ==========
    # 5. Metrics
    metric_mlt = Accuracy(task="multiclass", num_classes=10).to(device)
    metric_mlt = wl_exp.watch(metric_mlt, flag='metric/mlt_metric')
    metric_bin = Accuracy(task="binary", num_classes=1).to(device)
    metric_bin = wl_exp.watch(metric_bin, flag='metric/bin_metric')

    # ================
    # 6. Training Loop
    print("\nStarting Training...")
    nb_epochs = wl_exp.get_global_hyperparam('epochs')
    for epoch in range(
        1, nb_epochs + 1
    ):
        # Train for one ep
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            bin_labels = (labels == 0).float().to(device)
            mlt_labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss_bin = criterion_bin(outputs[:, 0], bin_labels)
            loss_mlt = criterion_mlt(outputs, mlt_labels)
            loss = loss_bin + loss_mlt

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # --------------------------------------------------------------------
        # --------------------------------------------------------------------
        # Evaluate on the test set
        model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():  # Disable gradient calculation for evaluation
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                bin_labels = (labels == 0).float().to(device)
                mlt_labels = labels.to(device)

                outputs = model(inputs)
                loss_bin = criterion_bin(outputs[:, 0], bin_labels)
                loss_mlt = criterion_mlt(outputs, mlt_labels)
                loss = loss_bin + loss_mlt
                total_loss += loss.item()

                # _, predicted = torch.max(outputs.data, 1)
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()
                metric_bin.update(outputs[:, 0], bin_labels)
                metric_mlt.update(outputs, mlt_labels)

        avg_test_loss = total_loss / len(test_loader)
        test_acc_mlt = metric_mlt.compute() * 100
        test_acc_bin = metric_bin.compute() * 100

        print(
            f"Epoch {epoch}/{GLOBAL_HYPER_PARAMETERS.get('epochs').default_value}: " +
            f"| Train Loss: {train_loss:.4f} " +
            f"| Test Loss: {avg_test_loss:.4f} " +
            f"| Test Acc mlt: {test_acc_mlt:.2f}%"
            f"| Test Acc bin: {test_acc_bin:.2f}%"
        )

    print("\n--- Training Finished ---")
