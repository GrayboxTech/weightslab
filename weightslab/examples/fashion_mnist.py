import os
import time
import torch
import tempfile
import collections
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from typing import Dict

from torchvision import datasets, transforms

from torch.utils.data import DataLoader

from torchmetrics.classification import Accuracy

from weightslab.experiment.next_gen_experiment import Experiment as wl_exp


# --- Configuration Constants ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TMP_DIR = tempfile.mkdtemp()

HyperParam = collections.namedtuple(
    "HyperParam",
    ["data_type", "default_value"]
)

# Using a standard Python Set for the global list of configuration schemas
GLOBAL_HYPER_PARAMETERS: Dict[str, HyperParam] = {
    "batch_size": HyperParam(data_type="int", default_value=64),
    "learning_rate": HyperParam(data_type="float", default_value=0.001),
    "optimizer": HyperParam(data_type="str", default_value="Adam"),
    "epochs": HyperParam(data_type="int", default_value=10),
}


# --- 2. Model Definition (FashionCNN) ---
class FashionCNN(nn.Module):
    """
    A simple Convolutional Neural Network (CNN) architecture for Fashion-MNIST.
    Uses two convolutional layers followed by fully connected layers.
    """
    def __init__(self):
        super(FashionCNN, self).__init__()

        # First Convolutional Block: 1x28x28 -> 16x28x28 (Conv) ->
        # 16x14x14 (MaxPool)
        self.conv1 = nn.Conv2d(
            in_channels=1,  # 1 for grayscale
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=2  # Padding keeps the height/width the same (28x28)
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second Convolutional Block: 16x14x14 -> 32x14x14 (Conv) ->
        # 32x7x7 (MaxPool)
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flattened feature size: 32 channels * 7 * 7 spatial dimensions
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Pass through Conv Block 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)

        # Pass through Conv Block 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2(x)

        # Flatten the output for the fully connected layers
        # x.size(0) is the batch dimension
        x = x.view(x.size(0), -1)

        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        # Output logits (no Softmax here, as CrossEntropyLoss will apply it)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    # --- 1. Data Loading and Preprocessing ---
    # The images are 28x28 grayscale. We normalize them to the range [-1, 1].
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Mean 0.5, Std Dev 0.5
    ])

    # Download and load the training data
    train_dataset = datasets.FashionMNIST(
        root=os.path.join(TMP_DIR, 'data'),
        train=True,
        download=True,
        transform=transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=GLOBAL_HYPER_PARAMETERS.get('batch_size').default_value,
        shuffle=True,
        num_workers=8
    )

    # Download and load the test data
    test_dataset = datasets.FashionMNIST(
        root=os.path.join(TMP_DIR, 'data'),
        train=False,
        download=True,
        transform=transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=GLOBAL_HYPER_PARAMETERS.get('batch_size').default_value,
        shuffle=False,
        num_workers=8
    )

    # Run experiment
    start_time = time.time()

    # 0. Initialize the model, move to device
    model = FashionCNN().to(device)
    wl_exp.watch_or_edit(model, flag='model')  # Watch or Edit model

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=GLOBAL_HYPER_PARAMETERS.get('learning_rate').default_value)
    # wl_exp.watch_or_edit(optimizer, flag='optimizer')  # Watch or Edit optimizer

    # Losses
    criterion_mlt = nn.CrossEntropyLoss()
    criterion_bin = nn.CrossEntropyLoss()  # Use binary cross-entropy loss for binary classification
    # wl_exp.watch(criterion_mlt, flag='loss/mlt_loss')  # Watch loss
    # wl_exp.watch(criterion_bin, flag='loss/bin_loss')  # Watch loss

    # Metrics
    metric_mlt = Accuracy(task="multiclass", num_classes=10).to(device)
    metric_bin = Accuracy(task="binary", num_classes=1).to(device)
    # wl_exp.watch(metric_mlt, flag='metric/mlt_metric')  # Watch metric
    # wl_exp.watch(metric_bin, flag='metric/bin_metric')  # Watch metric

    # Training Loop
    print("\nStarting Training...")
    for epoch in range(1, GLOBAL_HYPER_PARAMETERS.get('epochs').default_value + 1):
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
