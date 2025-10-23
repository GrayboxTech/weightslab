import torch as th
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

from weightslab.utils.logs import print, setup_logging
from weightslab.components.tracking import add_tracked_attrs_to_input_tensor


class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        print(f'Initializing model named: ', self._get_name())

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
        self.fc3 = nn.Linear(in_features=4 * 6 * 6, out_features=10)
        self.s3 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.m1(self.r1(self.b1(self.c1(x))))
        x = self.m2(self.r2(self.b2(self.c2(x))))
        x = self.s3(self.fc3(self.f3(x)))
        return x


class FashionCNNSequential(nn.Module):
    def __init__(self):
        super().__init__()

        # Set input shape
        self.input_shape = (2, 1, 28, 28)

        # Feature Blocks (Same as before)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 4, 3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(4, 4, 3),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Classifier Block (Includes Flatten)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=4 * 6 * 6, out_features=128),
            nn.Linear(in_features=128, out_features=10),
            nn.Linear(in_features=10, out_features=1)
        )
        self.out_softmax = nn.Softmax(dim=1)

    def forward(self, y):
        x = self.features(y)
        x = self.classifier(x)

        if not isinstance(x, th.fx._symbolic_trace.Proxy):
            one_hot = F.one_hot(
                x.argmax(dim=1), num_classes=self.classifier[-1].out_features
            )

            if hasattr(x, 'in_id_batch') and \
                    hasattr(x, 'label_batch'):
                add_tracked_attrs_to_input_tensor(
                    one_hot, in_id_batch=input.in_id_batch,
                    label_batch=input.label_batch)
            self.classifier[-1].register(one_hot) if hasattr(self.classifier[-1], 'register') else None

        out = self.out_softmax(x)

        return out


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "TestArchitecture"

        # Set input shape
        self.input_shape = (1, 1, 28, 28)

        # L1
        self.c1 = nn.Conv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=3,
            padding=1
        )
        self.b1 = nn.BatchNorm2d(4)
        self.m1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # L2
        self.c2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3)
        self.b2 = nn.BatchNorm2d(4)
        self.m2 = nn.MaxPool2d(2)

        # L3
        self.l3 = nn.Linear(in_features=4*6*6, out_features=10)
        self.s = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.m1(self.b1(self.c1(x)))
        x = self.m2(self.b2(self.c2(x)))
        x = x.view(x.size(0), -1)
        x = self.s(self.l3(x))
        return x


class GraphMLP_res_test_A(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "Test Architecture Model Res. Co."

        # Set input shape
        self.input_shape = (1, 1, 28, 28)

        # Block 1 (Path A)
        self.c1 = nn.Conv2d(1, 4, 3, padding=1)  # Id 0

        # Block 2 (Residual/Skip Path)
        # Note: c2 takes b1's output. c3 takes c2's output.
        self.c2 = nn.Conv2d(4, 8, 3, padding=1)  # Id 2
        self.c3 = nn.Conv2d(8, 4, 3, padding=1)  # Id 3

    def forward(self, x):
        # Path A
        x1 = self.c1(x)  # [4, 28, 28]
        x2 = self.c2(x1)  # [8, 28, 28]
        x3 = self.c3(x2)  # [4, 28, 28]

        # Residual Connection (Add operation)
        x_out = x1 + x3  # The output of b1 and c3 both flow into the add op

        return x_out


class GraphMLP_res_test_B(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "Test Architecture Model Res. Co."

        # Set input shape
        self.input_shape = (1, 1, 28, 28)

        # Block 1 (Path A) - Stays the same
        self.c1 = nn.Conv2d(1, 4, 3, padding=1)

        # Block 2 (Main Path) - Stays the same
        self.c2 = nn.Conv2d(4, 8, 3, padding=1)
        self.c3 = nn.Conv2d(8, 4, 3, padding=1)

        # Block 3 (Residual/Skip Path)
        self.c4 = nn.Conv2d(4, 12, 3, padding=1)
        self.b1 = nn.BatchNorm2d(12)
        self.c5 = nn.Conv2d(12, 4, 3, padding=1)

    def forward(self, x):
        # Path A (Skip connection input)
        x1 = self.c1(x)

        # Main Path (where the skip connection comes from)
        x2 = self.c2(x1)
        x3 = self.c3(x2)  # [4, 28, 28]

        # Residual connection path (Transform x1 to match x3)
        x4 = self.c4(x1)
        x5 = self.c5(self.b1(x4))  # [4, 28, 28]

        # Residual Connection (Add operation)
        # Now x3 and x5 have the same shape: B x 4 x 28 x 28
        x_out = x3 + x5  # Assuming you intended to add x3 and x5/x4

        return x_out


class GraphMLP_res_test_C(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "Test Architecture Model Res. Co."

        # Set input shape
        self.input_shape = (1, 1, 28, 28)

        # Block 1 (Path A) - Stays the same
        self.c1 = nn.Conv2d(1, 4, 3, padding=1)

        # Block 2 (Main Path) - Stays the same
        self.c2 = nn.Conv2d(4, 8, 3, padding=1)
        self.c3 = nn.Conv2d(8, 12, 3, padding=1)

        # Block 3 (Residual/Skip Path)
        self.c4 = nn.Conv2d(4, 12, 3, padding=1)
        self.b1 = nn.BatchNorm2d(12)

    def forward(self, x):
        # Path A (Skip connection input)
        x1 = self.c1(x)

        # Main Path (where the skip connection comes from)
        x2 = self.c2(x1)
        x3 = self.c3(x2)  # [4, 28, 28]

        # Residual connection path (Transform x1 to match x3)
        x4 = self.c4(x1)
        x5 = self.b1(x4)  # [4, 28, 28]

        # Residual Connection (Add operation)
        # Now x3 and x5 have the same shape: B x 4 x 28 x 28
        x_out = x3 + x5  # Assuming you intended to add x3 and x5/x4

        return x_out


class GraphMLP_res_test_D(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "Test Architecture Model Res. Co."

        # Set input shape
        self.input_shape = (1, 1, 28, 28)

        # Block 1 (Path A) - Stays the same
        self.c1 = nn.Conv2d(1, 4, 3, padding=1)  # Input (1), Output (4)

        # Block 2 (Main Path) - Stays the same
        self.c2 = nn.Conv2d(4, 8, 3, padding=1)
        self.c3 = nn.Conv2d(8, 12, 3, padding=1)

        # Block 3 (Residual/Skip Path)
        self.c4 = nn.Conv2d(4, 12, 3, padding=1)
        self.b1 = nn.BatchNorm2d(12)

        # Block 4 (Residual/Skip Path)
        self.c5 = nn.Conv2d(4, 12, 3, padding=1)
        self.b2 = nn.BatchNorm2d(12)

    def forward(self, x):
        # Path A (Skip connection input)
        x1 = self.c1(x)

        # Main Path (where the skip connection comes from)
        x2 = self.c2(x1)
        x3 = self.c3(x2)  # [4, 28, 28]

        # Residual connection path (Transform x1 to match x3)
        x4 = self.c4(x1)
        x5 = self.b1(x4)  # [4, 28, 28]

        # Residual connection path (Transform x1 to match x3)
        x6 = self.c5(x1)
        x7 = self.b2(x6)  # [4, 28, 28]

        # Residual Connection (Add operation)
        # Now x3 and x5 have the same shape: B x 4 x 28 x 28
        x_out = x3 + x5 - x7  # Assuming you intended to add x3 and x5/x4

        return x_out


# --- Residual model - subpart ---
# --- The Core Residual Block for ResNet-18 and ResNet-34 ---
class SingleBlockResNetTruncated(nn.Module):
    """
    Implements the full architecture of the ResNet-18 start block
    (initial layers + one BasicBlock) within a single class.

    The BasicBlock logic is directly translated into the __init__ and forward
    methods.
    """

    def __init__(self, in_channels=3):
        super(SingleBlockResNetTruncated, self).__init__()

        # Set input shape
        self.input_shape = (1, 3, 28, 28)

        # Initial large convolution
        self.conv1_head = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1_head = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )

        # --- 2. BasicBlock Logic (Layer 1, Block 1) ---
        block_in_channels = 64
        block_out_channels = 64
        block_stride = 1

        # Block: Conv1 (3x3)
        self.block_conv1 = nn.Conv2d(
            block_in_channels,
            block_out_channels,
            kernel_size=3,
            stride=block_stride,
            padding=1,
            bias=False
        )
        self.block_bn1 = nn.BatchNorm2d(block_out_channels)

        # Note: The original BasicBlock had two BN layers (bn1 and bn3) right
        # before ReLU. This is non-standard for ResNet; a typical BasicBlock
        # only has one BN per Conv. We will keep the second BN (bn3) here to
        # match the provided code exactly.
        self.block_bn3 = nn.BatchNorm2d(block_out_channels)

        # Block: Conv2 (3x3)
        self.block_conv2 = nn.Conv2d(
            block_out_channels,
            block_out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.block_bn2 = nn.BatchNorm2d(block_out_channels)

        # Downsample Logic (Identity Mapping for this block)
        # Since block_stride=1 and block_in_channels=block_out_channels=64,
        # downsample is None (identity).
        self.downsample = None

    def forward(self, x):
        # --- 1. ResNet Head Forward Pass ---
        x = self.conv1_head(x)
        x = self.bn1_head(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # --- 2. BasicBlock Forward Pass (The Residual Block) ---

        identity = x

        # Main path
        out = self.block_conv1(x)
        out = self.block_bn1(out)
        out = self.block_bn3(out)  # Second BN to match original code
        out = self.relu(out)

        out = self.block_conv2(out)
        out = self.block_bn2(out)

        # Skip connection (identity is just x in this specific case)
        if self.downsample is not None:
            identity = self.downsample(x)

        # Residual connection (Addition)
        out += identity
        out = self.relu(out)

        # The model stops here
        return out


class ResNet18_L1_Extractor(nn.Module):
    """
    Trunks a pre-trained ResNet-18 model to only include the initial layers
    and the first residual block (layer1).

    The output features size will be (Batch, 64, H/4, W/4).
    """
    def __init__(self, pretrained=True):
        super().__init__()

        # Set input shape
        self.input_shape = (1, 3, 224, 224)

        # Load the full ResNet-18 model
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            resnet = models.resnet18(weights=weights)
        else:
            resnet = models.resnet18(weights=None)

        # 1. Initial Convolution (conv1) and Batch Norm (bn1)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu

        # 2. Max Pooling
        self.maxpool = resnet.maxpool

        # 3. First Residual Block Layer (Layer 1)
        # This layer contains two BasicBlocks and the first set of 64 channels.
        self.layer1 = resnet.layer1

        # We discard layer2, layer3, layer4, avgpool, and fc.

    def forward(self, x):
        # Initial Feature Extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Downsampling by MaxPool (H/2, W/2)
        x = self.maxpool(x)

        # First Residual Block (Layer 1)
        # Spatial size remains H/4, W/4 (relative to original H, W)
        x = self.layer1(x)

        # Output: B x 64 x (H/4) x (W/4)
        return x


class TinyUNet_Straightforward(nn.Module):
    """
    Implémentation UNet ultra-minimaliste (1 niveau d'encodage/décodage)
    utilisant l'interpolation pour l'upsampling.

    Architecture:
    Input (H, W) -> Enc1 -> Bottleneck -> Up1 -> Output (H, W)
    """
    def __init__(self, in_channels=3, out_classes=1):
        super().__init__()
        print('Initializing model named: ', self._get_name())

        # Set input shape
        self.input_shape = (1, 3, 256, 256)

        # Hyperparamètres (Canaux à chaque étape)
        # c[1]=8 (Encodage/Décodage), c[2]=16 (Bottleneck)
        c = [in_channels, 8, 16]

        # --- A. ENCODER (Down Path) ---
        # 1. ENCODER 1: Conv -> 8 canaux (Génère le skip connection x1)
        self.enc1 = nn.Sequential(
            nn.Conv2d(c[0], c[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(c[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(c[1], c[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(c[1]),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)  # Downsample 1

        # --- B. BOTTLENECK ---
        # 2. BOTTLENECK: Conv -> 16 canaux
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c[1], c[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(c[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(c[2], c[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(c[2]),
            nn.ReLU(inplace=True)
        )

        # --- C. DECODER (Up Path) ---

        # 3. UPSAMPLE 1 (Transition Bottleneck -> Up1)
        # Interpolation du Bottleneck (16 -> 16)
        self.up_interp1 = nn.Upsample(
            scale_factor=2,
            mode='bilinear',
            align_corners=True
        )
        # Dual conv after cat (In: 16 (bottleneck) + 8 (skip) = 24 -> Out: 8)
        self.up_conv1 = nn.Sequential(
            nn.Conv2d(c[2] + c[1], c[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(c[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(c[1], c[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(c[1]),
            nn.ReLU(inplace=True)
        )

        # --- D. OUTPUT ---
        # 4. OUTPUT: Ramène à out_classes
        self.out_conv = nn.Conv2d(c[1], out_classes, kernel_size=1)

    def forward(self, x):
        # 1. ENCODER
        x1 = self.enc1(x)
        p1 = self.pool1(x1)  # Skip x1

        # 2. BOTTLENECK
        bottleneck = self.bottleneck(p1)

        # 3. DECODER 1: Interp + Concat (x1) + Conv
        up_b = self.up_interp1(bottleneck)
        merged1 = th.cat([x1, up_b], dim=1)
        d1 = self.up_conv1(merged1)

        # 4. OUTPUT
        logits = self.out_conv(d1)

        return logits


# -------------------
# ---- VGG-xx ----
class VGG13(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set input shape
        self.input_shape = (1, 3, 520, 520)

        # Get the pre-trained VGG-13
        self.model = models.vgg13(
            weights=models.VGG13_Weights.IMAGENET1K_V1
        )

    def forward(self, input):
        return self.model(input)


class VGG11(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set input shape
        self.input_shape = (1, 3, 520, 520)

        # Get the pre-trained VGG-11
        self.model = models.vgg11(
            weights=models.VGG11_Weights.IMAGENET1K_V1
        )

    def forward(self, input):
        return self.model(input)


class VGG16(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set input shape
        self.input_shape = (1, 3, 520, 520)

        # Get the pre-trained VGG-16
        self.model = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1
        )

    def forward(self, input):
        return self.model(input)


class VGG19(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set input shape
        self.input_shape = (1, 3, 520, 520)

        # Get the pre-trained VGG-19
        self.model = models.vgg19(
            weights=models.VGG19_Weights.IMAGENET1K_V1
        )

    def forward(self, input):
        return self.model(input)


# -------------------
# ---- ResNet-xx ----
class ResNet18(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set input shape
        self.input_shape = (1, 3, 224, 224)

        # Get the pre-trained ResNet-18
        self.model = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

    def forward(self, input):
        return self.model(input)


class ResNet34(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set input shape
        self.input_shape = (1, 3, 224, 224)

        # Get the pre-trained ResNet-34
        self.model = models.resnet34(
            weights=models.ResNet34_Weights.IMAGENET1K_V1
        )

    def forward(self, input):
        return self.model(input)


class ResNet50(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set input shape
        self.input_shape = (1, 3, 224, 224)

        # Get the pre-trained ResNet-50
        self.model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        )

    def forward(self, input):
        return self.model(input)


class FCNResNet50(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set input shape
        self.input_shape = (1, 3, 224, 224)

        # Get the pre-trained ResNet-50
        self.model = models.segmentation.fcn_resnet50(
            weights=models.segmentation.FCN_ResNet50_Weights.
            COCO_WITH_VOC_LABELS_V1
        )

    def forward(self, input):
        return self.model(input)


if __name__ == "__main__":
    from weightslab.backend.watcher_editor import WatcherEditor

    # Init logs
    setup_logging('INFO')

    # Gen. the model
    dummy_input = th.randn(1, 3, 256, 256)
    model = TinyUNet_Straightforward()
    model(dummy_input)

    # Watcher implementation
    model = WatcherEditor(model, dummy_input=dummy_input, print_graph=True)
    print(f'Inference results {model(dummy_input)}')  # infer
    print(model)

    # Model Operations
    # # Test: add neurons
    print("--- Test: Add Neurons ---")

    def model_add_neurons(model, x=None, dummy_input=None):
        """
            Test function to iteratively update neurons for each layer,
            then test inference.
        """
        n_layers = len(model.layers)
        for n in range(n_layers-1, -1, -1):
            if x is not None:
                if x >= 0:
                    if n != x:
                        continue
                else:
                    if n != n_layers + x:  # - -x != + -x
                        continue
            print(f'Add neuron at layer {n}', level='DEBUG')
            model.add_neurons(n, neuron_count=1)
            model.visited_nodes = set()  # reinitialisation
        model(dummy_input)
        print('Inference done!\n', level='DEBUG')

    model_add_neurons(model, x=2, dummy_input=dummy_input)
    print('#'+'-'*50)
