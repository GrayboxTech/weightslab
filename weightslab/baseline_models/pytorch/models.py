"""
    Batch of Torch models that use several types of architectures and can be
    used for different tasks.
    These architectures include several types of convolutional layers,
    batch normalization, and max pooling that are useful to stress test
    our neuron operations.
"""
import warnings; warnings.filterwarnings("ignore")
import torch
import logging
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

from typing import Tuple, List
from torch.nn import functional as F

from weightslab.components.tracking import add_tracked_attrs_to_input_tensor


# Define logger
logger = logging.getLogger(__name__)


# --- Basic CNN & MLP Models ---
class FashionCNN(nn.Module):
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
        x = self.fc4(self.fc3(self.f3(x)))
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
            nn.Linear(in_features=128, out_features=10)
        )
        self.out_softmax = nn.Softmax(dim=1)

    def forward(self, y):
        x = self.features(y)
        x = self.classifier(x)

        if not isinstance(x, torch.fx._symbolic_trace.Proxy):
            one_hot = F.one_hot(
                x.argmax(dim=1), num_classes=self.classifier[-1].out_features
            )

            if hasattr(x, 'in_id_batch') and \
                    hasattr(x, 'label_batch'):
                add_tracked_attrs_to_input_tensor(
                    one_hot, in_id_batch=input.in_id_batch,
                    label_batch=input.label_batch)
            self.classifier[-1].register(one_hot) \
                if hasattr(self.classifier[-1], 'register') else None

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


# --- Residual model - subpart ---
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


# --- The Core Residual Block for ResNet-18 and ResNet-34 ---
class SingleBlockResNetTruncated(nn.Module):
    """
    Implements the full architecture of the ResNet-18 start block
    (initial layers + one BasicBlock) within a single class.

    The BasicBlock logic is directly translated into the __init__ and forward
    methods.
    """

    def __init__(self, in_channels=1):
        super(SingleBlockResNetTruncated, self).__init__()

        # Set input shape
        self.input_shape = (1, 1, 28, 28)

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
    def __init__(self, in_channels=1, out_classes=1):
        super().__init__()

        # Set input shape
        self.input_shape = (1, 1, 256, 256)

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
        # Interpolation du Bottleneck (16 -> 32)
        self.up_interp1 = nn.Upsample(
            scale_factor=2,
            mode='bilinear',
            align_corners=True
        )
        # Dual conv after cat (In: 16 (bottleneck) + 8 (skip) = 24 -> Out: 8)
        self.up_conv1 = nn.Sequential(
            nn.Conv2d(c[2] + c[1] + c[1], c[1], kernel_size=3, padding=1),
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
        merged1 = torch.cat([x1, self.up_interp1(p1), up_b], dim=1)
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
        self.input_shape = (1, 1, 224, 224)

        # Get the pre-trained ResNet-18
        self.model = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

    def forward(self, input):
        input = torch.cat([input,]*3, dim=1)  # Add channels dim
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


class FlexibleCNNBlock(nn.Module):
    """
    A PyTorch module designed to handle 1D, 2D, or 3D convolutions,
    normalizations, and pooling layers based on the 'dim' parameter.

    This is useful for creating flexible architectures that work across
    time series (1D), images (2D), or volumes (3D).
    """

    # --- Static Mappings for Layer Classes ---

    # Map dimension (int) to the corresponding Convolution class
    CONV_MAP = {
        1: nn.Conv1d,
        2: nn.Conv2d,
        3: nn.Conv3d
    }

    # Map dimension (int) to the corresponding Transposed Convolution class
    CONV_TRANSPOSED_MAP = {
        1: nn.ConvTranspose1d,
        2: nn.ConvTranspose2d,
        3: nn.ConvTranspose3d
    }

    # Map dimension (int) to the corresponding Batch Normalization class
    BATCH_NORM_MAP = {
        1: nn.BatchNorm1d,
        2: nn.BatchNorm2d,
        3: nn.BatchNorm3d
    }

    # Map dimension (int) to the corresponding Instance Normalization class
    INSTANCE_NORM_MAP = {
        1: nn.InstanceNorm1d,
        2: nn.InstanceNorm2d,
        3: nn.InstanceNorm3d
    }

    # Map dimension (int) to the corresponding MaxPool class
    MAX_POOL_MAP = {
        1: nn.MaxPool1d,
        2: nn.MaxPool2d,
        3: nn.MaxPool3d
    }

    # Map dimension (int) to the corresponding Lazy Convolution class
    LAZY_CONV_MAP = {
        1: nn.LazyConv1d,
        2: nn.LazyConv2d,
        3: nn.LazyConv3d
    }

    def __init__(self,
                 dim: int = 3,
                 in_channels: int | None = 1,
                 out_channels: int = 8,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 norm_type: str = 'BatchNorm',
                 is_transposed: bool = False,
                 use_lazy: bool = False):

        super().__init__()
        self.input_shape = (1, 1,) + (16,) * dim

        if dim not in [1, 2, 3]:
            raise ValueError("Dimension (dim) must be 1, 2, or 3.")

        self.dim = dim
        self.out_channels = out_channels
        self.norm_type = norm_type

        # --- Helper for dynamic class selection ---
        def _get_conv_layer(is_transposed, use_lazy, in_channels):
            """Selects the correct Conv layer class based on parameters."""
            if use_lazy:
                # Lazy layers do not require in_channels
                return self.LAZY_CONV_MAP[self.dim]

            if is_transposed:
                return self.CONV_TRANSPOSED_MAP[self.dim]
            else:
                return self.CONV_MAP[self.dim]

        def _get_norm_layer(norm_type):
            """Selects the correct Normalization layer class."""
            norm_type = norm_type.lower()
            if 'batch' in norm_type:
                return self.BATCH_NORM_MAP[self.dim]
            elif 'instance' in norm_type:
                return self.INSTANCE_NORM_MAP[self.dim]
            else:
                raise ValueError(f"Unknown norm_type: {norm_type}")

        # --- Instantiate Layers Dynamically ---

        # 1. Convolution Layer
        ConvClass = _get_conv_layer(is_transposed, use_lazy, in_channels)

        # If using LazyConv, we only pass out_channels
        if use_lazy:
            self.conv = ConvClass(
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
            self.conv1 = ConvClass(
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
        else:
            # For standard Conv, we must specify in_channels
            if in_channels is None:
                raise ValueError(
                    "in_channels must be specified when use_lazy is False."
                )

            self.conv = ConvClass(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
            self.conv1 = ConvClass(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )

        # 2. Normalization Layer
        NormClass = _get_norm_layer(norm_type)
        # Normalization layers take num_features (which is our out_channels)
        self.norm = NormClass(out_channels, affine=True)

        # 3. Activation and Max Pooling
        self.relu = nn.ReLU(inplace=True)
        # Using a fixed MaxPool of size 2
        self.pool = self.MAX_POOL_MAP[self.dim](kernel_size=2)

    def forward(self, x):
        # Sequential execution
        x = self.conv(x)
        x = self.conv1(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.pool(x)

        return x


class DCGAN(nn.Module):
    """
    A single class encapsulating both the Generator (G) and Discriminator (D)
    of a Deep Convolutional Generative Adversarial Network (DCGAN).

    The standard forward method is dedicated to the Generator, while the
    Discriminator's logic is exposed via the 'discriminate' method.
    """
    def __init__(self, z_dim=100, img_channels=3,
                 features_g=64, features_d=64):
        super(DCGAN, self).__init__()

        self.input_shape = (16, z_dim, 1, 1)
        self.z_dim = z_dim

        # Initialize Generator and Discriminator as nn.Sequential blocks
        self.generator = self._init_generator_sequential(
            z_dim,
            img_channels,
            features_g
        )
        self.discriminator = self._init_discriminator_sequential(
            img_channels,
            features_d
        )

    def _init_generator_sequential(self, z_dim, img_channels, features_g):
        """Defines the Generator network using nn.Sequential."""
        return nn.Sequential(
            # Block 1: Input: N x Z_DIM x 1 x 1 -> N x 512 x 4 x 4
            nn.ConvTranspose2d(z_dim, features_g * 8, kernel_size=4, stride=1,
                               padding=0, bias=False),
            nn.BatchNorm2d(features_g * 8),
            nn.ReLU(inplace=True),

            # Block 2: N x 512 x 4 x 4 -> N x 256 x 8 x 8
            nn.ConvTranspose2d(features_g * 8, features_g * 4, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(inplace=True),

            # Block 3: N x 256 x 8 x 8 -> N x 128 x 16 x 16
            nn.ConvTranspose2d(features_g * 4, features_g * 2, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(inplace=True),

            # Block 4: N x 128 x 16 x 16 -> N x 64 x 32 x 32
            nn.ConvTranspose2d(features_g * 2, features_g, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_g),
            nn.ReLU(inplace=True),

            # Final Conv: N x 64 x 32 x 32 -> N x 3 x 64 x 64
            nn.ConvTranspose2d(features_g, img_channels, kernel_size=4,
                               stride=2, padding=1),
            nn.Tanh()  # Output range [-1, 1]
        )

    def _init_discriminator_sequential(self, img_channels, features_d):
        """Defines the Discriminator network using nn.Sequential."""
        return nn.Sequential(
            # Input: N x C x 64 x 64
            nn.Conv2d(img_channels, features_d, kernel_size=4, stride=2,
                      padding=1),  # Output: N x 64 x 32 x 32
            nn.LeakyReLU(0.2, inplace=True),

            # Block 2: N x 64 x 32 x 32 -> N x 128 x 16 x 16
            nn.Conv2d(features_d, features_d * 2, kernel_size=4, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 3: N x 128 x 16 x 16 -> N x 256 x 8 x 8
            nn.Conv2d(features_d * 2, features_d * 4, kernel_size=4, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 4: N x 256 x 8 x 8 -> N x 512 x 4 x 4
            nn.Conv2d(features_d * 4, features_d * 8, kernel_size=4, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(features_d * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Final Conv: N x 512 x 4 x 4 -> N x 1 x 1 x 1
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=1, padding=0),
        )

    def forward(self, noise):
        """
        Standard forward pass, dedicated to the Generator.
        Takes noise and returns a fake image.
        """
        return self.generator(noise)

    def discriminate(self, image):
        """
        Dedicated method to run the Discriminator.
        Takes an image and returns a single logit (N x 1).
        """
        output = self.discriminator(image)
        # Squeeze the output (N x 1 x 1 x 1) to (N x 1)
        return output.view(image.size(0), -1)


class SimpleVAE(nn.Module):
    """
    A minimal Variational Autoencoder (VAE) using fully-connected layers.
    Designed for 28x28 grayscale images (784 features).
    """
    def __init__(self, image_size=784, h_dim=200, z_dim=20):
        super(SimpleVAE, self).__init__()
        self.image_size = image_size
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.input_shape = (8, 1, 28, 28)

        # --- 1. Encoder ---
        # The Encoder maps the input image to the parameters of the latent
        # distribution.
        self.encoder = nn.Sequential(
            nn.Linear(image_size, h_dim),
            nn.ReLU(),
        )
        # Separate layers to output the mean (mu) and log variance (log_var)
        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_log_var = nn.Linear(h_dim, z_dim)

        # --- 2. Decoder ---
        # The Decoder maps the sampled latent vector (z) back to the image
        # space.
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, image_size),
            nn.Sigmoid()  # Sigmoid to output pixel values in the range [0, 1]
        )

    def reparameterize(self, mu, log_var):
        """
        The Reparameterization Trick: samples z = mu + sigma * epsilon.
        This allows gradients to flow back through the sampling process.
        """
        # Calculate standard deviation (sigma) from log variance
        std = torch.exp(0.5 * log_var)

        # Sample epsilon from standard normal distribution
        eps = torch.randn_like(std)

        # Return the sampled latent vector z
        return mu + std * eps

    def forward(self, x):
        """
        The forward pass performs the encoding, sampling, and decoding steps.
        """
        # 1. Flatten the input image (e.g., N x 1 x 28 x 28 -> N x 784)
        # We use x.view(-1, IMAGE_SIZE) or x.flatten(1) if dimensions are known
        x = x.view(x.size(0), -1)

        # 2. Encode
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        log_var = self.fc_log_var(encoded)

        # 3. Reparameterize and Sample Latent Vector (z)
        z = self.reparameterize(mu, log_var)

        # 4. Decode
        reconstruction = self.decoder(z)

        # Return the reconstruction along with mu and log_var
        # (needed for the VAE loss)
        return reconstruction, mu, log_var


class TwoLayerUnflattenNet(torch.nn.Module):
    def __init__(self, D_in=1000, D_out=10):
        super().__init__()
        self.input_shape = (64, D_in, 8, 8)

        # --- Layer 1: Conv2d (3 -> 16 channels) ---
        # Output spatial size is maintained (32x32 -> 32x32)
        self.conv1 = torch.nn.Conv2d(in_channels=D_in, out_channels=16,
                                  kernel_size=3, padding=1)

        # --- Layer 2: Conv2d (16 -> 32 channels) ---
        # Spatial size is halved (32x32 -> 16x16)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32,
                                  kernel_size=3, stride=2, padding=1)

        self.linear1 = torch.nn.Linear(512, 128)
        self.linear2 = torch.nn.Linear(128, 8192)

        # --- Unflattening (Required before final Conv2d) ---
        # Reshapes the 8192 tensor back to (32, 16, 16)
        # dim=1 means we are reshaping starting from the channel dimension
        self.unflatten = torch.nn.Unflatten(
            dim=1,
            unflattened_size=(32, 16, 16)
        )

        # --- Layer 5: Conv2d (32 -> 10 channels) ---
        # Final convolution layer for 10 classes
        self.conv3 = torch.nn.Conv2d(
            in_channels=32,
            out_channels=D_out,
            kernel_size=1
        )

    def forward(self, x):
        # Convolution layers
        x = self.conv1(x)
        x = self.conv2(x)

        # Flatten layer
        x = x.view(x.shape[0], -1)  # Flatten the tensor

        # Linear layers and ReLU activation function
        h_relu = self.linear1(x).clamp(min=0)
        x = self.linear2(h_relu)

        # Reshape for a final convolution
        x_reshaped = self.unflatten(x)
        y_pred = self.conv3(x_reshaped)

        return y_pred


class ToyAvgPoolNet(nn.Module):
    """
    A simple Convolutional Neural Network (CNN) demonstrating the use of
    nn.AvgPool2d to downsample feature maps.
    """
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.input_shape = (1, 3, 32, 32)

        # 1. Feature Extraction Layers
        self.features = nn.Sequential(
            # Input: B x 3 x 32 x 32
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            # Output: B x 16 x 32 x 32

            # Max Pooling (Standard method for downsampling)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Output: B x 16 x 16 x 16 (Spatial dimensions halved)

            # Groups = 1
            nn.Conv2d(16, 32, kernel_size=3, padding=1, groups=1),
            nn.ReLU(),
            # Output: B x 32 x 16 x 16

            # 1 < Groups > In_channels
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=4),
            nn.ReLU(),
            # Output: B x 32 x 16 x 16

            # # Groups = In_channels
            # nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),
            # nn.ReLU(),
            # # Output: B x 32 x 16 x 16

            # Average Pooling (The requested layer)
            nn.AvgPool2d(kernel_size=2, stride=2)
            # Output: B x 32 x 8 x 8 (Spatial dimensions halved again)
        )

        # Calculate the size of the flattened tensor for the Linear layer
        # 32 channels * 8 H * 8 W = 2048 features
        self.flatten_size = 32 * 8 * 8

        # 2. Classifier Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Pass through convolutional features
        x = self.features(x)

        # Pass through classifier head
        x = self.classifier(x)
        return x


class TinyUNet(nn.Module):
    """
    A full U-Net implementation contained within a single class,
    using nn.Sequential for all core DoubleConv blocks.

    This architecture is designed for semantic segmentation tasks.
    """
    def __init__(self, n_channels=3, n_classes=1):
        super(TinyUNet, self).__init__()
        self.input_shape = (1, n_channels, 256, 256)
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Helper function to define the DoubleConv block as an nn.Sequential
        def double_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        # ------------------ ENCODER (Downsampling Path) ------------------
        # 0. Initial Layer
        self.inc = double_conv(n_channels, 64)

        # 1. Down 1 (Max pool + Conv)
        self.down1_pool = nn.MaxPool2d(2)
        self.down1_conv = double_conv(64, 128)

        # ------------------ DECODER (Upsampling Path) ------------------
        # 4. Up 4
        self.up4_up = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=1)
        self.up4_conv = double_conv(192, 64)  # Input channels: 64 + 64 = 128

        # ------------------ OUTPUT Layer ------------------
        # 1x1 convolution to map the final feature channels (64) to the number of classes
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # ------------------ ENCODER (Store for skip connections) ------------------
        x1 = self.inc(x)

        x2 = self.down1_pool(x1)
        x2 = self.down1_conv(x2)

        # ------------------ DECODER (Concatenate and Convolve) ------------------
        # Up 4
        x = self.up4_up(x2)  # B3
        x = self._align_and_concat(x, x1)
        x = self.up4_conv(x)

        # ------------------ OUTPUT ------------------
        logits = self.outc(x)

        return logits

    def _align_and_concat(self, upsampled, skip):
        """
        Helper method to align upsampled tensor size to match the skip connection.
        Uses F.interpolate, which is tracer-friendly.

        Args:
            upsampled (Tensor): The upsampled tensor from the decoder patorch.
            skip (Tensor): The higher-resolution tensor from the encoder (skip connection).

        Returns:
            Tensor: The concatenated tensor.
        """
        # Explicitly resize the upsampled tensor to match the spatial dimensions of the skip connection
        # skip.shape[-2:] extracts (H, W)
        upsampled = F.interpolate(
            upsampled,
            size=skip.shape[-2:],
            mode='bilinear',
            align_corners=False  # Set to False for compatibility and best practice
        )

        # Concatenate along the channel dimension (dim=1)
        return torch.cat([skip, upsampled], dim=1)


class UNet(nn.Module):
    """
    A full U-Net implementation contained within a single class,
    using nn.Sequential for all core DoubleConv blocks.

    This architecture is designed for semantic segmentation tasks.
    """
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        self.input_shape = (1, n_channels, 256, 256)
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Helper function to define the DoubleConv block as an nn.Sequential
        def double_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        # ------------------ ENCODER (Downsampling Path) ------------------
        # 0. Initial Layer
        self.inc = double_conv(n_channels, 64)

        # 1. Down 1 (Max pool + Conv)
        self.down1_pool = nn.MaxPool2d(2)
        self.down1_conv = double_conv(64, 128)

        # 2. Down 2
        self.down2_pool = nn.MaxPool2d(2)
        self.down2_conv = double_conv(128, 256)

        # 3. Down 3
        self.down3_pool = nn.MaxPool2d(2)
        self.down3_conv = double_conv(256, 512)

        # 4. Down 4 (Bottleneck)
        self.down4_pool = nn.MaxPool2d(2)
        self.down4_conv = double_conv(512, 1024)

        # ------------------ DECODER (Upsampling Path) ------------------
        # 1. Up 1 (Upsample + Conv + Skip Connection)
        self.up1_up = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up1_conv = double_conv(1024, 512)  # Input channels: 512 (from up) + 512 (from skip) = 1024

        # 2. Up 2
        self.up2_up = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2_conv = double_conv(512, 256)  # Input channels: 256 + 256 = 512

        # 3. Up 3
        self.up3_up = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up3_conv = double_conv(256, 128)  # Input channels: 128 + 128 = 256

        # 4. Up 4
        self.up4_up = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up4_conv = double_conv(128, 64)  # Input channels: 64 + 64 = 128

        # ------------------ OUTPUT Layer ------------------
        # 1x1 convolution to map the final feature channels (64) to the number of classes
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # ------------------ ENCODER (Store for skip connections) ------------------
        x1 = self.inc(x)

        x2 = self.down1_pool(x1)
        x2 = self.down1_conv(x2)

        x3 = self.down2_pool(x2)
        x3 = self.down2_conv(x3)

        x4 = self.down3_pool(x3)
        x4 = self.down3_conv(x4)

        x5 = self.down4_pool(x4)
        x5 = self.down4_conv(x5)  # This is the bottleneck feature map (lowest resolution)

        # ------------------ DECODER (Concatenate and Convolve) ------------------
        # Up 1
        x = self.up1_up(x5)  # Upsample x5
        x = self._align_and_concat(x, x4)
        x = self.up1_conv(x)

        # Up 2
        x = self.up2_up(x)
        x = self._align_and_concat(x, x3)
        x = self.up2_conv(x)

        # Up 3
        x = self.up3_up(x)
        x = self._align_and_concat(x, x2)
        x = self.up3_conv(x)

        # Up 4
        x = self.up4_up(x)
        x = self._align_and_concat(x, x1)
        x = self.up4_conv(x)

        # ------------------ OUTPUT ------------------
        logits = self.outc(x)

        return logits

    def _align_and_concat(self, upsampled, skip):
        """
        Helper method to align upsampled tensor size to match the skip connection.
        Uses F.interpolate, which is tracer-friendly.

        Args:
            upsampled (Tensor): The upsampled tensor from the decoder patorch.
            skip (Tensor): The higher-resolution tensor from the encoder (skip connection).

        Returns:
            Tensor: The concatenated tensor.
        """
        # Explicitly resize the upsampled tensor to match the spatial dimensions of the skip connection
        # skip.shape[-2:] extracts (H, W)
        upsampled = F.interpolate(
            upsampled,
            size=skip.shape[-2:],
            mode='bilinear',
            align_corners=False  # Set to False for compatibility and best practice
        )

        # Concatenate along the channel dimension (dim=1)
        return torch.cat([skip, upsampled], dim=1)


class UNet3p(nn.Module):
    """
    UNet 3+ Architecture for Semantic Segmentation.

    This single-class implementation uses full-scale skip connections to
    combine multi-scale features for high-resolution output.

    Note: Deep Supervision is often used but is omitted here for simplicity
    and focusing on the core feature fusion logic.
    """
    def __init__(self, n_channels=3, n_classes=1, filter_list=[64, 128, 256, 512, 1024]):
        super(UNet3p, self).__init__()
        self.input_shape = (1, n_channels, 256, 256)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.filters = filter_list  # [F1, F2, F3, F4, F5]
        self.F_cat = self.filters[0] * 5  # Total channels in feature concatenation (e.g., 64 * 5 = 320)

        # ------------------- Internal Building Blocks -------------------

        # Helper for a simple 3x3 Conv -> BN -> ReLU block
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
        self.ConvBlock = conv_block

        # Helper for a Double Conv block (used for standard encoder layers)
        def double_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        # ------------------- ENCODER (Downsampling Path) -------------------
        # E1, E2, E3, E4 are standard double convolutions
        self.conv1 = double_conv(n_channels, self.filters[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = double_conv(self.filters[0], self.filters[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = double_conv(self.filters[1], self.filters[2])
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = double_conv(self.filters[2], self.filters[3])
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        # E5 is the Bottleneck
        self.conv5 = double_conv(self.filters[3], self.filters[4])


        # ------------------- DECODER (Upsampling Path with Full-Scale Fusion) -------------------
        # Decoder D4 (Input: E1, E2, E3 (Downsampled); E5 (Upsampled); D5 (Upsampled, from D5 in paper) - simplified here)
        # To match E1, E2, E3: need to downsample E1, E2, E3
        self.h1_L4 = conv_block(self.filters[0], self.F_cat) # E1 -> D4 (down by 8x)
        self.h2_L4 = conv_block(self.filters[1], self.F_cat) # E2 -> D4 (down by 4x)
        self.h3_L4 = conv_block(self.filters[2], self.F_cat) # E3 -> D4 (down by 2x)
        self.h4_L4 = conv_block(self.filters[3], self.F_cat) # E4 -> D4 (same scale)
        self.h5_L4 = conv_block(self.filters[4], self.F_cat) # E5 -> D4 (up by 2x)
        self.up_conv4 = double_conv(5 * self.F_cat, self.filters[3]) # Final D4 convolution

        # Decoder D3 (Input: E1, E2 (Downsampled); E4, E5 (Upsampled); D4 (Upsampled))
        self.h1_L3 = conv_block(self.filters[0], self.F_cat) # E1 -> D3 (down by 4x)
        self.h2_L3 = conv_block(self.filters[1], self.F_cat) # E2 -> D3 (down by 2x)
        self.h3_L3 = conv_block(self.filters[2], self.F_cat) # E3 -> D3 (same scale)
        self.h4_L3 = conv_block(self.filters[3], self.F_cat) # D4 -> D3 (up by 2x)
        self.h5_L3 = conv_block(self.filters[4], self.F_cat) # E5 -> D3 (up by 4x)
        self.up_conv3 = double_conv(5 * self.F_cat, self.filters[2]) # Final D3 convolution

        # Decoder D2 (Input: E1 (Downsampled); E3, E4, E5 (Upsampled); D3 (Upsampled))
        self.h1_L2 = conv_block(self.filters[0], self.F_cat) # E1 -> D2 (down by 2x)
        self.h2_L2 = conv_block(self.filters[1], self.F_cat) # E2 -> D2 (same scale)
        self.h3_L2 = conv_block(self.filters[2], self.F_cat) # D3 -> D2 (up by 2x)
        self.h4_L2 = conv_block(self.filters[3], self.F_cat) # D4 -> D2 (up by 4x)
        self.h5_L2 = conv_block(self.filters[4], self.F_cat) # E5 -> D2 (up by 8x)
        self.up_conv2 = double_conv(5 * self.F_cat, self.filters[1]) # Final D2 convolution

        # Decoder D1 (Input: E2, E3, E4, E5 (Upsampled); D2 (Upsampled))
        self.h1_L1 = conv_block(self.filters[0], self.F_cat) # E1 -> D1 (same scale)
        self.h2_L1 = conv_block(self.filters[1], self.F_cat) # D2 -> D1 (up by 2x)
        self.h3_L1 = conv_block(self.filters[2], self.F_cat) # D3 -> D1 (up by 4x)
        self.h4_L1 = conv_block(self.filters[3], self.F_cat) # D4 -> D1 (up by 8x)
        self.h5_L1 = conv_block(self.filters[4], self.F_cat) # E5 -> D1 (up by 16x)
        self.up_conv1 = double_conv(5 * self.F_cat, self.filters[0]) # Final D1 convolution

        # ------------------- FINAL Output Layer -------------------
        self.outc = nn.Conv2d(self.filters[0], n_classes, kernel_size=1)

    # Helper function for resizing tensors
    def _interpolate_to_size(self, x, size):
        return F.interpolate(x, size=size, mode='bilinear', align_corners=True)

    def forward(self, x):
        # ------------------- ENCODER -------------------
        # E1 (64 channels) - Size H/1
        e1 = self.conv1(x)
        p1 = self.pool1(e1) # Size H/2

        # E2 (128 channels)
        e2 = self.conv2(p1)
        p2 = self.pool2(e2) # Size H/4

        # E3 (256 channels)
        e3 = self.conv3(p2)
        p3 = self.pool3(e3) # Size H/8

        # E4 (512 channels)
        e4 = self.conv4(p3)
        p4 = self.pool4(e4) # Size H/16

        # E5 (1024 channels) - Bottleneck
        e5 = self.conv5(p4) # Size H/16

        # ------------------- DECODER D4 (Size H/8) -------------------
        # All inputs are brought to the spatial size of E3 (H/8)

        # 1. Input from E1 (Down 8x)
        h1_d4 = self.h1_L4(self._interpolate_to_size(e1, e3.shape[-2:]))

        # 2. Input from E2 (Down 4x)
        h2_d4 = self.h2_L4(self._interpolate_to_size(e2, e3.shape[-2:]))

        # 3. Input from E3 (Down 2x)
        h3_d4 = self.h3_L4(self._interpolate_to_size(e3, e3.shape[-2:]))

        # 4. Input from E4 (Same Scale)
        h4_d4 = self.h4_L4(self._interpolate_to_size(e4, e3.shape[-2:]))

        # 5. Input from E5 (Up 2x)
        h5_d4 = self.h5_L4(self._interpolate_to_size(e5, e3.shape[-2:]))

        # Concatenate and convolve
        d4 = torch.cat((h1_d4, h2_d4, h3_d4, h4_d4, h5_d4), dim=1)
        d4 = self.up_conv4(d4) # Output D4 (512 channels, Size H/8)

        # ------------------- DECODER D3 (Size H/4) -------------------
        # All inputs are brought to the spatial size of E2 (H/4)

        # 1. Input from E1 (Down 4x)
        h1_d3 = self.h1_L3(self._interpolate_to_size(e1, e2.shape[-2:]))

        # 2. Input from E2 (Down 2x)
        h2_d3 = self.h2_L3(self._interpolate_to_size(e2, e2.shape[-2:]))

        # 3. Input from E3 (Same Scale)
        h3_d3 = self.h3_L3(self._interpolate_to_size(e3, e2.shape[-2:]))

        # 4. Input from D4 (Up 2x)
        h4_d3 = self.h4_L3(self._interpolate_to_size(d4, e2.shape[-2:]))

        # 5. Input from E5 (Up 4x)
        h5_d3 = self.h5_L3(self._interpolate_to_size(e5, e2.shape[-2:]))

        # Concatenate and convolve
        d3 = torch.cat((h1_d3, h2_d3, h3_d3, h4_d3, h5_d3), dim=1)
        d3 = self.up_conv3(d3) # Output D3 (256 channels, Size H/4)

        # ------------------- DECODER D2 (Size H/2) -------------------
        # All inputs are brought to the spatial size of E1 (H/2)

        # 1. Input from E1 (Down 2x)
        h1_d2 = self.h1_L2(self._interpolate_to_size(e1, e1.shape[-2:]))

        # 2. Input from E2 (Same Scale)
        h2_d2 = self.h2_L2(self._interpolate_to_size(e2, e1.shape[-2:]))

        # 3. Input from D3 (Up 2x)
        h3_d2 = self.h3_L2(self._interpolate_to_size(d3, e1.shape[-2:]))

        # 4. Input from D4 (Up 4x)
        h4_d2 = self.h4_L2(self._interpolate_to_size(d4, e1.shape[-2:]))

        # 5. Input from E5 (Up 8x)
        h5_d2 = self.h5_L2(self._interpolate_to_size(e5, e1.shape[-2:]))

        # Concatenate and convolve
        d2 = torch.cat((h1_d2, h2_d2, h3_d2, h4_d2, h5_d2), dim=1)
        d2 = self.up_conv2(d2) # Output D2 (128 channels, Size H/2)

        # ------------------- DECODER D1 (Size H/1) -------------------
        # All inputs are brought to the spatial size of the original input x (H/1)

        # 1. Input from E1 (Same Scale)
        h1_d1 = self.h1_L1(e1)

        # 2. Input from D2 (Up 2x)
        h2_d1 = self.h2_L1(self._interpolate_to_size(d2, x.shape[-2:]))

        # 3. Input from D3 (Up 4x)
        h3_d1 = self.h3_L1(self._interpolate_to_size(d3, x.shape[-2:]))

        # 4. Input from D4 (Up 8x)
        h4_d1 = self.h4_L1(self._interpolate_to_size(d4, x.shape[-2:]))

        # 5. Input from E5 (Up 16x)
        h5_d1 = self.h5_L1(self._interpolate_to_size(e5, x.shape[-2:]))

        # Concatenate and convolve
        d1 = torch.cat((h1_d1, h2_d1, h3_d1, h4_d1, h5_d1), dim=1)
        d1 = self.up_conv1(d1)  # Output D1 (64 channels, Size H/1)

        # ------------------- OUTPUT -------------------
        logits = self.outc(d1)
        return logits


class UNet3D(nn.Module):
    """
    A single-class implementation of the 3D U-Net architecture
    for volumetric image segmentation.
    """
    def __init__(self, input_channels=1, output_classes=1, base_channels=32, v_size=64):
        super().__init__()
        self.input_shape = (2, input_channels, v_size, v_size, v_size)
        self.input_channels = input_channels
        self.output_classes = output_classes

        def double_conv_3d(in_c, out_c):
            """Returns a sequential block of two Conv3d layers with ReLU."""
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True)
            )

        # --- ENCODER (Contracting Path) ---

        # Initial convolution and first block
        self.inc = double_conv_3d(input_channels, base_channels)  # C -> 32

        # Down 1
        self.down1_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.down1_conv = double_conv_3d(base_channels, base_channels * 2) # 32 -> 64

        # Down 2
        self.down2_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.down2_conv = double_conv_3d(base_channels * 2, base_channels * 4) # 64 -> 128

        # Down 3 (Bottleneck)
        self.down3_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.down3_conv = double_conv_3d(base_channels * 4, base_channels * 8) # 128 -> 256

        # --- DECODER (Expansive Path) ---

        # Up 3
        # Transposed conv to double D, H, W while halving channels
        self.up3_upsample = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2) # 256 -> 128
        # DoubleConv takes (skip_c + upsampled_c) -> base_channels * 4
        self.up3_conv = double_conv_3d(base_channels * 8, base_channels * 4) # (128 + 128) -> 128

        # Up 2
        self.up2_upsample = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2) # 128 -> 64
        self.up2_conv = double_conv_3d(base_channels * 4, base_channels * 2) # (64 + 64) -> 64

        # Up 1
        self.up1_upsample = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2) # 64 -> 32
        self.up1_conv = double_conv_3d(base_channels * 2, base_channels) # (32 + 32) -> 32

        # Final Output Layer (maps channels to class count)
        self.out_conv = nn.Conv3d(base_channels, output_classes, kernel_size=1) # 32 -> C_out

    def forward(self, x):
        # x shape: (B, C, D, H, W)

        # --- ENCODER ---
        x1 = self.inc(x)                 # B x 32 x D x H x W (Skip 1)

        x2 = self.down1_pool(x1)
        x2 = self.down1_conv(x2)         # B x 64 x D/2 x H/2 x W/2 (Skip 2)

        x3 = self.down2_pool(x2)
        x3 = self.down2_conv(x3)         # B x 128 x D/4 x H/4 x W/4 (Skip 3)

        x4 = self.down3_pool(x3)
        x4 = self.down3_conv(x4)         # B x 256 x D/8 x H/8 x W/8 (Bottleneck)

        # --- DECODER ---

        # Up 3
        up3 = self.up3_upsample(x4)      # B x 128 x D/4 x H/4 x W/4 (Upsampled)
        # Skip connection: Concatenate with x3 (128 channels)
        cat3 = torch.cat([x3, up3], dim=1) # B x 256 x D/4 x H/4 x W/4
        x = self.up3_conv(cat3)          # B x 128 x D/4 x H/4 x W/4

        # Up 2
        up2 = self.up2_upsample(x)       # B x 64 x D/2 x H/2 x W/2
        # Skip connection: Concatenate with x2 (64 channels)
        cat2 = torch.cat([x2, up2], dim=1) # B x 128 x D/2 x H/2 x W/2
        x = self.up2_conv(cat2)          # B x 64 x D/2 x H/2 x W/2

        # Up 1
        up1 = self.up1_upsample(x)       # B x 32 x D x H x W
        # Skip connection: Concatenate with x1 (32 channels)
        cat1 = torch.cat([x1, up1], dim=1) # B x 64 x D x H x W
        x = self.up1_conv(cat1)          # B x 32 x D x H x W

        # Final Output
        logits = self.out_conv(x)        # B x C_out x D x H x W

        return logits


class TinyYOLO(nn.Module):
    """
    Tiny YOLO-like single-class detector (all-in-one class).

    Outputs tensor shape (N, S, S, B*5 + C) where:
      - for each cell and each box: tx, ty, tw, th, conf
      - plus class logits/prob (here C=1 by default)

    decode/predict helpers are provided as methods of this class.
    """

    def __init__(self, S: int = 7, B: int = 2, C: int = 1, image_size: int = 224):
        """
        S: grid size (SxS)
        B: boxes per cell
        C: number of classes (1 for single class)
        image_size: input image size (square) used for inference preprocessing / visualization
        """
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.image_size = image_size
        self.input_shape = (1, 3, image_size, image_size)

        # Tiny backbone: conv stack -> feature map of size (N, feat_ch, S, S)
        # We make sure adaptive pooling yields SxS.
        self.backbone = nn.Sequential(
            self._conv_bn_act(3, 16, 3, 1, 1),
            nn.MaxPool2d(2),
            self._conv_bn_act(16, 32, 3, 1, 1),
            nn.MaxPool2d(2),
            self._conv_bn_act(32, 64, 3, 1, 1),
            nn.MaxPool2d(2),
            self._conv_bn_act(64, 128, 3, 1, 1),
            nn.MaxPool2d(2),
            self._conv_bn_act(128, 256, 3, 1, 1),
            nn.AdaptiveAvgPool2d((S, S)),
        )

        # head: project features -> (B*5 + C) channels
        out_ch = B * 5 + C
        self.head = nn.Sequential(
            self._conv_bn_act(256, 512, 3, 1, 1),
            nn.Conv2d(512, out_ch, kernel_size=1)
        )

        # ImageNet normalization (used on input images)
        self.preprocess = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean/std
                        std=[0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def _conv_bn_act(in_ch, out_ch, k=3, s=1, p=1):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, 3, H, W)
        returns: (N, S, S, B*5 + C)
        """
        feat = self.backbone(x)
        out = self.head(feat)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out

    def decode_preds(self, pred: torch.Tensor, conf_thresh: float = 0.25) -> List[List[Tuple[float, float, float, float, float, int]]]:
        """
        Decode raw predictions into boxes per image.

        pred: (N, S, S, B*5 + C)
        returns: list length N, each element is a list of boxes (x1,y1,x2,y2,score,class)
                 coordinates normalized in [0,1] relative to image width & height.
        """
        device = pred.device
        N, S, _, _ = pred.shape
        boxes_per_image = []

        # prepare grid coords
        grid_y, grid_x = torch.meshgrid(torch.arange(S, device=device), torch.arange(S, device=device))
        grid_x = grid_x.float()
        grid_y = grid_y.float()

        for i in range(N):
            img_pred = pred[i]  # S x S x (B*5 + C)
            img_boxes = []

            # class score per cell (for single class we use sigmoid)
            if self.C == 1:
                cls_score = torch.sigmoid(img_pred[..., -1])  # (S, S)
            else:
                cls_logits = img_pred[..., self.B * 5:]
                cls_score_all = F.softmax(cls_logits, dim=-1)
                # take score for class 0 ("cat") by default
                cls_score = cls_score_all[..., 0]

            # iterate boxes
            for b in range(self.B):
                offset = b * 5
                tx = torch.sigmoid(img_pred[..., offset + 0])
                ty = torch.sigmoid(img_pred[..., offset + 1])
                tw = img_pred[..., offset + 2].exp()
                th = img_pred[..., offset + 3].exp()
                conf = torch.sigmoid(img_pred[..., offset + 4])

                # center coordinates normalized
                cx = (grid_x + tx) / S
                cy = (grid_y + ty) / S
                # width/height normalized (simple scheme, no anchors)
                w = tw / S
                h = th / S

                final_conf = conf * cls_score  # class-aware confidence

                mask = final_conf > conf_thresh
                if mask.any():
                    cx_m = cx[mask]
                    cy_m = cy[mask]
                    w_m = w[mask]
                    h_m = h[mask]
                    conf_m = final_conf[mask]

                    x1 = (cx_m - 0.5 * w_m).clamp(0.0, 1.0)
                    y1 = (cy_m - 0.5 * h_m).clamp(0.0, 1.0)
                    x2 = (cx_m + 0.5 * w_m).clamp(0.0, 1.0)
                    y2 = (cy_m + 0.5 * h_m).clamp(0.0, 1.0)

                    for xi, yi, xj, yj, cf in zip(x1.tolist(), y1.tolist(), x2.tolist(), y2.tolist(), conf_m.tolist()):
                        img_boxes.append((xi, yi, xj, yj, cf, 0))

            boxes_per_image.append(img_boxes)
        return boxes_per_image

    @staticmethod
    def nms(boxes: List[Tuple[float, float, float, float, float, int]], iou_thresh: float = 0.5) -> List[Tuple[float, float, float, float, float, int]]:
        """
        Simple class-agnostic NMS on boxes list of (x1,y1,x2,y2,score,cls).
        Returns list of kept boxes.
        """
        if not boxes:
            return []
        # convert to tensors for convenience
        boxes_arr = torch.tensor([b[:4] for b in boxes], dtype=torch.float32)
        scores = torch.tensor([b[4] for b in boxes], dtype=torch.float32)
        order = scores.argsort(descending=True)
        keep = []
        suppressed = torch.zeros(len(boxes), dtype=torch.bool)
        for idx in order.tolist():
            if suppressed[idx]:
                continue
            keep.append(boxes[idx])
            # suppress IoU > thresh
            box_i = boxes_arr[idx]
            xi1, yi1, xi2, yi2 = box_i
            area_i = (xi2 - xi1).clamp(min=0) * (yi2 - yi1).clamp(min=0)
            for j in order.tolist():
                if j == idx or suppressed[j]:
                    continue
                box_j = boxes_arr[j]
                xj1, yj1, xj2, yj2 = box_j
                inter_x1 = max(xi1.item(), xj1.item())
                inter_y1 = max(yi1.item(), yj1.item())
                inter_x2 = min(xi2.item(), xj2.item())
                inter_y2 = min(yi2.item(), yj2.item())
                inter_w = max(0.0, inter_x2 - inter_x1)
                inter_h = max(0.0, inter_y2 - inter_y1)
                inter = inter_w * inter_h
                area_j = (xj2 - xj1).clamp(min=0) * (yj2 - yj1).clamp(min=0)
                union = area_i + area_j - inter
                if union <= 0:
                    continue
                iou = inter / union
                if iou > iou_thresh:
                    suppressed[j] = True
        return keep
