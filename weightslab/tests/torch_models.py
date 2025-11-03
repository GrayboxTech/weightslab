"""
    Batch of Torch models that use several types of architectures and can be
    used for different tasks.
    These architectures include several types of convolutional layers,
    batch normalization, and max pooling that are useful to stress test
    our neuron operations.
"""
import torch as th
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

from weightslab.components.tracking import add_tracked_attrs_to_input_tensor


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
        std = th.exp(0.5 * log_var)

        # Sample epsilon from standard normal distribution
        eps = th.randn_like(std)

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


class TwoLayerUnflattenNet(th.nn.Module):
    def __init__(self, D_in=1000, D_out=10):
        super().__init__()
        self.input_shape = (64, D_in, 8, 8)

        # --- Layer 1: Conv2d (3 -> 16 channels) ---
        # Output spatial size is maintained (32x32 -> 32x32)
        self.conv1 = th.nn.Conv2d(in_channels=D_in, out_channels=16,
                                  kernel_size=3, padding=1)

        # --- Layer 2: Conv2d (16 -> 32 channels) ---
        # Spatial size is halved (32x32 -> 16x16)
        self.conv2 = th.nn.Conv2d(in_channels=16, out_channels=32,
                                  kernel_size=3, stride=2, padding=1)

        self.linear1 = th.nn.Linear(512, 128)
        self.linear2 = th.nn.Linear(128, 8192)

        # --- Unflattening (Required before final Conv2d) ---
        # Reshapes the 8192 tensor back to (32, 16, 16)
        # dim=1 means we are reshaping starting from the channel dimension
        self.unflatten = th.nn.Unflatten(
            dim=1,
            unflattened_size=(32, 16, 16)
        )

        # --- Layer 5: Conv2d (32 -> 10 channels) ---
        # Final convolution layer for 10 classes
        self.conv3 = th.nn.Conv2d(
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
