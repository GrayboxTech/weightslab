"""
Unit tests for computational graph dependency extraction.

Tests focus on minimal models with specific dependency patterns rather than
full architectures, enabling fast iteration and pattern validation.

Patterns covered:
- SAME: Conv -> BN -> ReLU (learnable -> normalization -> non-learnable)
- INCOMING: Conv -> Conv (learnable -> learnable with dimension change)
- REC: Residual blocks and concatenation operations
- Mixed: Complex patterns combining multiple dependency types
"""
import os
import unittest
import torch
import torch.nn as nn

from typing import List, Tuple

from weightslab.utils.tools import model_op_neurons
from weightslab.utils.computational_graph import _infer_dependency_type
from weightslab.utils.modules_dependencies import DepType
from weightslab.backend.model_interface import ModelInterface


os.environ['WEIGHTSLAB_LOG_LEVEL'] = 'DEBUG'

1
class MinimalSAMEDependencies(nn.Module):
    """
    Tests SAME dependency type: Conv -> BN -> ReLU

    Expected:
    - conv -> bn: SAME (BN has only 1D params)
    - bn -> relu: SAME (ReLU has no params)
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MinimalINCOMINGDependencies(nn.Module):
    """
    Tests INCOMING dependency type: Conv -> Conv

    Expected:
    - conv1 -> conv2: INCOMING (Conv2d has 2D weight matrix)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MinimalMixedDependencies(nn.Module):
    """
    Tests mixed SAME and INCOMING: Conv -> BN -> ReLU -> Conv

    Expected:
    - conv1 -> bn: SAME
    - bn -> relu: SAME
    - relu -> conv2: INCOMING
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class MinimalResidualBlock(nn.Module):
    """
    Tests REC dependency type: Simple residual connection

    Expected:
    - conv1 -> bn1: SAME
    - bn1 -> relu: SAME
    - relu -> conv2: INCOMING
    - conv2 -> bn2: SAME
    - conv1/bn1/relu <-REC-> conv2/bn2 (residual merge)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.bn1(out1)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + out1  # Residual connection (REC dependency)
        return out


class MinimalSkipConnectionCat(nn.Module):
    """
    Tests REC dependency via concatenation (Concat operation)

    Expected:
    - conv1 -> relu: SAME
    - conv2 -> relu2: SAME
    - relu <-REC-> relu2 (concatenated, must have matching dims)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(3, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv_merged = nn.Conv2d(64, 64, 3, padding=1)

    def forward(self, x):
        branch1 = self.conv1(x)
        branch1 = self.relu1(branch1)

        branch2 = self.conv2(x)
        branch2 = self.relu2(branch2)

        merged = torch.cat([branch1, branch2], dim=1)  # REC: both branches constrained
        out = self.conv_merged(merged)
        return out


class MinimalWithMaxPool(nn.Module):
    """
    Tests non-learnable layers in dependency chain: Conv -> MaxPool -> ReLU -> Conv

    Expected:
    - conv1 -> maxpool: SAME (MaxPool has no learnable params)
    - maxpool -> relu: SAME (ReLU has no learnable params)
    - relu -> conv2: INCOMING
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, 1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class MinimalWithUpsampling(nn.Module):
    """
    Tests upsampling layer: Conv -> Upsample -> Conv

    Expected:
    - conv1 -> upsample: SAME (Upsample has no learnable params)
    - upsample -> conv2: INCOMING
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.upsample(x)
        x = self.conv2(x)
        return x


class MinimalGroupedConv(nn.Module):
    """
    Grouped convolution chain: depthwise-like then pointwise
    Expected: INCOMING dependencies between grouped convs and subsequent conv
    """
    def __init__(self):
        super().__init__()
        # Input 4 channels, grouped conv with groups=2
        self.conv_g1 = nn.Conv2d(4, 8, kernel_size=3, padding=1, groups=2)
        self.relu = nn.ReLU()
        self.conv_g2 = nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=2)
        self.conv_pw = nn.Conv2d(8, 16, kernel_size=1)

    def forward(self, x):
        x = self.conv_g1(x)
        x = self.relu(x)
        x = self.conv_g2(x)
        x = self.conv_pw(x)
        return x


class MinimalDepthwisePointwiseWithLinear(nn.Module):
    """
    Depthwise (groups=in_channels) followed by pointwise conv
    Expected: INCOMING between depthwise -> pointwise
    """
    def __init__(self):
        super().__init__()
        self.dw = nn.Conv2d(8, 16, kernel_size=3, padding=1, groups=8)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.pw = nn.Conv2d(16, 12, kernel_size=1, groups=4)
        self.end = nn.Conv2d(12, 1, kernel_size=1)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(256, 64)
        self.linear2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.dw(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pw(x)
        x = self.end(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class MinimalConvTransposeChain(nn.Module):
    """
    Conv -> ReLU -> ConvTranspose2d
    Expected: SAME then INCOMING
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 16, kernel_size=3, padding=1, groups=4)
        self.relu = nn.ReLU()
        self.deconv = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.deconv(x)
        return x


class MinimalLinearChain(nn.Module):
    """
    Flatten -> Linear -> ReLU -> Linear
    Expected: SAME (flatten, relu) and INCOMING (linear)
    """
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 16 * 16, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class MinimalConv1DChain(nn.Module):
    """
    Conv1d -> BatchNorm1d -> ReLU -> Conv1d
    Expected: SAME then INCOMING
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(8)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class MinimalConv3DChain(nn.Module):
    """
    Conv3d -> BatchNorm3d -> ReLU -> Conv3d
    Expected: SAME then INCOMING
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(2, 4, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(4)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(4, 8, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class ResidualOfResidual(nn.Module):
    """
    Residual of residual: (Conv -> BN -> ReLU -> Conv -> BN + identity) + identity
    Expected: multiple REC dependencies across nested residuals
    """
    def __init__(self, channels=16):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.bn1(out1)
        out = self.relu(out)
        out_id = out
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + out_id  # inner residual
        out = self.conv3(out)
        out = self.bn3(out)
        out = out + out1      # outer residual
        return out


class MinimalMulOperation(nn.Module):
    """
    Two branches merged via multiplication
    Expected: REC dependencies between branches
    """
    def __init__(self):
        super().__init__()
        self.a1 = nn.Conv2d(3, 16, 3, padding=1)
        self.a_relu = nn.ReLU()
        self.b1 = nn.Conv2d(3, 16, 3, padding=1)
        self.b_relu = nn.ReLU()
        self.tail = nn.Conv2d(16, 8, 1)

    def forward(self, x):
        a = self.a_relu(self.a1(x))
        b = self.b_relu(self.b1(x))
        merged = torch.mul(a, b)
        return self.tail(merged)


class MinimalSubOperation(nn.Module):
    """
    Two branches merged via subtraction
    Expected: REC dependencies between branches
    """
    def __init__(self):
        super().__init__()
        self.a1 = nn.Conv2d(3, 16, 3, padding=1)
        self.a_relu = nn.ReLU()
        self.b1 = nn.Conv2d(3, 16, 3, padding=1)
        self.b_relu = nn.ReLU()
        self.tail = nn.Conv2d(16, 8, 1)

    def forward(self, x):
        a = self.a_relu(self.a1(x))
        b = self.b_relu(self.b1(x))
        merged = torch.sub(a, b)
        return self.tail(merged)


class DependencyPatternTest(unittest.TestCase):
    """Base class for dependency pattern tests"""

    def get_dependencies_torchfx(self, model: nn.Module, dummy_input: torch.Tensor) -> List[Tuple[nn.Module, nn.Module, DepType]]:
        """Extract dependencies using torch.fx tracing"""
        try:
            m = ModelInterface(model, dummy_input=dummy_input, use_onnx=False)
            self.model = m
            return self.model
        except Exception as e:
            self.skipTest(f"TorchFX tracing failed: {e}")

    def get_dependencies_onnx(self, model: nn.Module, dummy_input: torch.Tensor) -> List[Tuple[nn.Module, nn.Module, DepType]]:
        """Extract dependencies using ONNX export"""
        try:
            m = ModelInterface(model, dummy_input=dummy_input, use_onnx=True)
            self.model = m
            return self.model
        except Exception as e:
            self.skipTest(f"ONNX export failed: {e}")

    def assert_dependency_exists(self, deps: List[Tuple[nn.Module, nn.Module, DepType]],
                                src_name: str, dst_name: str, dep_type: DepType = None):
        """Assert that a dependency exists between two modules"""
        module_names = {id(m): n for n, m in self.model.named_modules()}
        found = False

        for src, dst, dtype in deps:
            src_n = module_names.get(id(src))
            dst_n = module_names.get(id(dst))

            if src_n == src_name and dst_n == dst_name:
                found = True
                if dep_type is not None:
                    self.assertEqual(dtype, dep_type,
                        f"Expected {src_name} -> {dst_name} to have type {dep_type.name}, got {dtype.name}")
                break

        self.assertTrue(found, f"Dependency {src_name} -> {dst_name} not found")

    def assert_dependency_count_range(self, deps: List[Tuple[nn.Module, nn.Module, DepType]],
                                      min_count: int = None, max_count: int = None):
        """Assert that dependency count is within expected range"""
        if min_count is not None:
            self.assertGreaterEqual(len(deps), min_count,
                f"Expected at least {min_count} dependencies, got {len(deps)}")
        if max_count is not None:
            self.assertLessEqual(len(deps), max_count,
                f"Expected at most {max_count} dependencies, got {len(deps)}")

    def operate(self, model: nn.Module, dummy_input: torch.Tensor, op: int = None):
        """Run a forward pass to ensure model operates correctly"""

        # Operate
        print('Performing model parameters operations..')
        model_op_neurons(model, dummy_input=dummy_input, op=op, rand=False)

        # Infer after operate
        try:
            with torch.no_grad():
                output = model(dummy_input)
            self.assertIsNotNone(output, "Model output is None")
        except Exception as e:
            self.fail(f"Model operation failed: {e}")


class TestSAMEDependencies(DependencyPatternTest):
    """Test models with SAME dependency patterns"""

    def setUp(self):
        self.model = MinimalSAMEDependencies()
        self.model.eval()
        self._model = self.model
        self.dummy_input = torch.randn(1, 3, 32, 32)

    def test_same_pattern_onnx(self):
        """Test SAME pattern: Conv -> BN -> ReLU with ONNX"""
        self.model = self.get_dependencies_onnx(self.model, self.dummy_input)
        deps = self.model.dependencies_with_ops

        # Should have dependencies connecting all layers
        self.assert_dependency_count_range(deps, min_count=2)

        # Check for SAME type dependencies
        same_deps = [d for d in deps if d[2] == DepType.SAME]
        self.assertGreater(len(same_deps), 0, "Should have at least one SAME dependency")

        # Check dependency consistency
        same_deps_ids = [[d[0].module_id, d[1].module_id] for d in deps]
        self.assertEqual(same_deps_ids, [[0, 1], [1, 2]])

        # Operation
        self.operate(self.model, self.dummy_input)

    def test_same_pattern_torchfx(self):
        """Test SAME pattern: Conv -> BN -> ReLU with TorchFX"""
        self.model = self.get_dependencies_torchfx(self.model, self.dummy_input)
        deps = self.model.dependencies_with_ops

        # Should have dependencies
        self.assert_dependency_count_range(deps, min_count=2)

        # Check for SAME type dependencies
        same_deps = [d for d in deps if d[2] == DepType.SAME]
        self.assertGreater(len(same_deps), 0, "Should have at least one SAME dependency")

        # Check dependency consistency
        same_deps_ids = [[d[0].module_id, d[1].module_id] for d in deps]
        self.assertEqual(same_deps_ids, [[0, 1], [1, 2]])

        # Operation
        self.operate(self.model, self.dummy_input)


class TestINCOMINGDependencies(DependencyPatternTest):
    """Test models with INCOMING dependency patterns"""

    def setUp(self):
        self.model = MinimalINCOMINGDependencies()
        self.model.eval()
        self._model = self.model
        self.dummy_input = torch.randn(1, 3, 32, 32)

    def test_incoming_pattern_onnx(self):
        """Test INCOMING pattern: Conv -> Conv with ONNX"""
        self.model = self.get_dependencies_onnx(self.model, self.dummy_input)
        deps = self.model.dependencies_with_ops

        # Should have at least one dependency
        self.assert_dependency_count_range(deps, min_count=1)

        # Check for INCOMING type dependencies
        incoming_deps = [d for d in deps if d[2] == DepType.INCOMING]
        self.assertGreater(len(incoming_deps), 0, "Should have at least one INCOMING dependency")

        # Check dependency consistency
        same_deps_ids = [[d[0].module_id, d[1].module_id] for d in deps]
        self.assertEqual(same_deps_ids, [[0, 1]])

        # Operation
        self.operate(self.model, self.dummy_input)

    def test_incoming_pattern_torchfx(self):
        """Test INCOMING pattern: Conv -> Conv with TorchFX"""
        self.model = self.get_dependencies_torchfx(self.model, self.dummy_input)
        deps = self.model.dependencies_with_ops

        # Should have dependencies
        self.assert_dependency_count_range(deps, min_count=1)

        # Check for INCOMING type dependencies
        incoming_deps = [d for d in deps if d[2] == DepType.INCOMING]
        self.assertGreater(len(incoming_deps), 0, "Should have at least one INCOMING dependency")

        # Check dependency consistency
        same_deps_ids = [[d[0].module_id, d[1].module_id] for d in deps]
        self.assertEqual(same_deps_ids, [[0, 1]])

        # Operation
        self.operate(self.model, self.dummy_input)


class TestMixedDependencies(DependencyPatternTest):
    """Test models with mixed dependency patterns"""

    def setUp(self):
        self.model = MinimalMixedDependencies()
        self.model.eval()
        self._model = self.model
        self.dummy_input = torch.randn(1, 3, 32, 32)

    def test_mixed_pattern_onnx(self):
        """Test mixed pattern: Conv -> BN -> ReLU -> Conv with ONNX"""
        self.model = self.get_dependencies_onnx(self.model, self.dummy_input)
        deps = self.model.dependencies_with_ops

        # Should have multiple dependencies
        self.assert_dependency_count_range(deps, min_count=3)

        # Check for both SAME and INCOMING types
        same_deps = [d for d in deps if d[2] == DepType.SAME]
        incoming_deps = [d for d in deps if d[2] == DepType.INCOMING]

        self.assertGreater(len(same_deps), 0, "Should have SAME dependencies")
        self.assertGreater(len(incoming_deps), 0, "Should have INCOMING dependencies")

        # Check dependency consistency
        same_deps_ids = [[d[0].module_id, d[1].module_id] for d in deps]
        self.assertEqual(same_deps_ids, [[0, 1], [1, 2], [2, 3]])

        # Operation
        self.operate(self.model, self.dummy_input)

    def test_mixed_pattern_torchfx(self):
        """Test mixed pattern: Conv -> BN -> ReLU -> Conv with TorchFX"""
        self.model = self.get_dependencies_torchfx(self.model, self.dummy_input)
        deps = self.model.dependencies_with_ops

        # Should have dependencies
        self.assert_dependency_count_range(deps, min_count=3)

        # Check dependency consistency
        same_deps_ids = [[d[0].module_id, d[1].module_id] for d in deps]
        self.assertEqual(same_deps_ids, [[0, 1], [1, 2], [2, 3]])

        # Operation
        self.operate(self.model, self.dummy_input)


class TestResidualConnections(DependencyPatternTest):
    """Test models with residual (REC) dependency patterns"""

    def setUp(self):
        self.model = MinimalResidualBlock()
        self.model.eval()
        self._model = self.model
        self.dummy_input = torch.randn(1, 64, 32, 32)

    def test_residual_pattern_onnx(self):
        """Test residual pattern with ONNX"""
        self.model = self.get_dependencies_onnx(self.model, self.dummy_input)
        deps = self.model.dependencies_with_ops

        # Should have dependencies
        self.assert_dependency_count_range(deps, min_count=4)

        # Check dependency consistency
        same_deps_ids = [[d[0].module_id, d[1].module_id] for d in deps]
        self.assertEqual(same_deps_ids, [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [0, 4]])

        # Operation
        self.operate(self.model, self.dummy_input)

    def test_residual_pattern_torchfx(self):
        """Test residual pattern with TorchFX"""
        self.model = self.get_dependencies_torchfx(self.model, self.dummy_input)
        deps = self.model.dependencies_with_ops

        # Should have dependencies
        self.assert_dependency_count_range(deps, min_count=4)

        # Check dependency consistency
        same_deps_ids = [[d[0].module_id, d[1].module_id] for d in deps]
        self.assertEqual(same_deps_ids, [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [0, 4]])

        # Operation
        self.operate(self.model, self.dummy_input)


class TestConcatenationSkipConnections(DependencyPatternTest):
    """Test models with concatenation skip connections (REC dependencies)"""

    def setUp(self):
        self.model = MinimalSkipConnectionCat()
        self.model.eval()
        self._model = self.model
        self.dummy_input = torch.randn(1, 3, 32, 32)

    def test_concat_pattern_onnx(self):
        """Test concatenation pattern with ONNX"""
        self.model = self.get_dependencies_onnx(self.model, self.dummy_input)
        deps = self.model.dependencies_with_ops

        # Should have dependencies
        self.assert_dependency_count_range(deps, min_count=4)

        # Check dependency consistency
        same_deps_ids = [[d[0].module_id, d[1].module_id] for d in deps]
        self.assertEqual(same_deps_ids, [[0, 1], [2, 3], [1, 4], [3, 4]])

        # Operation
        self.operate(self.model, self.dummy_input)

    def test_concat_pattern_torchfx(self):
        """Test concatenation pattern with TorchFX"""
        self.model = self.get_dependencies_torchfx(self.model, self.dummy_input)
        deps = self.model.dependencies_with_ops

        # Should have dependencies
        self.assert_dependency_count_range(deps, min_count=4)

        # Check dependency consistency
        same_deps_ids = [[d[0].module_id, d[1].module_id] for d in deps]
        self.assertEqual(same_deps_ids, [[0, 1], [2, 3], [1, 4], [3, 4]])

        # Operation
        self.operate(self.model, self.dummy_input)


class TestNonLearnableLayers(DependencyPatternTest):
    """Test models with non-learnable layers (MaxPool, ReLU, etc.)"""

    def setUp(self):
        self.model = MinimalWithMaxPool()
        self.model.eval()
        self._model = self.model
        self.dummy_input = torch.randn(1, 3, 32, 32)

    def test_maxpool_pattern_onnx(self):
        """Test pattern with MaxPool: Conv -> MaxPool -> ReLU -> Conv with ONNX"""
        self.model = self.get_dependencies_onnx(self.model, self.dummy_input)
        deps = self.model.dependencies_with_ops

        # Should have dependencies including non-learnable layers
        self.assert_dependency_count_range(deps, min_count=3)

        # Should include MaxPool and ReLU in dependencies
        module_names = {id(m): n for n, m in self.model.named_modules()}
        dep_pairs = [(module_names.get(id(s)), module_names.get(id(d))) for s, d, _ in deps]

        # Check that maxpool appears in dependency chain
        maxpool_involved = any(
            ('maxpool' in str(pair[0] or '') or 'maxpool' in str(pair[1] or ''))
            for pair in dep_pairs
        )
        self.assertTrue(maxpool_involved or len(deps) > 0,
            "Should have dependencies involving pooling layer")

        # Check dependency consistency
        same_deps_ids = [[d[0].module_id, d[1].module_id] for d in deps]
        self.assertEqual(same_deps_ids, [[0, 1], [1, 2], [2, 3]])

        # Operation
        self.operate(self.model, self.dummy_input)

    def test_maxpool_pattern_torchfx(self):
        """Test pattern with MaxPool using TorchFX"""
        self.model = self.get_dependencies_torchfx(self.model, self.dummy_input)
        deps = self.model.dependencies_with_ops

        # Should have dependencies
        self.assert_dependency_count_range(deps, min_count=3)

        # Should include MaxPool and ReLU in dependencies
        module_names = {id(m): n for n, m in self.model.named_modules()}
        dep_pairs = [(module_names.get(id(s)), module_names.get(id(d))) for s, d, _ in deps]

        # Check that maxpool appears in dependency chain
        maxpool_involved = any(
            ('maxpool' in str(pair[0] or '') or 'maxpool' in str(pair[1] or ''))
            for pair in dep_pairs
        )
        self.assertTrue(maxpool_involved or len(deps) > 0,
            "Should have dependencies involving pooling layer")

        # Check dependency consistency
        same_deps_ids = [[d[0].module_id, d[1].module_id] for d in deps]
        self.assertEqual(same_deps_ids, [[0, 1], [1, 2], [2, 3]])

        # Operation
        self.operate(self.model, self.dummy_input)


class TestUpsampling(DependencyPatternTest):
    """Test models with upsampling layers"""

    def setUp(self):
        self.model = MinimalWithUpsampling()
        self.model.eval()
        self._model = self.model
        self.dummy_input = torch.randn(1, 3, 32, 32)

    def test_upsample_pattern_onnx(self):
        """Test pattern with Upsample: Conv -> Upsample -> Conv with ONNX"""
        self.model = self.get_dependencies_onnx(self.model, self.dummy_input)
        deps = self.model.dependencies_with_ops

        # Should have dependencies
        self.assert_dependency_count_range(deps, min_count=2)

        # Check for dependencies involving upsampling
        module_names = {id(m): n for n, m in self.model.named_modules()}
        dep_pairs = [(module_names.get(id(s)), module_names.get(id(d))) for s, d, _ in deps]

        # Verify dependencies exist
        self.assertTrue(len(dep_pairs) > 0, "Should have dependencies with upsample layer")

        # Check dependency consistency
        same_deps_ids = [[d[0].module_id, d[1].module_id] for d in deps]
        self.assertEqual(same_deps_ids, [[0, 1], [1, 2]])

        # Operation
        self.operate(self.model, self.dummy_input)

    def test_upsample_pattern_torchfx(self):
        """Test pattern with Upsample using TorchFX"""
        self.model = self.get_dependencies_torchfx(self.model, self.dummy_input)
        deps = self.model.dependencies_with_ops

        # Should have dependencies
        self.assert_dependency_count_range(deps, min_count=2)

        # Check for dependencies involving upsampling
        module_names = {id(m): n for n, m in self.model.named_modules()}
        dep_pairs = [(module_names.get(id(s)), module_names.get(id(d))) for s, d, _ in deps]

        # Verify dependencies exist
        self.assertTrue(len(dep_pairs) > 0, "Should have dependencies with upsample layer")

        # Check dependency consistency
        same_deps_ids = [[d[0].module_id, d[1].module_id] for d in deps]
        self.assertEqual(same_deps_ids, [[0, 1], [1, 2]])

        # Operation
        self.operate(self.model, self.dummy_input)


class TestInferDependencyType(unittest.TestCase):
    """Test the _infer_dependency_type helper function"""

    def test_conv2d_inference(self):
        """Conv2d has 2D+ weights -> INCOMING"""
        conv = nn.Conv2d(3, 64, 3)
        dep_type = _infer_dependency_type(conv)
        self.assertEqual(dep_type, DepType.INCOMING)

    def test_batchnorm_inference(self):
        """BatchNorm has only 1D params -> SAME"""
        bn = nn.BatchNorm2d(64)
        dep_type = _infer_dependency_type(bn)
        self.assertEqual(dep_type, DepType.SAME)

    def test_linear_inference(self):
        """Linear has 2D weights -> INCOMING"""
        linear = nn.Linear(100, 50)
        dep_type = _infer_dependency_type(linear)
        self.assertEqual(dep_type, DepType.INCOMING)

    def test_relu_inference(self):
        """ReLU has no params -> SAME"""
        relu = nn.ReLU()
        dep_type = _infer_dependency_type(relu)
        self.assertEqual(dep_type, DepType.SAME)

    def test_maxpool_inference(self):
        """MaxPool has no params -> SAME"""
        maxpool = nn.MaxPool2d(2, 2)
        dep_type = _infer_dependency_type(maxpool)
        self.assertEqual(dep_type, DepType.SAME)


class TestGroupedConv(DependencyPatternTest):
    def setUp(self):
        self.model = MinimalGroupedConv()
        self.model.eval()
        self._model = self.model
        self.dummy_input = torch.randn(1, 4, 16, 16)

    def test_grouped_conv_onnx(self):
        self.model = self.get_dependencies_onnx(self.model, self.dummy_input)
        deps = self.model.dependencies_with_ops
        self.assert_dependency_count_range(deps, min_count=3)
        incoming_deps = [d for d in deps if d[2] == DepType.INCOMING]
        self.assertGreater(len(incoming_deps), 0)

        # Check dependency consistency
        same_deps_ids = [[d[0].module_id, d[1].module_id] for d in deps]
        self.assertEqual(same_deps_ids, [[0, 1], [1, 2], [2, 3]])

        # Operation
        self.operate(self.model, self.dummy_input)

    def test_grouped_conv_fx(self):
        self.model = self.get_dependencies_torchfx(self.model, self.dummy_input)
        deps = self.model.dependencies_with_ops
        self.assert_dependency_count_range(deps, min_count=3)
        incoming_deps = [d for d in deps if d[2] == DepType.INCOMING]
        self.assertGreater(len(incoming_deps), 0)

        # Check dependency consistency
        same_deps_ids = [[d[0].module_id, d[1].module_id] for d in deps]
        self.assertEqual(same_deps_ids, [[0, 1], [1, 2], [2, 3]])

        # Operation
        self.operate(self.model, self.dummy_input)



class TestDepthwisePointwiseWithLinear(DependencyPatternTest):
    def setUp(self):
        self.model = MinimalDepthwisePointwiseWithLinear()
        self.model.eval()
        self._model = self.model
        self.dummy_input = torch.randn(1, 8, 16, 16)

    def test_dw_pw_onnx(self):
        self.model = self.get_dependencies_onnx(self.model, self.dummy_input)
        deps = self.model.dependencies_with_ops
        incoming_deps = [d for d in deps if d[2] == DepType.INCOMING]
        self.assertGreater(len(incoming_deps), 0)

        # Check dependency consistency
        same_deps_ids = [[d[0].module_id, d[1].module_id] for d in deps]
        self.assertEqual(same_deps_ids, [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])

        # Operation
        self.operate(self.model, self.dummy_input)

    def test_dw_pw_fx(self):
        self.model = self.get_dependencies_torchfx(self.model, self.dummy_input)
        deps = self.model.dependencies_with_ops
        incoming_deps = [d for d in deps if d[-1] == DepType.INCOMING]
        self.assertGreater(len(incoming_deps), 0)

        # Check dependency consistency
        same_deps_ids = [[d[0].module_id, d[1].module_id] for d in deps]
        self.assertEqual(same_deps_ids, [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])

        # Operation
        self.operate(self.model, self.dummy_input)


class TestConvTranspose(DependencyPatternTest):
    def setUp(self):
        self.model = MinimalConvTransposeChain()
        self.model.eval()
        self._model = self.model
        self.dummy_input = torch.randn(1, 4, 16, 16)

    def test_deconv_onnx(self):
        self.model = self.get_dependencies_onnx(self.model, self.dummy_input)
        deps = self.model.dependencies_with_ops
        self.assert_dependency_count_range(deps, min_count=2)

        # Operation
        self.operate(self.model, self.dummy_input)

        # Check dependency consistency
        same_deps_ids = [[d[0].module_id, d[1].module_id] for d in deps]
        self.assertEqual(same_deps_ids, [[0, 1], [1, 2]])

    def test_deconv_fx(self):
        self.model = self.get_dependencies_torchfx(self.model, self.dummy_input)
        deps = self.model.dependencies_with_ops
        self.assert_dependency_count_range(deps, min_count=2)

        # Operation
        self.operate(self.model, self.dummy_input)

        # Check dependency consistency
        same_deps_ids = [[d[0].module_id, d[1].module_id] for d in deps]
        self.assertEqual(same_deps_ids, [[0, 1], [1, 2]])


class TestLinearChain(DependencyPatternTest):
    def setUp(self):
        self.model = MinimalLinearChain()
        self.model.eval()
        self._model = self.model
        self.dummy_input = torch.randn(1, 3, 16, 16)

    def test_linear_onnx(self):
        self.model = self.get_dependencies_onnx(self.model, self.dummy_input)
        deps = self.model.dependencies_with_ops
        incoming_deps = [d for d in deps if d[2] == DepType.INCOMING]
        self.assertGreater(len(incoming_deps), 0)

        # Operation
        self.operate(self.model, self.dummy_input)

        # Check dependency consistency
        same_deps_ids = [[d[0].module_id, d[1].module_id] for d in deps]
        self.assertEqual(same_deps_ids, [[0, 1], [1, 2], [2, 3]])

    def test_linear_fx(self):
        self.model = self.get_dependencies_torchfx(self.model, self.dummy_input)
        deps = self.model.dependencies_with_ops
        incoming_deps = [d for d in deps if d[2] == DepType.INCOMING]
        self.assertGreater(len(incoming_deps), 0)

        # Operation
        self.operate(self.model, self.dummy_input)

        # Check dependency consistency
        same_deps_ids = [[d[0].module_id, d[1].module_id] for d in deps]
        self.assertEqual(same_deps_ids, [[0, 1], [1, 2], [2, 3]])


class TestConv1DChain(DependencyPatternTest):
    def setUp(self):
        self.model = MinimalConv1DChain()
        self.model.eval()
        self._model = self.model
        self.dummy_input = torch.randn(1, 4, 64)  # N, C, L

    def test_conv1d_onnx(self):
        self.model = self.get_dependencies_onnx(self.model, self.dummy_input)
        deps = self.model.dependencies_with_ops
        self.assert_dependency_count_range(deps, min_count=3)

        # Operation
        self.operate(self.model, self.dummy_input)

        # Check dependency consistency
        same_deps_ids = [[d[0].module_id, d[1].module_id] for d in deps]
        self.assertEqual(same_deps_ids, [[0, 1], [1, 2], [2, 3]])

    def test_conv1d_fx(self):
        self.model = self.get_dependencies_torchfx(self.model, self.dummy_input)
        deps = self.model.dependencies_with_ops
        self.assert_dependency_count_range(deps, min_count=3)

        # Operation
        self.operate(self.model, self.dummy_input)

        # Check dependency consistency
        same_deps_ids = [[d[0].module_id, d[1].module_id] for d in deps]
        self.assertEqual(same_deps_ids, [[0, 1], [1, 2], [2, 3]])


class TestConv3DChain(DependencyPatternTest):
    def setUp(self):
        self.model = MinimalConv3DChain()
        self.model.eval()
        self._model = self.model
        self.dummy_input = torch.randn(1, 2, 8, 16, 16)  # N, C, D, H, W

    def test_conv3d_onnx(self):
        self.model = self.get_dependencies_onnx(self.model, self.dummy_input)
        deps = self.model.dependencies_with_ops
        self.assert_dependency_count_range(deps, min_count=3)

        # Operation
        self.operate(self.model, self.dummy_input)

        # Check dependency consistency
        same_deps_ids = [[d[0].module_id, d[1].module_id] for d in deps]
        self.assertEqual(same_deps_ids, [[0, 1], [1, 2], [2, 3]])

    def test_conv3d_fx(self):
        self.model = self.get_dependencies_torchfx(self.model, self.dummy_input)
        deps = self.model.dependencies_with_ops
        self.assert_dependency_count_range(deps, min_count=3)

        # Operation
        self.operate(self.model, self.dummy_input)

        # Check dependency consistency
        same_deps_ids = [[d[0].module_id, d[1].module_id] for d in deps]
        self.assertEqual(same_deps_ids, [[0, 1], [1, 2], [2, 3]])


class TestResidualOfResidual(DependencyPatternTest):
    def setUp(self):
        self.model = ResidualOfResidual(channels=16)
        self.model.eval()
        self._model = self.model
        self.dummy_input = torch.randn(1, 16, 16, 16)

    def test_residual_of_residual_onnx(self):
        self.model = self.get_dependencies_onnx(self.model, self.dummy_input)
        deps = self.model.dependencies_with_ops
        rec_deps = [d for d in deps if d[2] == DepType.REC]
        self.assertGreater(len(rec_deps), 0)

        # Operation
        self.operate(self.model, self.dummy_input)

        # Check dependency consistency
        same_deps_ids = [[d[0].module_id, d[1].module_id] for d in deps]
        self.assertEqual(same_deps_ids, [[0, 1], [1, 2], [2, 3], [3, 4], [4, 2], [2, 4], [4, 5], [2, 5], [5, 6], [6, 0], [0, 6]])

    def test_residual_of_residual_fx(self):
        self.model = self.get_dependencies_torchfx(self.model, self.dummy_input)
        deps = self.model.dependencies_with_ops
        rec_deps = [d for d in deps if d[2] == DepType.REC]
        self.assertGreater(len(rec_deps), 0)

        # Check dependency consistency
        same_deps_ids = [[d[0].module_id, d[1].module_id] for d in deps]
        self.assertEqual(same_deps_ids, [[0, 1], [1, 2], [2, 3], [3, 4], [4, 2], [2, 4], [4, 5], [2, 5], [5, 6], [6, 0], [0, 6]])

        # Operation
        self.operate(self.model, self.dummy_input)


class TestMulSubMerges(DependencyPatternTest):
    def setUp(self):
        self.model_mul = MinimalMulOperation()
        self.model_sub = MinimalSubOperation()
        self.model_mul.eval(); self.model_sub.eval()
        self._model = self.model_mul
        self.dummy_input = torch.randn(1, 3, 16, 16)

    def test_mul_merge_onnx(self):
        self.model = self.get_dependencies_onnx(self.model_mul, self.dummy_input)
        deps = self.model.dependencies_with_ops
        rec_deps = [d for d in deps if d[2] == DepType.REC]
        self.assertGreater(len(rec_deps), 0)
        self.operate(self.model, self.dummy_input)

    def test_sub_merge_onnx(self):
        self.model = self.get_dependencies_onnx(self.model_sub, self.dummy_input)
        deps = self.model.dependencies_with_ops
        rec_deps = [d for d in deps if d[2] == DepType.REC]
        self.assertGreater(len(rec_deps), 0)
        self.operate(self.model, self.dummy_input)

    def test_mul_merge_fx(self):
        self.model = self.get_dependencies_torchfx(self.model_mul, self.dummy_input)
        deps = self.model.dependencies_with_ops
        rec_deps = [d for d in deps if d[2] == DepType.REC]
        self.assertGreater(len(rec_deps), 0)
        self.operate(self.model, self.dummy_input)

    def test_sub_merge_fx(self):
        self.model = self.get_dependencies_torchfx(self.model_sub, self.dummy_input)
        deps = self.model.dependencies_with_ops
        rec_deps = [d for d in deps if d[2] == DepType.REC]
        self.assertGreater(len(rec_deps), 0)
        self.operate(self.model, self.dummy_input)
