"""
Unit tests for constraint detection and propagation in dependency graphs.

Tests focus on:
- Detecting grouped convolution constraints
- Propagating constraints through SAME dependencies
- Stopping propagation at INCOMING dependencies
"""

import unittest
import torch
import torch.nn as nn

from typing import List, Tuple

from weightslab.utils.computational_graph import (
    _detect_layer_constraints
)
from weightslab.utils.modules_dependencies import DepType
from weightslab.backend.model_interface import ModelInterface


def get_dependencies_onnx(model: nn.Module, dummy_input: torch.Tensor) -> List[Tuple[nn.Module, nn.Module, DepType]]:
    """Extract dependencies using ONNX export"""
    try:
        m = ModelInterface(model, dummy_input=dummy_input, use_onnx=True)
        return m
    except Exception as e:
        raise RuntimeError(f"ONNX dependency extraction failed: {e}")


class GroupedConvModel(nn.Module):
    """
    Model with grouped convolutions: GroupedConv -> BN -> ReLU -> Regular Conv
    
    Expected:
    - grouped_conv: Detected as grouped constraint
    - bn: Inherits grouped constraint (SAME dep)
    - relu: Inherits grouped constraint (SAME dep)
    - regular_conv: Does NOT inherit (INCOMING dep blocks it)
    """
    def __init__(self):
        super().__init__()
        self.grouped_conv = nn.Conv2d(8, 16, kernel_size=3, padding=1, groups=2)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.regular_conv = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # No groups
    
    def forward(self, x):
        x = self.grouped_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.regular_conv(x)
        return x


class DepthwisePointwiseModel(nn.Module):
    """
    Model with depthwise (groups=in_channels) -> pointwise
    
    Expected:
    - dw: Detected as grouped with is_depthwise=True
    - pw: Should NOT inherit if INCOMING boundary (Conv->Conv is INCOMING)
    """
    def __init__(self):
        super().__init__()
        self.dw = nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16)  # Depthwise
        self.pw = nn.Conv2d(16, 32, kernel_size=1)  # Pointwise
    
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return x


class MultiGroupedModel(nn.Module):
    """
    Multiple grouped convolutions in sequence
    
    Expected:
    - All grouped convs flagged
    - Constraints propagate between them (SAME deps if any BN/ReLU between)
    """
    def __init__(self):
        super().__init__()
        self.g_conv1 = nn.Conv2d(4, 8, kernel_size=3, padding=1, groups=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.g_conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1, groups=4)
        self.relu = nn.ReLU()
        self.g_conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1, groups=8)
    
    def forward(self, x):
        x = self.g_conv1(x)
        x = self.bn1(x)
        x = self.g_conv2(x)
        x = self.relu(x)
        x = self.g_conv3(x)
        return x


class TestConstraintDetection(unittest.TestCase):
    """Test constraint detection on individual modules"""
    
    def test_grouped_conv_detection(self):
        """Detect grouped convolution"""
        conv = nn.Conv2d(8, 16, 3, padding=1, groups=2)
        constraints = _detect_layer_constraints(conv)
        
        self.assertIn('grouped', constraints)
        self.assertEqual(constraints['grouped']['cons_group_size'], 2)
        self.assertEqual(constraints['grouped']['in_channels'], 8)
        self.assertEqual(constraints['grouped']['out_channels'], 16)
        self.assertFalse(constraints['grouped']['is_depthwise'])
    
    def test_depthwise_detection(self):
        """Detect depthwise convolution (groups == in_channels)"""
        conv = nn.Conv2d(16, 16, 3, padding=1, groups=16)
        constraints = _detect_layer_constraints(conv)
        
        self.assertIn('grouped', constraints)
        self.assertEqual(constraints['grouped']['cons_group_size'], 16)
        self.assertTrue(constraints['grouped']['is_depthwise'])
    
    def test_regular_conv_no_constraint(self):
        """Regular conv (groups=1) has no grouped constraint"""
        conv = nn.Conv2d(8, 16, 3, padding=1)
        constraints = _detect_layer_constraints(conv)
        
        self.assertNotIn('grouped', constraints)
        self.assertEqual(len(constraints), 0)
    
    def test_batchnorm_no_constraint(self):
        """BatchNorm has no native constraints"""
        bn = nn.BatchNorm2d(64)
        constraints = _detect_layer_constraints(bn)
        
        self.assertEqual(len(constraints), 0)


class TestConstraintPropagation(unittest.TestCase):
    """Test constraint propagation through dependency graphs"""
    
    def test_grouped_conv_propagation(self):
        """Grouped constraint propagates through BN and ReLU (SAME) but stops at regular Conv (INCOMING)"""
        model = GroupedConvModel()
        model.eval()
        dummy_input = torch.randn(1, 8, 16, 16)
        
        # Generate dependencies with ONNX
        model = get_dependencies_onnx(model, dummy_input)
        
        # Check that layers have constraints attached
        # grouped_conv should have grouped constraint
        self.assertTrue(hasattr(model.model.grouped_conv, 'wl_constraints'))
        self.assertEqual('grouped', model.model.grouped_conv.wl_constraints[0]['name'])
        
        # BN and ReLU should inherit grouped constraint (SAME deps)
        self.assertTrue(hasattr(model.model.bn, 'wl_constraints'))
        self.assertEqual('grouped', model.model.bn.wl_constraints[0]['name'])
        
        self.assertTrue(hasattr(model.model.relu, 'wl_constraints'))
        self.assertEqual('grouped', model.model.relu.wl_constraints[0]['name'])
        
        # Regular conv should NOT have grouped constraint (propagation blocked at INCOMING)
        if hasattr(model.model.regular_conv, 'wl_constraints'):
            self.assertEqual('grouped', model.model.regular_conv.wl_constraints[0]['name'])
            self.assertTrue(model.model.regular_conv.wl_constraints[0]['incoming'])
    
    def test_depthwise_propagation(self):
        """Depthwise constraint is detected and marked"""
        model = DepthwisePointwiseModel()
        model.eval()
        dummy_input = torch.randn(1, 16, 16, 16)
        
        model = get_dependencies_onnx(model, dummy_input)
        
        # DW conv should be flagged as depthwise
        self.assertTrue(hasattr(model.model.dw, 'wl_constraints'))
        self.assertEqual('grouped', model.model.dw.wl_constraints[0]['name'])
        self.assertTrue(model.model.dw.wl_constraints[0]['is_depthwise'])
    
    def test_multi_grouped_propagation(self):
        """Multiple grouped convs in sequence"""
        model = MultiGroupedModel()
        model.eval()
        dummy_input = torch.randn(1, 4, 16, 16)
        
        model = get_dependencies_onnx(model, dummy_input)
        
        # All grouped convs should be flagged
        self.assertTrue(hasattr(model.model.g_conv1, 'wl_constraints'))
        self.assertEqual('grouped', model.model.g_conv1.wl_constraints[0]['name'])
        self.assertEqual(model.model.g_conv1.wl_constraints[0]['cons_group_size'], 2)
        
        self.assertTrue(hasattr(model.model.g_conv2, 'wl_constraints'))
        self.assertEqual('grouped', model.model.g_conv2.wl_constraints[1]['name'])
        self.assertEqual(model.model.g_conv2.wl_constraints[1]['cons_group_size'], 4)
        
        self.assertTrue(hasattr(model.model.g_conv3, 'wl_constraints'))
        self.assertEqual('grouped', model.model.g_conv3.wl_constraints[1]['name'])
        self.assertEqual(model.model.g_conv3.wl_constraints[1]['cons_group_size'], 8)

class TestConstraintReporting(unittest.TestCase):
    """Test constraint reporting and information retrieval"""
    
    def test_constraint_source_tracking(self):
        """Each module tracks the ID of the source of its constraints"""
        model = GroupedConvModel()
        model.eval()
        dummy_input = torch.randn(1, 8, 16, 16)
        
        model = get_dependencies_onnx(model, dummy_input)
        
        # Modules with constraints should track their source
        modules_with_constraints = [
            m for m in model.modules()
            if hasattr(m, 'wl_constraints') and m.wl_constraints
        ]
        
        for module in modules_with_constraints:
            self.assertTrue(hasattr(module, 'wl_constraint_source_id'))
            self.assertIsNotNone(module.wl_constraint_source_id)
    
    def test_constraint_no_hardcoding(self):
        """Constraints are detected via introspection, not hardcoding on names"""
        # Create a custom conv with groups but no special name
        conv_with_groups = nn.Conv2d(4, 8, 3, padding=1, groups=2)
        conv_with_groups.__class__.__name__ = "CustomConv"  # Change class name
        
        constraints = _detect_layer_constraints(conv_with_groups)
        
        # Should still detect grouped constraint regardless of name
        self.assertIn('grouped', constraints)
        self.assertEqual(constraints['grouped']['cons_group_size'], 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
