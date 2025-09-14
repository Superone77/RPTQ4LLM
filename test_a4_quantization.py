#!/usr/bin/env python3
"""
Unit tests for A4 quantization with reordering functionality.
"""

import torch
import torch.nn as nn
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quantize.reorder_blocking import (
    compute_perm,
    apply_permutation_to_tensor,
    validate_kernel_block_size,
    test_perm_equiv,
    test_kernel_validation
)
from quantize.a4_quantizer import A4Quantizer
from quantize.perm_absorption import (
    absorb_linear_permutation,
    absorb_layernorm_permutation,
    validate_permutation_equivalence
)


def test_permutation_computation():
    """Test permutation computation with different parameters."""
    print("Testing permutation computation...")
    
    # Test basic permutation computation
    torch.manual_seed(42)
    max_abs = torch.randn(64) * 10
    P, blocks = compute_perm(max_abs, B=32, headwise=False, local_swap=0)
    
    assert P.shape == (64,), f"Expected permutation shape (64,), got {P.shape}"
    assert torch.all(torch.sort(P)[0] == torch.arange(64)), "Permutation should contain all indices"
    assert blocks['num_blocks'] == 2, f"Expected 2 blocks, got {blocks['num_blocks']}"
    assert blocks['block_size'] == 32, f"Expected block size 32, got {blocks['block_size']}"
    
    print("✓ Basic permutation computation test passed")
    
    # Test headwise permutation
    P_headwise, blocks_headwise = compute_perm(
        max_abs, B=16, headwise=True, local_swap=2
    )
    assert P_headwise.shape == (64,), f"Expected permutation shape (64,), got {P_headwise.shape}"
    assert torch.all(torch.sort(P_headwise)[0] == torch.arange(64)), "Headwise permutation should contain all indices"
    
    print("✓ Headwise permutation computation test passed")


def test_permutation_application():
    """Test applying permutations to tensors."""
    print("Testing permutation application...")
    
    torch.manual_seed(42)
    x = torch.randn(2, 64, 32)
    P = torch.randperm(64)
    
    # Test permutation application
    x_perm = apply_permutation_to_tensor(x, P, dim=-1)
    assert x_perm.shape == x.shape, f"Shape mismatch: {x_perm.shape} vs {x.shape}"
    
    # Test that permutation is correctly applied
    for i in range(64):
        assert torch.allclose(x[..., i], x_perm[..., P[i]]), f"Permutation not correctly applied at index {i}"
    
    print("✓ Permutation application test passed")


def test_a4_quantizer():
    """Test A4 quantizer functionality."""
    print("Testing A4 quantizer...")
    
    # Test MXFP4 quantizer
    quantizer_mxfp4 = A4Quantizer(
        aformat="mxfp4",
        mx_block_size=32,
        mx_clip_quantile=0.999,
        stochastic_rounding=False
    )
    
    torch.manual_seed(42)
    x = torch.randn(2, 64, 32)
    
    # Test calibration mode
    quantizer_mxfp4.set_calibration_mode()
    x_calib = quantizer_mxfp4(x)
    assert torch.allclose(x, x_calib), "Calibration mode should return input unchanged"
    
    # Test permutation computation
    quantizer_mxfp4.compute_permutation()
    assert quantizer_mxfp4.permutation is not None, "Permutation should be computed"
    
    # Test eval mode
    quantizer_mxfp4.set_eval_mode()
    x_quant = quantizer_mxfp4(x)
    assert x_quant.shape == x.shape, f"Quantized output shape mismatch: {x_quant.shape} vs {x.shape}"
    
    print("✓ A4 quantizer test passed")


def test_linear_permutation_absorption():
    """Test permutation absorption into Linear modules."""
    print("Testing Linear permutation absorption...")
    
    torch.manual_seed(42)
    
    # Create a Linear module
    linear = nn.Linear(64, 32)
    original_weight = linear.weight.data.clone()
    original_bias = linear.bias.data.clone()
    
    # Create permutation
    P = torch.randperm(64)
    
    # Test input permutation absorption
    class MockArgs:
        pass
    args = MockArgs()
    
    absorb_linear_permutation(linear, P, "test_layer", args)
    
    # Check that weights were permuted
    assert not torch.allclose(linear.weight.data, original_weight), "Weights should be permuted"
    assert torch.allclose(linear.bias.data, original_bias), "Bias should not change for input permutation"
    
    print("✓ Linear permutation absorption test passed")


def test_layernorm_permutation_absorption():
    """Test permutation absorption into LayerNorm modules."""
    print("Testing LayerNorm permutation absorption...")
    
    torch.manual_seed(42)
    
    # Create a LayerNorm module
    layernorm = nn.LayerNorm(64)
    original_weight = layernorm.weight.data.clone()
    original_bias = layernorm.bias.data.clone()
    
    # Create permutation
    P = torch.randperm(64)
    
    # Test permutation absorption
    class MockArgs:
        pass
    args = MockArgs()
    
    absorb_layernorm_permutation(layernorm, P, "test_layernorm", args)
    
    # Check that parameters were permuted
    assert not torch.allclose(layernorm.weight.data, original_weight), "Weight should be permuted"
    assert not torch.allclose(layernorm.bias.data, original_bias), "Bias should be permuted"
    
    print("✓ LayerNorm permutation absorption test passed")


def test_kernel_validation():
    """Test kernel block size validation."""
    print("Testing kernel validation...")
    
    # Test MXFP4 validation
    assert validate_kernel_block_size(32, 'mxfp4') == 32
    assert validate_kernel_block_size(16, 'mxfp4') == 32  # Should warn and return 32
    
    # Test NVFP4 validation
    assert validate_kernel_block_size(16, 'nvfp4') == 16
    assert validate_kernel_block_size(32, 'nvfp4') == 16  # Should warn and return 16
    
    print("✓ Kernel validation test passed")


def test_permutation_equivalence():
    """Test permutation equivalence validation."""
    print("Testing permutation equivalence...")
    
    # Run the built-in permutation equivalence test
    test_perm_equiv()
    
    print("✓ Permutation equivalence test passed")


def test_end_to_end_quantization():
    """Test end-to-end A4 quantization pipeline."""
    print("Testing end-to-end A4 quantization...")
    
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(64, 32)
            self.layernorm = nn.LayerNorm(32)
            self.linear2 = nn.Linear(32, 16)
        
        def forward(self, x):
            x = self.linear1(x)
            x = self.layernorm(x)
            x = self.linear2(x)
            return x
    
    model = SimpleModel()
    torch.manual_seed(42)
    x = torch.randn(2, 64)
    
    # Test with A4 quantization
    quantizer = A4Quantizer(
        aformat="mxfp4",
        mx_block_size=32,
        stochastic_rounding=False
    )
    
    # Calibration
    quantizer.set_calibration_mode()
    for _ in range(5):
        _ = quantizer(x)
    
    # Compute permutation
    quantizer.compute_permutation()
    
    # Evaluation
    quantizer.set_eval_mode()
    x_quant = quantizer(x)
    
    assert x_quant.shape == x.shape, f"Quantized output shape mismatch: {x_quant.shape} vs {x.shape}"
    assert torch.isfinite(x_quant).all(), "Quantized output should be finite"
    
    print("✓ End-to-end A4 quantization test passed")


def run_all_tests():
    """Run all unit tests."""
    print("Running A4 quantization unit tests...\n")
    
    try:
        test_permutation_computation()
        test_permutation_application()
        test_a4_quantizer()
        test_linear_permutation_absorption()
        test_layernorm_permutation_absorption()
        test_kernel_validation()
        test_permutation_equivalence()
        test_end_to_end_quantization()
        
        print("\n🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
