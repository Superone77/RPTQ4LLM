#!/usr/bin/env python3
"""
Demo script showing A4 quantization with reordering functionality.
"""

import torch
import torch.nn as nn
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quantize.a4_quantizer import A4Quantizer
from quantize.reorder_blocking import compute_perm, validate_kernel_block_size


def demo_basic_a4_quantization():
    """Demonstrate basic A4 quantization functionality."""
    print("=== Basic A4 Quantization Demo ===")
    
    # Create test data
    torch.manual_seed(42)
    x = torch.randn(2, 64, 32) * 5  # [batch, seq_len, hidden_dim]
    
    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
    
    # Test MXFP4 quantization
    print("\n--- MXFP4 Quantization ---")
    quantizer_mxfp4 = A4Quantizer(
        aformat="mxfp4",
        mx_block_size=32,
        mx_clip_quantile=0.999,
        stochastic_rounding=False
    )
    
    # Calibration phase
    print("Calibration phase...")
    quantizer_mxfp4.set_calibration_mode()
    for _ in range(3):
        _ = quantizer_mxfp4(x)
    
    # Compute permutation
    print("Computing permutation...")
    quantizer_mxfp4.compute_permutation()
    
    if quantizer_mxfp4.permutation is not None:
        print(f"Permutation shape: {quantizer_mxfp4.permutation.shape}")
        print(f"First 10 permutation indices: {quantizer_mxfp4.permutation[:10].tolist()}")
    
    # Quantization phase
    print("Quantization phase...")
    quantizer_mxfp4.set_eval_mode()
    x_quant_mxfp4 = quantizer_mxfp4(x)
    
    print(f"Quantized shape: {x_quant_mxfp4.shape}")
    print(f"Quantized range: [{x_quant_mxfp4.min():.3f}, {x_quant_mxfp4.max():.3f}]")
    print(f"MSE: {torch.nn.functional.mse_loss(x, x_quant_mxfp4):.6f}")
    
    # Test NVFP4 quantization
    print("\n--- NVFP4 Quantization ---")
    quantizer_nvfp4 = A4Quantizer(
        aformat="nvfp4",
        mx_block_size=16,  # Will be corrected to 16 by kernel validation
        mx_clip_quantile=0.999,
        stochastic_rounding=False
    )
    
    # Calibration phase
    print("Calibration phase...")
    quantizer_nvfp4.set_calibration_mode()
    for _ in range(3):
        _ = quantizer_nvfp4(x)
    
    # Compute permutation
    print("Computing permutation...")
    quantizer_nvfp4.compute_permutation()
    
    # Quantization phase
    print("Quantization phase...")
    quantizer_nvfp4.set_eval_mode()
    x_quant_nvfp4 = quantizer_nvfp4(x)
    
    print(f"Quantized shape: {x_quant_nvfp4.shape}")
    print(f"Quantized range: [{x_quant_nvfp4.min():.3f}, {x_quant_nvfp4.max():.3f}]")
    print(f"MSE: {torch.nn.functional.mse_loss(x, x_quant_nvfp4):.6f}")


def demo_headwise_reordering():
    """Demonstrate headwise reordering for attention mechanisms."""
    print("\n=== Headwise Reordering Demo ===")
    
    # Simulate attention head data
    torch.manual_seed(42)
    num_heads = 8
    head_dim = 64
    seq_len = 32
    batch_size = 2
    
    x = torch.randn(batch_size, seq_len, num_heads, head_dim) * 3
    x = x.view(batch_size, seq_len, -1)  # Flatten to [batch, seq_len, hidden_dim]
    
    print(f"Input shape: {x.shape}")
    print(f"Number of heads: {num_heads}")
    print(f"Head dimension: {head_dim}")
    
    # Create headwise quantizer
    quantizer_headwise = A4Quantizer(
        aformat="mxfp4",
        mx_block_size=32,
        mx_headwise=True,
        num_heads=num_heads,
        head_dim=head_dim,
        stochastic_rounding=False
    )
    
    # Calibration phase
    print("Calibration phase...")
    quantizer_headwise.set_calibration_mode()
    for _ in range(3):
        _ = quantizer_headwise(x)
    
    # Compute permutation
    print("Computing headwise permutation...")
    quantizer_headwise.compute_permutation()
    
    if quantizer_headwise.permutation is not None:
        print(f"Permutation shape: {quantizer_headwise.permutation.shape}")
        print(f"First 10 permutation indices: {quantizer_headwise.permutation[:10].tolist()}")
    
    # Quantization phase
    print("Quantization phase...")
    quantizer_headwise.set_eval_mode()
    x_quant_headwise = quantizer_headwise(x)
    
    print(f"Quantized shape: {x_quant_headwise.shape}")
    print(f"MSE: {torch.nn.functional.mse_loss(x, x_quant_headwise):.6f}")


def demo_permutation_absorption():
    """Demonstrate permutation absorption into model weights."""
    print("\n=== Permutation Absorption Demo ===")
    
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
    
    print(f"Model: {model}")
    print(f"Input shape: {x.shape}")
    
    # Get original output
    with torch.no_grad():
        original_output = model(x)
    
    print(f"Original output shape: {original_output.shape}")
    print(f"Original output range: [{original_output.min():.3f}, {original_output.max():.3f}]")
    
    # Create permutation
    P = torch.randperm(64)
    print(f"Permutation shape: {P.shape}")
    print(f"First 10 permutation indices: {P[:10].tolist()}")
    
    # Absorb permutation into model weights
    from quantize.perm_absorption import absorb_linear_permutation, absorb_layernorm_permutation
    
    class MockArgs:
        pass
    args = MockArgs()
    
    print("Absorbing permutations into model weights...")
    absorb_linear_permutation(model.linear1, P, "linear1", args)
    absorb_layernorm_permutation(model.layernorm, P[:32], "layernorm", args)  # Use first 32 indices for layernorm
    
    # Get output after permutation absorption
    with torch.no_grad():
        permuted_output = model(x)
    
    print(f"Permuted output shape: {permuted_output.shape}")
    print(f"Permuted output range: [{permuted_output.min():.3f}, {original_output.max():.3f}]")
    print(f"Output difference MSE: {torch.nn.functional.mse_loss(original_output, permuted_output):.6f}")


def demo_kernel_validation():
    """Demonstrate kernel block size validation."""
    print("\n=== Kernel Validation Demo ===")
    
    # Test different block sizes
    test_cases = [
        (32, 'mxfp4'),
        (16, 'mxfp4'),
        (64, 'mxfp4'),
        (16, 'nvfp4'),
        (32, 'nvfp4'),
        (8, 'nvfp4'),
    ]
    
    for requested_size, kernel_name in test_cases:
        actual_size = validate_kernel_block_size(requested_size, kernel_name)
        print(f"{kernel_name}: requested={requested_size}, actual={actual_size}")


def main():
    """Run all demos."""
    print("A4 Quantization with Reordering - Demo")
    print("=" * 50)
    
    try:
        demo_basic_a4_quantization()
        demo_headwise_reordering()
        demo_permutation_absorption()
        demo_kernel_validation()
        
        print("\n" + "=" * 50)
        print("✅ All demos completed successfully!")
        print("\nTo use A4 quantization in your model:")
        print("1. Add --aformat mxfp4 or --aformat nvfp4 to your command line")
        print("2. Optionally add --mx-headwise for headwise reordering")
        print("3. Optionally add --mx-local-swap N for boundary optimization")
        print("4. The system will automatically handle permutation absorption")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
