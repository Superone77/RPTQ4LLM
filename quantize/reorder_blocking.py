import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any
import math


def compute_perm(
    max_abs: torch.Tensor, 
    B: int, 
    headwise: bool = False, 
    local_swap: int = 0
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Compute permutation and blocking information based on channel dynamic range.
    
    Args:
        max_abs: Tensor of maximum absolute values per channel
        B: Block size for quantization
        headwise: Whether to reorder within attention heads
        local_swap: Number of local swap iterations for boundary optimization
        
    Returns:
        P: Permutation indices
        blocks: Block information dictionary
    """
    device = max_abs.device
    dtype = max_abs.dtype
    
    # Apply quantile clipping to avoid outliers
    quantile_val = torch.quantile(max_abs, 0.999)
    max_abs_clipped = torch.clamp(max_abs, max=quantile_val)
    
    # Use log2 for more stable sorting
    eps = 1e-8
    log_max_abs = torch.log2(max_abs_clipped + eps)
    
    # Sort by log2(max_abs) in ascending order
    sorted_indices = torch.argsort(log_max_abs)
    
    # Apply local swap optimization if requested
    if local_swap > 0:
        sorted_indices = _local_swap_optimization(sorted_indices, log_max_abs, B, local_swap)
    
    # Create block information
    total_channels = max_abs.shape[0]
    num_blocks = math.ceil(total_channels / B)
    
    blocks = {
        'num_blocks': num_blocks,
        'block_size': B,
        'total_channels': total_channels,
        'block_boundaries': [i * B for i in range(num_blocks + 1)]
    }
    
    return sorted_indices, blocks


def _local_swap_optimization(
    sorted_indices: torch.Tensor, 
    log_max_abs: torch.Tensor, 
    B: int, 
    num_swaps: int
) -> torch.Tensor:
    """
    Apply local swap optimization to reduce approximation MSE at block boundaries.
    """
    indices = sorted_indices.clone()
    total_channels = indices.shape[0]
    num_blocks = math.ceil(total_channels / B)
    
    for _ in range(num_swaps):
        for block_idx in range(num_blocks - 1):
            # Get boundary channels
            block_start = block_idx * B
            block_end = min((block_idx + 1) * B, total_channels)
            next_block_start = block_end
            next_block_end = min((block_idx + 2) * B, total_channels)
            
            if next_block_start >= total_channels:
                break
                
            # Get the last channel of current block and first channel of next block
            last_channel_idx = block_end - 1
            first_next_channel_idx = next_block_start
            
            # Find their positions in the sorted indices
            last_pos = torch.where(indices == last_channel_idx)[0][0]
            first_next_pos = torch.where(indices == first_next_channel_idx)[0][0]
            
            # Check if swapping would reduce MSE
            last_val = log_max_abs[last_channel_idx]
            first_next_val = log_max_abs[first_next_channel_idx]
            
            # Simple heuristic: swap if the values are very close
            if abs(last_val - first_next_val) < 0.1:  # Threshold for swapping
                indices[last_pos], indices[first_next_pos] = indices[first_next_pos], indices[last_pos]
    
    return indices


def absorb_perm_into_module(
    module: nn.Module, 
    P_in: Optional[torch.Tensor] = None, 
    P_out: Optional[torch.Tensor] = None
) -> None:
    """
    Absorb permutation into module weights to avoid runtime gather/scatter.
    
    Args:
        module: The module to modify (Linear or LayerNorm)
        P_in: Input permutation indices
        P_out: Output permutation indices
    """
    if isinstance(module, nn.Linear):
        if P_in is not None:
            # Apply input permutation to weight columns
            module.weight.data = module.weight.data[:, P_in]
        if P_out is not None:
            # Apply output permutation to weight rows
            module.weight.data = module.weight.data[P_out, :]
            if module.bias is not None:
                module.bias.data = module.bias.data[P_out]
    
    elif isinstance(module, nn.LayerNorm):
        if P_out is not None:
            # Apply output permutation to LayerNorm parameters
            module.weight.data = module.weight.data[P_out]
            module.bias.data = module.bias.data[P_out]
    
    else:
        raise ValueError(f"Unsupported module type for permutation absorption: {type(module)}")


def apply_permutation_to_tensor(x: torch.Tensor, P: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Apply permutation to tensor along specified dimension.
    
    Args:
        x: Input tensor
        P: Permutation indices
        dim: Dimension to apply permutation to
        
    Returns:
        Permuted tensor
    """
    if dim == -1:
        return x[..., P]
    elif dim == 0:
        return x[P, ...]
    else:
        # For other dimensions, use advanced indexing
        indices = [slice(None)] * x.ndim
        indices[dim] = P
        return x[tuple(indices)]


def validate_kernel_block_size(
    requested_block_size: int, 
    kernel_name: str
) -> int:
    """
    Validate and return the actual block size used by the kernel.
    
    Args:
        requested_block_size: User-requested block size
        kernel_name: Name of the kernel ('mxfp4' or 'nvfp4')
        
    Returns:
        Actual block size used by the kernel
    """
    if kernel_name == 'mxfp4':
        actual_block_size = 32
    elif kernel_name == 'nvfp4':
        actual_block_size = 16
    else:
        raise ValueError(f"Unknown kernel name: {kernel_name}")
    
    if requested_block_size != actual_block_size:
        print(f"Warning: {kernel_name} kernel uses fixed block size {actual_block_size}, "
              f"ignoring requested size {requested_block_size}")
    
    return actual_block_size


def compute_headwise_perm(
    max_abs: torch.Tensor, 
    num_heads: int, 
    head_dim: int, 
    B: int, 
    local_swap: int = 0
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Compute permutation for headwise reordering within attention heads.
    
    Args:
        max_abs: Tensor of maximum absolute values per channel
        num_heads: Number of attention heads
        head_dim: Dimension of each head
        B: Block size for quantization
        local_swap: Number of local swap iterations
        
    Returns:
        P: Permutation indices
        blocks: Block information dictionary
    """
    total_dim = max_abs.shape[0]
    assert total_dim == num_heads * head_dim, f"Dimension mismatch: {total_dim} != {num_heads} * {head_dim}"
    
    # Reshape to [num_heads, head_dim]
    max_abs_heads = max_abs.view(num_heads, head_dim)
    
    # Compute permutation for each head independently
    all_perm_indices = []
    all_blocks = []
    
    for head_idx in range(num_heads):
        head_max_abs = max_abs_heads[head_idx]
        head_perm, head_blocks = compute_perm(head_max_abs, B, headwise=True, local_swap=local_swap)
        
        # Convert to global indices
        global_perm = head_perm + head_idx * head_dim
        all_perm_indices.append(global_perm)
        all_blocks.append(head_blocks)
    
    # Concatenate all head permutations
    P = torch.cat(all_perm_indices, dim=0)
    
    # Combine block information
    combined_blocks = {
        'num_heads': num_heads,
        'head_dim': head_dim,
        'head_blocks': all_blocks,
        'total_blocks': sum(block['num_blocks'] for block in all_blocks)
    }
    
    return P, combined_blocks


def save_permutation_metadata(
    permutations: Dict[str, torch.Tensor],
    blocks: Dict[str, Any],
    save_path: str
) -> None:
    """
    Save permutation and block metadata for offline analysis.
    
    Args:
        permutations: Dictionary of permutation tensors
        blocks: Dictionary of block information
        save_path: Path to save the metadata
    """
    metadata = {
        'permutations': {k: v.cpu() for k, v in permutations.items()},
        'blocks': blocks,
        'timestamp': torch.tensor(torch.cuda.Event(enable_timing=True).elapsed_time(torch.cuda.Event(enable_timing=True)))
    }
    torch.save(metadata, save_path)


def load_permutation_metadata(load_path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Load permutation and block metadata.
    
    Args:
        load_path: Path to load the metadata from
        
    Returns:
        Tuple of (permutations, blocks)
    """
    metadata = torch.load(load_path, map_location='cpu')
    return metadata['permutations'], metadata['blocks']


# Unit test functions
def test_perm_equiv():
    """
    Test that GEMM(X, W) is equivalent to GEMM(perm(X), perm(W_perm)).
    """
    torch.manual_seed(42)
    
    # Create test data
    B, M, N = 2, 64, 32
    X = torch.randn(B, M)
    W = torch.randn(N, M)
    
    # Create random permutation
    P_in = torch.randperm(M)
    P_out = torch.randperm(N)
    
    # Apply permutations
    X_perm = X[:, P_in]
    W_perm = W[P_out, :][:, P_in]  # Apply both input and output permutations
    
    # Compute both results
    result1 = torch.mm(X, W.T)
    result2 = torch.mm(X_perm, W_perm.T)
    
    # Check equivalence
    assert torch.allclose(result1, result2), "Permutation equivalence test failed!"
    print("✓ Permutation equivalence test passed")


def test_kernel_validation():
    """
    Test kernel block size validation.
    """
    # Test MXFP4
    assert validate_kernel_block_size(32, 'mxfp4') == 32
    assert validate_kernel_block_size(16, 'mxfp4') == 32  # Should warn and return 32
    
    # Test NVFP4
    assert validate_kernel_block_size(16, 'nvfp4') == 16
    assert validate_kernel_block_size(32, 'nvfp4') == 16  # Should warn and return 16
    
    print("✓ Kernel validation test passed")


if __name__ == "__main__":
    # Run unit tests
    test_perm_equiv()
    test_kernel_validation()
    print("All tests passed!")
