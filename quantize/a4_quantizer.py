import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from quantize.fp_torch import fake_quant_mxfp4, fake_quant_fp4
from quantize.reorder_blocking import (
    compute_perm, 
    apply_permutation_to_tensor,
    validate_kernel_block_size,
    compute_headwise_perm
)


class A4Quantizer(nn.Module):
    """
    A4 quantization with reordering support for MXFP4 and NVFP4 formats.
    """
    
    def __init__(
        self,
        aformat: str = "cluster",
        mx_block_size: int = 32,
        mx_clip_quantile: float = 0.999,
        mx_headwise: bool = False,
        mx_local_swap: int = 0,
        num_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        stochastic_rounding: bool = False,
    ):
        super().__init__()
        self.aformat = aformat
        self.mx_block_size = mx_block_size
        self.mx_clip_quantile = mx_clip_quantile
        self.mx_headwise = mx_headwise
        self.mx_local_swap = mx_local_swap
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.stochastic_rounding = stochastic_rounding
        
        # Permutation and block information
        self.permutation = None
        self.blocks = None
        self.max_abs_values = None
        
        # Calibration state
        self.mode = "calibration"
        self.enable = True
        
        # Validate block size for the chosen format
        if aformat in ["mxfp4", "nvfp4"]:
            self.actual_block_size = validate_kernel_block_size(mx_block_size, aformat)
        else:
            self.actual_block_size = mx_block_size
    
    def set_calibration_mode(self):
        """Set quantizer to calibration mode."""
        self.mode = "calibration"
    
    def set_eval_mode(self):
        """Set quantizer to evaluation mode."""
        self.mode = "eval"
    
    def calibration(self, x: torch.Tensor):
        """
        Collect max_abs values during calibration for permutation computation.
        """
        if not self.enable or self.aformat == "cluster":
            return
        
        # Compute max absolute values per channel
        if self.mx_headwise and self.num_heads is not None and self.head_dim is not None:
            # For headwise reordering, compute max_abs per head
            x_reshaped = x.view(-1, self.num_heads, self.head_dim)
            max_abs = x_reshaped.abs().max(dim=0)[0].max(dim=0)[0]  # [head_dim]
        else:
            # Standard per-channel max_abs
            max_abs = x.abs().max(dim=tuple(range(x.ndim - 1)))[0]  # [channels]
        
        # Accumulate max_abs values
        if self.max_abs_values is None:
            self.max_abs_values = max_abs
        else:
            self.max_abs_values = torch.maximum(self.max_abs_values, max_abs)
    
    def compute_permutation(self):
        """
        Compute permutation and block information based on collected max_abs values.
        """
        if self.max_abs_values is None or self.aformat == "cluster":
            return
        
        if self.mx_headwise and self.num_heads is not None and self.head_dim is not None:
            # Headwise reordering
            self.permutation, self.blocks = compute_headwise_perm(
                self.max_abs_values,
                self.num_heads,
                self.head_dim,
                self.actual_block_size,
                self.mx_local_swap
            )
        else:
            # Standard reordering
            self.permutation, self.blocks = compute_perm(
                self.max_abs_values,
                self.actual_block_size,
                self.mx_headwise,
                self.mx_local_swap
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply A4 quantization with reordering.
        """
        if not self.enable or self.aformat == "cluster":
            return x
        
        if self.mode == "calibration":
            self.calibration(x)
            return x
        
        # Apply permutation if available
        if self.permutation is not None:
            x_perm = apply_permutation_to_tensor(x, self.permutation, dim=-1)
        else:
            x_perm = x
        
        # Apply A4 quantization
        if self.aformat == "mxfp4":
            return fake_quant_mxfp4(x_perm, stochastic_rounding=self.stochastic_rounding)
        elif self.aformat == "nvfp4":
            return fake_quant_fp4(x_perm, stochastic_rounding=self.stochastic_rounding)
        else:
            return x
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get permutation and block metadata for saving.
        """
        return {
            'permutation': self.permutation.cpu() if self.permutation is not None else None,
            'blocks': self.blocks,
            'max_abs_values': self.max_abs_values.cpu() if self.max_abs_values is not None else None,
            'aformat': self.aformat,
            'actual_block_size': self.actual_block_size,
        }
    
    def load_metadata(self, metadata: Dict[str, Any]):
        """
        Load permutation and block metadata.
        """
        self.permutation = metadata['permutation']
        self.blocks = metadata['blocks']
        self.max_abs_values = metadata['max_abs_values']
        if self.permutation is not None:
            self.permutation = self.permutation.to(next(self.parameters()).device)


class A4QuantizerWrapper(nn.Module):
    """
    Wrapper to integrate A4Quantizer with existing quantization infrastructure.
    """
    
    def __init__(
        self,
        original_quantizer: nn.Module,
        a4_quantizer: A4Quantizer
    ):
        super().__init__()
        self.original_quantizer = original_quantizer
        self.a4_quantizer = a4_quantizer
        self.use_a4 = a4_quantizer.aformat != "cluster"
    
    def set_calibration_mode(self):
        """Set both quantizers to calibration mode."""
        if hasattr(self.original_quantizer, 'set_calibration_mode'):
            self.original_quantizer.set_calibration_mode()
        self.a4_quantizer.set_calibration_mode()
    
    def set_eval_mode(self):
        """Set both quantizers to evaluation mode."""
        if hasattr(self.original_quantizer, 'set_eval_mode'):
            self.original_quantizer.set_eval_mode()
        self.a4_quantizer.set_eval_mode()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply quantization using A4 or original quantizer."""
        if self.use_a4:
            return self.a4_quantizer(x)
        else:
            return self.original_quantizer(x)
    
    def compute_permutation(self):
        """Compute permutation for A4 quantizer."""
        if self.use_a4:
            self.a4_quantizer.compute_permutation()
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata from A4 quantizer."""
        if self.use_a4:
            return self.a4_quantizer.get_metadata()
        return {}


def create_a4_quantizer(
    args,
    original_quantizer: Optional[nn.Module] = None,
    num_heads: Optional[int] = None,
    head_dim: Optional[int] = None
) -> A4QuantizerWrapper:
    """
    Create an A4QuantizerWrapper based on command line arguments.
    
    Args:
        args: Command line arguments containing A4 format options
        original_quantizer: Original quantizer to wrap (if any)
        num_heads: Number of attention heads (for headwise reordering)
        head_dim: Dimension of each head (for headwise reordering)
        
    Returns:
        A4QuantizerWrapper instance
    """
    a4_quantizer = A4Quantizer(
        aformat=getattr(args, 'aformat', 'cluster'),
        mx_block_size=getattr(args, 'mx_block_size', 32),
        mx_clip_quantile=getattr(args, 'mx_clip_quantile', 0.999),
        mx_headwise=getattr(args, 'mx_headwise', False),
        mx_local_swap=getattr(args, 'mx_local_swap', 0),
        num_heads=num_heads,
        head_dim=head_dim,
        stochastic_rounding=getattr(args, 'sr', False)
    )
    
    return A4QuantizerWrapper(original_quantizer, a4_quantizer)
