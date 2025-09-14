# A4 Quantization with Reordering

This document describes the implementation of "重排 + microscaling A4" (reorder + microscaling A4) quantization path, which combines channel reordering with MXFP4/NVFP4 Triton kernels for block quantization.

## Overview

The A4 quantization path provides:
- **Channel Reordering**: Sorts channels by dynamic range for better quantization efficiency
- **Microscaling**: Uses MXFP4/NVFP4 Triton kernels with per-block scaling
- **Zero-Cost Permutation**: Absorbs permutations into model weights to avoid runtime overhead
- **Headwise Support**: Optional reordering within attention heads

## Features

### Supported Formats
- **MXFP4**: 4-bit floating point with 1-2-1 format, 32-element blocks
- **NVFP4**: 4-bit floating point with 1-2-1 format, 16-element blocks
- **Cluster**: Original cluster-based quantization (default)

### Key Components
- `quantize/reorder_blocking.py`: Core reordering and blocking logic
- `quantize/a4_quantizer.py`: A4 quantizer implementation
- `quantize/perm_absorption.py`: Permutation absorption into model weights
- `quantize/fp_torch.py`: MXFP4/NVFP4 Triton kernel interfaces

## Usage

### Command Line Arguments

```bash
python main.py opt-125m \
    --aformat mxfp4 \
    --mx-block-size 32 \
    --mx-clip-quantile 0.999 \
    --mx-headwise \
    --mx-local-swap 0
```

#### New Arguments

- `--aformat {cluster,mxfp4,nvfp4}`: Activation quantization format (default: cluster)
- `--mx-block-size 32`: Block size for MXFP4 quantization (default: 32)
- `--mx-clip-quantile 0.999`: Quantile for clipping max_abs values (default: 0.999)
- `--mx-headwise`: Enable headwise reordering within attention heads
- `--mx-local-swap 0`: Number of local swap iterations for boundary optimization (default: 0)

### Example Commands

#### Basic MXFP4 Quantization
```bash
python main.py opt-125m --aformat mxfp4 --wbits 4 --abits 4
```

#### Headwise Reordering with MXFP4
```bash
python main.py opt-125m --aformat mxfp4 --mx-headwise --wbits 4 --abits 4
```

#### NVFP4 with Boundary Optimization
```bash
python main.py opt-125m --aformat nvfp4 --mx-local-swap 5 --wbits 4 --abits 4
```

## Implementation Details

### Reordering Algorithm

1. **Calibration Phase**: Collect max_abs values per channel during forward passes
2. **Quantile Clipping**: Apply `mx-clip-quantile` to remove outliers
3. **Log2 Sorting**: Sort channels by `log2(max_abs + eps)` in ascending order
4. **Block Partitioning**: Divide sorted channels into blocks of size `mx-block-size`
5. **Local Optimization**: Apply `mx-local-swap` iterations to optimize block boundaries

### Permutation Absorption

The system automatically absorbs permutations into model weights to avoid runtime overhead:

- **Linear Layers**: Apply input/output permutations to weight matrices
- **LayerNorm**: Apply output permutations to weight and bias parameters
- **Attention**: Q/K/V projections use consistent permutation conventions

### Headwise Reordering

When `--mx-headwise` is enabled:
- Each attention head is reordered independently
- Permutations are computed per head dimension
- Maintains attention mechanism structure

## File Structure

```
quantize/
├── reorder_blocking.py      # Core reordering logic
├── a4_quantizer.py         # A4 quantizer implementation
├── perm_absorption.py      # Permutation absorption
├── fp_torch.py            # MXFP4/NVFP4 kernel interfaces
├── mxfp4_triton.py        # MXFP4 Triton kernel
├── nvfp4_triton.py        # NVFP4 Triton kernel
└── quant_transformer_layer.py  # Modified quantization pipeline
```

## Testing

### Unit Tests
```bash
python test_a4_quantization.py
```

### Demo Script
```bash
python test_a4_demo.py
```

### Validation Tests
- **Permutation Equivalence**: Verifies `GEMM(X, W) = GEMM(perm(X), perm(W_perm))`
- **Kernel Validation**: Ensures block sizes match Triton kernel requirements
- **End-to-End**: Tests complete quantization pipeline

## Performance Considerations

### Memory Usage
- Permutation metadata is saved during calibration
- No additional memory overhead during inference
- Permutations are absorbed into weights offline

### Computational Overhead
- Calibration phase: Minimal overhead for max_abs collection
- Inference phase: Zero overhead (permutations absorbed)
- Triton kernels: Optimized for GPU execution

### Accuracy
- Reordering improves quantization efficiency
- Per-block scaling reduces quantization error
- Headwise reordering preserves attention structure

## Troubleshooting

### Common Issues

1. **Block Size Mismatch**: The system automatically corrects block sizes to match Triton kernels
   - MXFP4: Always uses 32-element blocks
   - NVFP4: Always uses 16-element blocks

2. **CUDA Requirements**: MXFP4/NVFP4 kernels require CUDA
   - Falls back to cluster quantization if CUDA unavailable

3. **Memory Issues**: Large models may require gradient checkpointing
   - Use `--multigpu` for multi-GPU inference

### Debug Options

- Enable verbose logging to see permutation information
- Use `--mx-local-swap 0` to disable boundary optimization
- Check permutation metadata in `{output_path}/permutation_metadata.pth`

## Integration with Existing Code

The A4 quantization path is designed to be minimally invasive:

1. **Backward Compatibility**: Original cluster quantization remains unchanged
2. **Automatic Detection**: A4 quantizers are created automatically when `--aformat` is specified
3. **Transparent Integration**: No changes required to existing model code

## Future Enhancements

- Support for additional FP4 formats
- Dynamic block size selection
- Advanced boundary optimization algorithms
- Integration with other quantization methods

## References

- MXFP4: Microsoft's 4-bit floating point format
- NVFP4: NVIDIA's 4-bit floating point format
- Triton: GPU kernel compilation framework
- Microscaling: Per-block scaling for improved quantization accuracy
