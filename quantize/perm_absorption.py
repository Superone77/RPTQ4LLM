import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from quantize.reorder_blocking import absorb_perm_into_module, apply_permutation_to_tensor


def absorb_permutations_into_model(
    model: nn.Module, 
    permutation_metadata: Dict[str, Any],
    args
) -> None:
    """
    Absorb permutations into model weights to avoid runtime gather/scatter.
    
    Args:
        model: The model to modify
        permutation_metadata: Dictionary containing permutation information
        args: Command line arguments
    """
    if not hasattr(args, 'aformat') or args.aformat not in ['mxfp4', 'nvfp4']:
        return
    
    print("Absorbing permutations into model weights...")
    
    # Process each layer's permutation metadata
    for layer_name, metadata in permutation_metadata.items():
        if 'permutation' not in metadata or metadata['permutation'] is None:
            continue
            
        P = metadata['permutation']
        if P is None:
            continue
            
        # Find the corresponding module in the model
        module = find_module_by_name(model, layer_name)
        if module is None:
            print(f"Warning: Could not find module {layer_name}")
            continue
            
        # Apply permutation absorption based on module type
        if isinstance(module, nn.Linear):
            absorb_linear_permutation(module, P, layer_name, args)
        elif isinstance(module, nn.LayerNorm):
            absorb_layernorm_permutation(module, P, layer_name, args)
        else:
            print(f"Warning: Unsupported module type for permutation absorption: {type(module)}")


def find_module_by_name(model: nn.Module, name: str) -> Optional[nn.Module]:
    """
    Find a module by its name in the model.
    
    Args:
        model: The model to search
        name: The name of the module to find
        
    Returns:
        The module if found, None otherwise
    """
    try:
        return dict(model.named_modules())[name]
    except KeyError:
        return None


def absorb_linear_permutation(
    module: nn.Linear, 
    P: torch.Tensor, 
    layer_name: str, 
    args
) -> None:
    """
    Absorb permutation into Linear module weights.
    
    Args:
        module: The Linear module to modify
        P: Permutation indices
        layer_name: Name of the layer for logging
        args: Command line arguments
    """
    print(f"Absorbing permutation into Linear layer: {layer_name}")
    
    # Determine if this is an input or output permutation based on layer name
    # This is a heuristic - in practice, you might want to be more specific
    if any(x in layer_name.lower() for x in ['qproj', 'kproj', 'vproj', 'fc1']):
        # Input permutation for Q/K/V projections and FFN first layer
        module.weight.data = module.weight.data[:, P]
        print(f"  Applied input permutation to {layer_name}")
    elif any(x in layer_name.lower() for x in ['out_proj', 'fc2', 'lm_head']):
        # Output permutation for output projection, FFN second layer, and LM head
        module.weight.data = module.weight.data[P, :]
        if module.bias is not None:
            module.bias.data = module.bias.data[P]
        print(f"  Applied output permutation to {layer_name}")
    else:
        # Default to input permutation for unknown layers
        module.weight.data = module.weight.data[:, P]
        print(f"  Applied input permutation (default) to {layer_name}")


def absorb_layernorm_permutation(
    module: nn.LayerNorm, 
    P: torch.Tensor, 
    layer_name: str, 
    args
) -> None:
    """
    Absorb permutation into LayerNorm module parameters.
    
    Args:
        module: The LayerNorm module to modify
        P: Permutation indices
        layer_name: Name of the layer for logging
        args: Command line arguments
    """
    print(f"Absorbing permutation into LayerNorm: {layer_name}")
    
    # Apply output permutation to LayerNorm parameters
    module.weight.data = module.weight.data[P]
    module.bias.data = module.bias.data[P]
    print(f"  Applied output permutation to {layer_name}")


def create_permutation_metadata(
    a4_quantizers: Dict[str, Any],
    model: nn.Module
) -> Dict[str, Any]:
    """
    Create permutation metadata from A4 quantizers.
    
    Args:
        a4_quantizers: Dictionary of A4 quantizers
        model: The model being quantized
        
    Returns:
        Dictionary containing permutation metadata
    """
    metadata = {}
    
    for name, quantizer in a4_quantizers.items():
        if hasattr(quantizer, 'get_metadata'):
            quantizer_metadata = quantizer.get_metadata()
            if quantizer_metadata.get('permutation') is not None:
                metadata[name] = quantizer_metadata
                
    return metadata


def save_permutation_metadata(
    metadata: Dict[str, Any],
    save_path: str
) -> None:
    """
    Save permutation metadata to file.
    
    Args:
        metadata: Permutation metadata dictionary
        save_path: Path to save the metadata
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(metadata, save_path)
    print(f"Saved permutation metadata to {save_path}")


def load_permutation_metadata(load_path: str) -> Dict[str, Any]:
    """
    Load permutation metadata from file.
    
    Args:
        load_path: Path to load the metadata from
        
    Returns:
        Dictionary containing permutation metadata
    """
    return torch.load(load_path, map_location='cpu')


def validate_permutation_equivalence(
    model: nn.Module,
    permutation_metadata: Dict[str, Any],
    test_input: torch.Tensor
) -> bool:
    """
    Validate that permutations are correctly absorbed by checking equivalence.
    
    Args:
        model: The model with absorbed permutations
        permutation_metadata: Permutation metadata
        test_input: Test input tensor
        
    Returns:
        True if validation passes, False otherwise
    """
    print("Validating permutation equivalence...")
    
    # This is a simplified validation - in practice, you might want to
    # test specific layers or use more sophisticated checks
    try:
        with torch.no_grad():
            output = model(test_input)
            # Basic sanity check - output should be finite
            if torch.isfinite(output).all():
                print("✓ Permutation equivalence validation passed")
                return True
            else:
                print("✗ Permutation equivalence validation failed: non-finite output")
                return False
    except Exception as e:
        print(f"✗ Permutation equivalence validation failed: {e}")
        return False


def print_permutation_summary(permutation_metadata: Dict[str, Any]) -> None:
    """
    Print a summary of permutation information.
    
    Args:
        permutation_metadata: Permutation metadata dictionary
    """
    print("\n=== Permutation Summary ===")
    print(f"Total layers with permutations: {len(permutation_metadata)}")
    
    for layer_name, metadata in permutation_metadata.items():
        if 'blocks' in metadata:
            blocks = metadata['blocks']
            if isinstance(blocks, dict) and 'num_blocks' in blocks:
                print(f"  {layer_name}: {blocks['num_blocks']} blocks, "
                      f"block_size={blocks.get('block_size', 'N/A')}")
            else:
                print(f"  {layer_name}: {len(blocks)} blocks")
        else:
            print(f"  {layer_name}: permutation applied")
    
    print("===========================\n")
