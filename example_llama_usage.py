#!/usr/bin/env python3
"""
Example usage of RPTQ with LLaMA models
"""

import subprocess
import sys
import os

def run_llama_quantization_example():
    """Run LLaMA quantization examples"""
    
    print("=" * 60)
    print("RPTQ LLaMA Quantization Examples")
    print("=" * 60)
    
    examples = [
        {
            "name": "LLaMA-2-7B Quantization (W4A4)",
            "command": [
                "python", "main.py", "llama-2-7b",
                "--wbits", "4", "--abits", "4",
                "--eval_ppl", "--tasks", "lambada_openai,piqa,arc_easy",
                "--nsamples", "128", "--calib_dataset", "mix"
            ]
        },
        {
            "name": "LLaMA-3-8B Quantization (W4A4)",
            "command": [
                "python", "main.py", "llama-3-8b", 
                "--wbits", "4", "--abits", "4",
                "--eval_ppl", "--tasks", "lambada_openai,piqa,arc_easy",
                "--nsamples", "128", "--calib_dataset", "mix"
            ]
        },
        {
            "name": "LLaMA-2-7B Quantization (W4A3) - Only KV Cache",
            "command": [
                "python", "main.py", "llama-2-7b",
                "--wbits", "4", "--abits", "3",
                "--only_quant_kv", "--eval_ppl",
                "--nsamples", "128", "--calib_dataset", "mix"
            ]
        },
        {
            "name": "LLaMA-3-8B Quantization (W3A3) - Aggressive Quantization",
            "command": [
                "python", "main.py", "llama-3-8b",
                "--wbits", "3", "--abits", "3", 
                "--eval_ppl", "--tasks", "lambada_openai,piqa",
                "--nsamples", "128", "--calib_dataset", "mix"
            ]
        }
    ]
    
    print("Available LLaMA quantization examples:")
    print()
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['name']}")
        print(f"   Command: {' '.join(example['command'])}")
        print()
    
    print("To run an example, copy the command and execute it in the terminal.")
    print()
    print("Note: Make sure you have:")
    print("  - Downloaded the LLaMA models from HuggingFace")
    print("  - Set up the proper cache directory")
    print("  - Installed all required dependencies")
    print()
    print("For more options, run: python main.py --help")

def show_llama_model_mapping():
    """Show LLaMA model mapping"""
    print("=" * 60)
    print("LLaMA Model Mapping")
    print("=" * 60)
    
    mapping = {
        "llama-2-7b": "meta-llama/Llama-2-7b-hf",
        "llama-2-13b": "meta-llama/Llama-2-13b-hf",
        "llama-3-8b": "meta-llama/Llama-3-8B", 
        "llama-3-70b": "meta-llama/Llama-3-70B",
    }
    
    print("Command line argument -> HuggingFace model name")
    print("-" * 50)
    for cmd_arg, hf_name in mapping.items():
        print(f"{cmd_arg:<15} -> {hf_name}")
    
    print()
    print("You can use any of the command line arguments on the left")
    print("in the main.py script.")

def show_quantization_parameters():
    """Show quantization parameters"""
    print("=" * 60)
    print("LLaMA Quantization Parameters")
    print("=" * 60)
    
    print("Key parameters for LLaMA quantization:")
    print()
    print("Weight Quantization:")
    print("  --wbits 4          # 4-bit weight quantization")
    print("  --wbits 3          # 3-bit weight quantization")
    print()
    print("Activation Quantization:")
    print("  --abits 4          # 4-bit activation quantization")
    print("  --abits 3          # 3-bit activation quantization")
    print("  --only_quant_kv    # Only quantize KV cache")
    print()
    print("Reordering (R1-R5):")
    print("  --reorder 12345    # Enable all reordering stages")
    print("  --reorder 1234     # Skip R5 (MLP reordering)")
    print("  --R1_clusters 32   # R1 cluster count")
    print("  --R2_clusters 4    # R2 cluster count")
    print("  --R3_clusters 4    # R3 cluster count")
    print("  --R4_clusters 32   # R4 cluster count")
    print("  --R5_clusters 4    # R5 cluster count")
    print()
    print("Calibration:")
    print("  --nsamples 128     # Number of calibration samples")
    print("  --calib_dataset mix # Calibration dataset (mix/wikitext2/ptb/c4)")
    print()
    print("Evaluation:")
    print("  --eval_ppl         # Evaluate perplexity")
    print("  --tasks task1,task2 # Zero-shot tasks")
    print("  --multigpu         # Use multiple GPUs")

def main():
    """Main function"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "examples":
            run_llama_quantization_example()
        elif sys.argv[1] == "mapping":
            show_llama_model_mapping()
        elif sys.argv[1] == "params":
            show_quantization_parameters()
        else:
            print("Usage: python example_llama_usage.py [examples|mapping|params]")
            sys.exit(1)
    else:
        print("RPTQ LLaMA Usage Guide")
        print("=" * 60)
        print()
        print("Available commands:")
        print("  python example_llama_usage.py examples  # Show usage examples")
        print("  python example_llama_usage.py mapping   # Show model mapping")
        print("  python example_llama_usage.py params    # Show parameters")
        print()
        print("Quick start:")
        print("  python main.py llama-2-7b --wbits 4 --abits 4 --eval_ppl")

if __name__ == "__main__":
    main()
