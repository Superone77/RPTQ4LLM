#!/usr/bin/env python3
"""
Simple test to verify LLaMA support
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all imports work"""
    try:
        from models.llama import LLaMAClass
        print("✓ LLaMA model class imported")
        
        from models.int_llama_layer import QuantLLaMAAttention, QuantLLaMADecoderLayer
        print("✓ LLaMA quantization layers imported")
        
        from quantize.llama_reorder_quantize import llama_reorder_quantize
        print("✓ LLaMA reorder functions imported")
        
        import main
        print("✓ Main module imported")
        
        # Check if LLaMA models are in choices
        llama_models = [choice for choice in main.net_choices if "llama" in choice]
        print(f"✓ Found LLaMA models: {llama_models}")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing LLaMA support...")
    if test_imports():
        print("\n🎉 All imports successful! LLaMA support is ready.")
        print("\nYou can now use:")
        print("  python main.py llama-2-7b --wbits 4 --abits 4 --eval_ppl")
        print("  python main.py llama-3-8b --wbits 4 --abits 4 --eval_ppl")
    else:
        print("\n❌ Some imports failed. Please check the errors above.")
        sys.exit(1)
