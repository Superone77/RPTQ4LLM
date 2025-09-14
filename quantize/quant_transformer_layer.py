import torch
from quantize.int_linear import QuantLinear
from quantize.int_matmul import QuantMatMul
from quantize.reorder_layer_norm import ReorderLayerNorm
from quantize.gptq import GPTQ
from quantize.mem_packer import MemoryPacker
from quantize.a4_quantizer import A4QuantizerWrapper, create_a4_quantizer
from quantize.perm_absorption import (
    absorb_permutations_into_model,
    create_permutation_metadata,
    save_permutation_metadata,
    print_permutation_summary
)
from torch.nn import Parameter


def gptq_add_batch(m, i, o):
    if m.recorded_quant_input is not None:
        m.gptq.add_batch(m.recorded_quant_input, o.data)
    else:
        m.gptq.add_batch(i[0].data, o.data)


@torch.no_grad()
def quant_layer(qlayer, args, outs, inps, attention_mask, dev):
    handlers = []
    gptqs = {}
    qlayer.set_quant_state(weight_quant=True, act_quant=True)
    for name, module in qlayer.named_modules():
        if isinstance(module, QuantLinear):
            if args.disable_w_quant or module.weight_quantizer.n_bits>=16:
                continue
            if args.w_quantizer == "normal":
                module.weight_quantizer.set_calibration_mode()
                """caculate the step size and zero point for weight quantizer"""
                fake_quantized_weight = module.weight_quantizer(module.weight)
                module.weight_quantizer.set_eval_mode()
                del module.weight
                module.register_buffer('weight',fake_quantized_weight)
                # module.weight = fake_quantized_weight
                module.replace_weight_with_quantized = True
            elif args.w_quantizer == "gptq":
                gptqs[name] = GPTQ(module, module.weight_quantizer)
                module.gptq = gptqs[name]
                module.record_quant_input = True
                handler = module.register_forward_hook(gptq_add_batch)
                handlers.append(handler)
    qlayer.set_quant_state(weight_quant=False, act_quant=True)
    # for activation quantize
    a_quantizers = {}
    w_quantizers = {}
    a4_quantizers = {}
    
    # Create A4 quantizers for supported modules
    for name, m in qlayer.named_modules():
        if isinstance(m, (QuantLinear)) and not m.disable_input_quant:
            a_quantizers[name] = m.act_quantizer
            w_quantizers[name] = m.weight_quantizer
            
            # Create A4 quantizer wrapper and replace the original quantizer
            if hasattr(args, 'aformat') and args.aformat in ['mxfp4', 'nvfp4']:
                a4_quantizer = create_a4_quantizer(args, m.act_quantizer)
                m.a4_quantizer = a4_quantizer
                a4_quantizers[name] = a4_quantizer
                
        if isinstance(m, ReorderLayerNorm) and m.out_quantizer is not None:
            a_quantizers[name] = m.out_quantizer
            
            # Create A4 quantizer wrapper for LayerNorm output
            if hasattr(args, 'aformat') and args.aformat in ['mxfp4', 'nvfp4']:
                a4_quantizer = create_a4_quantizer(args, m.out_quantizer)
                # Store A4 quantizer in the layer norm module
                m.a4_out_quantizer = a4_quantizer
                a4_quantizers[name] = a4_quantizer
                
        if isinstance(m, QuantMatMul):
            a_quantizers[name + "x1"] = m.x1_quantizer
            a_quantizers[name + "x2"] = m.x2_quantizer
            
            # Create A4 quantizer wrappers for matmul inputs
            if hasattr(args, 'aformat') and args.aformat in ['mxfp4', 'nvfp4']:
                a4_quantizer_x1 = create_a4_quantizer(args, m.x1_quantizer)
                a4_quantizer_x2 = create_a4_quantizer(args, m.x2_quantizer)
                m.a4_x1_quantizer = a4_quantizer_x1
                m.a4_x2_quantizer = a4_quantizer_x2
                a4_quantizers[name + "x1"] = a4_quantizer_x1
                a4_quantizers[name + "x2"] = a4_quantizer_x2

    # Set calibration mode for all quantizers
    for name, quantizer in a_quantizers.items():
        quantizer.set_calibration_mode()
    
    for name, quantizer in a4_quantizers.items():
        quantizer.set_calibration_mode()
    for j in range(args.nsamples):
        outs[j] = qlayer(
            inps[j].unsqueeze(0).to(dev), attention_mask=attention_mask.to(dev)
        )[0]

    # Compute permutations for A4 quantizers
    for name, quantizer in a4_quantizers.items():
        quantizer.compute_permutation()
    
    # Set eval mode for all quantizers
    for name, quantizer in a_quantizers.items():
        quantizer.set_eval_mode()
        if args.metric == "layer_mse":
            quantizer.layer_mse_param_search_update(
                qlayer, a_quantizers, inps, outs, attention_mask
            )
        quantizer.free()
    
    for name, quantizer in a4_quantizers.items():
        quantizer.set_eval_mode()

    # Absorb permutations into model weights if using A4 formats
    if hasattr(args, 'aformat') and args.aformat in ['mxfp4', 'nvfp4'] and a4_quantizers:
        print("Computing and absorbing permutations...")
        
        # Create permutation metadata
        permutation_metadata = create_permutation_metadata(a4_quantizers, qlayer)
        
        # Print permutation summary
        print_permutation_summary(permutation_metadata)
        
        # Absorb permutations into model weights
        absorb_permutations_into_model(qlayer, permutation_metadata, args)
        
        # Save permutation metadata for later use
        if hasattr(args, 'output_path'):
            metadata_path = f"{args.output_path}/permutation_metadata.pth"
            save_permutation_metadata(permutation_metadata, metadata_path)

    if args.w_quantizer == "gptq" and not args.disable_w_quant:
        # for weight quantize
        for handler in handlers:
            handler.remove()
        gptq_losses = {}
        print(f"GPTQ Quantizing ...")
        for name, module in qlayer.named_modules():
            if isinstance(module, QuantLinear) and module.weight_quantizer.n_bits < 16:
                module.record_quant_input = False
                module.recorded_quant_input = None

                fake_quantized_weight, gptq_loss = gptqs[name].fasterquant(
                    percdamp=args.percdamp
                )
                
                gptq_losses[name] = gptq_loss
                if args.pack_weight:
                    module.mem_packer = MemoryPacker(
                        module.weight_quantizer.scale,
                        module.weight_quantizer.round_zero_point,
                        module.weight_quantizer.n_bits,
                    )
                    w_packed = module.mem_packer.pack_tensor(fake_quantized_weight)
                    del module.weight
                    module.register_buffer(w_packed,w_packed)
                    module.is_weight_packed = True
                else:
                    # module.weight.data = fake_quantized_weight.to(module.weight.dtype)
                    del module.weight
                    module.register_buffer("weight", fake_quantized_weight.half())
                    module.replace_weight_with_quantized = True
                gptqs[name].free()
                module.gptq = None
                module.weight_quantizer = None

        if len(gptq_losses):
            print("GPTQ losses", gptq_losses)

            for j in range(args.nsamples):
                outs[j] = qlayer(
                    inps[j].unsqueeze(0).to(dev), attention_mask=attention_mask.to(dev)
                )[0]
            gptqs.clear()
    for name, quantizer in w_quantizers.items():
        quantizer.free()

    return outs
