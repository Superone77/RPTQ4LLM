import torch
import torch.nn as nn
from quantize.int_linear import QuantLinear
from quantize.int_matmul import QuantMatMul
from quantize.reorder_layer_norm import ReorderLayerNorm
from models.int_llama_layer import QuantLLaMADecoderLayer
from quantize.quant_transformer_layer import quant_layer
from quantize.reorder_utils import (
    tensor_calc_reorder_index,
    ic_maxmin_dict,
    oc_maxmin_dict,
    layer_i0max_hook,
    layer_omax_hook,
)


R_DEBUG_BIT = 0
DEBUG_BREAK_LAYER = -1


def LLaMA_R1_reorder(input_layernorm, q_proj, k_proj, v_proj, index, counts):
    """R1: Input layer norm reordering for LLaMA"""
    input_layernorm.register_buffer("reorder_index", index)
    input_layernorm.out_quantizer.cluster_dim = 2
    input_layernorm.out_quantizer.cluster_counts = counts
    if R_DEBUG_BIT:
        input_layernorm.out_quantizer.change_n_bits(R_DEBUG_BIT)

    q_proj.weight.data = torch.index_select(q_proj.weight.data, 1, index)
    q_proj.set_ic_cluster_counts(counts, a_dim=None)

    k_proj.weight.data = torch.index_select(k_proj.weight.data, 1, index)
    k_proj.set_ic_cluster_counts(counts, a_dim=None)
    
    v_proj.weight.data = torch.index_select(v_proj.weight.data, 1, index)
    v_proj.set_ic_cluster_counts(counts, a_dim=None)


def LLaMA_R2_reorder(q_proj, k_proj, qk_matmul, index, counts):
    """R2: Q, K projection reordering for LLaMA"""
    q_proj.weight.data = torch.index_select(q_proj.weight.data, 0, index)
    k_proj.weight.data = torch.index_select(k_proj.weight.data, 0, index)

    qk_matmul.set_ic_cluster_counts(counts, x1_dim=2, x2_dim=2)
    if R_DEBUG_BIT:
        qk_matmul.x1_quantizer.change_n_bits(R_DEBUG_BIT)
        qk_matmul.x2_quantizer.change_n_bits(R_DEBUG_BIT)


def LLaMA_R3_reorder(v_proj, pv_matmul, o_proj, index, counts):
    """R3: V projection and output projection reordering for LLaMA"""
    v_proj.weight.data = torch.index_select(v_proj.weight.data, 0, index)
    pv_matmul.set_ic_cluster_counts(counts, cluster_x1=False)
    o_proj.weight.data = torch.index_select(o_proj.weight.data, 1, index)
    o_proj.set_ic_cluster_counts(counts)
    if R_DEBUG_BIT:
        pv_matmul.x2_quantizer.change_n_bits(R_DEBUG_BIT)
        o_proj.act_quantizer.change_n_bits(R_DEBUG_BIT)


def LLaMA_R4_reorder(post_attention_layernorm, gate_proj, up_proj, index, counts):
    """R4: Post-attention layer norm reordering for LLaMA"""
    post_attention_layernorm.register_buffer("reorder_index", index)

    post_attention_layernorm.out_quantizer.cluster_dim = 1
    post_attention_layernorm.out_quantizer.cluster_counts = counts

    gate_proj.weight.data = torch.index_select(gate_proj.weight.data, 1, index)
    gate_proj.set_ic_cluster_counts(counts, a_dim=None)
    
    up_proj.weight.data = torch.index_select(up_proj.weight.data, 1, index)
    up_proj.set_ic_cluster_counts(counts, a_dim=None)
    
    if R_DEBUG_BIT:
        post_attention_layernorm.out_quantizer.change_n_bits(R_DEBUG_BIT)


def LLaMA_R5_reorder(gate_proj, up_proj, down_proj, index, counts):
    """R5: MLP layer reordering for LLaMA"""
    gate_proj.weight.data = torch.index_select(gate_proj.weight.data, 0, index)
    up_proj.weight.data = torch.index_select(up_proj.weight.data, 0, index)

    down_proj.weight.data = torch.index_select(down_proj.weight.data, 1, index)
    down_proj.set_ic_cluster_counts(counts, a_dim=1)
    if R_DEBUG_BIT:
        down_proj.act_quantizer.change_n_bits(R_DEBUG_BIT)


@torch.no_grad()
def llama_reorder_quantize(
    lm,
    args,
    dataloader,
    n_clusters={"R1": 4, "R2": 4, "R3": 4, "R4": 32, "R5": 4},
    reorder="12345",
):
    print("Starting LLaMA quantization...")

    model = lm.model
    dev = lm.device

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    # only catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask", None)
            raise ValueError

    layers[0] = Catcher(layers[0])

    for batch in dataloader:
        if cache["i"] >= args.nsamples:
            break
        try:
            model(batch[0].to(dev))

        except ValueError:
            pass

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    enable_R1 = True if "1" in reorder else False
    enable_R2 = True if "2" in reorder else False
    enable_R3 = True if "3" in reorder else False
    enable_R4 = True if "4" in reorder else False
    enable_R5 = True if "5" in reorder else False
    print(f"Ready for LLaMA reorder {reorder}.")

    for i in range(len(layers)):
        if i == DEBUG_BREAK_LAYER:
            break
        print(f"=== Start quantize LLaMA layer {i} ===")
        layer = layers[i].to(dev)
        qlayer = QuantLLaMADecoderLayer(lm.model.config, layer, args)

        # register hook for data
        handlers = []
        for name, module in layer.named_modules():
            if (
                enable_R1
                and isinstance(module, nn.LayerNorm)
                and "input_layernorm" in name
            ):
                module.name = name
                handler = module.register_forward_hook(layer_omax_hook)
                handlers.append(handler)
            if (
                enable_R2
                and isinstance(module, nn.Linear)
                and ("q_proj" in name or "k_proj" in name)
            ):
                module.name = name
                handler = module.register_forward_hook(layer_omax_hook)
                handlers.append(handler)
            if enable_R3 and isinstance(module, nn.Linear) and "o_proj" in name:
                module.name = name
                handler = module.register_forward_hook(layer_i0max_hook)
                handlers.append(handler)
            if (
                enable_R4
                and isinstance(module, nn.LayerNorm)
                and "post_attention_layernorm" in name
            ):
                module.name = name
                handler = module.register_forward_hook(layer_omax_hook)
                handlers.append(handler)
            if enable_R5 and isinstance(module, nn.Linear) and "down_proj" in name:
                module.name = name
                handler = module.register_forward_hook(layer_i0max_hook)
                handlers.append(handler)

        # inference to collect data for reordering
        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0).to(dev), 
                attention_mask=attention_mask.to(dev) if attention_mask is not None else None
            )[0]
        for handler in handlers:
            handler.remove()

        if enable_R1:
            feature_max, feature_min = oc_maxmin_dict[f"input_layernorm"]

            R1_index, counts = tensor_calc_reorder_index(
                feature_max, feature_min, n_clusters["R1"]
            )
            LLaMA_R1_reorder(
                qlayer.input_layernorm,
                qlayer.self_attn.q_proj,
                qlayer.self_attn.k_proj,
                qlayer.self_attn.v_proj,
                R1_index,
                counts,
            )

        if enable_R2:
            qmax, qmin = oc_maxmin_dict[f"q_proj"]
            kmax, kmin = oc_maxmin_dict[f"k_proj"]
            R2_index, counts = tensor_calc_reorder_index(
                [qmax, kmax], [qmin, kmin], n_clusters["R2"], qlayer.self_attn.num_heads
            )
            LLaMA_R2_reorder(
                qlayer.self_attn.q_proj,
                qlayer.self_attn.k_proj,
                qlayer.self_attn.qk_matmul,
                R2_index,
                counts,
            )

        if enable_R3:
            feature_max, feature_min = ic_maxmin_dict[f"o_proj"]
            R3_index, counts = tensor_calc_reorder_index(
                feature_max, feature_min, n_clusters["R3"], qlayer.self_attn.num_heads
            )
            LLaMA_R3_reorder(
                qlayer.self_attn.v_proj,
                qlayer.self_attn.pv_matmul,
                qlayer.self_attn.o_proj,
                R3_index,
                counts,
            )

        if enable_R4:
            feature_max, feature_min = oc_maxmin_dict[f"post_attention_layernorm"]

            R4_index, counts = tensor_calc_reorder_index(
                feature_max, feature_min, n_clusters["R4"]
            )
            LLaMA_R4_reorder(
                qlayer.post_attention_layernorm,
                qlayer.mlp.gate_proj,
                qlayer.mlp.up_proj,
                R4_index,
                counts,
            )

        if enable_R5:
            feature_max, feature_min = ic_maxmin_dict[f"down_proj"]
            R5_index, counts = tensor_calc_reorder_index(
                feature_max, feature_min, n_clusters["R5"]
            )
            LLaMA_R5_reorder(
                qlayer.mlp.gate_proj,
                qlayer.mlp.up_proj,
                qlayer.mlp.down_proj,
                R5_index,
                counts,
            )

        outs = quant_layer(qlayer, args, outs, inps, attention_mask, dev)

        ic_maxmin_dict.clear()
        oc_maxmin_dict.clear()
        layers[i] = qlayer.to("cpu")
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps
        print(
            lm._device,
            "memory_allocated",
            i,
            torch.cuda.memory_allocated(lm._device) / 1024 / 1024,
            "max memory_allocated",
            torch.cuda.max_memory_allocated(lm._device) / 1024**2,
        )

    del inps, outs
    model.config.use_cache = use_cache
    return model


if __name__ == "__main__":
    tensor = torch.rand([30])
    index, counts = tensor_calc_reorder_index(tensor, 2, 3)
    print(index, counts)
