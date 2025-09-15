import torch
import torch.nn as nn
from quantize.int_linear import QuantLinear
from quantize.int_matmul import QuantMatMul
from quantize.reorder_layer_norm import ReorderLayerNorm
from models.int_llama_layer import QuantLlamaDecoderLayer
from quantize.quant_transformer_layer import quant_layer
from quantize.reorder_utils import (
    tensor_calc_reorder_index,
    ic_maxmin_dict,
    oc_maxmin_dict,
    oc_maxmin_dict_debug,
    layer_i0max_hook,
    layer_omax_hook,
)

R_DEBUG_BIT = 0
DEBUG_BREAK_LAYER = -1


def R1_reorder(layer_norm, qproj, kproj, vproj, index, counts):
    layer_norm.register_buffer("reorder_index", index)
    layer_norm.out_quantizer.cluster_dim = 2
    layer_norm.out_quantizer.cluster_counts = counts
    if R_DEBUG_BIT:
        layer_norm.out_quantizer.change_n_bits(R_DEBUG_BIT)

    qproj.weight.data = torch.index_select(qproj.weight.data, 1, index)
    qproj.set_ic_cluster_counts(counts, a_dim=None)

    kproj.weight.data = torch.index_select(kproj.weight.data, 1, index)
    kproj.set_ic_cluster_counts(counts, a_dim=None)
    vproj.weight.data = torch.index_select(vproj.weight.data, 1, index)
    vproj.set_ic_cluster_counts(counts, a_dim=None)


def R2_reorder(qproj, kproj, qkt_matmul, index, counts):
    qproj.weight.data = torch.index_select(qproj.weight.data, 0, index)
    qproj.bias.data = torch.index_select(qproj.bias.data, 0, index)
    kproj.weight.data = torch.index_select(kproj.weight.data, 0, index)
    kproj.bias.data = torch.index_select(kproj.bias.data, 0, index)

    qkt_matmul.set_ic_cluster_counts(counts, x1_dim=2, x2_dim=2)
    if R_DEBUG_BIT:
        qkt_matmul.x1_quantizer.change_n_bits(R_DEBUG_BIT)
        qkt_matmul.x2_quantizer.change_n_bits(R_DEBUG_BIT)


def R3_reorder(vproj, pv_matmul, oproj, index, counts):
    vproj.weight.data = torch.index_select(vproj.weight.data, 0, index)
    vproj.bias.data = torch.index_select(vproj.bias.data, 0, index)
    pv_matmul.set_ic_cluster_counts(counts, cluster_x1=False)
    oproj.weight.data = torch.index_select(oproj.weight.data, 1, index)
    oproj.set_ic_cluster_counts(counts)
    if R_DEBUG_BIT:
        pv_matmul.x2_quantizer.change_n_bits(R_DEBUG_BIT)
        oproj.act_quantizer.change_n_bits(R_DEBUG_BIT)


def R4_reorder(layer_norm, gate_proj, up_proj, index, counts):
    layer_norm.register_buffer("reorder_index", index)
    layer_norm.out_quantizer.cluster_dim = 1
    layer_norm.out_quantizer.cluster_counts = counts
    gate_proj.weight.data = torch.index_select(gate_proj.weight.data, 1, index)
    gate_proj.set_ic_cluster_counts(counts, a_dim=None)
    up_proj.weight.data = torch.index_select(up_proj.weight.data, 1, index)
    up_proj.set_ic_cluster_counts(counts, a_dim=None)
    if R_DEBUG_BIT:
        layer_norm.out_quantizer.change_n_bits(R_DEBUG_BIT)


def R5_reorder(gate_proj, up_proj, down_proj, index, counts):
    gate_proj.weight.data = torch.index_select(gate_proj.weight.data, 0, index)
    gate_proj.bias.data = torch.index_select(gate_proj.bias.data, 0, index)
    up_proj.weight.data = torch.index_select(up_proj.weight.data, 0, index)
    up_proj.bias.data = torch.index_select(up_proj.bias.data, 0, index)
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
    print("Starting ...")
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

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
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

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        qlayer = QuantLlamaDecoderLayer(model.config, layer, args)
        enable_R1 = "1" in reorder
        enable_R2 = "2" in reorder
        enable_R3 = "3" in reorder
        enable_R4 = "4" in reorder
        enable_R5 = "5" in reorder
        handlers = []
        if DEBUG_BREAK_LAYER >= 0 and i == DEBUG_BREAK_LAYER:
            import pdb; pdb.set_trace()
        for name, module in qlayer.named_modules():
            if enable_R1 and isinstance(module, ReorderLayerNorm) and "input_layernorm" in name:
                module.name = name
                handler = module.register_forward_hook(layer_omax_hook)
                handlers.append(handler)
            if enable_R1 and isinstance(module, QuantLinear) and "self_attn.q_proj" in name:
                module.name = name
                handler = module.register_forward_hook(layer_i0max_hook)
                handlers.append(handler)
            if enable_R1 and isinstance(module, QuantLinear) and "self_attn.k_proj" in name:
                module.name = name
                handler = module.register_forward_hook(layer_i0max_hook)
                handlers.append(handler)
            if enable_R1 and isinstance(module, QuantLinear) and "self_attn.v_proj" in name:
                module.name = name
                handler = module.register_forward_hook(layer_i0max_hook)
                handlers.append(handler)
            if enable_R2 and isinstance(module, QuantLinear) and "self_attn.q_proj" in name:
                module.name = name
                handler = module.register_forward_hook(layer_omax_hook)
                handlers.append(handler)
            if enable_R2 and isinstance(module, QuantLinear) and "self_attn.k_proj" in name:
                module.name = name
                handler = module.register_forward_hook(layer_omax_hook)
                handlers.append(handler)
            if enable_R3 and isinstance(module, QuantLinear) and "self_attn.v_proj" in name:
                module.name = name
                handler = module.register_forward_hook(layer_i0max_hook)
                handlers.append(handler)
            if enable_R3 and isinstance(module, QuantLinear) and "self_attn.o_proj" in name:
                module.name = name
                handler = module.register_forward_hook(layer_omax_hook)
                handlers.append(handler)
            if enable_R4 and isinstance(module, ReorderLayerNorm) and "post_attention_layernorm" in name:
                module.name = name
                handler = module.register_forward_hook(layer_omax_hook)
                handlers.append(handler)
            if enable_R4 and isinstance(module, QuantLinear) and "gate_proj" in name:
                module.name = name
                handler = module.register_forward_hook(layer_i0max_hook)
                handlers.append(handler)
            if enable_R4 and isinstance(module, QuantLinear) and "up_proj" in name:
                module.name = name
                handler = module.register_forward_hook(layer_i0max_hook)
                handlers.append(handler)
            if enable_R5 and isinstance(module, QuantLinear) and "down_proj" in name:
                module.name = name
                handler = module.register_forward_hook(layer_i0max_hook)
                handlers.append(handler)

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0).to(dev), attention_mask=attention_mask.to(dev))[0]
        for handler in handlers:
            handler.remove()

        if enable_R1:
            feature_max, feature_min = oc_maxmin_dict[f"input_layernorm"]
            R1_index, counts = tensor_calc_reorder_index(feature_max, feature_min, n_clusters["R1"])
            R1_reorder(
                qlayer.input_layernorm,
                qlayer.self_attn.q_proj,
                qlayer.self_attn.k_proj,
                qlayer.self_attn.v_proj,
                R1_index,
                counts,
            )

        if enable_R2:
            qmax, qmin = oc_maxmin_dict[f"self_attn.q_proj"]
            kmax, kmin = oc_maxmin_dict[f"self_attn.k_proj"]
            R2_index, counts = tensor_calc_reorder_index([qmax, kmax], [qmin, kmin], n_clusters["R2"], qlayer.self_attn.num_heads)
            R2_reorder(
                qlayer.self_attn.q_proj,
                qlayer.self_attn.k_proj,
                qlayer.self_attn.qkt_matmul,
                R2_index,
                counts,
            )

        if enable_R3:
            feature_max, feature_min = ic_maxmin_dict[f"self_attn.o_proj"]
            R3_index, counts = tensor_calc_reorder_index(feature_max, feature_min, n_clusters["R3"], qlayer.self_attn.num_heads)
            R3_reorder(
                qlayer.self_attn.v_proj,
                qlayer.self_attn.pv_matmul,
                qlayer.self_attn.o_proj,
                R3_index,
                counts,
            )

        if enable_R4:
            feature_max, feature_min = oc_maxmin_dict[f"post_attention_layernorm"]
            R4_index, counts = tensor_calc_reorder_index(feature_max, feature_min, n_clusters["R4"])
            R4_reorder(
                qlayer.post_attention_layernorm,
                qlayer.gate_proj,
                qlayer.up_proj,
                R4_index,
                counts,
            )

        if enable_R5:
            feature_max, feature_min = ic_maxmin_dict[f"down_proj"]
            R5_index, counts = tensor_calc_reorder_index(feature_max, feature_min, n_clusters["R5"])
            R5_reorder(
                qlayer.gate_proj,
                qlayer.up_proj,
                qlayer.down_proj,
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
            torch.cuda.max_memory_allocated(lm._device) / 1024 ** 2,
        )

    del inps, outs
    model.config.use_cache = use_cache
    return model
