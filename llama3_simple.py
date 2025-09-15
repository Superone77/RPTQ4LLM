import types
import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HuggingFaceCausalLM
from datautils import get_loaders
from quantize.int_linear import QuantLinear
from quantize.reorder_layer_norm import ReorderLayerNorm
from quantize.quant_transformer_layer import quant_layer

# Hard coded parameters
MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
SEQLEN = 2048
NSAMPLES = 128

class Args:
    wbits = 4
    abits = 4
    nsamples = NSAMPLES
    percdamp = 0.01
    metric = "ema_minmax"
    seed = 2
    disable_w_quant = False
    disable_a_quant = False
    w_quantizer = "gptq"
    pack_weight = False
    a_dynamic = False

args = Args()
args.weight_quant_params = {
    "n_bits": args.wbits,
    "per_channel_axes": [0],
    "symmetric": False,
    "metric": "minmax",
}
args.act_quant_params = {
    "n_bits": args.abits,
    "per_channel_axes": [],
    "symmetric": False,
    "metric": args.metric,
    "dynamic": args.a_dynamic,
}
args.q_quant_params = args.act_quant_params
args.k_quant_params = args.act_quant_params
args.v_quant_params = args.act_quant_params
args.layer_norm_out_quant_params = {
    "n_bits": max(8, args.abits),
    "per_channel_axes": [],
    "symmetric": False,
    "metric": args.metric,
    "dynamic": args.a_dynamic,
}
args.p_quant_params = {"n_bits": max(8, args.abits), "metric": "fix0to1"}

# Load model via lm_eval harness
lm = HuggingFaceCausalLM(pretrained=MODEL_NAME, dtype="float16", device_map="auto")
model = lm.model
model.eval()
lm.seqlen = SEQLEN

device = next(model.parameters()).device

# Replace layers with quantized modules
for layer in model.model.layers:
    attn = layer.self_attn
    attn.q_proj = QuantLinear(attn.q_proj, args.weight_quant_params, args.act_quant_params, disable_input_quant=True)
    attn.k_proj = QuantLinear(attn.k_proj, args.weight_quant_params, args.act_quant_params, disable_input_quant=True)
    attn.v_proj = QuantLinear(attn.v_proj, args.weight_quant_params, args.act_quant_params, disable_input_quant=True)
    attn.o_proj = QuantLinear(attn.o_proj, args.weight_quant_params, args.act_quant_params)

    mlp = layer.mlp
    mlp.gate_proj = QuantLinear(mlp.gate_proj, args.weight_quant_params, args.act_quant_params)
    mlp.up_proj = QuantLinear(mlp.up_proj, args.weight_quant_params, args.act_quant_params)
    mlp.down_proj = QuantLinear(mlp.down_proj, args.weight_quant_params, args.act_quant_params)

    layer.input_layernorm = ReorderLayerNorm(layer.input_layernorm, args.layer_norm_out_quant_params)
    layer.post_attention_layernorm = ReorderLayerNorm(layer.post_attention_layernorm, args.layer_norm_out_quant_params)

    def set_quant_state(self, weight_quant, act_quant):
        for m in self.modules():
            if isinstance(m, (QuantLinear, ReorderLayerNorm)):
                m.set_quant_state(weight_quant, act_quant)
    layer.set_quant_state = types.MethodType(set_quant_state, layer)

model.model.norm = ReorderLayerNorm(model.model.norm, args.layer_norm_out_quant_params)

# Prepare calibration data
dataloader, _ = get_loaders("wikitext2", nsamples=NSAMPLES, seed=args.seed, model=MODEL_NAME, seqlen=SEQLEN, cache_dir=MODEL_NAME)


layers = model.model.layers
dtype = next(model.parameters()).dtype
inps = torch.zeros((args.nsamples, SEQLEN, model.config.hidden_size), dtype=dtype, device=device)
cache = {"i": 0, "attention_mask": None}

class Catcher(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    def forward(self, inp, **kwargs):
        inps[cache["i"]] = inp
        cache["i"] += 1
        cache["attention_mask"] = kwargs.get("attention_mask")
        raise ValueError

layers[0] = Catcher(layers[0])
for batch in dataloader:
    try:
        model(batch[0].to(device))
    except ValueError:
        pass
    if cache["i"] >= args.nsamples:
        break
layers[0] = layers[0].module
outs = torch.zeros_like(inps)
attention_mask = cache["attention_mask"]

for layer in layers:
    layer.to(device)
    outs = quant_layer(layer, args, outs, inps, attention_mask, device)
    inps, outs = outs, inps
    layer.cpu()

torch.cuda.empty_cache()
for layer in layers:
    layer.set_quant_state(True, True)
model.model.norm.set_quant_state(False, True)

# Evaluate PPL on wikitext2 using lm_eval
results = evaluator.simple_evaluate(lm, tasks=["wikitext"], num_fewshot=0)
print("wikitext ppl:", results["results"]["wikitext"]["word_perplexity"])
