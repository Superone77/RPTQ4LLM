import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from quantize.int_linear import QuantLinear
from quantize.quant_transformer_layer import quant_layer
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from types import MethodType

# Hard-coded parameters
MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
SEQLEN = 2048
NSAMPLES = 32
WBITS = 4
ABITS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Prepare tokenizer and model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
model.to(DEVICE)
model.eval()

# Load dataset
print("Preparing wikitext2 dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_text = "\n\n".join(dataset["train"]["text"])
val_text = "\n\n".join(dataset["validation"]["text"])
train_enc = tokenizer(train_text, return_tensors="pt")
val_enc = tokenizer(val_text, return_tensors="pt")

# Build calibration samples
print("Building calibration samples...")
calib_data = []
for _ in range(NSAMPLES):
    i = random.randint(0, train_enc.input_ids.shape[1] - SEQLEN - 1)
    calib_data.append(train_enc.input_ids[:, i:i+SEQLEN])
calib_data = torch.cat(calib_data, dim=0)

# Replace linear layers with quantized versions
def replace_linear_with_quant(module):
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            qlinear = QuantLinear(
                child,
                weight_quant_params={"n_bits": WBITS, "per_channel_axes": [0], "symmetric": False, "metric": "minmax"},
                act_quant_params={"n_bits": ABITS, "per_channel_axes": [], "symmetric": False, "metric": "minmax", "dynamic": False},
            )
            setattr(module, name, qlinear)
        else:
            replace_linear_with_quant(child)

replace_linear_with_quant(model)

# Add set_quant_state to transformer blocks
def _set_quant_state(self, weight_quant=True, act_quant=True):
    for m in self.modules():
        if isinstance(m, QuantLinear):
            m.set_quant_state(weight_quant, act_quant)

for layer in model.model.layers:
    layer.set_quant_state = MethodType(_set_quant_state, layer)

# Collect inputs for the first layer
print("Collecting layer inputs...")
model.config.use_cache = False
layers = model.model.layers
inps = torch.zeros((NSAMPLES, SEQLEN, model.config.hidden_size), dtype=torch.float16, device=DEVICE)
outs = torch.zeros_like(inps)
attention_mask = torch.ones((1, SEQLEN), dtype=torch.long, device=DEVICE)
cache = {"i": 0}

class Catcher(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    def forward(self, inp, **kwargs):
        inps[cache["i"]] = inp
        cache["i"] += 1
        return self.module(inp, **kwargs)

layers[0] = Catcher(layers[0])
with torch.no_grad():
    for j in range(NSAMPLES):
        model(calib_data[j:j+1].to(DEVICE))
layers[0] = layers[0].module

# Quantize each layer
print("Quantizing layers...")
for i, layer in enumerate(layers):
    outs = quant_layer(layer, type("Args", (), {"nsamples": NSAMPLES, "metric": "minmax", "disable_w_quant": False, "disable_a_quant": False, "w_quantizer": "normal", "pack_weight": False, "percdamp": 0.01})(), inps, outs, attention_mask, DEVICE)
    inps, outs = outs, inps

# Evaluate perplexity using lm_eval
print("Evaluating perplexity...")
quant_lm = HFLM(model=model, tokenizer=tokenizer)
results = evaluator.simple_evaluate(model=quant_lm, tasks=["wikitext"], num_fewshot=0)
print("Wikitext2 perplexity:", results["results"]["wikitext"]["ppl"])
