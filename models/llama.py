import os
import torch
from .models_utils import BaseLM, find_layers
from transformers import LlamaForCausalLM, AutoTokenizer, AutoConfig
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm


class LlamaClass(BaseLM):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self._device = torch.device("cpu")
        self.model_name = args.model
        self.batch_size_per_gpu = args.batch_size

        model_kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
        tokenizer_kwargs = {"use_fast": False}
        config_kwargs = {}
        if args.cache_dir and not os.path.isdir(self.model_name):
            model_kwargs["cache_dir"] = args.cache_dir
            tokenizer_kwargs["cache_dir"] = args.cache_dir
            config_kwargs["cache_dir"] = args.cache_dir

        hidden_layers = getattr(args, "hidden_layer_num", None)
        if hidden_layers:
            config = AutoConfig.from_pretrained(self.model_name, **config_kwargs)
            config.num_hidden_layers = hidden_layers
            self.model = LlamaForCausalLM.from_pretrained(
                self.model_name, config=config, **model_kwargs
            )
        else:
            self.model = LlamaForCausalLM.from_pretrained(
                self.model_name, **model_kwargs
            )
        self.seqlen = self.model.config.max_position_embeddings
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, **tokenizer_kwargs
        )
        self.vocab_size = self.tokenizer.vocab_size
        print("Llama vocab size: ", self.vocab_size)

    def truncate_hidden_layers(self, hidden_layer_num: int):
        total_layers = len(self.model.model.layers)
        if hidden_layer_num is not None and hidden_layer_num < total_layers:
            self.model.model.layers = nn.ModuleList(
                self.model.model.layers[:hidden_layer_num]
            )
            self.model.config.num_hidden_layers = hidden_layer_num
            print(
                f"Using first {hidden_layer_num} of {total_layers} Llama layers for debugging"
            )

    @property
    def eot_token(self) -> str:
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_encode_batch(self, strings):
        return self.tokenizer(
            strings,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        )

    def tok_decode(self, tokens):
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def _model_call(self, inps):
        with torch.no_grad():
            return self.model(inps)["logits"]

    def model_batched_set(self, inps):
        dataset_logits = []
        for batch in inps:
            multi_logits = F.log_softmax(self._model_call(batch), dim=-1).cpu()
            dataset_logits.append(multi_logits)
        return dataset_logits

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )


# for backwards compatibility
Llama = LlamaClass
