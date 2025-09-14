import transformers
import torch
from .models_utils import BaseLM, find_layers
from transformers import LlamaForCausalLM, AutoTokenizer
import torch.nn.functional as F
from torch import nn
import torch
from tqdm import tqdm
import pdb


class LLaMAClass(BaseLM):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = args.model
        self.batch_size_per_gpu = args.batch_size
        
        self.model_config = args.model
        self.model = LlamaForCausalLM.from_pretrained(
            self.model_name, 
            cache_dir=args.cache_dir, 
            torch_dtype="auto",
            trust_remote_code=True
        )
        self.seqlen = self.model.config.max_position_embeddings
        self.model.eval()
        
        # LLaMA tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            cache_dir=args.cache_dir, 
            use_fast=False,
            trust_remote_code=True
        )
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.vocab_size = self.tokenizer.vocab_size
        print("LLaMA vocab size: ", self.vocab_size)
        
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
        print("max_gen_toks fn")
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
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call
        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.model(inps)["logits"]
            
    def model_batched_set(self, inps):
        dataset_logits = []
        for batch in inps:
            multi_logits = F.log_softmax(
                self._model_call(batch), dim=-1
            ).cpu()  # [batch, padding_length, vocab]
            dataset_logits.append(multi_logits)
        return dataset_logits
        
    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context, 
            max_length=max_length, 
            eos_token_id=eos_token_id, 
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id
        )


# for backwards compatibility
LLaMA = LLaMAClass
