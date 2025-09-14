import torch
from torch import nn
from typing import Optional, Tuple, List
from quantize.int_linear import QuantLinear
from quantize.int_matmul import QuantMatMul
import torch.nn.functional as F
from quantize.reorder_layer_norm import ReorderLayerNorm
import pdb
import math


class QuantLLaMAAttention(nn.Module):
    """Multi-headed attention for LLaMA models"""

    def __init__(
        self,
        org_module: nn.Module,
        config,
        args=None,
        disable_act_quant=False,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = getattr(config, 'rope_theta', 10000.0)
        
        if (self.hidden_size % self.num_heads) != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.head_dim = self.hidden_size // self.num_heads
        self.scaling = self.head_dim**-0.5

        # input is quantized by LayerNorm, set disable_input_quant=True
        self.q_proj = QuantLinear(
            org_module.q_proj,
            args.weight_quant_params,
            args.act_quant_params,
            disable_input_quant=True,
        )
        self.k_proj = QuantLinear(
            org_module.k_proj,
            args.weight_quant_params,
            args.act_quant_params,
            disable_input_quant=True,
        )
        self.v_proj = QuantLinear(
            org_module.v_proj,
            args.weight_quant_params,
            args.act_quant_params,
            disable_input_quant=True,
        )
        self.o_proj = QuantLinear(
            org_module.o_proj, 
            args.weight_quant_params, 
            args.act_quant_params
        )
        
        # For LLaMA, we need to handle QK and PV matrix multiplications
        self.qk_matmul = QuantMatMul(
            args.q_quant_params, 
            args.k_quant_params, 
            matmul_func=torch.bmm
        )
        self.pv_matmul = QuantMatMul(
            args.p_quant_params, 
            args.v_quant_params, 
            matmul_func=torch.bmm
        )

        # Add rotary embedding
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

        self.use_weight_quant = False
        self.use_act_quant = False

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _shape_kv(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2).contiguous()

    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_pos_emb(self, q, k, cos, sin, position_ids):
        # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
        cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
        sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = self._apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # Use quantized matrix multiplication for QK
        query_states_flat = query_states.view(bsz * self.num_heads, q_len, self.head_dim)
        key_states_flat = key_states.view(bsz * self.num_heads, kv_seq_len, self.head_dim)
        attn_weights = self.qk_matmul(query_states_flat, key_states_flat.transpose(1, 2)) * self.scaling
        attn_weights = attn_weights.view(bsz, self.num_heads, q_len, kv_seq_len)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Use quantized matrix multiplication for PV
        attn_weights_flat = attn_weights.view(bsz * self.num_heads, q_len, kv_seq_len)
        value_states_flat = value_states.view(bsz * self.num_heads, kv_seq_len, self.head_dim)
        attn_output = self.pv_matmul(attn_weights_flat, value_states_flat)
        attn_output = attn_output.view(bsz, self.num_heads, q_len, self.head_dim)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                m.set_quant_state(weight_quant, act_quant)


class QuantLLaMADecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        ori_layer,
        args,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = QuantLLaMAAttention(
            org_module=ori_layer.self_attn,
            config=config,
            args=args,
        )
        
        self.mlp = QuantLLaMAMLP(
            org_module=ori_layer.mlp,
            config=config,
            args=args,
        )
        
        self.input_layernorm = ReorderLayerNorm(
            ori_layer.input_layernorm, 
            args.layer_norm_out_quant_params
        )
        self.post_attention_layernorm = ReorderLayerNorm(
            ori_layer.post_attention_layernorm, 
            args.layer_norm_out_quant_params
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for name, m in self.named_modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                m.set_quant_state(weight_quant, act_quant)
            if isinstance(m, ReorderLayerNorm):
                m.set_quant_state(weight_quant, act_quant)


class QuantLLaMAMLP(nn.Module):
    def __init__(
        self,
        org_module: nn.Module,
        config,
        args,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = QuantLinear(
            org_module.gate_proj,
            args.weight_quant_params,
            args.act_quant_params,
            disable_input_quant=True,
        )
        self.up_proj = QuantLinear(
            org_module.up_proj,
            args.weight_quant_params,
            args.act_quant_params,
            disable_input_quant=True,
        )
        self.down_proj = QuantLinear(
            org_module.down_proj,
            args.weight_quant_params,
            args.act_quant_params,
        )

    def forward(self, x):
        gate_proj = self.gate_proj(x)
        up_proj = self.up_proj(x)
        return self.down_proj(F.silu(gate_proj) * up_proj)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.modules():
            if isinstance(m, QuantLinear):
                m.set_quant_state(weight_quant, act_quant)
