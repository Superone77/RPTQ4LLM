import torch
from torch import nn
from typing import Optional, Tuple
import torch.nn.functional as F
from quantize.int_linear import QuantLinear
from quantize.int_matmul import QuantMatMul
from quantize.reorder_layer_norm import ReorderLayerNorm
from quantize.int_rotary_emb import IntRotaryEmbedding


class QuantLlamaAttention(nn.Module):
    def __init__(
        self,
        org_module: nn.Module,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        args=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim ** -0.5

        # quantized linear projections
        self.k_proj = QuantLinear(
            org_module.k_proj, args.weight_quant_params, args.act_quant_params, disable_input_quant=True
        )
        self.v_proj = QuantLinear(
            org_module.v_proj, args.weight_quant_params, args.act_quant_params, disable_input_quant=True
        )
        self.q_proj = QuantLinear(
            org_module.q_proj, args.weight_quant_params, args.act_quant_params, disable_input_quant=True
        )
        self.o_proj = QuantLinear(
            org_module.o_proj, args.weight_quant_params, args.act_quant_params
        )

        self.qkt_matmul = QuantMatMul(args.q_quant_params, args.k_quant_params, matmul_func=torch.bmm)
        self.pv_matmul = QuantMatMul(args.p_quant_params, args.v_quant_params, matmul_func=torch.bmm)
        self.rotary_emb = IntRotaryEmbedding(org_module.rotary_emb, self.head_dim, num_heads)
        self.is_decoder = True

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, tgt_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self.qkt_matmul.quant_x1(query_states)
        key_states = self.qkt_matmul.quant_x2(key_states)
        value_states = self.pv_matmul.quant_x2(value_states)

        query_states = self._shape(query_states, tgt_len, bsz)
        key_states = self._shape(key_states, tgt_len, bsz)
        value_states = self._shape(value_states, tgt_len, bsz)

        query_states, key_states = self.rotary_emb(query_states, seq_len=tgt_len, q=query_states, k=key_states)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = query_states.view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        attn_weights = self.qkt_matmul(query_states, key_states.transpose(1, 2))
        if attention_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, -1)
            attn_weights = attn_weights + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, -1)

        attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_probs = attn_probs.to(query_states.dtype)

        attn_output = self.pv_matmul(attn_probs, value_states)
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.o_proj(attn_output)

        if output_attentions:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, -1)
        else:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for name, m in self.named_modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                m.set_quant_state(weight_quant, act_quant)
            if isinstance(m, ReorderLayerNorm):
                m.set_quant_state(weight_quant, act_quant)


class QuantLlamaDecoderLayer(nn.Module):
    def __init__(self, config, ori_layer, args):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = QuantLlamaAttention(
            org_module=ori_layer.self_attn,
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            args=args,
        )
        self.input_layernorm = ReorderLayerNorm(
            ori_layer.input_layernorm, args.layer_norm_out_quant_params
        )
        self.post_attention_layernorm = ReorderLayerNorm(
            ori_layer.post_attention_layernorm, args.layer_norm_out_quant_params
        )
        self.gate_proj = QuantLinear(
            ori_layer.mlp.gate_proj,
            weight_quant_params=args.weight_quant_params,
            act_quant_params=args.act_quant_params,
            disable_input_quant=True,
        )
        self.up_proj = QuantLinear(
            ori_layer.mlp.up_proj,
            weight_quant_params=args.weight_quant_params,
            act_quant_params=args.act_quant_params,
            disable_input_quant=True,
        )
        self.down_proj = QuantLinear(
            ori_layer.mlp.down_proj,
            weight_quant_params=args.weight_quant_params,
            act_quant_params=args.act_quant_params,
        )
        self.dropout = config.dropout

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        hidden_states = F.silu(gate) * up
        hidden_states = self.down_proj(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for name, m in self.named_modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                m.set_quant_state(weight_quant, act_quant)
            if isinstance(m, ReorderLayerNorm):
                m.set_quant_state(weight_quant, act_quant)
