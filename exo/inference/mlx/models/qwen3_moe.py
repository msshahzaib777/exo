# Copyright © 2025
# Layer-parallel shim for Qwen-3-MoE (MLX)

from dataclasses import dataclass, field
from typing import Optional, Any

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.qwen3_moe import  ModelArgs, Qwen3MoeDecoderLayer

from ...shard import Shard          # same helper object you already use
from .base import IdentityBlock     # trivial x -> x module


# --------------------------------------------------------------------
# 1.  Extend ModelArgs with a shard description
# --------------------------------------------------------------------
@dataclass
class ModelArgs(ModelArgs):
    shard: Shard = field(default_factory=lambda: Shard("", 0, 0, 0))

    def __post_init__(self):
        parent_post = getattr(super(), "__post_init__", None)
        if parent_post is not None:
            parent_post()
        if isinstance(self.shard, Shard):
            return
        if not isinstance(self.shard, dict):
            raise TypeError("shard must be Shard or dict")
        self.shard = Shard(**self.shard)


# --------------------------------------------------------------------
# 2.  Shard-aware backbone
# --------------------------------------------------------------------
class Qwen3MoeModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        # first shard owns embeddings (and LM-head if tied)
        if args.shard.is_first_layer() or (
            args.shard.is_last_layer() and args.tie_word_embeddings
        ):
            self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)

        # build decoder stack with identity pass-through where needed
        self.layers = []
        for i in range(args.num_hidden_layers):
            if args.shard.start_layer <= i <= args.shard.end_layer:
                self.layers.append(Qwen3MoeDecoderLayer(args, layer_idx=i))
            else:
                self.layers.append(IdentityBlock())

        # last shard owns the final RMSNorm
        if args.shard.is_last_layer():
            self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    # ---- forward ----
    def __call__(self, input_ids: mx.array, cache=None):
        # embeddings on first shard only
        hidden = (
            self.embed_tokens(input_ids)
            if self.args.shard.is_first_layer()
            else input_ids
        )

        mask = create_attention_mask(hidden, cache) if hidden.shape[1] > 1 else None
        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            hidden = layer(hidden, mask, c)

        if self.args.shard.is_last_layer():
            hidden = self.norm(hidden)
        return hidden


# --------------------------------------------------------------------
# 3.  Full Causal-LM wrapper
# --------------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen3MoeModel(args)

        if args.shard.is_last_layer() and not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, input_ids: mx.array, cache=None):
        out = self.model(input_ids, cache)
        if self.args.shard.is_last_layer():
            out = (
                self.model.embed_tokens.as_linear(out)
                if self.args.tie_word_embeddings
                else self.lm_head(out)
            )
        return out.astype(mx.float32)

    # ----------------------------------------------------------------
    # 4.  Strip weights to just this shard
    # ----------------------------------------------------------------
    def sanitize(self, weights):
        shard_state_dict = {}
        for k, v in weights.items():
            if "self_attn.rotary_emb.inv_freq" in k:
                continue
            if k.startswith("model.layers."):
                layer_num = int(k.split(".")[2])
                if self.args.shard.start_layer <= layer_num <= self.args.shard.end_layer:
                    shard_state_dict[k] = v
            elif self.args.shard.is_first_layer() and k.startswith("model.embed_tokens"):
                shard_state_dict[k] = v
            elif (
                self.args.shard.is_last_layer() and self.args.tie_word_embeddings
                and k.startswith("model.embed_tokens")
            ):
                shard_state_dict[k] = v
            elif (
                self.args.shard.is_last_layer()
                and not self.args.tie_word_embeddings
                and k.startswith("lm_head")
            ):
                shard_state_dict[k] = v
            elif self.args.shard.is_last_layer() and k.startswith("model.norm"):
                shard_state_dict[k] = v
        if self.args.tie_word_embeddings:
            shard_state_dict.pop("lm_head.weight", None)
        return shard_state_dict

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.hidden_size // self.args.num_attention_heads

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads
