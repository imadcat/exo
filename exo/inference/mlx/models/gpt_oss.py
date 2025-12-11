from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.gpt_oss import TransformerBlock, ModelArgs

from ...shard import Shard
from .base import IdentityBlock


@dataclass
class ModelArgs(ModelArgs):
  shard: Shard = field(default_factory=lambda: Shard("", 0, 0, 0))

  def __post_init__(self):
    super().__post_init__()

    if isinstance(self.shard, Shard):
      return
    if not isinstance(self.shard, dict):
      raise TypeError(f"Expected shard to be a Shard instance or a dict, got {type(self.shard)} instead")

    self.shard = Shard(**self.shard)


class GptOssMoeModel(nn.Module):
  def __init__(self, args: ModelArgs):
    super().__init__()
    self.args = args
    self.vocab_size = args.vocab_size
    self.num_hidden_layers = args.num_hidden_layers
    assert self.vocab_size > 0

    if self.args.shard.is_first_layer():
      self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)

    # Initialize layer types (alternating sliding and full attention)
    if args.layer_types:
      self.layer_types = args.layer_types
    else:
      # Create alternating pattern to match num_hidden_layers
      pattern = ["sliding_attention", "full_attention"]
      self.layer_types = [pattern[i % 2] for i in range(args.num_hidden_layers)]
    
    self.window_size = args.sliding_window
    # Find the first occurrence of each attention type for cache indexing
    self.swa_idx = next((i for i, lt in enumerate(self.layer_types) if lt == "sliding_attention"), 0)
    self.ga_idx = next((i for i, lt in enumerate(self.layer_types) if lt == "full_attention"), 1)

    self.layers = []
    for i in range(self.num_hidden_layers):
      if self.args.shard.start_layer <= i <= self.args.shard.end_layer:
        self.layers.append(TransformerBlock(args))
      else:
        self.layers.append(IdentityBlock())

    if self.args.shard.is_last_layer():
      self.norm = nn.RMSNorm(args.hidden_size, args.rms_norm_eps)

  def __call__(
    self,
    inputs: mx.array,
    cache=None,
  ):
    if self.args.shard.is_first_layer():
      h = self.embed_tokens(inputs)
    else:
      h = inputs

    if cache is None:
      cache = [None]*len(self.layers)

    # Ensure cache matches the number of layers
    while len(cache) < len(self.layers):
      cache.append(None)
    cache = cache[:len(self.layers)]

    # Create masks for both full and sliding window attention
    # Use the first occurrence of each attention type for mask creation
    ga_cache = cache[self.ga_idx] if 0 <= self.ga_idx < len(cache) else None
    swa_cache = cache[self.swa_idx] if 0 <= self.swa_idx < len(cache) else None
    
    full_mask = create_attention_mask(h, ga_cache)
    swa_mask = create_attention_mask(h, swa_cache, window_size=self.window_size)

    # Ensure layer_types matches the number of layers
    layer_types_iter = self.layer_types[:len(self.layers)]
    if len(layer_types_iter) < len(self.layers):
      # Extend with the pattern if needed
      pattern = ["sliding_attention", "full_attention"]
      for i in range(len(layer_types_iter), len(self.layers)):
        layer_types_iter.append(pattern[i % 2])

    for layer, c, layer_type in zip(self.layers, cache, layer_types_iter):
      mask = full_mask if layer_type == "full_attention" else swa_mask
      h = layer(h, mask, c)

    if self.args.shard.is_last_layer():
      h = self.norm(h)
    return h


class Model(nn.Module):
  def __init__(self, args: ModelArgs):
    super().__init__()
    self.args = args
    self.model_type = args.model_type
    self.model = GptOssMoeModel(args)
    if self.args.shard.is_last_layer():
      self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

  def __call__(
    self,
    inputs: mx.array,
    cache=None,
  ):
    out = self.model(inputs, cache)
    if self.args.shard.is_last_layer():
      out = self.lm_head(out)
    return out

  def sanitize(self, weights):
    # Check if weights are already sanitized by looking for the split gate/up projections
    # In gpt_oss, gate_up_proj is split into gate_proj and up_proj during sanitization
    is_sanitized = any("gate_proj.weight" in k for k in weights.keys()) or not any("gate_up_proj" in k for k in weights.keys())
    
    if is_sanitized:
      # Already sanitized, now filter by shard
      shard_state_dict = {}
      for key, value in weights.items():
        if "self_attn.rotary_emb.inv_freq" in key:
          continue
        if key.startswith('model.layers.'):
          layer_num = int(key.split('.')[2])
          if self.args.shard.start_layer <= layer_num <= self.args.shard.end_layer:
            shard_state_dict[key] = value
        elif self.args.shard.is_first_layer() and key.startswith('model.embed_tokens'):
          shard_state_dict[key] = value
        elif self.args.shard.is_last_layer() and key.startswith('lm_head'):
          shard_state_dict[key] = value
        elif self.args.shard.is_last_layer() and key.startswith('model.norm'):
          shard_state_dict[key] = value
      return shard_state_dict

    # If not sanitized, apply the standard gpt_oss sanitization first
    new_weights = {}
    for k, v in weights.items():
      if "gate_up_proj" in k and "bias" not in k:
        if "_blocks" in k:
          v = v.view(mx.uint32).flatten(-2)
          k = k.replace("_blocks", ".weight")
        if "_scales" in k:
          k = k.replace("_scales", ".scales")
        new_weights[k.replace("gate_up_proj", "gate_proj")] = mx.contiguous(
          v[..., ::2, :]
        )
        new_weights[k.replace("gate_up_proj", "up_proj")] = mx.contiguous(
          v[..., 1::2, :]
        )
      elif "down_proj" in k and "bias" not in k:
        if "_blocks" in k:
          v = v.view(mx.uint32).flatten(-2)
          k = k.replace("_blocks", ".weight")
        if "_scales" in k:
          k = k.replace("_scales", ".scales")
        new_weights[k] = v
      elif "gate_up_proj_bias" in k:
        new_weights[k.replace("gate_up_proj_bias", "gate_proj.bias")] = (
          mx.contiguous(v[..., ::2])
        )
        new_weights[k.replace("gate_up_proj_bias", "up_proj.bias")] = (
          mx.contiguous(v[..., 1::2])
        )
      elif "down_proj_bias" in k:
        new_weights[k.replace("down_proj_bias", "down_proj.bias")] = v
      else:
        new_weights[k] = v

    # Now filter by shard
    shard_state_dict = {}
    for key, value in new_weights.items():
      if "self_attn.rotary_emb.inv_freq" in key:
        continue
      if key.startswith('model.layers.'):
        layer_num = int(key.split('.')[2])
        if self.args.shard.start_layer <= layer_num <= self.args.shard.end_layer:
          shard_state_dict[key] = value
      elif self.args.shard.is_first_layer() and key.startswith('model.embed_tokens'):
        shard_state_dict[key] = value
      elif self.args.shard.is_last_layer() and key.startswith('lm_head'):
        shard_state_dict[key] = value
      elif self.args.shard.is_last_layer() and key.startswith('model.norm'):
        shard_state_dict[key] = value

    return shard_state_dict

  @property
  def layers(self):
    return self.model.layers

  @property
  def head_dim(self):
    return self.args.head_dim

  @property
  def n_kv_heads(self):
    return self.args.num_key_value_heads
