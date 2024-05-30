# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import jax

from typing import Dict, List
from jax import numpy as jnp
from safetensors import safe_open

from jetstream_pt import torchjax
from jetstream_pt import environment


@dataclasses.dataclass
class LoraAdapterState:
  current_slot_number: int


class LoraAdapter:
  """LoraAdapter information"""

  name: str
  ckpt_path: str
  rank: int
  alpha: int
  target_modules: List[str]
  state: LoraAdapterState

  def __init__(self, cfg: environment.LoraAdapterConfig):
    self.name = cfg.name
    self.ckpt_path = cfg.checkpoint_path
    self.rank = cfg.rank
    self.alpha = cfg.alpha
    self.state = LoraAdapterState(-1)
    self.scaling = (
        1.0 if self.alpha / self.rank <= 0 else self.alpha / self.rank
    )
    self.target_modules = cfg.target_modules

  def load_params(self) -> Dict[str, jax.Array]:
    weights = {}
    with safe_open(self.ckpt_path, framework="pt", device="cpu") as f:
      for key in f.keys():
        weights[key] = torchjax.from_torch(f.get_tensor(key))
    return weights


class LoraAdapterManager:
  """Lora Adapter Manager"""

  def __init__(self, env: environment.JetEngineEnvironment):
    self.env = env
    cfgs = env.lora_adapter_configs
    self.adapters = {cfg.name: LoraAdapter(cfg) for _, cfg in enumerate(cfgs)}
    self.model_dim = env.model_dim
    self.num_layers = env.num_layers
    # Leave the first slot for non lora case.
    self._num_slot = len(cfgs) + 1
    self._next_available_slot_index = 1
    self.max_rank = max([adapter.rank for adapter in self.adapters.values()])
    self.target_modules = set(
        [
            module
            for adapter in self.adapters.values()
            for module in adapter.target_modules
        ]
    )

  def load_all_adapters(self):
    """Load all adapters to the device memory"""
    self.batched_weights = self._initialize_batched_weights(
        num_layers=self.num_layers, model_dim=self.model_dim
    )
    self.batched_scaling = jnp.zeros((self._num_slot))
    for key in self.adapters.key():
      self._load(key)

  def adapter_index(self, adapter_name: str) -> int:
    """Get the slot index for the adapter"""
    return (
        self.adapters[adapter_name].state.current_slot_number
        if adapter_name in self.adapters
        else 0
    )

  def _load(self, adapter_name: str):
    adapter = self.adapters[adapter_name]
    weight = adapter.load_params()
    adapter.state.current_slot_number = self._next_available_slot_index
    slot = adapter.state.current_slot_number
    with jax.named_scope("InsertAdapter"):
      self._insert_weight(weight, slot, adapter.target_module)
    self.batched_scaling = self.batched_scaling.at[slot].set(adapter.scaling)

    self._next_available_slot_index += 1

  def _lora_weight_key_conversion(self, key: str):
    """Convert user input weight key to algin with the current model implementation for easy lookup."""

    def _get_layer_number(path):
      s = path.split(".")
      for t in s:
        if t.isdigit():
          return int(t)
      raise LookupError(f"Not found layer number in {path}")

    layer_number = _get_layer_number(key)

    if "q_proj" in key:
      if "lora_A" in key:
        return f"layers.{layer_number}.wq.lora_A"
      else:
        return f"layers.{layer_number}.wq.lora_B"
    if "v_proj" in key:
      if "lora_A" in key:
        return f"layers.{layer_number}.v_proj.lora_A"
      else:
        return f"layers.{layer_number}.v_proj.lora_B"
    raise NotImplementedError("unknown weight key from user")

  def _initialize_batched_weights(
      self, num_layers: int, model_dim: int
  ) -> Dict[str, jax.Array]:
    batched_weights = {}
    batch_size = self._num_slot
    for i in range(len(num_layers)):
      for module in self.target_modules:
        wa_key = self._lora_weight_key_conversion(f"layers.{i}.{module}.lora_A")
        wb_key = self._lora_weight_key_conversion(f"layers.{i}.{module}.lora_B")
        match module:
          case "q_proj" | "v_proj":
            batched_weights[wa_key] = jax.device_put(
                jnp.zeros(shape=(batch_size, model_dim, self.max_rank)),
                self.env.sharding_by_axis(1),
            )

            batched_weights[wb_key] = jax.device_put(
                jnp.zeros(shape=(batch_size, self.max_rank, model_dim)),
                self.env.sharding_by_axis(2),
            )
          case _:
            raise NotImplementedError(
                f"{module} module is not supported in lora inference yet"
            )

    return batched_weights

  def _insert_weight(self, weight: Dict[str, jax.Array], slot: int):
    for key, tensor in weight.items():
      target_key = self._lora_weight_key_conversion(key)

      if "lora_A" in target_key:
        tensor = jax.device_put(tensor, self.env.sharding_by_axis(1))
      else:
        tensor = jax.device_put(tensor, self.env.sharding_by_axis(2))

      self.batched_weights[target_key] = jax.lax.dynamic_update_slice(
          self.batched_weights[target_key].T, weight, (slot, 0, 0)
      )
