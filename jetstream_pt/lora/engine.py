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

"""Implement Jet Engine API for Lora."""

import functools
from typing import Any
import threading
import os

import numpy as np
from etils import epath
from flax import struct
import jax
import jax.numpy as jnp
import torch

from jetstream.engine import engine_api
from jetstream_pt import engine as default_engine
from jetstream_pt import environment
from jetstream_pt.lora import lora_manager
from jetstream_pt import cache_manager
from jetstream_pt import torchjax
from jetstream_pt.environment import JetEngineEnvironment, JetEngineEnvironmentData, QuantizationConfig
from jetstream_pt.third_party.llama import model_exportable as llama_model, model_args
from jetstream_pt.third_party.gemma import config as gemma_config, model_lora as gemma_model_lora


Mesh = jax.sharding.Mesh
P = jax.sharding.PartitionSpec

Params = jax.Array
PrefillInputs = jax.Array


@struct.dataclass
# pylint: disable-next=all
class Prefix(default_engine.Prefix):
  # token: jax.Array  # [1, seqlen]
  # caches: List[Tuple[jax.Array, jax.Array]]
  # seq_len: int  # true seqlen front pad
  lora_index: int


@struct.dataclass
# pylint: disable-next=all
class DecodeState(default_engine.DecodeState):
  # tokens: jax.Array  # [batch_size, seqlen]
  # caches: List[Tuple[jax.Array, jax.Array]]
  # cache_scales: List[
  #     Tuple[jax.Array, jax.Array]
  # ]  # only present in quantized kv
  # current_position: int
  # lens: jax.Array  # [batch_size, 1]
  # start: jax.Array  # [batch_size, 1], the starting pos for each slot
  # input_pos: jax.Array  # [batch_size, 1] input pos for each slot
  # mask: jax.Array  # [batch_size, seqlen] -inf for invalid; 0 for valid
  lora_indics: jax.Array  # [batch_size]


# NOTE model specific


# pylint: disable-next=all
class LoraPyTorchEngine(default_engine.PyTorchEngine):
  """LoraPyTorchEngine implementation based on PyTorchEngine."""

  def __init__(
      self,
      pt_model: torch.nn.Module,
      env: environment.JetEngineEnvironment,
  ):
    super().__init__(pt_model, env)
    # self.pt_model = pt_model
    # self.env = env
    # self.default_dtype = jnp.bfloat16 if env.bf16_enable else jnp.float32

    # self.y_sharding = env.sharding_by_axis(1)
    # self.x_sharding = env.sharding_by_axis(0)
    # self.replicated = env.sharding_by_axis(-1)  # replicated

    # self.cache_sharding = self.env.cache_sharding

    # jax.config.update("jax_enable_x64", False)
    self.lora_manager = lora_manager.LoraAdapterManager(env=env)

    self.prefill = jax.jit(
        self.prefill, out_shardings=self.get_prefix_destination_sharding()
    )
    self.insert = jax.jit(
        self.insert,
        donate_argnums=(0, 1),
        out_shardings=self.get_decode_state_sharding(),
    )
    self.generate = jax.jit(
        self.generate,
        donate_argnums=(1,),
        out_shardings=(self.get_decode_state_sharding(), None),
    )
    # self._insert_wrap = jax.jit(self._insert_wrap, donate_argnums=(0, 1),
    #                              out_shardings=self.get_decode_state_sharding())

    # self._insert_no_wrap = jax.jit(
    #      self._insert_no_wrap,
    #      donate_argnums=(0, 1),
    #      out_shardings=self.get_decode_state_sharding())
    self._lock = threading.RLock()

  def init_decode_state(
      self,
  ) -> DecodeState:
    super_state = super().init_decode_state()
    return DecodeState(
        **vars(super_state),
        lora_indics=jnp.zeros((self.env.batch_size,), dtype=jnp.int32),
    )

  @functools.partial(
      jax.jit,
      static_argnums=(0,),
  )
  def _call_model_prefill(self, weights, tokens, input_indexes, lora_index):
    caches = [
        cache_manager.KVCachePrefill(
            self.env.quant_config.enable_kv_quantization
        )
        for _ in self.pt_model.layers
    ]
    mask = jnp.full(
        (1, 1, tokens.shape[1], tokens.shape[1]),
        float("-inf"),
        dtype=self.default_dtype,
    )
    mask = jnp.triu(mask, k=1)
    lora_weight = self.lora_manager.batched_weights
    lora_scaling = self.lora_manager.batched_scaling
    args = (
        tokens,
        input_indexes,
        caches,
        mask,
        lora_index,
        lora_weight,
        lora_scaling,
    )

    paramst, argst = torchjax.to_torch((weights, args))
    with self._lock:
      with torchjax.jax_mode:
        res = torch.func.functional_call(self.pt_model, paramst, argst)[0]
    caches_res = [c.state() for c in caches]
    return torchjax.from_torch((res, caches_res))

  def prefill(
      self,
      *,
      params: Any,  # Weights
      per_request_hyperparams: Any | None = None,
      padded_tokens: PrefillInputs,  # PrefillInputs[jax.Array],
      true_length: int,
  ) -> Prefix:
    if isinstance(padded_tokens, jax.Array):
      batched_token = padded_tokens.reshape(1, -1)
    else:
      raise TypeError(
          "Input tokens should be of type Jax Array, but receiving:"
          " {prefill_inputs}"
      )
    seq_len = padded_tokens.shape[0]
    input_indexes = jnp.arange(0, seq_len)
    adapter_name = ""
    if per_request_hyperparams.adapter_name:
      adapter_name = per_request_hyperparams.adapter_name
    lora_index = self.lora_manager.adapter_index(adapter_name)
    logits, updated_caches = self._call_model_prefill(
        params, batched_token, input_indexes, lora_index
    )
    if len(logits.shape) == 3:  # b, seqlen, num words
      logits = logits[0]

    token = jnp.argmax(logits[true_length - 1])

    # truncate to true_length didnt work need to be out side of jit
    # caches = [
    #   (jax.lax.dynamic_slice_in_dim(
    #       k, seq_len - true_length, true_length, axis=2),
    #    jax.lax.dynamic_slice_in_dim(
    #       v, seq_len - true_length, true_length, axis=2))
    #   for k, v in updated_caches
    # ]
    return Prefix(token, updated_caches, true_length, lora_index)

  def insert(
      self,
      prefix: Prefix,
      decode_state: DecodeState,
      slot: int,
  ) -> DecodeState:
    # logging.info(
    #     'Jet input prefix: %s, decode state before insert: %s',
    #     prefix,
    #     decode_state,
    # )
    super_decode_state = super().insert(Prefix, decode_state, slot)
    lora_indices = decode_state.lora_indics.at[slot].set(prefix.lora_index)
    return DecodeState(
        **vars(super_decode_state),
        lora_indics=lora_indices,
    )

  # pylint: disable-next=all
  def _call_model_generate(
      self,
      weights,
      tokens,
      input_indexes,
      caches,
      cache_scales,
      mask,
      start,
      input_pos,
      ragged_batch_index,
      ragged_block_index,
      lora_indices,
  ):
    if self.env.quant_config.enable_kv_quantization:
      caches_obj = [
          cache_manager.Int8KVCacheGenerate(k, v, ks, vs, input_indexes)
          for (k, v), (ks, vs) in torchjax.to_torch(
              list(zip(caches, cache_scales))
          )
      ]
    else:
      caches_obj = [
          cache_manager.KVCacheGenerate(
              k, v, input_indexes, self.cache_sharding
          )
          for k, v in torchjax.to_torch(caches)
      ]
    mask = jnp.expand_dims(mask, (1, 2))

    lora_weight = self.lora_manager.batched_weights
    lora_scaling = self.lora_manager.batched_scaling
    args = (
        tokens,
        input_pos,
        caches_obj,
        mask,
        start,
        ragged_batch_index,
        ragged_block_index,
        lora_indices,
        lora_weight,
        lora_scaling,
    )
    paramst, argst = torchjax.to_torch((weights, args))
    with self._lock:
      with torchjax.jax_mode:
        # The mode is needed so that tensors created inside of
        # the model (such as via torch.ones etc) also have the right type
        res = torch.func.functional_call(self.pt_model, paramst, argst)
    updated_caches = [c.state() for c in caches_obj]
    scales = []
    if self.env.quant_config.enable_kv_quantization:
      scales = [c.scalers() for c in caches_obj]
    return torchjax.from_torch((res, updated_caches, scales))

  def generate(
      self, params: Any, decode_state: DecodeState
  ) -> tuple[DecodeState, engine_api.ResultTokens]:
    # seq_len = padded_tokens.shape[0]
    pos = decode_state.current_position
    input_indexes = jnp.full((1,), pos)

    # fill mask first
    mask = decode_state.mask.at[:, decode_state.current_position].set(0)
    ragged_batch_index, ragged_block_index = (
        self.precompute_ragged_block_indices(decode_state)
    )
    ragged_batch_index, ragged_block_index = ragged_batch_index.reshape(
        (-1)
    ), ragged_block_index.reshape((-1))

    logits, new_caches, new_scales = self._call_model_generate(
        params,
        decode_state.tokens,
        input_indexes,
        decode_state.caches,
        decode_state.cache_scales,
        mask,
        decode_state.start,
        decode_state.input_pos,
        ragged_batch_index,
        ragged_block_index,
        decode_state.lora_indics,
    )

    next_token = self._sampling(logits, self.env.batch_size)
    lens = decode_state.lens + 1
    data = jnp.concatenate(
        [
            decode_state.tokens,
            jnp.ones_like(next_token),
            lens,
        ],
        axis=-1,
    )

    # [0] is the batch dimension, [1] normally should be 1
    length = next_token.shape[1]
    result_tokens = engine_api.ResultTokens(
        data=data,
        tokens_idx=(0, length),
        valid_idx=(length, 2 * length),
        length_idx=(2 * length, 2 * length + 1),
        samples_per_slot=1,
    )

    new_decode_state = DecodeState(
        next_token,
        new_caches,
        new_scales,
        (decode_state.current_position + 1) % self.env.cache_sequence_length,
        lens,
        decode_state.start,
        decode_state.input_pos + 1,
        mask,
    )
    print(
        "new_pos",
        (decode_state.current_position + 1) % self.env.cache_sequence_length,
    )
    print("cache_seq_len", self.env.cache_sequence_length)

    return new_decode_state, result_tokens

  # pylint: disable-next=all
  def load_params(self) -> Params:
    self.lora_manager.load_all_adapters()
    return super().load_params()

  def get_prefix_destination_sharding(self) -> Prefix:
    """Returns the shardings necessary to transfer data between engines."""
    return Prefix(
        self.replicated,
        self.replicated if self.env.shard_on_batch else self.cache_sharding,
        self.replicated,
        self.replicated,
    )

  def get_decode_state_sharding(self) -> DecodeState:
    """Gets the shardings corresponding to the decode state."""
    return DecodeState(
        self.x_sharding if self.env.shard_on_batch else self.replicated,
        self.cache_sharding,
        self.replicated,
        self.replicated,
        self.replicated,
        self.replicated,
        self.replicated,
        self.replicated,
        self.replicated,
    )


# pylint: disable-next=all
def create_lora_pytorch_engine(
    # pylint: disable-next=all
    devices: list[Any],
    tokenizer_path: str,
    ckpt_path: str | None = None,
    samples_per_slot: int = 1,  # pylint: disable=unused-argument
    bf16_enable: bool = False,
    param_size: str = "7b",
    context_length: int = 1024,
    batch_size: int = 1,
    max_decode_length: int = 4096,
    model_name="llama-2",
    quant_config: QuantizationConfig = QuantizationConfig(),
    max_cache_length=1024,
    sharding_config=None,
    shard_on_batch=False,
    ragged_mha=False,
    starting_position=512,
    lora_adapter_configs=[],
) -> LoraPyTorchEngine:
  """Returns: The pytorch engine with lora support."""

  supported_models = ["gemma"]
  if model_name not in supported_models:
    raise NotImplementedError(
        f"Model name should be one of{','.join(supported_models)}"
    )
  # See issue b/309529778 if it's turned on.
  jax.config.update("jax_dynamic_shapes", False)
  # Pytorch exports has int64 constants.
  # jax.config.update('jax_enable_x64', True)
  jax.config.update("jax_traceback_filtering", "off")
  torch_dtype = torch.bfloat16 if bf16_enable else torch.float32
  torch.set_default_dtype(torch_dtype)

  checkpoint_format = ""
  checkpoint_path = ""

  if not ckpt_path or ckpt_path is None:
    print("WARNING: Using random weights instead of checkpoints.")
  elif ".safetensors" in ckpt_path:
    checkpoint_format = "safetensors"
    checkpoint_path = ckpt_path
  elif ".pth" in ckpt_path or ".ckpt" in ckpt_path:
    checkpoint_format = "state_dict"
    checkpoint_path = ckpt_path
  else:
    path = epath.Path(ckpt_path) if ckpt_path and ckpt_path is not None else ""
    if not path.exists():
      raise ValueError(f"Checkpoint path {ckpt_path} not exists!")
    paths = list(path.glob("*.safetensors"))
    assert (
        len(paths) == 1
    ), f"Expects 1 *.safetensors in the checkpoint dir, see {len(paths)}"
    checkpoint_format = "safetensors"
    checkpoint_path = paths[0]

  pt_model = None

  if not sharding_config:
    sharding_file_name = "llama" if model_name.startswith("llama") else "gemma"
    sharding_config = os.path.join(
        "default_shardings", sharding_file_name + ".yaml"
    )

  env_data = JetEngineEnvironmentData(
      tokenizer_path=tokenizer_path,
      checkpoint_path=checkpoint_path,
      checkpoint_format=checkpoint_format,
      batch_size=batch_size,
      max_decode_length=max_decode_length,
      max_input_sequence_length=context_length,
      quant_config=quant_config,
      cache_sequence_length=max_cache_length,
      bf16_enable=bf16_enable,
      sharding_config_path=sharding_config,
      shard_on_batch=shard_on_batch,
      ragged_mha=ragged_mha,
      starting_position=starting_position,
      lora_adapter_configs=lora_adapter_configs,
  )

  if shard_on_batch and sharding_config:
    print("WARNING: with sharding_on_batch sharding config is ignored.")

  if model_name.startswith("llama"):

    args = model_args.get_model_args(
        model_name + "-" + param_size, context_length, batch_size, bf16_enable
    )
    args.device = "meta"
    env_data.cache_shape = (
        batch_size,
        args.n_kv_heads,
        max_cache_length,
        args.dim // args.n_heads,
    )
    env_data.model_type = model_name + "-" + param_size
    env_data.num_layers = args.n_layers
    env = JetEngineEnvironment(env_data)
    pt_model = llama_model.Transformer(args, env)
  elif model_name == "gemma":
    args = gemma_config.get_model_config(param_size)
    env_data.cache_shape = (
        batch_size,
        args.num_key_value_heads,
        max_cache_length,
        args.head_dim,
    )
    env_data.model_type = model_name + "-" + param_size
    env_data.num_layers = args.num_hidden_layers
    env = JetEngineEnvironment(env_data)
    print(f"Enviroment variables: {vars(env)}")
    pt_model = gemma_model_lora.GemmaModel(args, env)
  else:
    raise RuntimeError(f"Model with name {model_name} not found")

  num_params_size = 0
  num_params = 0
  for _, v in pt_model.state_dict().items():
    num_params += 1
    num_params_size += np.prod(v.shape) * (1 if v.dtype == torch.int8 else 2)
  print("Number of param Gbytes:", num_params_size / (1 << 30))
  print("Number of param: ", num_params)

  return LoraPyTorchEngine(pt_model=pt_model, env=env)
