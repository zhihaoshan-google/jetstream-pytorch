from absl import app
from absl import flags
from absl import logging
import sys
import jax
import jax.numpy as jnp
import numpy as np

from jetstream.engine import token_utils
from absl.testing import absltest

import os
import sys

from petstream import jet_engine2 as je
import time
import logging

logging.getLogger().setLevel(logging.ERROR)


FLAGS = flags.FLAGS

_TOKENIZER_PATH = flags.DEFINE_string(
    'tokenizer_path',
    'petstream/pets/tokenizer.model',
    'The tokenizer model path',
    required=False,
)
_CKPT_PATH = flags.DEFINE_string(
    'checkpoint_path', None, 'Directory for .pth checkpoints', required=False
)
_BF16_ENABLE = flags.DEFINE_bool(
    'bf16_enable', False, 'Whether to enable bf16', required=False
)
_CONTEXT_LENGTH = flags.DEFINE_integer(
    'context_length', 1024, 'The context length', required=False
)
_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size', 32, 'The batch size', required=False
)
_PROFILING_OUTPUT =flags.DEFINE_string(
    'profiling_output',
    '',
    'The profiling output',
    required=False,
)

_SIZE = flags.DEFINE_string('size', 'tiny', 'size of model')

_QUANTIZE_WEIGHTS = flags.DEFINE_bool('quantize_weights', False, 'weight quantization')
# _QUANTIZE_KV_CACHE= flags.DEFINE_bool('quantize_weights', False, 'weight quantization')


WEIGHT_SHARDING_OVERRIDE = {
  # int is axis; -1 is replicated
  # 0 is column replicated for linears
  "tok_embeddings.weight": 1,
  "tok_embeddings.weight_scaler": 0,
  "attention.wq.weight": 0,
  "attention.wq.weight_scaler": 0,
  "attention.wk.weight": 0,
  "attention.wk.weight_scaler": 0,
  "attention.wv.weight": 0,
  "attention.wv.weight_scaler": 0,
  "attention.wo.weight": 1,
  "attention.wo.weight_scaler": 0,
  "feed_forward.w1.weight": 0,
  "feed_forward.w1.weight_scaler": 0,
  "feed_forward.w2.weight": 1,
  "feed_forward.w2.weight_scaler": 1,
  "feed_forward.w3.weight": 0,
  "feed_forward.w3.weight_scaler": 1,
  "attention_norm.weight":  -1,
  "ffn_norm.weight": -1,
}



def main(argv):
  del argv
  jax.config.update('jax_default_prng_impl', 'unsafe_rbg')
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  max_prefill_predict_length = 1024
  max_target_length = max_prefill_predict_length + 256

  devices = jax.devices()
  start = time.perf_counter()
  engine = je.create_pytorch_engine(
        devices=devices,
        tokenizer_path=_TOKENIZER_PATH.value,
        ckpt_path=_CKPT_PATH.value,
        bf16_enable=True,
        param_size=_SIZE.value,
        context_length=_CONTEXT_LENGTH.value,
        batch_size=_BATCH_SIZE.value,
        quantize_weights=_QUANTIZE_WEIGHTS.value,
  )
  print('Initialize engine', time.perf_counter() - start)
  start = time.perf_counter()
  params = engine.load_params()
  print('Load params ', time.perf_counter() - start)

  text = 'This is a beautiful day'
  metadata = engine.get_tokenizer()
  vocab = token_utils.load_vocab(
    metadata.path, metadata.extra_ids)
  tokenizer = vocab.tokenizer
  tokens, true_length = token_utils.tokenize_and_pad(
    text, vocab, is_bos=True, prefill_lengths=[max_prefill_predict_length])
  assert tokens.size <= max_prefill_predict_length, "can't take too many tokens"

  start = time.perf_counter()
  prefill_result = engine.prefill(
      params=params, padded_tokens=tokens, true_length=true_length
  )
  end = time.perf_counter()
  print('Prefill time', end - start)


  slot = jnp.int32(1)

  decode_state = engine.init_decode_state()
  decode_state = engine.insert(
      prefill_result, decode_state, slot=slot
  )

  steps = range(max_prefill_predict_length, max_target_length)
  sampled_tokens_list = []

  for i in range(3): # warm up
    decode_state, sampled_tokens = engine.generate(
      params, decode_state
    )
    sampled_tokens_list.append(sampled_tokens)
  print('======= decode starting ===')
  for i in range(10):
    start = time.perf_counter()
    decode_state, sampled_tokens = engine.generate(
      params, decode_state
    )
    decode_state.tokens.block_until_ready()
    sampled_tokens_list.append(sampled_tokens)
    end = time.perf_counter()
    print(i, 'decode time', (end - start))

  results = [sampled_tokens.get_result_at_slot(slot).tokens.item() for sampled_tokens in sampled_tokens_list]
  output = tokenizer.detokenize(results)
  print(f"Input `{text}` -> `{output}`")


if __name__ == "__main__":
  app.run(main)