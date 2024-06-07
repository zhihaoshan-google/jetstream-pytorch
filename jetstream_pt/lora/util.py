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

from typing import Dict
import torch
import re


def apply_lora(
    input: torch.Tensor,
    indices: torch.Tensor,
    batched_lora_a: torch.Tensor,
    batched_lora_b: torch.Tensor,
    batched_scaling: torch.Tensor,
):
  selected_batched_lora_a = batched_lora_a.index_select(0, indices)
  selected_batched_lora_b = batched_lora_b.index_select(0, indices)
  selected_scaling = batched_scaling.index_select(0, indices)[:, None, None]
  output = (
      torch.bmm(
          torch.bmm(input, selected_batched_lora_a), selected_batched_lora_b
      )
      * selected_scaling
  )
  return output


def get_lora_weight_keys(weights: Dict[str, torch.Tensor], search_word: str):
  pattern = re.compile(search_word)
  return [key for key in weights.keys() if pattern.match(key)]
