#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Timothy Dozat
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





import numpy as np
import tensorflow as tf

from nparser.neural.models.embeds.base_embed import BaseEmbed

#***************************************************************
class MLPEmbed(BaseEmbed):
  """ """
  
  #=============================================================
  def __call__(self, vocab, **kwargs):
    """ """
    
    # (n x b x d)
    embeddings = super(MLPEmbed, self).__call__(vocab, **kwargs)
    # (n x b x d) -> (n x d)
    with tf.compat.v1.variable_scope('Attn'):
      attn = self.linear_attention(embeddings)
    # (n x d) -> (n x h)
    with tf.compat.v1.variable_scope('MLP'):
      hidden = self.MLP(attn, self.mlp_size)
    # (n x h) -> (n x o)
    linear = self.linear(hidden, vocab.token_embed_size)
    return linear 