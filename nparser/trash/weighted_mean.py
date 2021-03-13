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

from nparser.neural.models import NN

#***************************************************************
class WeightedMean(NN):
  """ """
  
  #=============================================================
  def __call__(self, vocab, output_size, moving_params=None):
    """ """
    
    inputs = tf.compat.v1.placeholder(tf.int32, shape=(None,None), name='inputs-%s' % self.name)
    
    self.tokens_to_keep = tf.cast(tf.greater(inputs, vocab.PAD), dtype=tf.float32)
    self.sequence_lengths = tf.reduce_sum(input_tensor=self.tokens_to_keep, axis=1, keepdims=True)
    self.n_tokens = tf.reduce_sum(input_tensor=self.sequence_lengths)
    self.batch_size = tf.shape(input=inputs)[0]
    self.bucket_size = tf.shape(input=inputs)[1]
    self.moving_params = moving_params
    
    embeddings = vocab.embedding_lookup(inputs, moving_params=self.moving_params)
    weighted_embeddings = self.linear_attention(embeddings)
    mlp = self.MLP(weighted_embeddings, self.mlp_size)
    lin = self.linear(mlp, output_size)
    
    return {'output': lin, 'inputs': inputs}
