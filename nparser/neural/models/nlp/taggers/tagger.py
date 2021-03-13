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

from nparser.neural.models.nlp.taggers.base_tagger import BaseTagger

#***************************************************************
class Tagger(BaseTagger):
  """ """
  
  #=============================================================
  def __call__(self, vocabs, moving_params=None):
    """ """
    
    top_recur = super(Tagger, self).__call__(vocabs, moving_params=moving_params)
    int_tokens_to_keep = tf.cast(self.tokens_to_keep, dtype=tf.int32)
    
    with tf.compat.v1.variable_scope('MLP'):
      mlp = self.MLP(top_recur, self.mlp_size)
    
    with tf.compat.v1.variable_scope('Tag'):
      logits = self.linear(mlp, len(self.vocabs['tags']))
      probs = tf.nn.softmax(logits)
      preds = tf.cast(tf.argmax(input=logits, axis=-1), dtype=tf.int32)
      targets = self.vocabs['tags'].placeholder
      correct = tf.cast(tf.equal(preds, targets), dtype=tf.int32)*int_tokens_to_keep
      loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(targets, logits, self.tokens_to_keep)
    
    
    n_correct = tf.reduce_sum(input_tensor=correct)
    n_seqs_correct = tf.reduce_sum(input_tensor=tf.cast(tf.equal(tf.reduce_sum(input_tensor=correct, axis=1), self.sequence_lengths-1), dtype=tf.int32))
    
    outputs = {
      'logits': logits,
      'probs': probs,
      'preds': preds,
      'targets': targets,
      'correct': correct,
      'loss': loss,
      'n_correct': n_correct,
      
      'n_tokens': self.n_tokens,
      'n_seqs': self.batch_size,
      'tokens_to_keep': self.tokens_to_keep,
      'n_correct': n_correct,
      'n_seqs_correct': n_seqs_correct,
      'loss': loss
    }
    
    return outputs
