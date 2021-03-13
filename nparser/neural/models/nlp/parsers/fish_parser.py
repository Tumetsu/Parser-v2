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

from nparser.neural.models.nlp.parsers.base_parser import BaseParser

#***************************************************************
class FishParser(BaseParser):
  """ """
  
  #=============================================================
  def __call__(self, vocabs, moving_params=None):
    """ """
    
    top_recur = super(FishParser, self).__call__(vocabs, moving_params=moving_params)
    int_tokens_to_keep = tf.cast(self.tokens_to_keep, dtype=tf.int32)
    
    with tf.compat.v1.variable_scope('MLP'):
      dep_mlp, head_mlp = self.MLP(top_recur, self.arc_mlp_size + self.rel_mlp_size + self.lambda_mlp_size,
                                   n_splits=2)
      arc_dep_mlp, rel_dep_mlp, lambda_dep_mlp = tf.split(dep_mlp, [self.arc_mlp_size, self.rel_mlp_size, self.lambda_mlp_size], axis=2)
      arc_head_mlp, rel_head_mlp, lambda_head_mlp = tf.split(head_mlp, [self.arc_mlp_size, self.rel_mlp_size, self.lambda_mlp_size], axis=2)
    
    with tf.compat.v1.variable_scope('Lambda'):
      # (n x b x d) o (d x 1 x d) o (n x b x d).T -> (n x b x b)
      arc_lambdas = self.bilinear(lambda_dep_mlp, lambda_head_mlp, 1) + 5
      # (b x 1)
      i_mat = tf.expand_dims(tf.expand_dims(tf.range(self.bucket_size), 1), 0)
      # (1 x b)
      j_mat = tf.expand_dims(tf.expand_dims(tf.range(self.bucket_size), 0), 0)
      # (b x 1) - (1 x b) -> (b x b)
      k_mat = tf.abs(i_mat - j_mat)
      # (b x 1)
      n_mat = tf.expand_dims(tf.expand_dims(self.sequence_lengths, 1), 1) - 1 - i_mat
      # (b x b) * (n x b x b) - (n x b x b) - (b x b) -> (n x b x b)
      arc_logits = tf.cast(k_mat, dtype=tf.float32)*arc_lambdas - tf.exp(arc_lambdas) - tf.math.lgamma(tf.cast(k_mat+1, dtype=tf.float32))

    with tf.compat.v1.variable_scope('Arc'):
      # (n x b x d) o (d x 1 x d) o (n x b x d).T -> (n x b x b)
      arc_logits += self.bilinear(arc_dep_mlp, arc_head_mlp, 1, add_bias2=False)
      # (n x b x b)
      arc_probs = tf.nn.softmax(arc_logits)
      # (n x b)
      arc_preds = tf.cast(tf.argmax(input=arc_logits, axis=-1), dtype=tf.int32)
      # (n x b)
      arc_targets = self.vocabs['heads'].placeholder
      # (n x b)
      arc_correct = tf.cast(tf.equal(arc_preds, arc_targets), dtype=tf.int32)*int_tokens_to_keep
      # ()
      arc_loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(arc_targets, arc_logits, self.tokens_to_keep)
    
    with tf.compat.v1.variable_scope('Rel'):
      # (n x b x d) o (d x r x d) o (n x b x d).T -> (n x b x r x b)
      rel_logits = self.bilinear(rel_dep_mlp, rel_head_mlp, len(self.vocabs['rels']))
      # (n x b x r x b)
      rel_probs = tf.nn.softmax(rel_logits, axis=2)
      # (n x b x b)
      one_hot = tf.one_hot(arc_preds if moving_params is not None else arc_targets, self.bucket_size)
      # (n x b x b) -> (n x b x b x 1)
      one_hot = tf.expand_dims(one_hot, axis=3)
      # (n x b x r x b) o (n x b x b x 1) -> (n x b x r x 1)
      select_rel_logits = tf.matmul(rel_logits, one_hot)
      # (n x b x r x 1) -> (n x b x r)
      select_rel_logits = tf.squeeze(select_rel_logits, axis=3)
      # (n x b)
      rel_preds = tf.cast(tf.argmax(input=select_rel_logits, axis=-1), dtype=tf.int32)
      # (n x b)
      rel_targets = self.vocabs['rels'].placeholder
      # (n x b)
      rel_correct = tf.cast(tf.equal(rel_preds, rel_targets), dtype=tf.int32)*int_tokens_to_keep
      # ()
      rel_loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(rel_targets, select_rel_logits, self.tokens_to_keep)
    
    n_arc_correct = tf.reduce_sum(input_tensor=arc_correct)
    n_rel_correct = tf.reduce_sum(input_tensor=rel_correct)
    correct = arc_correct * rel_correct
    n_correct = tf.reduce_sum(input_tensor=correct)
    n_seqs_correct = tf.reduce_sum(input_tensor=tf.cast(tf.equal(tf.reduce_sum(input_tensor=correct, axis=1), self.sequence_lengths-1), dtype=tf.int32))
    loss = arc_loss + rel_loss
    
    outputs = {
      'arc_logits': arc_logits,
      'arc_lambdas': arc_lambdas,
      'arc_probs': arc_probs,
      'arc_preds': arc_preds,
      'arc_targets': arc_targets,
      'arc_correct': arc_correct,
      'arc_loss': arc_loss,
      'n_arc_correct': n_arc_correct,
      
      'rel_logits': rel_logits,
      'rel_probs': rel_probs,
      'rel_preds': rel_preds,
      'rel_targets': rel_targets,
      'rel_correct': rel_correct,
      'rel_loss': rel_loss,
      'n_rel_correct': n_rel_correct,
      
      'n_tokens': self.n_tokens,
      'n_seqs': self.batch_size,
      'tokens_to_keep': self.tokens_to_keep,
      'n_correct': n_correct,
      'n_seqs_correct': n_seqs_correct,
      'loss': loss
    }
    
    return outputs
