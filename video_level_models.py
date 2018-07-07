# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains model definitions."""
import math

import models
import tensorflow as tf
import utils

from tensorflow import flags
import tensorflow.contrib.slim as slim

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "moe_num_mixtures", 2,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")

class LogisticModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    
    fc_1 = slim.fully_connected(
        model_input, 2048, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    output = slim.fully_connected(
        fc_1, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}

class MoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")
    expert_activations = slim.fully_connected(
        model_input,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}

class OnePointAttn(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""
  self.ln_params = {'center':True, 'scale':True}
  def create_model(self,
                   model_input,
                   vocab_size,
                   l2_penalty=1e-8,
                   is_training=True,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      dim_attn: The dimensionality of of each attention head (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    dim_attn = dim_attn or FLAGS.opa_dim_attn
    num_heads = num_heads or FLAGS.opa_num_heads
    num_features = 1024 +128

    value_activations = tf.reshape(slim.fully_connected(
        model_input,
        dim_attn * num_heads,
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="values"),
        [-1, num_heads, dim_attn])

    query_activations = tf.reshape(slim.fully_connected(
        model_input,
        dim_attn * num_heads,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="queries"),
        [-1, num_heads, dim_attn])

    key_activations = tf.reshape(slim.fully_connected(
        model_input,
        dim_attn * num_heads,
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="keys"),
        [-1, num_heads, dim_attn])
    
    attention_point_distribution = tf.nn.softmax(
      tf.reduce_sum(query_activations * key_activations, -1, keepdims=True) / sqrt(dim_attn), -1) # scaled dot-product attention (softmax over each output entry)

    attention_final_dist = tf.reshape(
      tf.reduce_sum(attention_point_distribution * value_activations, -1, keepdims=True), [-1, dim_attn * num_heads])
    
    attention_residual_addend = slim.fully_connected(
      attention_final_dist,
      num_features,
      activation_fn=None,
      biases_initializer=None,
      weights_regularizer=slim.l2_regularizer(l2_penalty),
      scope="residual_ffn")

    attention_residual_dropout = tf.contrib.layers.dropout(attention_resid_addend, keep_prob=self.keep_prob, is_training=is_training)
    
    residual_output = tf.contrib.layers.layer_norm(model_input + attention_residual_dropout, center=True, scale=True)

    final_probabilities = slim.fully_connected(
        residual_output, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    
    return {"predictions": final_probabilities}