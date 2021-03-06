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

"""Contains a collection of models which operate on variable-length sequences.
"""
import math

import models
import video_level_models
import tensorflow as tf
import model_utils as utils

import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers
from tensorflow import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("iterations", 30,
                     "Number of frames per batch for DBoF.")
flags.DEFINE_bool("dbof_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")
flags.DEFINE_bool(
    "sample_random_frames", True,
    "If true samples random frames (for frame level models). If false, a random"
    "sequence of frames is sampled instead.")
flags.DEFINE_integer("dbof_cluster_size", 8192,
                     "Number of units in the DBoF cluster layer.")
flags.DEFINE_integer("dbof_hidden_size", 1024,
                     "Number of units in the DBoF hidden layer.")
flags.DEFINE_string("dbof_pooling_method", "max",
                    "The pooling method used in the DBoF cluster layer. "
                    "Choices are 'average' and 'max'.")
flags.DEFINE_string("video_level_classifier_model", "MoeModel",
                    "Some Frame-Level models can be decomposed into a "
                    "generalized pooling operation followed by a "
                    "classifier layer")
flags.DEFINE_integer("lstm_cells", 1024, "Number of LSTM cells.")
flags.DEFINE_integer("lstm_layers", 2, "Number of LSTM layers.")

flags.DEFINE_integer("tcn_bottleneck", 1024, "Number of channels in TCN bottleneck.")
flags.DEFINE_integer("tcn_layers", 7, "Number of residual blocks in TCN.")
flags.DEFINE_integer("tcn_kernel", 5, "Width of TCN kernel.")
flags.DEFINE_float("tcn_dropout_prob", 0.1, "Probability of dropout in training TCN.")

flags.DEFINE_integer("attn_layers", 7, "Number of blocks in Transformer Attn model.")
flags.DEFINE_integer("attn_heads", 5, "Number of heads in Transformer Multi-Head Attn model.")
flags.DEFINE_float("attn_dropout_prob", 0.1, "Probability of dropout in training Transformer.")

class FrameLevelLogisticModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a logistic classifier over the average of the
    frame-level features.

    This class is intended to be an example for implementors of frame level
    models. If you want to train a model over averaged features it is more
    efficient to average them beforehand rather than on the fly.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    feature_size = model_input.get_shape().as_list()[2]

    denominators = tf.reshape(
        tf.tile(num_frames, [1, feature_size]), [-1, feature_size])
    avg_pooled = tf.reduce_sum(model_input,
                               axis=[1]) / denominators

    output = slim.fully_connected(
        avg_pooled, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(1e-8))
    return {"predictions": output}

class DbofModel(models.BaseModel):
  """Creates a Deep Bag of Frames model.

  The model projects the features for each frame into a higher dimensional
  'clustering' space, pools across frames in that space, and then
  uses a configurable video-level model to classify the now aggregated features.

  The model will randomly sample either frames or sequences of frames during
  training to speed up convergence.

  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).

  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.dbof_cluster_size
    hidden1_size = hidden_size or FLAGS.dbof_hidden_size

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    tf.summary.histogram("input_hist", reshaped_input)

    if add_batch_norm:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    cluster_weights = tf.get_variable("cluster_weights",
      [feature_size, cluster_size],
      initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
    tf.summary.histogram("cluster_weights", cluster_weights)
    activation = tf.matmul(reshaped_input, cluster_weights)
    if add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="cluster_bn")
    else:
      cluster_biases = tf.get_variable("cluster_biases",
        [cluster_size],
        initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
      tf.summary.histogram("cluster_biases", cluster_biases)
      activation += cluster_biases
    activation = tf.nn.relu6(activation)
    tf.summary.histogram("cluster_output", activation)

    activation = tf.reshape(activation, [-1, max_frames, cluster_size])
    activation = utils.FramePooling(activation, FLAGS.dbof_pooling_method)

    hidden1_weights = tf.get_variable("hidden1_weights",
      [cluster_size, hidden1_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
    tf.summary.histogram("hidden1_weights", hidden1_weights)
    activation = tf.matmul(activation, hidden1_weights)
    if add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")
    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases
    activation = tf.nn.relu6(activation)
    tf.summary.histogram("hidden1_output", activation)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        **unused_params)

class LstmModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, is_training=True, **unused_params):
    """Creates a model which uses a stack of LSTMs to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0)
                for _ in range(number_of_layers)
                ])

    loss = 0.0

    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                       sequence_length=num_frames,
                                       dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=state[-1].h,
        vocab_size=vocab_size,
        **unused_params)
class TcnModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, is_training=True, **unused_params):
    """Creates a model which uses a TCN with residual connections to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    self.number_of_layers = FLAGS.tcn_layers
    self.kernel_size = FLAGS.tcn_kernel
    self.keep_prob = 1 -FLAGS.tcn_dropout_prob
    self.bn_params = {'center':True, 'scale':True, 'is_training':is_training}
    
    def TCNBlock(inputs, hidden_channels, out_channels, dilation, **unused_params):
      
      conv1 = layers.conv2d(inputs, hidden_channels, 1, 
        data_format='NWC', stride=1, padding='SAME', rate=dilation, 
        normalizer_fn=layers.batch_norm, normalizer_params=self.bn_params)
      dropout1 = layers.dropout(conv1, keep_prob=self.keep_prob, is_training=is_training)
      
      conv2 = layers.conv2d(conv1, hidden_channels, self.kernel_size, 
        data_format='NWC', stride=1, padding='SAME', rate=dilation, 
        normalizer_fn=layers.batch_norm, normalizer_params=self.bn_params)
      dropout2 = layers.dropout(conv2, keep_prob=self.keep_prob, is_training=is_training)

      conv3 = layers.conv2d(dropout2, out_channels, 1, 
        data_format='NWC', stride=1, padding='SAME', rate=dilation, 
        normalizer_fn=layers.batch_norm, normalizer_params=self.bn_params)
      dropout3 = layers.dropout(conv3, keep_prob=self.keep_prob, is_training=is_training)

      res = layers.conv2d(inputs, out_channels, 1) if inputs.shape[-1] != out_channels else inputs
      return layers.layer_norm(tf.nn.relu(dropout3 + res), center=True, scale=True)

    hidden_size = [1024, 512, 256, 128, 64, 32, 16]
    tcn_params = [[hidden_size[i], 2*hidden_size[i]*(self.kernel_size -1), 2 ** i] for i in range(self.number_of_layers)]
    tcn_out = layers.stack(model_input, TCNBlock, tcn_params)
    #fc_1 = layers.fully_connected(tcn_out, 8192, tf.nn.relu, layers.layer_norm, self.bn_params)
    #dropout4 = layers.dropout(fc_1, keep_prob=self.keep_prob, is_training=is_training)
    fc_out = layers.fully_connected(tcn_out, vocab_size, tf.sigmoid, layers.batch_norm, self.bn_params)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=tcn_out,
        vocab_size=vocab_size,
        **unused_params)

class AttnModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, is_training=True, **unused_params):
    """Creates a model which uses a TCN with residual connections to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    self.num_frames = 300
    self.num_features = 1152

    self.num_layers = FLAGS.attn_layers
    self.num_heads = FLAGS.attn_heads 
    self.dim_head = int(self.num_features/self.num_heads)
    
    self.keep_prob = 1 -FLAGS.attn_dropout_prob
    self.ln_params = {'center':True, 'scale':True}
    def AttnBlock(inputs, num_heads, dim_head, is_training):

      values = tf.transpose(tf.reshape(
        tf.layers.dense(model_input, dim_head * num_heads, activation=None, use_bias=False,
          weights_regularizer=slim.l2_regularizer(l2_penalty), name="values"),
        [-1, self.num_frames, num_heads, dim_head]), [0, 2, 1, 3])

      querys = tf.transpose(tf.reshape(
          tf.layers.dense(model_input, dim_head * num_heads, activation=None,
          weights_regularizer=slim.l2_regularizer(l2_penalty), name="querys"),
        [-1, self.num_frames, num_heads, dim_head]), [0, 2, 1, 3])

      keys = tf.transpose(tf.reshape(
        tf.layers.dense(model_input, dim_head * num_heads, activation=None, use_bias=False,
          weights_regularizer=slim.l2_regularizer(l2_penalty), name="keys"),
        [-1, self.num_frames, num_heads, dim_head]), [0, 2, 1, 3])
      
      # scaled dot-product attention in shape
      # shape is [-1, num_heads, self.num_frames, self.num_frames]
      attn_spatial_act = tf.nn.softmax(tf.matmul(querys, keys, transpose_b=True) / sqrt(dim_head), -1)
      attn_spatial_dropout_act = layers.dropout(attn_spatial_act, keep_prob=self.keep_prob, is_training=is_training)

      attention_concat_heads = tf.reshape(tf.transpose(
        tf.matmul(attn_spatial_dropout_act, values), 
        [0, 2, 1, 3]), [-1, self.num_frames, num_heads * dim_head])

      attention_mixed_heads = tf.layers.dense(attention_combined_heads, dim_head * num_heads, activation=None, use_bias=False,
        weights_regularizer=slim.l2_regularizer(l2_penalty), name="head_mixers")
      attention_dropout_mixed_heads = layers.dropout(attention_mixed_heads, keep_prob=self.keep_prob, is_training=is_training)

      attention_final_output = layers.layer_norm(model_input + attention_dropout_mixed_heads, **self.ln_params)

      feedforward_1 = layers.conv2d(attention_final_output, 4 * num_heads * dim_head, 1, data_format='NWC', stride=1, name="ff_1")
      feedforward_2 = layers.conv2d(feedforward_1, num_heads * dim_head, 1, data_format='NWC', stride=1, activation_fn=None, name="ff_2")
      feedforward_dropout = layers.dropout(feedforward_2, keep_prob=self.keep_prob, is_training=is_training)

      return layers.layer_norm(feedforward_dropout + attention_final_output, **self.ln_params)

    attn_params = [[num_heads, dim_head, is_training] for _ in range(self.num_layers)]
    attn_out = layers.stack(model_input, AttnBlock, attn_params)
    #pre-flattens attn_out
    class_probs = layers.fully_connected(class_act, vocab_size, tf.sigmoid, layers.layer_norm, ln_params)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=fc_out,
        vocab_size=vocab_size,
        **unused_params)