import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
from tensorflow.contrib import seq2seq
from tensorflow.python.util import nest

import custom_tf

try:
  from tensorflow.contrib.layers.python.layers import utils
except:
  from tensorflow.contrib.layers import utils

smart_cond = utils.smart_cond

class PointerWrapper(tf.contrib.seq2seq.AttentionWrapper):
  """Customized AttentionWrapper for PointerNet."""

  def __init__(self,cell,attention_size,memory,initial_cell_state=None,name=None):
    # In the paper, Bahdanau Attention Mechanism is used
    # We want the scores rather than the probabilities of alignments
    # Hence, we customize the probability_fn to return scores directly
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(attention_size, memory, probability_fn=lambda x: x )
    # According to the paper, no need to concatenate the input and attention
    # Therefore, we make cell_input_fn to return input only
    cell_input_fn=lambda input, attention: input
    # Call super __init__
    super(PointerWrapper, self).__init__(cell,
                                         attention_mechanism=attention_mechanism,
                                         attention_layer_size=None,
                                         alignment_history=False,
                                         cell_input_fn=cell_input_fn,
                                         output_attention=True,
                                         initial_cell_state=initial_cell_state,
                                         name=name)
  @property
  def output_size(self):
    return self.state_size.alignments

  def call(self, inputs, state):
    _, next_state = super(PointerWrapper, self).call(inputs, state)
    return next_state.alignments, next_state

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops

try:
  LSTMCell = rnn.LSTMCell
  MultiRNNCell = rnn.MultiRNNCell
except:
  LSTMCell = tf.contrib.rnn.LSTMCell
  MultiRNNCell = tf.contrib.rnn.MultiRNNCell

try:
  from tensorflow.python.ops.gen_array_ops import _concat_v2 as concat_v2
except:
  concat_v2 = tf.concat

def decoder_rnn(cell, inputs,
                enc_outputs, enc_final_states,
                seq_length, hidden_dim,
                num_glimpse, batch_size, is_train,
                end_of_sequence_id=0, initializer=None,
                max_length=None):
  with tf.variable_scope("decoder_rnn") as scope:
    pointer_cell = PointerWrapper(cell, attention_size=hidden_dim, memory=enc_outputs)


    if is_train:
      # Training Helper
      helper = tf.contrib.seq2seq.TrainingHelper(inputs, seq_length)
      # Basic Decoder
      initial_state = pointer_cell.zero_state(batch_size, tf.float32).clone(cell_state=enc_final_states)

      decoder = tf.contrib.seq2seq.BasicDecoder(pointer_cell, helper, initial_state)
      # Decode
      output, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True)
      # logits
      outputs = output.rnn_output
      sample_id = output.sample_id
    else:
      def initial_fn():
          initial_elements_finished = (0 >= seq_length)  # all False at the initial step
          initial_input = enc_outputs[:,0,:]
          return initial_elements_finished, initial_input

      def sample_fn(time, outputs, state):
          prediction_id = tf.to_int32(tf.argmax(outputs, axis=1))
          return prediction_id

      def next_inputs_fn(time, outputs, state, sample_ids):
          idx_pairs = index_matrix_to_pairs(sample_ids)
          next_inputs = tf.gather_nd(enc_outputs, idx_pairs)
          elements_finished = (time >= seq_length)  # this operation produces boolean tensor of [batch_size]
          all_finished = tf.reduce_all(elements_finished)  # -> boolean scalar
          next_state = state
          return elements_finished, next_inputs, next_state

      helper = tf.contrib.seq2seq.CustomHelper(initial_fn, sample_fn, next_inputs_fn)

      # Basic Decoder
      initial_state = pointer_cell.zero_state(batch_size, tf.float32).clone(cell_state=enc_final_states)

      decoder = custom_tf.Mydecoder(pointer_cell, helper, initial_state, batch_size, max_length)
      # decoder = Mydecoder(pointer_cell, helper, initial_state)

      # Decode
      output, final_state, _ = custom_tf.dynamic_decode(decoder, impute_finished=True)
      # logits
      outputs = output.rnn_output
      sample_id = output.sample_id

    return outputs, sample_id, helper

def trainable_initial_state(batch_size, state_size,
                            initializer=None, name="initial_state"):
  flat_state_size = nest.flatten(state_size)

  if not initializer:
    flat_initializer = tuple(tf.zeros_initializer for _ in flat_state_size)
  else:
    flat_initializer = tuple(tf.zeros_initializer for initializer in flat_state_size)

  names = ["{}_{}".format(name, i) for i in range(len(flat_state_size))]
  print (names)
  tiled_states = []

  for name, size, init in zip(names, flat_state_size, flat_initializer):
    shape_with_batch_dim = [1, size]
    initial_state_variable = tf.get_variable(
        name, shape=shape_with_batch_dim, initializer=init())

    tiled_state = tf.tile(initial_state_variable,
                          [batch_size, 1], name=(name + "_tiled"))
    tiled_states.append(tiled_state)

  return nest.pack_sequence_as(structure=state_size,
                               flat_sequence=tiled_states)

def index_matrix_to_pairs(index_matrix):
  # [[3,1,2], [2,3,1]] -> [[[0, 3], [0, 1], [0, 2]],
  #                        [[1, 2], [1, 3], [1, 1]]]
  replicated_first_indices = tf.range(tf.shape(index_matrix)[0])
  rank = len(index_matrix.get_shape())
  if rank == 2:
    replicated_first_indices = tf.tile(
        tf.expand_dims(replicated_first_indices, dim=1),
        [1, tf.shape(index_matrix)[1]])
  return tf.stack([replicated_first_indices, index_matrix], axis=rank)
