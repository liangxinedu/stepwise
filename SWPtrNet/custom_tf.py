from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

import tensorflow as tf


__all__ = ["Decoder", "dynamic_decode"]


_transpose_batch_time = rnn._transpose_batch_time  # pylint: disable=protected-access

def _create_zero_outputs(size, dtype, batch_size):
  """Create a zero outputs Tensor structure."""
  def _t(s):
    return (s if isinstance(s, ops.Tensor) else constant_op.constant(
        tensor_shape.TensorShape(s).as_list(),
        dtype=dtypes.int32,
        name="zero_suffix_shape"))

  def _create(s, d):
    return array_ops.zeros(
        array_ops.concat(
            ([batch_size], _t(s)), axis=0), dtype=d)

  return nest.map_structure(_create, size, dtype)


def dynamic_decode(decoder,
                   output_time_major=False,
                   impute_finished=False,
                   maximum_iterations=None,
                   parallel_iterations=32,
                   swap_memory=False,
                   scope=None):
  """Perform dynamic decoding with `decoder`.
  Calls initialize() once and step() repeatedly on the Decoder object.
  Args:
    decoder: A `Decoder` instance.
    output_time_major: Python boolean.  Default: `False` (batch major).  If
      `True`, outputs are returned as time major tensors (this mode is faster).
      Otherwise, outputs are returned as batch major tensors (this adds extra
      time to the computation).
    impute_finished: Python boolean.  If `True`, then states for batch
      entries which are marked as finished get copied through and the
      corresponding outputs get zeroed out.  This causes some slowdown at
      each time step, but ensures that the final state and outputs have
      the correct values and that backprop ignores time steps that were
      marked as finished.
    maximum_iterations: `int32` scalar, maximum allowed number of decoding
       steps.  Default is `None` (decode until the decoder is fully done).
    parallel_iterations: Argument passed to `tf.while_loop`.
    swap_memory: Argument passed to `tf.while_loop`.
    scope: Optional variable scope to use.
  Returns:
    `(final_outputs, final_state, final_sequence_lengths)`.
  Raises:
    TypeError: if `decoder` is not an instance of `Decoder`.
    ValueError: if `maximum_iterations` is provided but is not a scalar.
  """
  if not isinstance(decoder, tf.contrib.seq2seq.Decoder):
    raise TypeError("Expected decoder to be type Decoder, but saw: %s" %
                    type(decoder))

  with variable_scope.variable_scope(scope, "decoder") as varscope:
    # Properly cache variable values inside the while_loop
    if varscope.caching_device is None:
      varscope.set_caching_device(lambda op: op.device)

    if maximum_iterations is not None:
      maximum_iterations = ops.convert_to_tensor(
          maximum_iterations, dtype=dtypes.int32, name="maximum_iterations")
      if maximum_iterations.get_shape().ndims != 0:
        raise ValueError("maximum_iterations must be a scalar")

    initial_finished, initial_inputs, initial_state, hist = decoder.initialize()

    zero_outputs = _create_zero_outputs(decoder.output_size,
                                        decoder.output_dtype,
                                        decoder.batch_size)

    if maximum_iterations is not None:
      initial_finished = math_ops.logical_or(
          initial_finished, 0 >= maximum_iterations)
    initial_sequence_lengths = array_ops.zeros_like(
        initial_finished, dtype=dtypes.int32)
    initial_time = constant_op.constant(0, dtype=dtypes.int32)

    def _shape(batch_size, from_shape):
      if not isinstance(from_shape, tensor_shape.TensorShape):
        return tensor_shape.TensorShape(None)
      else:
        batch_size = tensor_util.constant_value(
            ops.convert_to_tensor(
                batch_size, name="batch_size"))
        return tensor_shape.TensorShape([batch_size]).concatenate(from_shape)

    def _create_ta(s, d):
      return tensor_array_ops.TensorArray(
          dtype=d,
          size=0,
          dynamic_size=True,
          element_shape=_shape(decoder.batch_size, s))

    initial_outputs_ta = nest.map_structure(_create_ta, decoder.output_size,
                                            decoder.output_dtype)

    def condition(unused_time, unused_outputs_ta, unused_state, unused_inputs,
                  finished, unused_sequence_lengths, unused_hist):
      return math_ops.logical_not(math_ops.reduce_all(finished))

    def body(time, outputs_ta, state, inputs, finished, sequence_lengths, hist):
      """Internal while_loop body.
      Args:
        time: scalar int32 tensor.
        outputs_ta: structure of TensorArray.
        state: (structure of) state tensors and TensorArrays.
        inputs: (structure of) input tensors.
        finished: bool tensor (keeping track of what's finished).
        sequence_lengths: int32 tensor (keeping track of time of finish).
      Returns:
        `(time + 1, outputs_ta, next_state, next_inputs, next_finished,
          next_sequence_lengths)`.
        ```
      """
      (next_outputs, decoder_state, next_inputs,
       decoder_finished, hist) = decoder.step(time, inputs, state, hist)
      if decoder.tracks_own_finished:
        next_finished = decoder_finished
      else:
        next_finished = math_ops.logical_or(decoder_finished, finished)
      if maximum_iterations is not None:
        next_finished = math_ops.logical_or(
            next_finished, time + 1 >= maximum_iterations)
      next_sequence_lengths = array_ops.where(
          math_ops.logical_and(math_ops.logical_not(finished), next_finished),
          array_ops.fill(array_ops.shape(sequence_lengths), time + 1),
          sequence_lengths)

      nest.assert_same_structure(state, decoder_state)
      nest.assert_same_structure(outputs_ta, next_outputs)
      nest.assert_same_structure(inputs, next_inputs)

      # Zero out output values past finish
      if impute_finished:
        emit = nest.map_structure(
            lambda out, zero: array_ops.where(finished, zero, out),
            next_outputs,
            zero_outputs)
      else:
        emit = next_outputs

      # Copy through states past finish
      def _maybe_copy_state(new, cur):
        # TensorArrays and scalar states get passed through.
        if isinstance(cur, tensor_array_ops.TensorArray):
          pass_through = True
        else:
          new.set_shape(cur.shape)
          pass_through = (new.shape.ndims == 0)
        return new if pass_through else array_ops.where(finished, cur, new)

      if impute_finished:
        next_state = nest.map_structure(
            _maybe_copy_state, decoder_state, state)
      else:
        next_state = decoder_state

      outputs_ta = nest.map_structure(lambda ta, out: ta.write(time, out),
                                      outputs_ta, emit)
      return (time + 1, outputs_ta, next_state, next_inputs, next_finished,
              next_sequence_lengths, hist)

    res = control_flow_ops.while_loop(
        condition,
        body,
        loop_vars=[
            initial_time, initial_outputs_ta, initial_state, initial_inputs,
            initial_finished, initial_sequence_lengths, hist
        ],
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory)

    final_outputs_ta = res[1]
    final_state = res[2]
    final_sequence_lengths = res[5]

    final_outputs = nest.map_structure(lambda ta: ta.stack(), final_outputs_ta)

    try:
      final_outputs, final_state = decoder.finalize(
          final_outputs, final_state, final_sequence_lengths)
    except NotImplementedError:
      pass

    if not output_time_major:
      final_outputs = nest.map_structure(_transpose_batch_time, final_outputs)

  return final_outputs, final_state, final_sequence_lengths

from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.contrib.seq2seq.python.ops import helper as helper_py
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util import nest

class Mydecoder(tf.contrib.seq2seq.BasicDecoder):
  def __init__(self, cell, helper, initial_state, batch_size, max_length, output_layer=None):
    """Initialize BasicDecoder.
    Args:
      cell: An `RNNCell` instance.
      helper: A `Helper` instance.
      initial_state: A (possibly nested tuple of...) tensors and TensorArrays.
        The initial state of the RNNCell.
      output_layer: (Optional) An instance of `tf.layers.Layer`, i.e.,
        `tf.layers.Dense`. Optional layer to apply to the RNN output prior
        to storing the result or sampling.
    Raises:
      TypeError: if `cell`, `helper` or `output_layer` have an incorrect type.
    """
    # rnn_cell_impl.assert_like_rnncell("cell", cell)
    if not isinstance(helper, helper_py.Helper):
      raise TypeError("helper must be a Helper, received: %s" % type(helper))
    if (output_layer is not None
        and not isinstance(output_layer, layers_base.Layer)):
      raise TypeError(
          "output_layer must be a Layer, received: %s" % type(output_layer))
    self._cell = cell
    self._helper = helper
    self._initial_state = initial_state
    self._output_layer = output_layer
    self.max_length = tf.convert_to_tensor(max_length, tf.int32)
    self.batch_size1 = batch_size

  def initialize(self, name=None):
    """Initialize the decoder.
       Args:
           name: Name scope for any created operations.
      Returns:
        `(finished, first_inputs, initial_state)`.
    """
    maximum_length = tf.convert_to_tensor(self.max_length, tf.int32)
    hist = tf.concat([tf.zeros([self.batch_size1,1]),tf.ones([self.batch_size1, maximum_length])], axis=1)
    return self._helper.initialize() + (self._initial_state, hist)

  def step(self, time, inputs, state, hist, name=None):
    """Perform a decoding step.
      Args:
        time: scalar `int32` tensor.
        inputs: A (structure of) input tensors.
        state: A (structure of) state tensors and TensorArrays.
        name: Name scope for any created operations.
      Returns:
        `(outputs, next_state, next_inputs, finished)`.
      """
    with ops.name_scope(name, "BasicDecoderStep", (time, inputs, state)):
      cell_outputs, cell_state = self._cell(inputs, state)
      if self._output_layer is not None:
        cell_outputs = self._output_layer(cell_outputs)
      sample_ids = self._helper.sample(time=time, outputs=cell_outputs, state=cell_state)
      maximum_length = tf.convert_to_tensor(self.max_length+1, tf.int32)
      hist = tf.subtract(hist, tf.one_hot(sample_ids, maximum_length))
      (finished, next_inputs, next_state) = self._helper.next_inputs(
          time=time,
          outputs=cell_outputs,
          state=cell_state,
          sample_ids=sample_ids)
      outputs = tf.contrib.seq2seq.BasicDecoderOutput(cell_outputs, sample_ids)
      return (outputs, next_state, next_inputs, finished, hist)
