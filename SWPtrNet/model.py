import tensorflow as tf
from tensorflow.contrib.framework import arg_scope

from layers import *

class Model(object):
  def __init__(self, config, 
               inputs, labels, enc_seq_length, dec_seq_length, mask,
               reuse=False, is_critic=False):
    self.task = config.task
    self.debug = config.debug
    self.config = config

    self.input_dim = config.input_dim
    self.hidden_dim = config.hidden_dim
    self.num_layers = config.num_layers

    self.max_enc_length = config.max_enc_length
    self.max_dec_length = config.max_dec_length
    self.num_glimpse = config.num_glimpse

    self.init_min_val = config.init_min_val
    self.init_max_val = config.init_max_val
    self.initializer = \
        tf.random_uniform_initializer(self.init_min_val, self.init_max_val)

    self.use_terminal_symbol = config.use_terminal_symbol

    self.lr_start = config.lr_start
    self.lr_decay_step = 18 * config.lr_decay_step
    self.lr_decay_rate = config.lr_decay_rate
    self.max_grad_norm = config.max_grad_norm

    self.layer_dict = {}

    self.is_training = tf.placeholder(tf.bool, (), name="is_training")

    self.enc_inputs, self.dec_targets, self.enc_seq_length, self.dec_seq_length, self.mask = inputs['train'], labels['train'], enc_seq_length['train'], dec_seq_length['train'], mask['train']
    self.dec_targets = self.dec_targets - 1
    self.dec_seq_length -= 1
    self.mask = self.mask[:, 1:]

    self._build_model()

    if not reuse:
      self._build_optim()

    self.train_summary = tf.summary.merge([
        tf.summary.scalar("train/total_loss", self.total_loss),
        tf.summary.scalar("train/lr", self.lr),
    ])

    self.test_summary = tf.summary.merge([
        tf.summary.scalar("test/total_loss", self.total_loss),
    ])
    self.fetch={"x":self.enc_inputs, "y":self.dec_targets, "len_x":self.enc_seq_length, "len_y":self.dec_seq_length, "mask":self.mask}
    
  def run(self, sess, fetch, feed_dict):
    fetch['step'] = self.global_step
    result = sess.run(fetch,feed_dict)
    return result
  
  def get_data(self, sess, train):
    if train:
        return self.run(sess, self.fetch, feed_dict={self.is_training: True})
    else:
        return None
  
  def _build_model(self):
    tf.logging.info("Create a model..")
    self.global_step = tf.Variable(0, trainable=False)
    self.x=tf.placeholder(tf.float32, [None, 20, 2], "input_x")
    self.y=tf.placeholder(tf.int32, [None], "output_y")
    self.seq_length=tf.placeholder(tf.int32, [None], "seq_length")
    self.decoder_input=tf.placeholder(tf.float32, [None, 1, 2], "decoder_input")
    self.step_mask=tf.placeholder(tf.float32, [None, 20],"step_mask")

    self.num_nodes = tf.shape(self.x)[1]

    input_embed = tf.get_variable(
        "input_embed", [1, self.input_dim, self.hidden_dim],
        initializer=self.initializer)

    with tf.variable_scope("encoder"):
      self.embeded_enc_inputs = tf.nn.conv1d(
          self.x, input_embed, 1, "VALID")
      self.embeded_decoder_input = tf.nn.conv1d(
          self.decoder_input, input_embed, 1, "VALID")

    batch_size = tf.shape(self.x)[0]
    with tf.variable_scope("encoder"):
      self.enc_cell = LSTMCell(
          self.hidden_dim,
          initializer=self.initializer)

      if self.num_layers > 1:
        cells = [self.enc_cell] * self.num_layers
        self.enc_cell = MultiRNNCell(cells)

      self.enc_init_state = trainable_initial_state(
          batch_size, self.enc_cell.state_size)

      self.enc_outputs_raw, self.enc_final_states = tf.nn.dynamic_rnn(
          self.enc_cell, self.embeded_enc_inputs,
          self.seq_length, self.enc_init_state)
      self.enc_outputs = self.embeded_enc_inputs

    with tf.variable_scope("decoder"):

      self.dec_cell = LSTMCell(
          self.hidden_dim,
          initializer=self.initializer)

      if self.num_layers > 1:
        cells = [self.dec_cell] * self.num_layers
        self.dec_cell = MultiRNNCell(cells)

      self.dec_pred_logits, self.sample_id, _ = decoder_rnn(
          self.dec_cell, self.embeded_decoder_input, 
          self.enc_outputs_raw, self.enc_final_states,
          tf.cast(tf.ones(batch_size),tf.int32), self.hidden_dim,
          self.num_glimpse, batch_size, is_train=True,
          initializer=self.initializer)
      self.dec_pred_prob = tf.nn.softmax(
          self.dec_pred_logits, 2, name="dec_pred_prob")
      self.dec_pred = tf.argmax(
          self.dec_pred_logits, 2, name="dec_pred")

  def _build_optim(self):
    batch_size = 128
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.y, logits=tf.reshape(self.dec_pred_logits, (batch_size, 20)) * self.step_mask)
    batch_loss = tf.reduce_mean(losses)

    tf.losses.add_loss(batch_loss)
    total_loss = tf.losses.get_total_loss()

    self.total_loss = total_loss

    self.lr = tf.train.exponential_decay(
        self.lr_start, self.global_step, self.lr_decay_step,
        self.lr_decay_rate, staircase=True, name="learning_rate")

    optimizer = tf.train.AdamOptimizer(self.lr)

    if self.max_grad_norm != None:
      grads_and_vars = optimizer.compute_gradients(self.total_loss)
      for idx, (grad, var) in enumerate(grads_and_vars):
        if grad is not None:
          grads_and_vars[idx] = (tf.clip_by_norm(grad, self.max_grad_norm), var)
      self.optim = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
    else:
      self.optim = optimizer.minimize(self.total_loss, global_step=self.global_step)
