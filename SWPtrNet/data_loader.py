# Most of the codes are from 
# https://github.com/vshallc/PtrNets/blob/master/pointer/misc/tsp.py
import os
import re
import zipfile
import itertools
import threading
import numpy as np
# from tqdm import trange, tqdm
from collections import namedtuple

import tensorflow as tf
import glob

TSP = namedtuple('TSP', ['x', 'y', 'length', 'name'])

def length(x, y):
  return np.linalg.norm(np.asarray(x) - np.asarray(y))

def read_paper_dataset(paths, max_length, is_train):
  x, y = [], []
  for path in paths:
    tf.logging.info("Read dataset {} which is used in the paper..".format(path))
    with open(path) as f:
      for l in f:
        inputs, outputs = l.split(' output ')
        x.append(np.array(inputs.split(), dtype=np.float32).reshape([-1, 2]))
        y.append(np.array(outputs.split(), dtype=np.int32)[:-1]) # skip the last one
        if not is_train:
            return x, y
  return x, y

class TSPDataLoader(object):
  def __init__(self, config, rng=None):
    self.config = config
    self.rng = rng

    self.task = config.task.lower()
    self.batch_size = config.batch_size
    self.min_length = config.min_data_length
    self.max_length = config.max_data_length

    self.is_train = config.is_train
    self.use_terminal_symbol = config.use_terminal_symbol
    self.random_seed = config.random_seed

    self.data_num = {}
    self.data_num['train'] = config.train_num

    self.data_dir = config.data_dir
    self.task_name = "{}_({},{})".format(
        self.task, self.min_length, self.max_length)

    self.data = None
    self.coord = None
    self.threads = None
    self.input_ops, self.target_ops, self.length_ops = None, None, None
    self.queue_ops, self.enqueue_ops = None, None
    self.x, self.y, self.seq_length, self.mask = None, None, None, None

    paths = glob.glob("data/*")
    self.read_zip_and_update_data(paths, "train")


    self._create_input_queue()

  def _create_input_queue(self, queue_capacity_factor=16):
    self.input_ops, self.target_ops, self.length_ops = {}, {}, {}
    self.queue_ops, self.enqueue_ops = {}, {}
    self.x, self.y, self.seq_length, self.mask = {}, {}, {}, {}

    for name in self.data_num.keys():
      self.input_ops[name] = tf.placeholder(tf.float32, shape=[None, None])
      self.target_ops[name] = tf.placeholder(tf.int32, shape=[None])
      self.length_ops[name] = tf.placeholder(tf.int32, shape=[])

      min_after_dequeue = 1000
      capacity = min_after_dequeue + 3 * self.batch_size

      if name == "train":
        self.queue_ops[name] = tf.RandomShuffleQueue(
            capacity=capacity,
            min_after_dequeue=min_after_dequeue,
            dtypes=[tf.float32, tf.int32, tf.int32],
            shapes=[[self.max_length, 2,], [self.max_length], []],
            seed=self.random_seed,
            name="random_queue_{}".format(name))
      self.enqueue_ops[name] = \
          self.queue_ops[name].enqueue([self.input_ops[name], self.target_ops[name], self.length_ops[name]])

      inputs, labels, length = self.queue_ops[name].dequeue()

      seq_length = tf.shape(inputs)[0]
      if self.use_terminal_symbol:
        mask = tf.ones([length], dtype=tf.float32) # terminal symbol
      else:
        mask = tf.ones([length], dtype=tf.float32)
      if name=="train":
        batch_size=self.batch_size

      self.x[name], self.y[name], self.seq_length[name], self.mask[name] = \
          tf.train.batch(
              [inputs, labels, seq_length, mask],
              batch_size=batch_size,
              capacity=capacity,
              dynamic_pad=True,
              allow_smaller_final_batch=True,
              name="batch_and_pad")


  def run_input_queue(self, sess):
    self.threads = []
    self.coord = tf.train.Coordinator()

    for name in self.data_num.keys():
      def load_and_enqueue(sess, name, input_ops, target_ops, length_ops, enqueue_ops, coord):
        idx = 0
        while not coord.should_stop():
          feed_dict = {
              input_ops[name]: self.data[name].x[idx],
              target_ops[name]: self.data[name].y[idx],
              length_ops[name]: self.data[name].length[idx]
          }
          sess.run(self.enqueue_ops[name], feed_dict=feed_dict)
          idx = idx+1 if idx+1 <= len(self.data[name].x) - 1 else 0

      args = (sess, name, self.input_ops, self.target_ops, self.length_ops, self.enqueue_ops, self.coord)
      t = threading.Thread(target=load_and_enqueue, args=args)
      t.start()
      self.threads.append(t)
      tf.logging.info("Thread for [{}] start".format(name))

  def stop_input_queue(self):
    self.coord.request_stop()
    self.coord.join(self.threads)
    tf.logging.info("All threads stopped")

  def get_path(self, name):
    return os.path.join(
        self.data_dir, "{}_{}={}.npz".format(
            self.task_name, name, self.data_num[name]))

  def read_zip_and_update_data(self, path, name):


    x_list, y_list = read_paper_dataset(path, self.max_length, self.config.is_train)

    x = np.zeros([len(x_list), self.max_length, 2], dtype=np.float32)
    y = np.zeros([len(y_list), self.max_length], dtype=np.int32)
    length = np.zeros([len(y_list)], dtype=np.int32)

    for idx, (nodes, res) in enumerate(zip(x_list, y_list)):
      x[idx,:len(nodes)] = nodes
      y[idx,:len(res)] = res
      length[idx] = len(res)

    if name=="train":

      index=np.random.permutation(x.shape[0])
      x=x[index]
      y=y[index]
      length=length[index]


    if self.data is None:
      self.data = {}

    self.data[name] = TSP(x=x, y=y, length=length, name=name)
