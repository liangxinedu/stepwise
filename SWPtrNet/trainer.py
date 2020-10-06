import os
import numpy as np
from tqdm import trange
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope

from model import Model
from utils import show_all_variables
from data_loader import TSPDataLoader
import time

class Trainer(object):
  def __init__(self, config, rng):
    self.config = config
    self.rng = rng

    self.task = config.task
    self.model_dir = config.model_dir
    self.gpu_memory_fraction = config.gpu_memory_fraction

    self.log_step = config.log_step
    self.max_step = config.max_step
    self.num_log_samples = config.num_log_samples
    self.checkpoint_secs = config.checkpoint_secs

    if config.task.lower().startswith('tsp'):
      self.data_loader = TSPDataLoader(config, rng=self.rng)
    else:
      raise Exception("[!] Unknown task: {}".format(config.task))

    self.summary_writer = tf.summary.FileWriter(self.model_dir)
    self.model = Model(
        config,
        inputs=self.data_loader.x,
        labels=self.data_loader.y,
        enc_seq_length=self.data_loader.seq_length,
        dec_seq_length=self.data_loader.seq_length,
        mask=self.data_loader.mask)

    self.build_session()
    show_all_variables()

  def build_session(self):
    self.saver = tf.train.Saver()

    sv = tf.train.Supervisor(logdir=self.model_dir,
                             is_chief=True,
                             saver=self.saver,
                             summary_op=None,
                             summary_writer=self.summary_writer,
                             save_summaries_secs=300,
                             save_model_secs=self.checkpoint_secs,
                             global_step=self.model.global_step)

    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=self.gpu_memory_fraction,
        allow_growth=True) # seems to be not working
    sess_config = tf.ConfigProto(allow_soft_placement=True,
                                 gpu_options=gpu_options)

    sess_config.gpu_options.allow_growth = True
    self.sess = sv.prepare_or_wait_for_session(config=sess_config)
    if not self.config.is_train:
        self.saver.restore(self.sess, self.config.load_path)
  
  def train(self):
    tf.logging.info("Training starts...")
    self.data_loader.run_input_queue(self.sess)
    test_data=np.load("test.data.npy")
    loss_list = []
    distance_list = []
    accuracy_list = []
    n_classes = 20

    fetch = {'optim': self.model.optim,
             'loss': self.model.total_loss,
             'logits': self.model.dec_pred_logits}
    

    data=self.model.get_data(self.sess,train=True)


    for k in trange(self.max_step):
        data=self.model.get_data(self.sess,train=True)
        optimal_distance=np.zeros(128)
        predict_distance=np.zeros(128)
        num_nodes=20
        data_x=data["x"]
        data_y=data["y"]
        mask=np.ones((128,20),dtype="bool")
        distance=np.zeros(128)
        for i in range(18):
            decoder_input=data_x[np.arange(128).astype("int32"),data_y[:,i].astype("int32")].reshape(128,1,2)
            step_data_x=data_x[mask].reshape(128,-1,2)
            one_hot_targets = np.eye(n_classes)[data_y[:,i+1]][mask].reshape(128,-1)
            step_data_y=np.argmax(one_hot_targets,-1)

            if num_nodes < 20:
                step_data_x_padding=np.concatenate([step_data_x, np.zeros((128,20-num_nodes,2))],1)
            else:
                step_data_x_padding=step_data_x

            mask[np.arange(128).astype("int32"),data_y[:,i+1].astype("int32")]=0

            step_mask=np.zeros((128,20))
            step_mask[:,1:num_nodes]=1
            feed_dict={self.model.x:step_data_x_padding,
                       self.model.y:step_data_y, 
                       self.model.seq_length:np.ones(128)*num_nodes,
                       self.model.decoder_input:decoder_input,
                       self.model.step_mask:step_mask}

            result = self.model.run(self.sess, fetch, feed_dict)

            num_nodes-=1
            loss_list.append(result['loss'])
            accuracy_list.append(np.mean(step_data_y==np.argmax(result['logits'].reshape(128,20)*step_mask,-1)))
            next_input=data_x[np.arange(128).astype("int32"),data_y[:,i+1].astype("int32")].reshape(128,1,2)
            distance+=np.sqrt(np.sum(((next_input.reshape(128,2)-decoder_input.reshape(128,2))**2), -1))
        input1=data_x[np.arange(128).astype("int32"),data_y[:,-2].astype("int32")].reshape(128,1,2)
        input2=data_x[np.arange(128).astype("int32"),data_y[:,-1].astype("int32")].reshape(128,1,2)
        distance+=np.sqrt(np.sum((input1.reshape(128,2)-input2.reshape(128,2))**2, -1))
        input1=data_x[np.arange(128).astype("int32"),data_y[:,-1].astype("int32")].reshape(128,1,2)
        input2=data_x[np.arange(128).astype("int32"),data_y[:,0].astype("int32")].reshape(128,1,2)
        distance+=np.sqrt(np.sum((input1.reshape(128,2)-input2.reshape(128,2))**2, -1))
        distance_list.append(np.mean(distance))

        if k % self.log_step == 0:
            print ("")
            print ("loss", np.mean(loss_list), "opt_dist", np.mean(distance_list), "accuracy", np.mean(accuracy_list))
            loss_list=[]
            distance_list=[]
            accuracy_list=[]
            self._test(test_data)
    self.data_loader.stop_input_queue()

  def test(self):
    tf.logging.info("Testing starts...")


    test_data=np.load("test.data.npy")
    self._test(test_data)

    print ("------ done -------")

  def _test(self, test_data):
    start = time.time()

    distance_list = []
    fetch = {'logits': self.model.dec_pred_logits}
    batch_size = 128
    for j in trange(test_data.shape[0] // batch_size):
        data = test_data[j * batch_size:(j + 1) * batch_size]
        num_nodes = data.shape[1]
        mask = np.ones((batch_size, num_nodes), dtype="bool")
        decoder_input = data[:,0].reshape(batch_size, 1, 2)
        index_matrix = np.concatenate([np.arange(20).reshape(1, 20) for i in range(batch_size)], 0)

        result_index = [np.zeros(data.shape[0])]
        for i in range(18):
            step_data_x = data[mask].reshape(batch_size, -1, 2)
            if num_nodes < 20:
                step_data_x_padding = np.concatenate([step_data_x, np.zeros((batch_size, 20 - num_nodes, 2))], 1)
            else:
                step_data_x_padding = step_data_x
            index_matrix_step = index_matrix[mask].reshape(batch_size, -1)
            step_mask = np.zeros((batch_size, 20))
            step_mask[:, 1:num_nodes] = 1
            feed_dict={self.model.x:step_data_x_padding,
                       self.model.seq_length:np.ones(batch_size) * num_nodes,
                       self.model.decoder_input:decoder_input,
                       self.model.step_mask:step_mask}
            
            result = self.model.run(self.sess, fetch, feed_dict)

            logits = result['logits'].reshape(batch_size, 20) * step_mask
            prediction = np.argmax(logits, 1)
            prediction = index_matrix_step[np.arange(batch_size).astype("int32"), prediction.astype("int32")]
    
            mask[np.arange(batch_size).astype("int32"), prediction.astype("int32")] = 0
            num_nodes -= 1
            next_input = data[np.arange(batch_size).astype("int32"), prediction.astype("int32")].reshape(batch_size, 1, 2)
            decoder_input = next_input
            result_index.append(prediction.reshape(-1))

        mask[:, 0] = 0
        last_ones_index = np.argmax(mask, -1)
        last_selected_ones = data[np.arange(batch_size).astype("int"), last_ones_index.astype("int")].reshape(batch_size, 1, 2)

        result_index.append(last_ones_index.reshape(-1))
        result_index = np.stack(result_index, -1)

        assert np.array_equal(
                np.arange(20).reshape(1, -1).repeat(data.shape[0], 0),
                np.sort(result_index, 1)
        ), "Invalid tour"

        d = data[np.arange(data.shape[0]).reshape(-1, 1), result_index.astype(int)]
        distance = np.sum(np.sqrt(np.sum((d - np.roll(d, 1, 1)) ** 2, -1)), -1)
        distance_list.append(np.mean(distance))

    print ("test result", np.mean(distance_list), time.time() - start)



  def _get_summary_writer(self, result):
    if result['step'] % self.log_step == 0:
      return self.summary_writer
    else:
      return None
