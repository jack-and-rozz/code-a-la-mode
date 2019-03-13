# coding: utf-8
import sys, os, re, time
import numpy as np
import tensorflow as tf
from dataset import Dataset
def _create_dir(path):
  if not os.path.exists(path):
    os.makedirs(path)

class Trainer:
  def __init__(self, config, sess, model):
    self.config = config
    self.sess = sess
    self.model = model
    self.setup_dirs(config)
    self.setup_params_for_train(config)
    self.restore_model(config)

  def train(self):
    for epoch in range(self.epoch.eval(), self.config.max_epoch):
      self.output_parameters()
      log_dir = self.games_path + '/%03d' % self.epoch.eval()
      _create_dir(log_dir)
      sys.stderr.write('<Epoch %d>' % epoch)
      # Wait until the similator finishes enough number of games. 
      while True:
        log_files = [log_dir + '/' + x for x in os.listdir(log_dir) if x[-5:] == '.json']
        sys.stderr.write("\rWaiting simulator finishes enough number of games (%d/%d)." % (len(log_files), self.config.games_per_epoch))
        time.sleep(1)
        if len(log_files) >= self.config.games_per_epoch:
          break
      sys.stderr.write('Start training.\n')
      data = Dataset(log_files)
      is_best = self.log_summary(data)
      self.run_epoch(data)
      self.add_epoch()
      self.save_model(self.model, save_as_best=is_best)
    self.output_parameters()

  def run_epoch(self, data):
    pass

  def log_summary(data):
    is_best = True
    return is_best

  def output_parameters(self):
    epoch = self.epoch.eval()
    variables_path = self.root_path + '/variables.list'
    with open(variables_path, 'w') as f:
      variable_names = sorted([v.name + ' ' + str(v.get_shape()) for v in tf.global_variables()])
      variable_names = [name for name in variable_names if not re.search('Adam', name)]
      f.write('\n'.join(variable_names) + '\n')

    variables = sorted([(v.name, v) for v in tf.global_variables()])
    parameters_path = self.parameters_path + '/%03d' % epoch
    _create_dir(parameters_path)
    for name, v in variables:
      if not re.search('Adam', name) and len(v.shape) > 0 and len(v.shape) < 3:
        name = name[:-2].replace('/', '_')
        np.savetxt(parameters_path + '/' + name, v.eval(), 
                   fmt='%.11e')

  def setup_params_for_train(self, config):
    self.optimizer_type = config.optimizer_type
    self.decay_rate_per_epoch = config.decay_rate_per_epoch
    self.max_gradient_norm = config.max_gradient_norm

    self.is_training = tf.placeholder(tf.bool, name='is_training', shape=[]) 
    with tf.name_scope('keep_prob'):
      self.keep_prob = 1.0 - tf.to_float(self.is_training) * config.dropout_rate

    with tf.name_scope('global_variables'):
      self.global_step = tf.get_variable(
        "global_step", trainable=False, shape=[],  dtype=tf.int32,
        initializer=tf.constant_initializer(0, dtype=tf.int32)) 

      self.epoch = tf.get_variable(
        "epoch", trainable=False, shape=[], dtype=tf.int32,
        initializer=tf.constant_initializer(0, dtype=tf.int32)) 

      # Decay by epoch.
      self.learning_rate = tf.train.exponential_decay(
        config.learning_rate, self.epoch,
        1, self.decay_rate_per_epoch, staircase=True)

      self.optimizer = getattr(tf.train, self.optimizer_type)(self.learning_rate)

    # Define operations in advance not to create ops in the loop.
    with tf.name_scope('add_epoch'):
      self._add_epoch = tf.assign(self.epoch, tf.add(self.epoch, tf.constant(1, dtype=tf.int32)))

    with tf.name_scope('update_max_score'):
      self._next_score = tf.placeholder(tf.float32, name='max_score_ph', shape=[])
      self.max_score = tf.get_variable(
        "max_score", trainable=False, shape=[],  dtype=tf.float32,
        initializer=tf.constant_initializer(0.0, dtype=tf.float32)) 

      self._update_max_score = tf.assign(self.max_score, self._next_score)


  def setup_dirs(self, config):
    self.root_path = config.model_root_path
    self.checkpoints_path = config.model_root_path +'/checkpoints'
    self.parameters_path = config.model_root_path + '/parameters'
    self.summaries_path = config.model_root_path + '/summaries'
    self.games_path = config.model_root_path + '/games'

    _create_dir(self.checkpoints_path)
    _create_dir(self.parameters_path)
    _create_dir(self.summaries_path)
    _create_dir(self.games_path)


  def add_epoch(self):
    self.sess.run(self._add_epoch)

  def restore_model(self, config):
    checkpoint_path = self.checkpoints_path
    self.saver = tf.train.Saver(tf.global_variables(), 
                                max_to_keep=config.max_to_keep)
    self.summary_writer = tf.summary.FileWriter(self.summaries_path, 
                                                self.sess.graph)
    if checkpoint_path and os.path.exists(checkpoint_path + '.index'):
      sys.stdout.write("Reading model parameters from %s\n" % checkpoint_path)
      self.saver.restore(self.sess, checkpoint_path)
    else:
      sys.stdout.write("Created model with fresh parameters.\n")
      tf.global_variables_initializer().run()

  def save_model(self, model, save_as_best=False):
    checkpoint_path = self.checkpoints_path + '/model.ckpt'
    self.saver.save(self.sess, checkpoint_path, global_step=model.epoch)
    if save_as_best:
      suffixes = ['data-00000-of-00001', 'index', 'meta']

      # Keep the older best checkpoint to handle failures in saving.
      for sfx in suffixes:
        target_path = self.checkpoints_path + "/%s.%s" % (BEST_CHECKPOINT_NAME, sfx)
        target_path_bak = self.checkpoints_path + "/%s.%s.old" % (BEST_CHECKPOINT_NAME, sfx)
        if os.path.exists(target_path):
          cmd = "mv %s %s" % (target_path, target_path_bak)
          os.system(cmd)

      # Copy the current best checkpoint.
      for sfx in suffixes:
        source_path = self.checkpoints_path + "/model.ckpt-%d.%s" % (model.epoch.eval(), sfx)
        target_path = self.checkpoints_path + "/%s.%s" % (BEST_CHECKPOINT_NAME, sfx)
        if os.path.exists(source_path):
          cmd = "cp %s %s" % (source_path, target_path)
          os.system(cmd)

      # Remove the older one.
      for sfx in suffixes:
        target_path_bak = self.checkpoints_path + "/%s.%s.old" % (BEST_CHECKPOINT_NAME, sfx)
        if os.path.exists(target_path_bak):
          cmd = "rm %s" % (target_path_bak)
          os.system(cmd)

