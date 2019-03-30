# coding: utf-8
import sys, os, re, time
import numpy as np
import tensorflow as tf
from dataset import Dataset, print_batch
from utils import make_summary, logManager, dbgprint
from logging import FileHandler

def _create_dir(path):
  if not os.path.exists(path):
    os.makedirs(path)

BEST_CHECKPOINT_NAME = 'model.ckpt.best'

def get_logger(logfile_path=None):
  logger = logManager(handler=FileHandler(logfile_path)) if logfile_path else logManager()
  return logger


class Trainer:
  def __init__(self, config, sess, model):
    self.config = config
    self.sess = sess
    self.model = model
    self.setup_dirs(config)
    self.setup_params_for_train(config)
    self.updates = self.get_updates(config, model.loss, self.global_step, self.epoch) if config.mode == 'train' else None
    self.restore_model(config)
    self.logger = get_logger(self.config.model_root_path + '/train.log')

  def get_updates(self, config, loss, global_step, epoch):
    with tf.name_scope("update"):
      params = tf.contrib.framework.get_trainable_variables()
      opt = self.optimizer
      return opt.minimize(loss, global_step=global_step)
      
      ######################
      # clippingがおかしい？
      if config.max_gradient_norm:
        gradients = [grad for grad, _ in opt.compute_gradients(loss)]
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 
                                                      config.max_gradient_norm)
        grad_and_vars = [(g, v) for g, v in zip(clipped_gradients, params)]
        updates = opt.apply_gradients(
          grad_and_vars, global_step=global_step)
        return updates
      else:
        return opt.minimize(loss)

  def train(self):
    for epoch in range(self.epoch.eval(), self.config.max_epoch):
      self.output_parameters()
      log_dir = self.games_path + '/%03d' % self.epoch.eval()
      _create_dir(log_dir)
      # Wait until the similator finishes enough number of games. 
      self.logger.info('<Epoch %d>\tWaiting for sampling.' % epoch)
      while True:
        log_files = [log_dir + '/' + x for x in os.listdir(log_dir) if x[-5:] == '.json']
        sys.stderr.write("\r<Epoch %d>\tWaiting the simulator finishes enough number of games (%d/%d)." % (epoch, len(log_files), self.config.games_per_epoch))
        time.sleep(1)
        if len(log_files) >= self.config.games_per_epoch:
          print()
          break
      self.logger.info('<Epoch %d>\tStart training.' % epoch)
      sys.stderr.write('<Epoch %d>\tStart training.\n' % epoch)
      data = Dataset(log_files, self.config.batch_size)
      loss = self.run_epoch(data)
      is_best = self.log_summary(data, loss)
      self.add_epoch()
      self.save_model(self.model, save_as_best=is_best)
    self.output_parameters()

  def run_epoch(self, data):
    losses = []
    model = self.model
    for step, batch in enumerate(data):
      #print_batch(batch)
      input_feed = self.model.get_input_feed(batch)
      outputs = self.sess.run(self.model.debug_ops, input_feed)
      # for out, ops in zip(outputs, model.debug_ops):
      #   dbgprint(ops, np.any(np.isnan(out)))

      loss, _ = self.sess.run([self.model.loss, self.updates], input_feed)
      sys.stderr.write('Epoch %d, Step %d:\t loss=%e\n' % (self.epoch.eval(), step, loss))
      if np.isnan(loss):
        sys.stderr.write('NaN loss error.\n')

      losses.append(loss)
    return sum(losses) / len(losses)

  def log_summary(self, data, loss):
    self_score = data.average_score
    enemy_score = data.enemy_average_score
    score_ratio = 1.0 * self_score / enemy_score
    summary = make_summary({
      'loss': loss,
      'AveScore/Self': self_score, 
      'AveScore/Enemy': enemy_score,
      'AveScore/Ratio': score_ratio,
    })
    self.logger.info("<Epoch %d>\tScore(self, enemy, ratio)=(%d, %d, %.2f), loss=%.3f, learning_rate=%e" % (
      self.epoch.eval(), 
      self_score,
      enemy_score,
      score_ratio,
      loss,
      self.learning_rate.eval()
    ))
    self.summary_writer.add_summary(summary, self.epoch.eval())
    is_best = False

    if self_score >= self.max_score.eval():
      is_best = True
      prev_score = self.max_score.eval()
      self.logger.info("<Epoch %d>\tMax average score was updated (%d->%d)" % (
        self.epoch.eval(),
        prev_score,
        self_score,
      ))
      self.update_max_score(self_score)
    return is_best

  def output_parameters(self):
    epoch = self.epoch.eval()
    variables_path = self.root_path + '/variables.list'      

    def n_params(v):
      xx = 1
      for x in v.shape:
        xx *= x
      return xx

    with open(variables_path, 'w') as f:
      variable_names = sorted([v.name + ' ' + str(v.get_shape()) for v in tf.global_variables()])
      variable_names = [name for name in variable_names if not re.search('Adam', name)]
      f.write('\n'.join(variable_names) + '\n')

      n_total_parameters = sum([n_params(v) for v in tf.global_variables() if not re.search('Adam', v.name) and len(v.shape) > 0 ])
      f.write('# total parameters: %d\n' % n_total_parameters)

    variables = sorted([(v.name, v) for v in tf.global_variables()])
    parameters_path = self.parameters_path + '/%03d' % epoch
    _create_dir(parameters_path)
    for name, v in variables:
      if not re.search('Adam', name) and len(v.shape) > 0 and len(v.shape) < 3:
        name = name[:-2].replace('/', '_')
        np.savetxt(parameters_path + '/' + name, v.eval(), 
                   fmt='%.8e')

  def setup_params_for_train(self, config):
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
        1, config.decay_rate_per_epoch, staircase=True)

      self.optimizer = getattr(tf.train, config.optimizer_type)(
        self.learning_rate)
        #epsilon=1e-4)

    # Define operations in advance not to create ops in the loop.
    with tf.name_scope('add_epoch'):
      self._add_epoch = tf.assign(self.epoch, tf.add(self.epoch, tf.constant(1, dtype=tf.int32)))

    with tf.name_scope('update_max_score'):
      self._next_score = tf.placeholder(tf.float32, name='max_score_ph', shape=[])
      self.max_score = tf.get_variable(
        "max_score", trainable=False, shape=[],  dtype=tf.float32,
        initializer=tf.constant_initializer(0.0, dtype=tf.float32)) 

      self._update_max_score = tf.assign(self.max_score, self._next_score)

  def add_epoch(self):
    self.sess.run(self._add_epoch)

  def update_max_score(self, score):
    self.sess.run(self._update_max_score, feed_dict={self._next_score:score})

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


  def restore_model(self, config, load_best=False):
    if load_best:
      checkpoint_path = self.checkpoints_path + '/' + BEST_CHECKPOINT_NAME 
    else:
      ckpt = tf.train.get_checkpoint_state(self.checkpoints_path)
      checkpoint_path = ckpt.model_checkpoint_path if ckpt else None
    

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
    self.saver.save(self.sess, checkpoint_path, global_step=self.epoch)
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
        source_path = self.checkpoints_path + "/model.ckpt-%d.%s" % (self.epoch.eval(), sfx)
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



