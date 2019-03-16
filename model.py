# coding: utf-8
import tensorflow as tf 
from utils import dbgprint, dotDict, shape, flatten_timeseries_tensor, batch_gather

from play import recDotDefaultDict, action_vocab, tile_vocab, item_vocab, board_vocab, NPCNNBased, Y, X

PARAM_DTYPE = tf.float32 # Pass epsilon=1e-4 when constructing AdamOptimizer if this model have it parameters as tf.float16. Otherwise some elements of gradients become NaN.
CALC_DTYPE = tf.float32
def clipped_loss(q_values, target_values):
  error = tf.abs(target_values - q_values)
  quadratic = tf.clip_by_value(error, 0.0, 1.0)
  linear = error - quadratic
  losses = 0.5 * tf.square(quadratic) + linear
  return losses

class TFCNNBased:
  def __init__(self, config, sess):
    self.config = config
    self.sess = sess
    self.ph = dotDict()
    self.td_lambda = config.td_lambda
    self.td_gamma = config.td_gamma
    self.max_steps = config.max_steps if config.mode == 'train' else 0
    self.max_turns = config.max_turns if config.mode == 'train' else 1

    with tf.name_scope('placeholder'):
      self.ph.is_training = tf.placeholder(tf.bool, name='is_training', shape=[]) 
      self.ph.board = tf.placeholder( 
        tf.int32, name='board', 
        shape=[None, self.max_turns, Y, X, board_vocab.size])
      self.ph.orders = tf.placeholder( 
        tf.int32, name='orders', 
        shape=[None, self.max_turns, 3, item_vocab.size])
      self.ph.rewards = tf.placeholder( 
        CALC_DTYPE, name='rewards', 
        shape=[None, self.max_turns])
      self.ph.actions = tf.placeholder( 
        tf.int32, name='actions', 
        shape=[None, self.max_turns])

      with tf.name_scope('keep_prob'):
        self.keep_prob = 1.0 - tf.to_float(self.ph.is_training) * config.dropout_rate

    batch_size = shape(self.ph.board, 0)
    num_turns = shape(self.ph.board , 1)

    with tf.name_scope('flatten'):
      flatten_board, _ = flatten_timeseries_tensor(self.ph.board)
      flatten_orders, _ = flatten_timeseries_tensor(self.ph.orders)
      flatten_actions, _ = flatten_timeseries_tensor(self.ph.actions)

    with tf.name_scope('calc_q_values'):
      flatten_q_values = self.calc_q_values(config, flatten_board, flatten_orders)
      q_values = tf.reshape(flatten_q_values, [batch_size, num_turns, -1], 
                            name='q_values')

    with tf.name_scope('q_values_of_selected_action'):
      actions = self.ph.actions
      q_values_of_selected_action =  tf.batch_gather(
        q_values, 
        tf.expand_dims(actions, -1))
      q_values_of_selected_action = tf.reshape(
        q_values_of_selected_action, 
        [batch_size, num_turns]) 

    if config.mode != 'train':
      self.q_values = q_values
      self.q_values_of_selected_action = q_values_of_selected_action
      self.loss_by_example = tf.constant(0.0)
      self.loss = tf.constant(0.0)
      return

    target_values = []
    for n in range(1, self.max_steps+1):
      with tf.name_scope('calc_target_values/%d' % n):
        # calc target values by N-step TD for each N from 1 to self.max_steps.
        _target_values = self.calc_n_step_target_values(
          q_values, 
          self.ph.actions, 
          self.ph.rewards, n,
          self.max_turns - self.max_steps)
        target_values.append(_target_values)

    with tf.name_scope('lambda_discounts'):
      lambda_discounts = [self.td_lambda ** i for i in range(self.max_steps)]
      lambda_discounts = [x/sum(lambda_discounts) for x in lambda_discounts]
      target_values = [lambda_discounts[i] * target_values[i] for i in range(self.max_steps)]
      target_values = tf.stack(target_values, axis=-1)
      target_values = tf.reduce_sum(target_values, axis=-1)


    
    self.q_values = q_values
    self.q_values_of_selected_action = q_values_of_selected_action
    self.loss_by_example = clipped_loss(
      q_values_of_selected_action[:, :self.max_turns - self.max_steps], 
      target_values)
    self.loss = tf.reduce_sum(self.loss_by_example, name='loss')

    self.debug_ops = [self.q_values, self.q_values_of_selected_action, self.loss, self.loss_by_example, self.hidden_state, self.board_emb, self.action_emb]
  def calc_n_step_target_values(self, q_values, actions, rewards, N, max_turns):
    # Q(t) = R(t) + gamma * R(t+1) + ... + gamma^{N-1} * R(t+N) + gamma^N Q(t+N)
    target_values = []
    with tf.name_scope(''):
      gamma_discounts = tf.constant([self.td_gamma ** i for i in range(N+1)])
      for i in range(max_turns):
        discounted_rewards = tf.reduce_sum(gamma_discounts[:N] * rewards[:, i:i+N], axis=-1)
        _target_values = discounted_rewards + gamma_discounts[-1] * tf.reduce_max(q_values[:, i+N, :], axis=-1)

        target_values.append(_target_values)
    target_values = tf.stack(target_values, axis=1)
    return target_values


  def inference(self, inp):
    return self.sess.run(self.q_values, {
      self.ph.is_training: False,
      self.ph.board: [[inp.board]],
      self.ph.orders: [[inp.orders]],
      self.ph.rewards: [[inp.rewards]],
    })

  def cnn_2d(self, inputs, filter_w, filter_b, strides=[1, 1, 1, 1], 
             padding='SAME', activation=tf.nn.tanh,
             scope=None):
    '''
    - inputs: [batch, in_height, in_width, in_channels]
    - filter: [filter_h, filter_w, in_channel, out_channel]
    '''
    with tf.variable_scope(scope or '2DCNN'):
      outputs = tf.nn.conv2d(inputs, filter_w, strides, padding) + filter_b
      outputs = activation(outputs)
    return tf.nn.dropout(outputs, keep_prob=self.keep_prob)

  # def max_pool(self, inputs, filter_w, strides=[1, 1, 1, 1], padding='SAME'):
  #   fh = shape(filter_w, 0)
  #   fw = shape(filter_w, 1)
  #   ksize = [1, fh, fw, 1]
  #   outputs = tf.nn.max_pool(inputs, ksize, strides, padding)
  #   return inputs

  def calc_q_values(self, config, board_ph, orders_ph):
    '''
    board_ph: [None, Y, X, board_vocab.size]
    orders_ph: [None, 3, item_vocab.size]
    '''
    batch_size = shape(board_ph, 0)
    board_ph = tf.cast(board_ph, CALC_DTYPE)
    orders_ph = tf.cast(orders_ph, CALC_DTYPE)

    with tf.variable_scope('embeddings'):
      board_emb = tf.get_variable('board', 
                                  shape=[board_vocab.size, config.emb_size], 
                                  dtype=PARAM_DTYPE)
      board_emb = tf.cast(board_emb, dtype=CALC_DTYPE)
      action_emb = tf.get_variable('action', 
                                   shape=[3, config.out_channels],
                                   dtype=PARAM_DTYPE)
      action_emb = tf.cast(action_emb, dtype=CALC_DTYPE)
      wait_emb = action_emb[0]
      move_emb = action_emb[1]
      use_emb = action_emb[2]

    with tf.variable_scope('L1_CNN'):
      filter_w = tf.get_variable('w', [config.filter_h * config.filter_w * config.emb_size, config.out_channels], dtype=PARAM_DTYPE) # save as 2D array.
      filter_w = tf.cast(filter_w, dtype=CALC_DTYPE)

      filter_w = tf.reshape(filter_w, [config.filter_h, config.filter_w, config.emb_size, config.out_channels]) 
      filter_b = tf.get_variable('b', [config.out_channels],
                                 dtype=PARAM_DTYPE)
      filter_b = tf.cast(filter_b, dtype=CALC_DTYPE)

    with tf.name_scope('encode_board'):
      emb_size = shape(board_emb, -1)
      board_repls = tf.matmul(tf.reshape(board_ph, [batch_size*Y*X, board_vocab.size]), board_emb)
      board_repls = tf.reshape(board_repls, [batch_size, Y, X, emb_size])
      
      board_repls = self.cnn_2d(board_repls, filter_w, filter_b)
      hidden_state = tf.reduce_mean(board_repls, axis=(1,2),
                                    name='hidden_state')

    with tf.name_scope('encode_order'):
      order_repls = tf.matmul(tf.reshape(orders_ph, [batch_size * 3, -1]), 
                              board_emb[:item_vocab.size])
      order_repls = tf.reshape(order_repls, [batch_size, 3, emb_size])

    dbgprint(board_repls, order_repls, hidden_state)
    exit(1)
    with tf.variable_scope('output'):
      def _expand(state):
        state = tf.expand_dims(state, axis=1)
        state = tf.expand_dims(state, axis=1)
        return state
      state_for_move = _expand(hidden_state * move_emb)
      state_for_use = _expand(hidden_state * use_emb)

      
      wait_q_value = tf.reduce_sum(hidden_state * wait_emb, axis=-1)
      wait_q_value = tf.expand_dims(wait_q_value, 1)
      move_q_values = tf.reduce_sum(state_for_move * board_repls, axis=-1)
      move_q_values = tf.reshape(move_q_values, [batch_size, Y*X])
      use_q_values = tf.reduce_sum(state_for_use * board_repls, axis=-1)
      use_q_values = tf.reshape(use_q_values, [batch_size, Y*X])
      q_values = tf.concat([wait_q_value, move_q_values, use_q_values], axis=-1) #[batch_size, 1 + 2 * x * y] 
    self.hidden_state = hidden_state
    self.board_emb = board_emb
    self.action_emb = action_emb
    return q_values #tf.nn.relu(q_values)

  def get_input_feed(self, batch, is_training=False):
    input_feed = {}
    input_feed[self.ph.is_training] = is_training
    input_feed[self.ph.board] = batch.board[:, :self.max_turns, :, :, :]
    input_feed[self.ph.orders] = batch.orders[:, :self.max_turns, :, :]
    if hasattr(batch, 'rewards'):
      input_feed[self.ph.rewards] = batch.rewards[:, :self.max_turns] * 0.001
    if hasattr(batch, 'actions'):
      input_feed[self.ph.actions] = batch.actions[:, :self.max_turns]
    return input_feed