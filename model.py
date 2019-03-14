# coding: utf-8
import tensorflow as tf 
from utils import dotDict, shape

from play import recDotDefaultDict, action_vocab, tile_vocab, item_vocab, board_vocab, NPCNNBased, Y, X

class TFCNNBased:
  def __init__(self, config, sess):
    self.config = config
    self.sess = sess
    self.ph = dotDict()
    with tf.name_scope('placeholder'):
      self.ph.board = tf.placeholder( 
        tf.int32, name='board', 
        shape=[None, Y, X, board_vocab.size])
      self.ph.orders = tf.placeholder( 
        tf.int32, name='orders', 
        shape=[None, 3, item_vocab.size])
      self.ph.rewards = tf.placeholder( 
        tf.float32, name='rewards', 
        shape=[None])
    self.q_value = self.build_inference_graph(config, self.ph)

  def inference(self, inp):
    return self.sess.run(self.q_value, {self.ph.board: [inp.board]})

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
    return outputs

  # def max_pool(self, inputs, filter_w, strides=[1, 1, 1, 1], padding='SAME'):
  #   fh = shape(filter_w, 0)
  #   fw = shape(filter_w, 1)
  #   ksize = [1, fh, fw, 1]
  #   outputs = tf.nn.max_pool(inputs, ksize, strides, padding)
  #   return inputs

  def build_inference_graph(self, config, inp_ph):
    batch_size = shape(inp_ph.board, 0)
    board_ph = tf.cast(inp_ph.board, tf.float16)
    orders_ph = tf.cast(inp_ph.orders, tf.float16)

    with tf.variable_scope('embeddings'):
      board_emb = tf.get_variable('board', 
                                  shape=[board_vocab.size, config.emb_size], 
                                  dtype=tf.float16)
      action_emb = tf.get_variable('action', 
                                   shape=[3, config.out_channels],
                                   dtype=tf.float16)
      wait_emb = action_emb[0]
      move_emb = action_emb[1]
      use_emb = action_emb[2]

    with tf.variable_scope('L1_CNN'):
      filter_w = tf.get_variable('w', [config.filter_h * config.filter_w * config.emb_size, config.out_channels], dtype=tf.float16) # save as 2D array.
      filter_w = tf.reshape(filter_w, [config.filter_h, config.filter_w, config.emb_size, config.out_channels]) 
      filter_b = tf.get_variable('b', [config.out_channels],
                                 dtype=tf.float16)

    with tf.name_scope('encode_board'):
      emb_size = shape(board_emb, -1)
      board_repls = tf.matmul(tf.reshape(board_ph, [batch_size*Y*X, board_vocab.size]), board_emb)
      board_repls = tf.reshape(board_repls, [batch_size, Y, X, emb_size])
      
      board_repls = self.cnn_2d(board_repls, filter_w, filter_b)
      hidden_state = tf.reduce_mean(board_repls, axis=(1,2))

    with tf.name_scope('encode_order'):
      order_repls = tf.matmul(tf.reshape(orders_ph, [batch_size * 3, -1]), 
                              board_emb[:item_vocab.size])
      order_repls = tf.reshape(order_repls, [batch_size, 3, emb_size])

    print(board_repls, order_repls, hidden_state)
    #exit(1)

    with tf.variable_scope('output'):
      def _expand(state):
        state = tf.expand_dims(state, axis=1)
        state = tf.expand_dims(state, axis=1)
        return state
      state_for_move = _expand(hidden_state * move_emb)
      state_for_use = _expand(hidden_state * use_emb)

      
      wait_reward = tf.reduce_sum(hidden_state * wait_emb, axis=-1)
      wait_reward = tf.expand_dims(wait_reward, 1)
      move_rewards = tf.reduce_sum(state_for_move * board_repls, axis=-1)
      move_rewards = tf.reshape(move_rewards, [batch_size, Y*X])
      use_rewards = tf.reduce_sum(state_for_use * board_repls, axis=-1)
      use_rewards = tf.reshape(use_rewards, [batch_size, Y*X])
      rewards = tf.concat([wait_reward, move_rewards, use_rewards], axis=-1) #[batch_size, 1 + 2 * x * y] 
    outputs = rewards
    return outputs

  def get_input_feed(self, batch):
    input_feed = {}
    pass
