# coding: utf-8
import collections
from itertools import chain
from inspect import currentframe
import numpy as np
import tensorflow as tf
from play import action_vocab, tile_vocab, item_vocab, board_vocab, NPCNNBased, Y, X
from logging import getLogger, StreamHandler, FileHandler, Formatter, DEBUG, INFO, WARNING, ERROR, CRITICAL

def flatten(l):
  return list(chain.from_iterable(l))

def dbgprint(*args):
  names = {id(v):k for k,v in currentframe().f_back.f_locals.items()}
  print(', '.join(names.get(id(arg),'???')+' = '+repr(arg) for arg in args))

def generate_random_inp():
  inp = dotDict()
  board = np.random.randint(2, size=Y*X*board_vocab.size)
  board = np.reshape(board, [Y, X, board_vocab.size])
  board = board.astype(np.float16)
  
  orders = np.random.randint(2, size=3 * item_vocab.size)
  orders = np.reshape(orders, [3, -1])
  timer = np.zeros([11])
  timer[np.random.randint(11)] = 1
  inp.board = board
  inp.orders = orders
  inp.order_awards = np.random.randint(2000, size=3).astype(np.float16)
  inp.timer = timer
  inp.rewards = np.random.randint(3000)
  return inp



class dotDict(dict):
  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__

  def __getattr__(self, key):
    if key in self:
      return self[key]
    raise AttributeError("\'%s\' is not in %s" % (str(key), str(self.keys())))

class recDotDict(dict):
  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__
  def __init__(self, _dict={}):
    for k in _dict:
      if isinstance(_dict[k], dict):
        _dict[k] = recDotDict(_dict[k])
      if isinstance(_dict[k], list):
        for i,x in enumerate(_dict[k]):
          if isinstance(x, dict):
            _dict[k][i] = dotDict(x)
    super(recDotDict, self).__init__(_dict)

  def __getattr__(self, key):
    if key in self:
      return self[key]
    # else:
    #   return None
    raise AttributeError("\'%s\' is not in %s" % (str(key), str(self.keys())))

class recDotDefaultDict(collections.defaultdict):
  __getattr__ = collections.defaultdict.__getitem__
  __setattr__ = collections.defaultdict.__setitem__
  __delattr__ = collections.defaultdict.__delitem__
  def __init__(self, _=None):
    super(recDotDefaultDict, self).__init__(recDotDefaultDict)

def flatten_recdict(d):
  res = dotDict()
  for k in d:
    if isinstance(d[k], dict):
      subtrees = flatten_recdict(d[k])
      for kk, v in subtrees.items():
        res['%s.%s' % (k, kk)] = v
    else:
      res[k] = d[k]
  return res


def logManager(logger_name='main', 
              handler=StreamHandler(),
              log_format = "[%(levelname)s] %(asctime)s - %(message)s",
              level=DEBUG):
    formatter = Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
    logger = getLogger(logger_name)
    handler.setFormatter(formatter)
    handler.setLevel(level)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger




def shape(x, dim):
  with tf.name_scope('shape'):
    return x.get_shape()[dim].value or tf.shape(x)[dim]

def ffnn(inputs, output_size=None, initializer=None,
         activation=tf.nn.tanh, scope=None):
  """
  Args:
    inputs : Rank 2 or 3 Tensor of shape [batch_size, (sequence_size,) hidden_size].
    output_size : An integer.
  """
  if output_size is None:
    output_size = shape(inputs, -1)
  with tf.variable_scope(scope or "ffnn"):
    inputs_rank = len(inputs.get_shape().as_list())
    hidden_size = shape(inputs, -1)
    initializer = tf.initializers.truncated_normal(stddev=0.01)
    w = tf.get_variable('w', [hidden_size, output_size],
                        initializer=initializer)
    b = tf.get_variable('b', [output_size],
                        initializer=initializer)
    if inputs_rank == 3:
      batch_size = shape(inputs, 0)
      max_sequence_length = shape(inputs, 1)
      inputs = tf.reshape(inputs, [batch_size * max_sequence_length, hidden_size])
      outputs = activation(tf.nn.xw_plus_b(inputs, w, b))
      outputs = tf.reshape(outputs, [batch_size, max_sequence_length, output_size])
    elif inputs_rank == 2:
      outputs = activation(tf.nn.xw_plus_b(inputs, w, b))
    else:
      ValueError("linear with rank {} not supported".format(inputs_rank))

    return outputs

###########################################
##         Tensorflow Utils
###########################################

def batch_gather(emb, indices):
  '''
  e.g. arr = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]  # arr.shape = [2,2,2]
       indices = [[0], [1]] 
       res = [[[0, 1], [6, 7]]]

       indices = [[0, 0], [0, 1]]
       res = [[[0, 1], [0, 1]], [[4, 5], [6, 7]]]
  '''
  # When the rank of emb is N, indices of rank N-1 tensor is regarded as batch respective specifications.
  if len(indices.get_shape()) == 1:
    indices = tf.expand_dims(indices, 1)
  batch_size = shape(emb, 0)
  seqlen = shape(emb, 1)
  if len(emb.get_shape()) > 2:
    emb_size = shape(emb, 2)
  else:
    emb_size = 1
  flattened_emb = tf.reshape(emb, [batch_size * seqlen, emb_size])  # [batch_size * seqlen, emb]

  offset = tf.expand_dims(tf.range(batch_size) * seqlen, 1)  # [batch_size, 1]

  gathered = tf.gather(flattened_emb, indices + offset) # [batch_size, num_indices, emb]

  return gathered


def flatten_timeseries_tensor(t):
  # batch_size = shape(t, 0)
  # max_timestep = shape(t, 1)
  # other_shapes = [shape(t, i+2) for i in range(len(t.get_shape()) - 2)]
  original_shape = [shape(t, i) for i in range(len(t.get_shape()))]
  batch_size = original_shape[0]
  max_timestep = original_shape[1]
  return tf.reshape(t, [batch_size*max_timestep] + original_shape[2:]), original_shape

def make_summary(value_dict):
  return tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v) for k,v in list(value_dict.items())])

