# coding: utf-8
import collections
from itertools import chain
import numpy as np
import tensorflow as tf
from play import recDotDefaultDict, action_vocab, tile_vocab, item_vocab, board_vocab, NPCNNBased, Y, X
from logging import getLogger, StreamHandler, FileHandler, Formatter, DEBUG, INFO, WARNING, ERROR, CRITICAL

def make_summary(value_dict):
  return tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v) for k,v in list(value_dict.items())])

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
  inp.timer = timer
  return inp


def flatten(l):
  return list(chain.from_iterable(l))

def shape(x, dim):
  with tf.name_scope('shape'):
    return x.get_shape()[dim].value or tf.shape(x)[dim]

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
