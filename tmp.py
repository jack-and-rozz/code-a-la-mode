# coding: utf-8
import numpy as np
import tensorflow as tf
import json
import time, sys
from pprint import pprint

from play import state2tensor
from utils import flatten_recdict

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


def parse_views(i, view_string):
  jsonstr = view_string.split('\n')[1]
  view = json.loads(jsonstr)
  print('----- %d ----' % i)
  if not 'entitymodule' in view:
    return
  view = view['entitymodule']
  if 'T \'' in view:
    print(view)

if __name__ == "__main__":
  import argparse
  desc = ""
  parser = argparse.ArgumentParser(description=desc)
  # parser.add_argument('mode', type=str, help ='')
  parser.add_argument('log_file', type=str)
  args = parser.parse_args()
  path = args.log_file #= 'checkpoints/tmp/games/000/NGzZCMiiSe.detail.json'


  d = json.load(open(path))
  #logs = flatten_recdict(d)
  #print(logs.keys())

  #views = d['views']
  #setup_turns = set([0, 1, 202, 203, 404, 405])
  #views = [parse_views(i, view_str) for i, view_str in enumerate(views) if i not in setup_turns]
  #print (views)

  logs = [json.loads(x) for x in d['errors']['0'] if x]
  
  for i in range(len(logs)):
    print('-----------------')
    for k, v in logs[i].items():
      print(k, v)
      #print(v)
  exit(1)
  for k in d:
    if isinstance(d[k], dict):
      print(k, d[k].keys())
    else:
      print(k, type(d[k]))
  #print(d[''])
  logs = d['errors']['0']
  #print(len(logs))
  #logs = [x for x in d['errors']['0'] if x and x['turn'] == 100]
  logs = [json.loads(x) for x in d['errors']['0'] if x]
  #logs = [x for x in logs if x['turn'] == 100]
  logs = [x['turn'] for x in logs]
  print(logs)
  #print(len(logs))
  exit(1)
  #logs = [json.loads(x) for x in d['errors'][str(i)] if x is not None]
  for i in range(3):
    print(i)
    try:
      logs = [json.loads(x) for x in d['errors'][str(i)] if x is not None]
      print(len(logs))
      pprint(logs[-1])
      pprint(state2tensor(logs[-1]))
    except:
      pass
exit(1)

y, x, v = 7, 11, 23
emb_size = in_channels = 20
filter_h, filter_w, out_channels = 3, 3, 10

inp = np.random.randint(2, size=y*x*v)
inp = np.reshape(inp, [y, x, v])
np.random.seed(0)
var = 0.5
embeddings = np.random.rand(v, emb_size) * var - var/2
filter_w = np.random.rand(filter_h, filter_w, in_channels, out_channels) * var - var/2
filter_b = np.random.rand(out_channels) * var - var/2
inp = inp.astype(np.float16)

#####################
#    in np
#####################
def relu(x):
    y = np.where( x > 0, 1, 0)
    return y



def max_pool(inputs, filter_size, strides, padding):
  pass

  return outputs
np_result = np_encode(inp, embeddings, filter_w, filter_b)

#exit(1)
#####################
#    in tf
#####################

def shape(x, dim):
  with tf.name_scope('shape'):
    return x.get_shape()[dim].value or tf.shape(x)[dim]

def cnn_2d(inputs, filter_w,filter_b, strides=[1, 1, 1, 1], 
           padding = 'SAME', activation=tf.nn.tanh,
           scope=None):
  '''
  - inputs: [batch, in_height, in_width, in_channels]
  - filter: [filter_h, filter_w, in_channel, out_channel]
  '''
  with tf.variable_scope(scope or '2DCNN'):
    outputs = tf.nn.conv2d(inputs, filter_w, strides, padding) + filter_b
    outputs = activation(outputs)
    fh = shape(filter_w, 0)
    fw = shape(filter_w, 1)
    ksize = [1, fh, fw, 1]
    #outputs = tf.nn.max_pool(outputs, ksize, strides, padding)
  return outputs

def tf_encode(inp, embeddings, filter_w, filter_b, dtype=tf.float16):
  inp = tf.constant(inp, dtype=dtype)
  embeddings = tf.constant(embeddings, dtype=dtype)
  #embeddings = tf.get_variable('embeddings', shape=[v, emb_size], dtype=tf.float16)
  #embeddings = tf.cast(embeddings, tf.float32)
  #print(embeddings)
  #print(tf.matmul(inp, embeddings))

  emb_inp = tf.reshape(tf.matmul(tf.reshape(inp, shape=[y*x, v]), embeddings), 
                       shape=[y, x, emb_size])
  emb_inp = tf.expand_dims(emb_inp, 0)

  filter_w = tf.constant(filter_w, dtype=dtype)
  outputs = cnn_2d(emb_inp, filter_w, filter_b)
  return outputs[0]

with tf.Session() as sess:    
  tf_result = tf_encode(inp, embeddings, filter_w, filter_b).eval()
print(np_result[0][0])
print(tf_result[0][0])
