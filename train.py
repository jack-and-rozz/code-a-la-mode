# coding: utf-8
import sys, os
import numpy as np
import tensorflow as tf
from trainer import Trainer
from play import recDotDefaultDict, action_vocab, tile_vocab, item_vocab, NPCNNBased
from utils import generate_random_inp
from model import TFCNNBased

def main(args):
  inp = generate_random_inp()
  with tf.Session() as sess:
    tf_model = TFCNNBased(args, sess)
    trainer = Trainer(args, sess, tf_model)
    if args.mode == 'train':
      trainer.train()
    elif args.mode =='debug':
      trainer.output_parameters()
      np_model = NPCNNBased(args)
      inp = generate_random_inp()
      a = tf_model.inference(inp)
      b = np_model.inference(inp)
      print(a)
      print(b)
      #print(a.dtype)
      #print(b.dtype)

    # np_model = NPCNNBased(args)
    # #res = tf_model.inference([inp])
    # # print(res[0][0][0])
    # res = np_model.inference(inp)
    # print(res)
    # #print(res[0][0])
    # print(tf_model.inference([inp])[0])

def get_parser():
  desc = ""
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('model_root_path', default=None, type=str, help ='')
  parser.add_argument('-m', '--mode', default='train')

  # Training parameter
  parser.add_argument('-gpe', '--games_per_epoch', default=30, type=int)        
  parser.add_argument('--batch_size', default=5, type=int)
  parser.add_argument('--max_epoch', default=50, type=int)
  parser.add_argument('--max_to_keep', default=1, type=int)
  parser.add_argument('--optimizer_type', default='AdamOptimizer')
  parser.add_argument('--max_gradient_norm', default=1.0, type=float)
  parser.add_argument('--decay_rate_per_epoch', default=0.75, type=float)
  
  parser.add_argument('--learning_rate', default=0.0001, type=float)
  parser.add_argument('--dropout_rate', default=0.2, type=float)

  # NN parameters
  parser.add_argument('--filter_h', default=3, type=int)
  parser.add_argument('--filter_w', default=3, type=int)
  parser.add_argument('--emb_size', default=20, type=int)
  parser.add_argument('--out_channels', default=20, type=int)

  # RL parameters
  parser.add_argument('-mt', '--max_turns', default=100, type=int)
  parser.add_argument('-ms', '--max_steps', default=10, type=int)
  parser.add_argument('--td_gamma', default=0.99, type=float)
  parser.add_argument('--td_lambda', default=0.95, type=float)
  
  return parser

if __name__ == "__main__":
  import argparse
  parser = get_parser()
  args = parser.parse_args()
  main(args)
