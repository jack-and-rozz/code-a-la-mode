# coding: utf-8
import sys, os
import numpy as np

def compress_tensor_str(t):
  digits = "%.5f"
  is_mat = len(t.shape) == 2
  t = t.tolist()
  def comp_vec(l):
    return '[' + ','.join([digits % x for x in l]) + ']'
  if is_mat:
    return '['+','.join([comp_vec(l) for l in t])+']'
  else:
    return comp_vec(t)

def main(args):
  parameter_dir = args.model_root_path + '/parameters'
  parameter_dir = parameter_dir + '/%03d' % args.epoch if args.epoch else parameter_dir + '/' + os.listdir(parameter_dir)[-1] 
  print('PARAMS = dotDict()')
  for f in os.listdir(parameter_dir):
    p = np.loadtxt(parameter_dir + '/' + f)
    p = compress_tensor_str(p)
    print('PARAMS.%s =' % f, 'np.array(', p , ', dtype=np.float16)')

def get_parser():
  desc = ""
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('model_root_path', type=str, help ='')
  parser.add_argument('-e','--epoch', default=None, type=int)
  return parser


if __name__ == "__main__":
  import argparse
  parser = get_parser()
  args = parser.parse_args()
  main(args)
