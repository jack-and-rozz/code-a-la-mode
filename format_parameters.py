# coding: utf-8
import sys, os
import numpy as np

def main(args):
  parameter_dir = args.model_root_path + '/parameters'
  parameter_dir = parameter_dir + '/%03d' % args.epoch if args.epoch else parameter_dir + '/' + os.listdir(parameter_dir)[-1] 
  print('PARAMS = recDotDefaultDict()')
  for f in os.listdir(parameter_dir):
    p = np.loadtxt(parameter_dir + '/' + f).tolist()
    
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
