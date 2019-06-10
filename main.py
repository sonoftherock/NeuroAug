import time
import os
import argparse

# Using Tesla K80 on Yale CS cluster (tangra)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug

from train import train_VAE, train_VGAE

# Settings
parser = argparse.ArgumentParser()
parser.add_argument("data_dir", nargs='?', help="data directory", type=str, default="./data/BSNIP_left_full/original.npy")
parser.add_argument("learning_rate", nargs='?', type=float, default=0.0001)
parser.add_argument("epochs", nargs='?', type=int, default=1000000)
parser.add_argument("batch_size", nargs='?', type=int, default=32)
parser.add_argument("model_type", help='select augmentor model type. \
                    Options: [VAE, VGAE, BrainNetCNN_VAE]', type=str)
parser.add_argument("--hidden_dim_1", type=int, default=100)
parser.add_argument("--hidden_dim_2", type=int, default=50)
parser.add_argument("--hidden_dim_3", type=int, default=5)
parser.add_argument("dropout", nargs='?', type=float, default=0.)
parser.add_argument('--debug', help='turn on tf debugger', action="store_true")
parser.add_argument('--restore', help='restore or train new model', action="store_true")
parser.add_argument('--lambd', help='lagrange multiplier on constraint (MSE)', default=1.0)
parser.add_argument('--alpha', help='slowness of the constraint ma', default=0.99)

args = parser.parse_args()
print("Learning Rate: " + str(args.learning_rate))
print("Hidden dimensions: " + str(args.hidden_dim_1), str(args.hidden_dim_2), str(args.hidden_dim_3))
print("Augmentor model type: " + args.model_type)

def main():

    if args.model_type == 'VAE':
        train_VAE(model_name, data, args)

    elif args.model_type == 'VGAE':
        train_VGAE(model_name, data, args)

    else:
        print('beep beep model unspecified')

    print("Training Complete. Model name: %s" %(model_name))

main()
