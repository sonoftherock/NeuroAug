import time
import os
import argparse

# Using Tesla K80 on Yale CS cluster (tangra)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug

from optimizer import OptimizerVGAE, OptimizerVAE
from model import VGAE, VAE
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

def define_placeholders(args, input_dim):

    if args.model_type == 'VAE':
        placeholders = {
            'inputs': tf.placeholder(tf.float32, [args.batch_size, input_dim]),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'lambd': tf.placeholder(tf.float32, []),
        }

    elif args.model_type == 'VGAE':
        placeholders = {
            'features': tf.placeholder(tf.float32, [args.batch_size, num_nodes,
                                        num_features]),
            'adj_norm': tf.placeholder(tf.float32, [args.batch_size, num_nodes,
                                        num_nodes]),
            'adj_orig': tf.placeholder(tf.float32, [args.batch_size, num_nodes,
                                        num_nodes]),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'lambd': tf.placeholder(tf.float32, [])
        }
    else:
        placeholders = {}

    return placeholders

def define_model_and_optimizer(args, input_dim, placeholders):

    if args.model_type == 'VAE':
        model = VAE(placeholders, input_dim, args)

        with tf.name_scope('optimizer'):
            opt = OptimizerVAE(reconstructions=tf.reshape(model.reconstructions, [-1]),
                               inputs=tf.reshape(placeholders['inputs'], [-1]),
                               model=model, learning_rate=args.learning_rate,
                               lambd=placeholders['lambd'], tolerance=0.03)

    elif args.model_type == 'VGAE':
        model = VGAE(placeholders, num_features, num_nodes, args)

        with tf.name_scope('optimizer'):
            opt = OptimizerVGAE(preds=model.reconstructions,
                                labels=tf.reshape(placeholders['adj_orig'], [-1]),
                                model=model, num_nodes=num_nodes,
                                learning_rate=args.learning_rate,
                                lambd=placeholders['lambd'], tolerance=0.1)

    else:
        model, opt = None, None

    return model, opt

def main():

    # Trigger debugging mode
    if args.debug:
        session = tf_debug.LocalCLIDebugWrapperSession(session)

    # Load data and define placeholders
    print('Loading data from: ' + args.data_dir)
    data = np.load(args.data_dir)
    input_dim = data.shape[0]
    placeholders = define_placeholders(args, input_dim)

    # Create model and optimizer
    model, opt = define_model_and_optimizer(args, input_dim, placeholders)
    model_name = "%s_%s_%s_%s" % (args.model_type, str(args.hidden_dim_1),
                    str(args.hidden_dim_2), str(args.hidden_dim_3))
    model_path = "./models/%s.ckpt" % (model_name)

    # Initialize session and model saver
    session = tf.Session()
    saver = tf.train.Saver()

    with session as sess:
        sess.run(tf.global_variables_initializer())

        if args.restore:
            print("Restoring model from: ", model_path)
            saver.restore(sess, model_path)

        start_time = time.ctime(int(time.time()))
        print("Training %s... Current Time: %s" % (model_name, str(start_time)))

        if args.model_type == 'VAE':
            train_VAE(model_name, data, session, saver, placeholders,
                        model, optimizer)

        elif args.model_type == 'VGAE':
            train_VGAE(model_name, data, session, saver, placeholders,
                        model, optimizer)

        else:
            print('beep beep model unspecified')

main()
