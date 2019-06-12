import time
import os
import argparse

# Using Tesla K80 on Yale CS cluster (tangra)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug

from define import define_placeholders, define_model, define_optimizer
from train import train_VAE, train_VGAE

# Settings
parser = argparse.ArgumentParser()
parser.add_argument("data_dir", help="Data directory", type=str)
parser.add_argument("model_type", help='select augmentor model type. \
                        Options: [VAE, VGAE, BrainNetCNN_VAE]', type=str)
parser.add_argument('tol', help='tolerance value for GECO training', type=float)
parser.add_argument("--hidden_dim_1", type=int, default=512)
parser.add_argument("--hidden_dim_2", type=int, default=256)
parser.add_argument("--hidden_dim_3", type=int, default=10)
parser.add_argument("--batch_size", nargs='?', type=int, default=32)
parser.add_argument("--epochs", nargs='?', type=int, default=1000000)
parser.add_argument("--learning_rate", nargs='?', type=float, default=0.0001)
parser.add_argument("--dropout", nargs='?', type=float, default=0.)
parser.add_argument('--debug', help='turn on tf debugger', action="store_true")
parser.add_argument('--restore', help='restore or train new model', action="store_true")
parser.add_argument('--lambd', help='lagrange multiplier on constraint (MSE)', default=1.0)
parser.add_argument('--alpha', help='slowness of the constraint ma', default=0.99)

args = parser.parse_args()
print("Learning Rate: " + str(args.learning_rate))
print("Hidden dimensions: " + str(args.hidden_dim_1), str(args.hidden_dim_2), str(args.hidden_dim_3))
print("Augmentor model type: " + args.model_type)
print("Taming VAE: Lambda=%f, Alpha=%f, Tolerane=%f" % (args.lambd, args.alpha, args.tol))

def main():

    # Initialize session and trigger debugging mode
    session = tf.Session()
    if args.debug:
        session = tf_debug.LocalCLIDebugWrapperSession(session)

    # Load data and define placeholders
    print('Loading data from: ' + args.data_dir)
    data = np.load(args.data_dir)
    placeholders = define_placeholders(args, data.shape)

    # Create model and optimizer
    model = define_model(args, data.shape, placeholders)
    opt = define_optimizer(args, model, data.shape, placeholders)
    model_name = "%s_%s_%s_%s_%s_tol=%s" % (args.data_dir[8:-10], args.model_type, str(args.hidden_dim_1),
                    str(args.hidden_dim_2), str(args.hidden_dim_3), str(args.tol))
    model_path = "../models/%s.ckpt" % (model_name)

    saver = tf.train.Saver()

    with session as sess:
        sess.run(tf.global_variables_initializer())

        if args.restore:
            print("Restoring model from: ", model_path)
            saver.restore(sess, model_path)

        start_time = time.ctime(int(time.time()))
        print("Starting to train '%s'... \nStart Time: %s" % (model_name, str(start_time)))

        if args.model_type == 'VAE':
            train_VAE(model_path, data, session, saver, placeholders,
                        model, opt, args)

        elif args.model_type == 'VGAE':
            train_VGAE(model_path, data, session, saver, placeholders,
                        model, opt, args)

    print("Training Complete. Model name: %s" %(model_name))

main()
