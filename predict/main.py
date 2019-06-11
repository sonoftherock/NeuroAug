import time
import os
import argparse

# Using Tesla K80 on Yale CS cluster (tangra)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug

from define import define_placeholders, define_model, define_optimizer
from train import train

# Settings
parser = argparse.ArgumentParser()
parser.add_argument("data_dir", help="Data directory", type=str)
parser.add_argument("model_type", help='Select prediction type. Classification \
                    or Regression. Options: [Classifier, Predictor]', type=str)
parser.add_argument("--hidden_dim_1", type=int, default=512)
parser.add_argument("--hidden_dim_2", type=int, default=256)
parser.add_argument("--hidden_dim_3", type=int, default=10)
parser.add_argument("--batch_size", nargs='?', type=int, default=32)
parser.add_argument("--epochs", nargs='?', type=int, default=1000000)
parser.add_argument("--learning_rate", nargs='?', type=float, default=0.0001)
parser.add_argument("--dropout", nargs='?', type=float, default=0.)
parser.add_argument('--debug', help='turn on tf debugger', action="store_true")
parser.add_argument('--restore', help='restore or train new model', action="store_true")

args = parser.parse_args()
print("Learning Rate: " + str(args.learning_rate))
print("Hidden dimensions: " + str(args.hidden_dim_1), str(args.hidden_dim_2), str(args.hidden_dim_3))
print("Model type: " + args.model_type)

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
    model_name = "%s_%s_%s_%s_%s" % (args.data_dir[8:-10], args.model_type,
                        str(args.hidden_dim_1), str(args.hidden_dim_2),
                        str(args.hidden_dim_3))
    model_path = "../models/%s.ckpt" % (model_name)

    saver = tf.train.Saver()

    with session as sess:
        sess.run(tf.global_variables_initializer())

        if args.restore:
            print("Restoring model from: ", model_path)
            saver.restore(sess, model_path)

        start_time = time.ctime(int(time.time()))
        print("Starting to train '%s'... \nStart Time: %s" \
                % (model_name, str(start_time)))

        train(model_path, data, session, saver, placeholders,
                    model, opt, args)

    print("Training Complete. Model name: %s" %(model_name))

main()
