import math
import time
import argparse
import logging

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
import matplotlib.colors as clr
plt.switch_backend('agg')
from define import define_placeholders, define_model
from utils import visualize_triangular, visualize_matrix, \
                            visualize_latent_space_VAE, get_random_batch_VAE, \
                            get_consecutive_batch_VAE

# Settings
parser = argparse.ArgumentParser()
parser.add_argument("data_dir", help="data directory", type=str)
parser.add_argument("model_type", help='select augmentor model type. \
                        Options: [VAE, VGAE, BrainNetCNN_VAE]', type=str)
parser.add_argument("predict_type", help='select augmentor model type. \
                        Options: [Classifier]', type=str)
parser.add_argument("tol", help="tolerance for GECO procedure", type=float)
parser.add_argument("num_batches", help='number of batches to augment by', type=int)
parser.add_argument("--hidden_dim_1", type=int, default=512)
parser.add_argument("--hidden_dim_2", type=int, default=256)
parser.add_argument("--hidden_dim_3", type=int, default=10)
parser.add_argument("--batch_size", nargs='?', type=int, default=32)
parser.add_argument('--debug', help='turn on tf debugger', action="store_true")

args = parser.parse_args()

# Load data
print('Loading data from ' + args.data_dir)
train_data = np.load(args.data_dir)
input_dim = train_data.shape[0]

def augment():

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
    model_name = "%s_%s_%s_%s_%s_tol=%s" % (args.data_dir[8: -10],
                    args.model_type, str(args.hidden_dim_1),
                    str(args.hidden_dim_2), str(args.hidden_dim_3), str(args.tol))
    model_path = "../models/%s.ckpt" % (model_name)

    saver = tf.train.Saver()

    with session as sess:
        saver.restore(sess, model_path)

        start_time = time.ctime(int(time.time()))
        print("Augmenting '%s'... \nStart Time: %s" % (model_name, str(start_time)))

        gen_all = []

        for i in range(args.num_batches):
            randoms = np.random.normal(0.0, 1.0, (args.batch_size,
                                                    args.hidden_dim_3))
            [gen, gen_preds] = sess.run([model.reconstructions, model.preds], feed_dict={model.z: randoms})
            gen = np.concatenate((gen, gen_preds), axis=1)
            gen = gen.reshape(args.batch_size, -1)
            gen_all.append(gen)

        gen = np.array(gen_all).reshape(args.num_batches, args.batch_size, -1)
        gen = gen.reshape(-1, input_dim)

        if args.predict_type == 'Classifier':
            gen[:, 16110:] = np.clip(tf.round(gen[:, 16110:]).eval(), 0, 1)

        visualize_triangular(gen[:,:16110], 0, model_name, 'generated')
        print(gen[:, 16110])
        augmented_data = np.concatenate((np.transpose(gen), train_data), axis=1)
        np.save('../data/%s_augmented_train.npy'% model_name, augmented_data)

augment()
