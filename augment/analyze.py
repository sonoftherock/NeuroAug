import time
import os
import argparse

# Using Tesla K80 on Yale CS cluster (tangra)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug

from define import define_placeholders, define_model
from train import train_VAE, train_VGAE
from utils import visualize_triangular, visualize_matrix, \
                            visualize_latent_space_VAE, get_random_batch_VAE, \
                            get_consecutive_batch

# Settings
parser = argparse.ArgumentParser()
parser.add_argument("data_dir", nargs='?', help="data directory", type=str, default="../data/BSNIP_left_full/original.npy")
parser.add_argument("model_type", help='select augmentor model type. \
                        Options: [VAE, VGAE, BrainNetCNN_VAE]', type=str)
parser.add_argument("--hidden_dim_1", type=int, default=512)
parser.add_argument("--hidden_dim_2", type=int, default=256)
parser.add_argument("--hidden_dim_3", type=int, default=10)
parser.add_argument("--batch_size", nargs='?', type=int, default=32)
parser.add_argument('--debug', help='turn on tf debugger', action="store_true")

args = parser.parse_args()
print("Hidden dimensions: " + str(args.hidden_dim_1), str(args.hidden_dim_2), str(args.hidden_dim_3))
print("Augmentor model type: " + args.model_type)

def analyze_VAE(args, data, model):
    batch = get_random_batch_VAE(args.batch_size, data)
    feed_dict={placeholders['inputs']: batch}
    [rc] = sess.run([model.reconstructions], feed_dict=feed_dict)
    [z] = sess.run([model.z], feed_dict = feed_dict)

    # Visualize sample full matrix of original and reconstructed batches
    for i in range(batch.shape[0]):
        visualize_triangular(batch, i, model_name, 'original_' + str(i))
        visualize_triangular(rc, i, model_name, 'reconstruction_' + str(i))

    # Visualize Latent Space. Label format =[control_bool, subject_bool]
    onehot = np.array([0 if label[0] == 1 else 1 for label in batch[-2:]])
    visualize_latent_space_VAE(z, onehot, model_name)

def analyze():

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
    model_name = "%s_%s_%s_%s_%s" % (args.data_dir[11: -10], args.model_type, str(args.hidden_dim_1),
                    str(args.hidden_dim_2), str(args.hidden_dim_3))
    model_path = "../models/%s.ckpt" % (model_name)

    saver = tf.train.Saver()

    with session as sess:
        saver.restore(sess, model_path)

        start_time = time.ctime(int(time.time()))
        print("Analyzing '%s'... \nStart Time: %s" % (model_name, str(start_time)))

        if args.model_type == 'VAE':
            analyze_VAE(args, data, model, model_name)

        elif args.model_type == 'VGAE':
            train_VGAE(model_name, data, session, saver, placeholders,
                        model, opt, args)

analyze()
