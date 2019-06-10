import time
import os
import argparse

# Using Tesla K80 on Yale CS cluster (tangra)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug

from define import define_placeholders, define_model
from utils import visualize_triangular, visualize_matrix, \
                            visualize_latent_space_VAE, get_random_batch_VAE, \
                            get_consecutive_batch_VAE

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

def analyze_VAE(args, placeholders, data, model, model_name, sess):
    batch = get_random_batch_VAE(args.batch_size, data)
    feed_dict={placeholders['inputs']: batch}
    [rc] = sess.run([model.reconstructions], feed_dict=feed_dict)
    [z] = sess.run([model.z], feed_dict = feed_dict)

    # Visualize sample full matrix of original and reconstructed batches
    for i in range(batch.shape[0]):
        visualize_triangular(batch, i, model_name, 'original_' + str(i))
        visualize_triangular(rc, i, model_name, 'reconstruction_' + str(i))

    # Visualize Latent Space. Label format =[control_bool, subject_bool]
    onehot = np.array([0 if label[0] == 1 else 1 for label in batch[:,-2:]])
    visualize_latent_space_VAE(z, onehot, model_name)

def score_VAE(args, placeholders, data, model, model_name, sess):
    start, tot_rc_loss = 0, 0
    og_all, gen_all = [], []

    # Get average reconstruction loss on training set
    while start + 32 < data.shape[1]:
        batch = get_consecutive_batch_VAE(start, args.batch_size, data)
        feed_dict={placeholders['inputs']: batch}
        [rc] = sess.run([model.reconstructions], feed_dict=feed_dict)

        tot_rc_loss += tf.reduce_mean(tf.square(batch - rc))
        start += args.batch_size
    avg_rc_loss = tot_rc_loss / math.floor(data.shape[1]/args.batch_size)
    f = open('./analysis/%s/score.txt' % (model_name), 'w')

    print('Writing scores at ./analysis/%s/score.txt' % (model_name))
    f.write("average reconstruction loss: %f\n" % avg_rc_loss.eval())

    for i in range(10):
        batch, idx = get_random_batch(args.batch_size)
        og_all.append(batch)

    og = np.array(og_all).reshape(10, 32, -1)
    og = og.reshape(-1, input_dim)
    og = og[:, :-2]
    og_mean = np.mean(og, axis=0)
    og_var = np.var(og, axis=0)

    # TODO: Get pearson coefficients of first and second moments (Only for variational models) - make sure latent space is N(0,0.1)?
    for i in range(10):
        randoms = np.random.normal(0, 1, (args.batch_size, args.hidden_dim_3))
        [gen] = sess.run([model.reconstructions], feed_dict={model.z: randoms})
        gen = gen.reshape(args.batch_size, -1)
        gen_all.append(gen)

    gen = np.array(gen_all).reshape(10, 32, -1)
    gen = gen.reshape(-1, input_dim)
    gen = gen[:,:-2]

    # Check one sample gen
    visualize_triangular(gen, 0, model_name, 'generated')

    gen_mean = np.mean(gen, axis=0)
    gen_var = np.var(gen, axis=0)

    plt.scatter(og_mean, gen_mean)
    plt.title('Feature Mean')
    plt.xlabel('original')
    plt.ylabel('generated')
    plt.savefig('./analysis/%s/mean.png'%(model_name))
    plt.clf()

    plt.scatter(og_var, gen_var)
    plt.title('Feature Variance')
    plt.xlabel('original')
    plt.ylabel('generated')
    plt.savefig('./analysis/%s/variance.png'%(model_name))
    plt.clf()

    f.write("Feature mean (180 * 180 features): %s\n" %(str(sp.pearsonr(og_mean, gen_mean))))
    f.write("Feature Variance (180*180 features): %s\n" %(str(sp.pearsonr(og_var, gen_var))))
    f.close()

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
            analyze_VAE(args, placeholders, data, model, model_name, sess)

        elif args.model_type == 'VGAE':
            train_VGAE(model_name, data, session, saver, placeholders,
                        model, opt, args)

analyze()
