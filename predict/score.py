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
from utils import get_consecutive_batch

# Settings
parser = argparse.ArgumentParser()
parser.add_argument("data_dir", help="Data directory", type=str)
parser.add_argument("test_data_dir", help="Test data directory", type=str)
parser.add_argument("model_type", help='Select prediction type. Classification \
                    or Regression. Options: [Classifier, Predictor]', type=str)
parser.add_argument("--hidden_dim_1", type=int, default=512)
parser.add_argument("--hidden_dim_2", type=int, default=256)
parser.add_argument("--hidden_dim_3", type=int, default=10)
parser.add_argument("--batch_size", nargs='?', type=int, default=32)
parser.add_argument('--debug', help='turn on tf debugger', action="store_true")

args = parser.parse_args()
print("Hidden dimensions: " + str(args.hidden_dim_1), str(args.hidden_dim_2), str(args.hidden_dim_3))
print("Model type: " + args.model_type)

def main():

    # Initialize session and trigger debugging mode
    session = tf.Session()
    if args.debug:
        session = tf_debug.LocalCLIDebugWrapperSession(session)

    # Load data and define placeholders
    print('Loading test data from: ' + args.test_data_dir)
    data = np.load(args.test_data_dir)
    placeholders = define_placeholders(args, data.shape)

    # Create model and optimizer
    model = define_model(args, data.shape, placeholders)
    model_name = "%s_%s_%s_%s_%s" % (args.data_dir[8:-10], args.model_type,
                        str(args.hidden_dim_1), str(args.hidden_dim_2),
                        str(args.hidden_dim_3))
    model_path = "../models/%s.ckpt" % (model_name)

    saver = tf.train.Saver()

    with session as sess:
        print('Scoring %s' % (model_name))
        saver.restore(sess, model_path)
        start, accuracy = 0, 0
        i = 0

        # Get average reconstruction loss on test set
        while start + args.batch_size <= data.shape[1]:
            batch, labels = get_consecutive_batch(start, args.batch_size, data)
            feed_dict = {placeholders['inputs']: batch}
            outs = sess.run([model.preds], feed_dict=feed_dict)

            if args.model_type == 'Classifier':
                preds = tf.nn.sigmoid(outs[0])
                correct_pred = tf.equal(tf.round(preds), labels)
                print(correct_pred.eval())
                accuracy += tf.reduce_mean(tf.cast(correct_pred, tf.float32))
                start += args.batch_size
                i += 1

            else:
                accuracy += tf.reduce_mean(tf.square(outs[0] - labels))
                start += args.batch_size
                i += 1

        print("total score:", accuracy.eval()/i)
        f = open('./scores/%s.txt' % (model_name), 'w')
        f.write("Score on %s: %f" % (args.test_data_dir, accuracy.eval()/i))
        f.close()

main()
