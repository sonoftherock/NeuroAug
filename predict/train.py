import time
import argparse

import tensorflow as tf
import numpy as np

from utils import get_random_batch

def train(model_path, data, sess, saver,
                placeholders, model, opt, args):

    for epoch in range(args.epochs):
        t = time.time()
        batch, labels = get_random_batch(args.batch_size)
        outs = sess.run([opt.opt_op, opt.cost], feed_dict={
                            placeholders['inputs']: batch,
                            placeholders['dropout']: args.dropout,
                            placeholders['labels']: labels})
        avg_cost = outs[1]

        if epoch % 100 == 0:
            print("Epoch:", '%04d' % (epoch + 1),
                    "train_loss=", "{:.5f}".format(avg_cost),
                    "time=", "{:.3f}".format(time.time() - t))

        if epoch % 1000 == 0 and epoch != 0:
            save_path = saver.save(sess, model_name)
            print('saving checkpoint at',save_path)
