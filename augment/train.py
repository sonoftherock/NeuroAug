import time
import argparse

import tensorflow as tf
import numpy as np

from utils import normalize_adj, construct_feed_dict_VGAE, \
                    get_random_batch_VGAE, get_random_batch_VAE

def train_VGAE(model_path, data, sess, saver,
                    placeholders, model, opt, args):

    # Normalize adjacency matrix (i.e. D^(.5)AD^(.5))
    adj = data
    adj_norm = normalize_adj(adj)

    # CHANGE TO features.shape[1] LATER
    num_nodes = adj.shape[1]
    num_features = adj.shape[1]

    # Use identity matrix for feature-less training
    features_batch = np.zeros([args.batch_size, num_nodes, num_features])
    for i in features_batch:
        np.fill_diagonal(i, 1)

    for epoch in range(args.epochs):
        t = time.time()
        random_batch = get_random_batch_VGAE(args.batch_size, adj, adj_norm)
        adj_norm_batch, adj_orig_batch, adj_idx = random_batch
        feed_dict = construct_feed_dict_VGAE(adj_norm_batch, adj_orig_batch,
                                    features_batch, args.dropout, placeholders)

        if epoch == 0:
            lambd = args.lambd
            feed_dict.update({placeholders['lambd']: lambd})
            [initial] = sess.run([opt.constraint], feed_dict=feed_dict)
            constraint_ma = initial
        else:
            feed_dict.update({placeholders['lambd']: lambd})
            outs = sess.run([opt.opt_op, opt.cost, opt.rc_loss, opt.kl,
                                opt.constraint], feed_dict=feed_dict)
            constraint = outs[4]
            constraint_ma = args.alpha * constraint_ma + (1 - args.alpha) * constraint
            lambd = np.clip(lambd, 0, 1e15)
            lambd *= np.clip(np.exp(constraint_ma), 0.9, 1.1)

            if epoch % 100 == 0:
                _, cost, rc_loss, kl_loss, constraint = outs
                print("Epoch:", '%04d' % (epoch + 1), "train_loss=",
                        "{:.5f}".format(cost), "kl_loss=%s" % (kl_loss),
                        "rc_loss=%s" % (rc_loss), "constraint=%s" % (constraint),
                        "lambd=%s" %(str(lambd)), "constraint_ma=%s" % (constraint_ma),
                        "time=", "{:.5f}".format(time.time() - t))

            # Save model every 500 epochs
            if epoch % 500 == 0 and epoch != 0:
                save_path = saver.save(sess, model_path)
                print('saving checkpoint at',save_path)

def train_VAE(model_path, data, sess, saver,
                placeholders, model, opt, args):

    for epoch in range(args.epochs):
        t = time.time()
        batch = get_random_batch_VAE(args.batch_size, data)
        if epoch == 0:
            [initial] = sess.run([opt.constraint], feed_dict={
                            placeholders['inputs']: batch,
                            placeholders['dropout']: args.dropout})
            lambd = args.lambd
            constraint_ma = initial
        else:
            outs = sess.run([opt.opt_op, opt.cost, opt.rc_loss, opt.kl,
                                opt.constraint], feed_dict={
                                placeholders['inputs']: batch,
                                placeholders['dropout']: args.dropout,
                                placeholders['lambd']: lambd})
            constraint = outs[4]
            constraint_ma = args.alpha * constraint_ma + (1 - args.alpha) * constraint
            lambd = np.clip(lambd, 0, 1e15)
            lambd *= np.clip(np.exp(constraint_ma), 0.9, 1.1)

            if epoch % 100 == 0:
                _, cost, rc_loss, kl_loss, constraint = outs
                print("Epoch:", '%04d' % (epoch + 1), "train_loss=",
                        "{:.5f}".format(cost), "kl_loss=%s" % (kl_loss),
                        "rc_loss=%s" % (rc_loss), "constraint=%s" % (constraint),
                        "lambd=%s" %(str(lambd)), "constraint_ma=%s" % (constraint_ma),
                        "time=", "{:.5f}".format(time.time() - t))

            if epoch % 1000 == 0 and epoch != 0:
                save_path = saver.save(sess, model_path)
                print('saving checkpoint at',save_path)
