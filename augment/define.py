import tensorflow as tf
import numpy as np

from optimizer import OptimizerVGAE, OptimizerVAE
from model import VGAE, VAE

def define_placeholders(args, data_shape):

    input_dim = data_shape[0]
    num_nodes, num_features = data_shape[1], data_shape[1]

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

def define_model(args, data_shape, placeholders):

    input_dim = data_shape[0]
    num_nodes, num_features = data_shape[1], data_shape[1]

    if args.model_type == 'VAE':
        model = VAE(placeholders, input_dim, args)

    elif args.model_type == 'VGAE':
        model = VGAE(placeholders, num_features, num_nodes, args)

    else:
        model = None

    return model

def define_optimizer(args, model, data_shape, placeholders):

    input_dim = data_shape[0]
    num_nodes, num_features = data_shape[1], data_shape[1]

    if args.model_type == 'VAE':
        with tf.name_scope('optimizer'):
            opt = OptimizerVAE(reconstructions=tf.reshape(model.reconstructions, [-1]),
                               inputs=tf.reshape(placeholders['inputs'], [-1]),
                               model=model, learning_rate=args.learning_rate,
                               lambd=placeholders['lambd'], tolerance=0.03)

    elif args.model_type == 'VGAE':
        with tf.name_scope('optimizer'):
            opt = OptimizerVGAE(preds=model.reconstructions,
                                labels=tf.reshape(placeholders['adj_orig'], [-1]),
                                model=model, num_nodes=num_nodes,
                                learning_rate=args.learning_rate,
                                lambd=placeholders['lambd'], tolerance=0.1)

    else:
        opt = None, None

    return opt
