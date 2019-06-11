import tensorflow as tf
import numpy as np

from optimizer import OptimizerClassifier, OptimizerPredictor
from model import Predictor

def define_placeholders(args, data_shape):

    input_dim = data_shape[0]
    num_nodes, num_features = data_shape[1], data_shape[1]

    placeholders = {
        'inputs': tf.placeholder(tf.float32, [args.batch_size, input_dim]),
        'labels': tf.placeholder(tf.float32, [args.batch_size, input_dim - 16110]),
        'dropout': tf.placeholder_with_default(0., shape=())
    }

    return placeholders

def define_model(args, data_shape, placeholders):

    input_dim = data_shape[0]
    num_nodes, num_features = data_shape[1], data_shape[1]

    if args.model_type == 'Predictor':
        model = Predictor(placeholders, num_features, num_nodes, args)

    else:
        model = None
        print("beep beep invalid model type")

    return model

def define_optimizer(args, model, data_shape, placeholders):

    input_dim = data_shape[0]
    num_nodes, num_features = data_shape[1], data_shape[1]

    if args.model_type == 'Classifier':
        with tf.name_scope('optimizer'):
            opt = OptimizerClassifier(logits=tf.reshape(model.preds, [-1]),
                       labels=tf.reshape(placeholders['labels'], [-1]),
                       model=model, learning_rate=args.learning_rate)

    elif args.model_type == 'Predictor':
        with tf.name_scope('optimizer'):
            opt = OptimizerPredictor(preds=tf.reshape(model.preds, [-1]),
                       labels=tf.reshape(placeholders['labels'], [-1]),
                       model=model, learning_rate=args.learning_rate)

    else:
        opt = None, None
        print("beep beep invalid model type")

    return opt
