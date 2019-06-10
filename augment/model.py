from layers import GraphConvolution, InnerProductDecoder, HiddenLayer
import tensorflow as tf
import math

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def build(self, args):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build(args)
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass

class VGAE(Model):
    def __init__(self, placeholders, num_features, num_nodes, args, **kwargs):
        super(VGAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.batch_size = args.batch_size
        self.input_dim = num_features
        self.n_samples = num_nodes
        self.adj = placeholders['adj_norm']
        self.dropout = placeholders['dropout']
        self.build(args)

    def _build(self, args):
        self.hidden1 = GraphConvolution(batch_size=self.batch_size,
                                              input_dim=self.input_dim,
                                              output_dim=args.hidden_dim_1,
                                              adj=self.adj,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)

        self.hidden2 = GraphConvolution(batch_size=self.batch_size,
                                      input_dim=args.hidden_dim_1,
                                      output_dim=args.hidden_dim_2,
                                      adj=self.adj,
                                      act=tf.nn.relu,
                                      dropout=self.dropout,
                                      logging=self.logging)(self.hidden1)

        self.z_mean = GraphConvolution(batch_size=self.batch_size,
                                       input_dim=args.hidden_dim_2,
                                       output_dim=args.hidden_dim_3,
                                       adj=self.adj,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.hidden2)

        self.z_log_std = GraphConvolution(batch_size=self.batch_size,
                                          input_dim=args.hidden_dim_2,
                                          output_dim=args.hidden_dim_3,
                                          adj=self.adj,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.hidden2)

        self.z = self.z_mean + tf.random_normal([self.n_samples, args.hidden_dim_3]) * tf.exp(self.z_log_std/2.)

        self.reconstructions = InnerProductDecoder(input_dim=args.hidden_dim_3,
                                      act=tf.nn.tanh,
                                      logging=self.logging)(self.z)

class VAE(Model):
    def __init__(self, placeholders, num_features, args, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.inputs = placeholders['inputs']
        self.batch_size = args.batch_size
        self.input_dim = num_features
        self.dropout = placeholders['dropout']
        self.build(args)

    def _build(self, args):
        self.encoder1 = HiddenLayer(input_dim=self.input_dim,
                                              output_dim=args.hidden_dim_1,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)

        self.encoder2 = HiddenLayer(input_dim=args.hidden_dim_1,
                                       output_dim=args.hidden_dim_2,
                                       act=tf.nn.relu,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.encoder1)

        self.z_mean = HiddenLayer(input_dim=args.hidden_dim_2,
                                          output_dim=args.hidden_dim_3,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.encoder2)

        self.z_log_std = HiddenLayer(input_dim=args.hidden_dim_2,
                                          output_dim=args.hidden_dim_3,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.encoder2)

        self.z = self.z_mean + tf.random_normal(shape=[self.batch_size, args.hidden_dim_3]) * tf.exp(self.z_log_std/2.)

        self.decoder1 = HiddenLayer(input_dim=args.hidden_dim_3,
                                          output_dim=args.hidden_dim_2,
                                          act=tf.nn.relu,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.z)

        self.decoder2 = HiddenLayer(input_dim=args.hidden_dim_2,
                                  output_dim=args.hidden_dim_1,
                                  act=tf.nn.relu,
                                  dropout=self.dropout,
                                  logging=self.logging)(self.decoder1)

        self.reconstructions = HiddenLayer(input_dim=args.hidden_dim_1,
                                  output_dim=16110,
                                  act=tf.nn.tanh,
                                  dropout=self.dropout,
                                  logging=self.logging)(self.decoder2)

        self.preds = HiddenLayer(input_dim=args.hidden_dim_1,
                                  output_dim= self.input_dim - 16110,
                                  act=lambda x: x,
                                  dropout=self.dropout,
                                  logging=self.logging)(self.decoder2)

class VAEwithFeatures(Model):
    def __init__(self, placeholders, num_features, args, **kwargs):
        super(VAEwithFeatures, self).__init__(**kwargs)
        self.inputs = placeholders['inputs']
        self.batch_size = args.batch_size
        self.input_dim = num_features
        self.dropout = placeholders['dropout']
        self.build(args)

    def _build(self, args):

        self.encoder1 = HiddenLayer(input_dim=self.input_dim,
                                              output_dim=args.hidden_dim_1,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)

        self.encoder2 = HiddenLayer(input_dim=args.hidden_dim_1,
                                       output_dim=args.hidden_dim_2,
                                       act=tf.nn.relu,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.encoder1)

        self.z_mean = HiddenLayer(input_dim=args.hidden_dim_2,
                                          output_dim=args.hidden_dim_3,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.encoder2)

        self.z_log_std = HiddenLayer(input_dim=args.hidden_dim_2,
                                          output_dim=args.hidden_dim_3,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.encoder2)

        self.z = self.z_mean + tf.random_normal(shape=[self.batch_size,
                                args.hidden_dim_3]) * tf.exp(self.z_log_std/2.)

        self.decoder1 = HiddenLayer(input_dim=args.hidden_dim_3,
                                          output_dim=args.hidden_dim_2,
                                          act=tf.nn.relu,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.z)

        self.decoder2 = HiddenLayer(input_dim=args.hidden_dim_2,
                                  output_dim=args.hidden_dim_1,
                                  act=tf.nn.relu,
                                  dropout=self.dropout,
                                  logging=self.logging)(self.decoder1)

        self.reconstructions = HiddenLayer(input_dim=args.hidden_dim_1,
                                  output_dim=self.input_dim,
                                  act=tf.nn.tanh,
                                  dropout=self.dropout,
                                  logging=self.logging)(self.decoder2)

        self.preds = HiddenLayer(input_dim=args.hidden_dim_1,
                          output_dim=2,
                          act=lambda x: x,
                          dropout=self.dropout,
                          logging=self.logging)(self.decoder2)

class BinaryClassifier(Model):
    def __init__(self, placeholders, num_features, args, **kwargs):
        super(BinaryClassifier, self).__init__(**kwargs)
        self.inputs = placeholders['inputs']
        self.batch_size = args.batch_size
        self.input_dim = num_features
        self.dropout = placeholders['dropout']
        self.build(args)

    def _build(self, args):
        self.layer1 = HiddenLayer(input_dim=self.input_dim,
                                              output_dim=args.hidden_dim_1,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)

        self.layer2 = HiddenLayer(input_dim=args.hidden_dim_1,
                                       output_dim=args.hidden_dim_2,
                                       act=tf.nn.relu,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.layer1)

        self.layer3 = HiddenLayer(input_dim=args.hidden_dim_2,
                                          output_dim=args.hidden_dim_3,
                                          act=tf.nn.relu,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.layer2)

        self.logits = HiddenLayer(input_dim=args.hidden_dim_3,
                                          output_dim=2,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.layer3)
