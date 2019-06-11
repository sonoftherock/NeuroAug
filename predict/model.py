from layers import HiddenLayer
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

class Predictor(Model):
    def __init__(self, placeholders, num_features, args, **kwargs):
        super(Predictor, self).__init__(**kwargs)
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

        self.preds = HiddenLayer(input_dim=args.hidden_dim_3,
                                          output_dim=2,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.layer3)
