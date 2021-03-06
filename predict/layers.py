from initializations import *
import tensorflow as tf

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

class Layer(object):
    """Base layer class. Defines basic API for all layer objects.

    # Properties
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs

class HiddenLayer(Layer):
    """VAE Encoder with two layers."""
    def __init__(self, input_dim, output_dim, dropout, act=tf.nn.leaky_relu, **kwargs):
        super(HiddenLayer, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
                self.vars['weights'] = weight_variable_glorot([input_dim, output_dim],
                    input_dim, output_dim, name="weights")
                self.vars['bias'] = tf.Variable(tf.random.normal([output_dim], dtype=tf.float32), name="bias")
        self.act = act
        self.dropout = dropout

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1-self.dropout)
        x = tf.matmul(x, self.vars['weights']) + self.vars['bias']
        outputs = self.act(x)
        return outputs
