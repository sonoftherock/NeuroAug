import tensorflow as tf

class OptimizerClassifier(object):
    def __init__(self, logits, labels, model, learning_rate):
        self.logits = logits
        self.labels = labels
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                                labels=labels, logits=logits)
        self.cost = tf.reduce_mean(cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

class OptimizerPredictor(object):
    def __init__(self, preds, labels, model, learning_rate):
        self.preds = preds
        self.labels = labels
        self.cost = tf.reduce_mean(tf.square(labels - preds))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
