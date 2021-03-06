import tensorflow as tf

class OptimizerVAE(object):
    def __init__(self, reconstructions, inputs, preds, labels, model,
                    learning_rate, lambd, tolerance):
        self.reconstructions = reconstructions
        self.preds = preds
        self.labels = labels
        self.inputs = inputs

        # Lagrange multiplier and slowness of the "constraint moving average"
        # Tolerance is set as minimum reconstruction loss of the vanilla
        # nonlinear autoencoder on training set.
        self.tolerance = tolerance

        self.fc_rc_loss = tf.reduce_mean(tf.square(self.inputs - self.reconstructions))
        self.label_rc_loss = tf.reduce_mean(tf.square(self.labels - self.preds))
        self.rc_loss = self.fc_rc_loss + self.label_rc_loss

        # constraint refers to MSE reconstruction loss
        self.constraint = self.rc_loss - tf.square(self.tolerance)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # Latent loss
        self.kl = -.5*tf.reduce_sum(1+model.z_log_std - tf.square(model.z_mean) - tf.exp(model.z_log_std))
        self.cost = lambd * self.rc_loss + self.kl
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

class OptimizerAE(object):
    def __init__(self, reconstructions, inputs, model, learning_rate):
        self.reconstructions = reconstructions
        self.inputs = inputs
        self.rc_loss = tf.reduce_mean(tf.square(self.inputs - self.reconstructions))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  # Adam Optimizer
        self.cost = self.rc_loss
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

class OptimizerVAEwithFeatures(object):
    def __init__(self, reconstructions, preds, inputs, labels, model, learning_rate, lambd, tolerance):
        self.reconstructions = reconstructions
        self.inputs = inputs

        # Lagrange multiplier and slowness of the "constraint moving average"
        # Tolerance is set as minimum reconstruction loss of the vanilla
        # nonlinear autoencoder on training set.
        self.tolerance = tolerance

        # constraint refers to MSE reconstruction loss
        self.rc_loss = tf.reduce_mean(tf.square(self.inputs - self.reconstructions))
        self.constraint = self.rc_loss - tf.square(self.tolerance)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # Latent loss
        self.kl = -.5*tf.reduce_sum(1+model.z_log_std - tf.square(model.z_mean) - tf.exp(model.z_log_std))
        self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=preds))
        self.cost = lambd * self.rc_loss + self.kl + self.cross_entropy
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

class OptimizerVGAE(object):
    def __init__(self, preds, labels, model, num_nodes, learning_rate, lambd, tolerance):
        self.preds_sub = preds
        self.labels_sub = labels

        # Lagrange multiplier and slowness of the "constraint moving average"
        # Tolerance is set as minimum reconstruction loss of the vanilla
        # nonlinear autoencoder on training set.
        self.tolerance = tolerance

        # constraint refers to MSE reconstruction loss
        self.rc_loss = tf.reduce_mean(tf.square(self.labels_sub - self.preds_sub))
        self.constraint = self.rc_loss - tf.square(self.tolerance)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # Latent loss
        self.kl = -(0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 +
                  model.z_log_std - tf.square(model.z_mean)
                  - tf.exp(model.z_log_std)))

        self.cost = lambd * self.rc_loss + self.kl
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

class OptimizerGAE(object):
    def __init__(self, preds, labels, model, num_nodes, learning_rate):
        self.preds_sub = preds
        self.labels_sub = labels
        self.rc_loss = tf.reduce_mean(tf.square(self.labels_sub - self.preds_sub))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # Latent loss
        self.cost = self.rc_loss

        # Just to let train script run, doesn't do anything.
        self.kl = tf.zeros([1,1])

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
