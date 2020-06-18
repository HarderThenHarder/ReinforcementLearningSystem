"""
@ Author: Szh
@ Time: 2020/2/21
@ Software: PyCharm
"""
import tensorflow as tf
import numpy as np
import os
import multiprocessing


def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        # lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:  # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v  # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init


def fc(x, scope, nh, *, init_scale=1.0, init_bias=0.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias))
        return tf.matmul(x, w)+b


def mlp(num_layers=2, num_hidden=64, activation=tf.tanh, layer_norm=False):
    """
    Stack of fully-connected layers to be used in a policy / q-function approximator

    Parameters:
    ----------
    num_layers: int                 number of fully-connected layers (default: 2)
    num_hidden: int                 size of fully-connected layers (default: 64)
    activation:                     activation function (default: tf.tanh)
    Returns:
    -------

    returns function that builds fully connected network with a given input tensor / placeholder
    """
    def network_fn(X):
        h = tf.layers.flatten(X)
        for i in range(num_layers):
            h = fc(h, 'mlp_fc{}'.format(i), nh=num_hidden, init_scale=np.sqrt(2))
            if layer_norm:
                h = tf.contrib.layers.layer_norm(h, center=True, scale=True)
            h = activation(h)

        return h

    return network_fn


def make_session(config=None, num_cpu=None, make_default=False, graph=None):
    """Returns a session that will use <num_cpu> CPU's only"""
    if num_cpu is None:
        num_cpu = int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count()))
    if config is None:
        config = tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=num_cpu,
            intra_op_parallelism_threads=num_cpu)
        config.gpu_options.allow_growth = True

    if make_default:
        return tf.InteractiveSession(config=config, graph=graph)
    else:
        return tf.Session(config=config, graph=graph)


class PolicywithValue(object):
    """
        Encapsulates RL networks for policy distribution and value function estimation with shared parameters
    """
    def __init__(self, observation_dim, action_dim, sess=None):
        self.obs = tf.placeholder(shape=(None, observation_dim), dtype=tf.float32, name='observation')
        # TODO: normalize observations
        with tf.variable_scope('latent', reuse=tf.AUTO_REUSE):
            latent = mlp()(self.obs)
        with tf.variable_scope('policy', reuse=tf.AUTO_REUSE):
            self.logits = fc(latent, 'a_out', action_dim, init_scale=0.01)  # (batch_size, action_dim)
            self.action = self._sample()  # (batch_size,) & int
            self.logp = self.logprob(self.action)  # (batch_size,) & float
        with tf.variable_scope('value', reuse=tf.AUTO_REUSE):
            self.v = fc(latent, 'v_out', 1)
            self.v = self.v[:, 0]  # (batch_size,)
        self.sess = sess or tf.get_default_session()

    def _sample(self):
        # in-graph: sample an action according to the current logits based on (0, 1) uniform distribution
        u = tf.random_uniform(tf.shape(self.logits), dtype=self.logits.dtype)
        return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)


    def entropy(self):
        # in-graph: calculate the entropy of the probability distribution according to the current logits
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)

    def logprob(self, action):
        # in-graph: calculate the log probability of the taken action according to the current logits
        assert action.dtype in {tf.uint8, tf.int32, tf.int64}
        action = tf.one_hot(action, self.logits.get_shape().as_list()[-1])  # one-hot encoding
        return -tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=action)

    def step(self, observation):
        # execution: take a step in the environment based on the received observation
        feed_dict = {self.obs: observation}
        return self.sess.run([self.action, self.v, self.logp], feed_dict)

    def value(self, observation):
        # execution: calculate the last value function at the end of an interaction loop for bootstrapping
        feed_dict = {self.obs: observation}
        return self.sess.run(self.v, feed_dict)


class PPO(object):
    """
        Implement the PPO_tensorflow algorithm, especially the training parts
    """
    def __init__(self, observation_dim, action_dim, vf_coef, ent_coef, max_grad_norm, policy=PolicywithValue, sess=None):
        self.obs_dim = observation_dim
        self.act_dim = action_dim
        self.sess = sess or tf.get_default_session()
        if self.sess is None:
            self.sess = make_session(config=None, make_default=True)

        # create two ppo models, one for sampling and one for training
        with tf.variable_scope('ppo_model', reuse=tf.AUTO_REUSE):
            act_model = policy(observation_dim, action_dim, self.sess)
            train_model = policy(observation_dim, action_dim, self.sess)

        # create the placeholders
        self.act = act = tf.placeholder(shape=(None,), dtype=tf.uint8, name='action')
        self.adv = adv = tf.placeholder(shape=(None,), dtype=tf.float32, name='advantage')
        self.ret = ret = tf.placeholder(shape=(None,), dtype=tf.float32, name='return')
        self.olp = olp = tf.placeholder(shape=(None,), dtype=tf.float32, name='old_log_prob')
        self.opv = opv = tf.placeholder(shape=(None,), dtype=tf.float32, name='old_predicted_value')
        self.lr = lr = tf.placeholder(shape=(), dtype=tf.float32, name='learning_rate')
        self.cr = cr = tf.placeholder(shape=(), dtype=tf.float32, name='clip_range')

        # calculate the current log probability
        clp = train_model.logprob(act)

        # calculate the entropy
        # entropy is used to improve exploration by limiting the premature convergence to suboptimal policy
        entropy = tf.reduce_mean(train_model.entropy())

        # clip the value to reduce variability during critic training
        # Get the current predicted value
        cpv = train_model.v
        cpv_clipped = opv + tf.clip_by_value(train_model.v - opv, -cr, cr)
        # unclipped value loss
        vf_loss_unclipped = tf.square(cpv - ret)
        # clipped value loss
        vf_loss_clipped = tf.square(cpv_clipped - ret)
        # the final value function loss
        vf_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_loss_unclipped, vf_loss_clipped))

        # clip the ratio to reduce variability during actor training
        # calculate the ratio (pi current policy / pi old policy)
        ratio = tf.exp(clp - olp)
        # unclipped policy loss
        pg_loss_unclipped = -adv * ratio
        # clipped policy loss
        pg_loss_clipped = -adv * tf.clip_by_value(ratio, 1.0 - cr, 1.0 + cr)
        # the final policy gradient loss
        pg_loss = tf.reduce_mean(tf.maximum(pg_loss_unclipped, pg_loss_clipped))
        approxkl = 0.5 * tf.reduce_mean(tf.square(olp - clp))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), cr)))

        # total loss
        loss = pg_loss + vf_coef * vf_loss - ent_coef * entropy

        # update the parameters using loss
        # get the model parameters
        params = tf.trainable_variables('ppo_model')
        # build the trainer
        self.trainer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-5)
        # calculate the gradients
        grads_and_var = self.trainer.compute_gradients(loss, params)
        grads, var = zip(*grads_and_var)
        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))
        self.grads = grads
        self.var = var
        self._train_op = self.trainer.apply_gradients(grads_and_var)
        self.stats_list = [pg_loss, vf_loss, entropy, approxkl, clipfrac]

        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value

        self.sess.run(tf.variables_initializer(tf.global_variables()))

    def train(self, lr, cr, obs, act, ret, opv, olp):
        # normalize the advantages
        adv = ret - opv
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # all the data needed for the training stage
        feed_dict = {
            self.lr: lr,
            self.cr: cr,
            self.train_model.obs: obs,
            self.act: act,
            self.ret: ret,
            self.opv: opv,
            self.olp: olp,
            self.adv: adv
        }

        return self.sess.run(self.stats_list + [self._train_op], feed_dict)[:-1]

