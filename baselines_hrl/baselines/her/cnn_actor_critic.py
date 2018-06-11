# Created by Xingyu Lin, 25/03/2018                                                                                  
import tensorflow as tf
from baselines.her.util import store_args, nn
import numpy as np


def cnn_one_stream(input_net, scope='phi', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        net = tf.layers.conv2d(name='conv1', inputs=input_net, filters=64, kernel_size=[2, 2], padding='same',
                               activation=tf.nn.relu)
        net = tf.layers.max_pooling2d(net, [3, 3], 2, padding='valid')
        net = tf.layers.conv2d(name='conv2', inputs=net, filters=64, kernel_size=[2, 2], padding='same',
                               activation=tf.nn.relu)
        net = tf.layers.max_pooling2d(net, [3, 3], 2, padding='valid')
        net = tf.layers.conv2d(name='conv3', inputs=net, filters=64, kernel_size=[2, 2], padding='same',
                               activation=tf.nn.relu)
        net = tf.layers.max_pooling2d(net, [3, 3], 2, padding='valid')
        net = tf.layers.conv2d(name='conv4', inputs=net, filters=64, kernel_size=[2, 2], padding='same',
                               activation=tf.nn.relu)
        shape = net.get_shape().as_list()  # a list: [None, 9, 2]
        dim = np.prod(shape[1:])  # dim = prod(9,2) = 18
        net = tf.reshape(net, [-1, dim])

        net = tf.layers.dense(name='conv_fc', inputs=net, units=64)
    return net


class CNNActorCritic:
    @store_args
    def __init__(self, inputs_tf, image_input_shapes, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                 **kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (baselines.her.Normalizer): normalizer for observations
            g_stats (baselines.her.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """
        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']

        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)
        o = tf.reshape(o, [-1, *image_input_shapes['o']])
        g = tf.reshape(g, [-1, *image_input_shapes['g']])

        # input_pi = tf.concat(axis=1, values=[o, g])  # for actor

        # Networks.

        x_o = cnn_one_stream(o, scope='phi', reuse=False)
        x_g = cnn_one_stream(g, scope='phi', reuse=True)

        x_concat = tf.concat(axis=1, values=[x_o, x_g])

        with tf.variable_scope('pi'):
            self.pi_tf = self.max_u * tf.tanh(nn(
                x_concat, [self.hidden] * self.layers + [self.dimu]))

        with tf.variable_scope('Q'):
            # for policy training
            input_Q = tf.concat(axis=1, values=[x_concat, self.pi_tf / self.max_u])
            self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1])
            # for critic training
            input_Q = tf.concat(axis=1, values=[x_concat, self.u_tf / self.max_u])
            self._input_Q = input_Q  # exposed for tests
            self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True)
