from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.contrib.slim as slim
import tensorflow as tf


#
# def blocker(blocker_num,channels,channel_padding=1):
#


def ChannelsPadding(input_tensor, channels):
    padding_tensor = tf.constant([[0, 0], [0, 0], [0, 0], [0, channels]])

    return tf.pad(input_tensor, padding_tensor)


def BlazeBlock(inputs, scope, num_block, output_channels):
    with tf.variable_scope(scope, [inputs]):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding='VALID',
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            net = inputs
            block_1_a = slim.separable_conv2d(net, output_channels, kernel_size=3, stride=2, padding='same',
                                              activation_fn=None, scope=scope + '/branch_a/depthwise_conv1')
            # branch a
            # block_1_a = slim.conv2d(block_1_a, output_channels, kernel_size=1, activation_fn=None,
            #                         scope=scope + '/branch_a/conv2')
            # branch b
            block_1_b = slim.max_pool2d(net, [2, 2], scope=scope + '/branch_b/pool1')
            #
            x = tf.nn.relu(block_1_b + block_1_a)

            for i in range(num_block):
                _x = slim.separable_conv2d(x, output_channels, kernel_size=3, stride=1,
                                           padding='same', activation_fn=None,
                                           scope=scope + '/depthwise_conv_' + str(i))
                # x = tf.nn.relu(x + slim.conv2d(_x, output_channels, kernel_size=1, activation_fn=None,
                #                                scope=scope + '/conv' + str(i)))
                x = tf.nn.relu(x + _x)
            return x


def BlazeBlockWithPadding(inputs, scope, num_block, output_channels, delta_channels):
    with tf.variable_scope(scope, [inputs]):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding='VALID',
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            net = inputs
            block_1_a = slim.separable_conv2d(net, output_channels, kernel_size=3, stride=2, padding='same',
                                              activation_fn=None, scope=scope + '/branch_a/depthwise_conv1')
            # branch a
            # block_1_a = slim.conv2d(block_1_a, output_channels, kernel_size=1, activation_fn=None,
            #                         scope=scope + '/branch_a/conv2')
            # branch b
            block_1_b = slim.max_pool2d(net, [2, 2], scope=scope + '/branch_b/pool1')
            block_1_b = ChannelsPadding(block_1_b, delta_channels)
            #
            x = tf.nn.relu(block_1_b + block_1_a)

            for i in range(num_block):
                _x = slim.separable_conv2d(x, output_channels, kernel_size=3, stride=1,
                                           padding='same', activation_fn=None,
                                           scope=scope + '/depthwise_conv_' + str(i))
                # x = tf.nn.relu(x + slim.conv2d(_x, output_channels, kernel_size=1, activation_fn=None,
                #                                scope=scope + '/conv' + str(i)))
                x = tf.nn.relu(x + _x)
            return x
