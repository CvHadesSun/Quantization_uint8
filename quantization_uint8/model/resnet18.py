'''
Author: cvhadessun
Date: 2021-08-11 15:57:06
LastEditTime: 2021-08-11 16:14:32
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Pose2D/src/model/resnet18.py
'''


import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.layers.python.layers import regularizers, \
    initializers, layers
from tensorflow.python.ops import nn_ops


def posenet_arg_scope(trainable=True,
                      weight_decay=0.000001,
                      weight_init = initializers.variance_scaling_initializer()):
    '''no bn layer arg scope'''

    with arg_scope(
            [slim.conv2d, slim.conv2d_transpose],
            weights_regularizer=regularizers.l2_regularizer(weight_decay),
            weights_initializer=weight_init,
            trainable=trainable,
            activation_fn=nn_ops.relu,
            normalizer_fn=layers.batch_norm) as arg_sc:
        return arg_sc




def res18_block(inp, output_channels, scope, stride=1):
    '''

    Args:
        inp:
        output_channels:
        stride:

    Returns:

    ----

    '''
    with slim.arg_scope(posenet_arg_scope()):
        net = slim.conv2d(inp, output_channels, kernel_size=3, stride=stride, activation_fn=None,
                          padding='same', scope=scope + '/conv1')
        # net = slim.separable_conv2d(inp, output_channels, kernel_size=3, stride=stride, activation_fn=None,
        #                   padding='same', scope=scope + '/conv1')
        net = slim.conv2d(net, output_channels, kernel_size=3, stride=1, activation_fn=None,
                          padding='same', scope=scope + '/conv2')
        # net = slim.separable_conv2d(net, output_channels, kernel_size=3, stride=1, activation_fn=None,
        #                   padding='same', scope=scope + '/conv2')
        if stride == 1:
            return tf.nn.relu(inp + net)
        else:
            identity = slim.conv2d(inp, output_channels, kernel_size=1, stride=stride,
                                   activation_fn=None, scope=scope + 'subsample')
            return tf.nn.relu(identity + net)


def make_block(inp, num_block, output_channels, scope, stride=1):
    with slim.arg_scope(posenet_arg_scope()):
        net = inp
        for i in range(num_block):
            sp = scope + '/block_' + str(i)
            net = res18_block(net, output_channels, sp, stride=stride)

        return net


def resnet18(inp):
    '''the net to detect the human 2d pose version 1:resnet-18,
    @params:
    - input:inp:the input batch image[B,H,W,C]
    - return:
        - landmark2d: the 2d pose landmarks[B,1,1,N*K];
        - pose_flag: the confidence of detected pose [B,1,1,1];
        - heatmap: the 2d heatmap of human pose [B,h,w,N].
    '''

    #
    with slim.arg_scope(posenet_arg_scope()):
        net = slim.conv2d(inp, 64, kernel_size=3, stride=1, activation_fn=tf.nn.relu, scope='conv1')

        net = make_block(net,1,64,'layer_1',stride=1)
        net = make_block(net,2,128,'layer_2',stride=2)
        net = make_block(net,2,256,'layer_3',stride=2)
        net = make_block(net,2,512,'layer_4',stride=2)

        return net


class Resnet18:

    def __init__(self, is_train=True):
        self.input_image = tf.placeholder(tf.float32, [None, 256, 192, 3], "inputs")
        self.is_train = is_train

        with tf.variable_scope("Resnet18") as scope:
            self.train_digits = self.construct_net(True)
            scope.reuse_variables()
            self.pred_digits = self.construct_net(False)
        self.predictions = self.pred_digits
      

    def construct_net(self, is_trained=True):
        net = resnet18(self.input_image)

        return net