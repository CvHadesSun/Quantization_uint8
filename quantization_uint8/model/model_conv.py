'''
Author: your name
Date: 2021-08-10 17:29:51
LastEditTime: 2021-08-11 18:46:35
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Pose2D/src/model/model_conv.py
'''
import tensorflow as tf
import sys
import tensorflow.contrib.slim as slim


def block(input, output_channels, stride, scope):
    with tf.variable_scope(scope, [input]):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding='same',
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            net = input
            net = slim.separable_conv2d(net, output_channels, kernel_size=3,
                                        stride=stride, padding='same',activation_fn=None,
                                        scope=scope + 'd_conv')
            return net

def fpn_head(inps,output_channels,scope):
    with tf.variable_scope(scope, [inps]):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding='same',
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            net1,net2,net3,net4 = inps
            p1 = slim.separable_conv2d(net4, 512, kernel_size=3, padding='same', activation_fn=None,
                                         scope='layer1/depthwise_conv1')
            p1 = tf.image.resize_images(p1, (16, 16), method=0)

            p2 = tf.nn.relu(p1+net3)

            p2 = slim.separable_conv2d(p2, 256, kernel_size=3, padding='same', activation_fn=None,
                                         scope='layer2/depthwise_conv1')

            p2 = tf.image.resize_images(p2, (32, 32), method=0)

            p3 = tf.nn.relu(p2+net2)

            p4 = slim.separable_conv2d(p3, 128, kernel_size=3, padding='same', activation_fn=None,
                                         scope='layer3/depthwise_conv1')

            p4 = tf.image.resize_images(p4, (64, 64), method=0)
            p4 = tf.nn.relu(p4+net1)

            net = slim.separable_conv2d(p4, 64, kernel_size=3, padding='same', activation_fn=None,
                                         scope='layer4/depthwise_conv1')
            return net



def upsample_head(inp,output_channels,scope):
    with tf.variable_scope(scope, [inp]):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding='same',
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            net = inp
            net = tf.image.resize_images(net,(16,16),method=0)
            net = slim.separable_conv2d(net, 512, kernel_size=3, padding='same', activation_fn=None,
                                         scope='conv1')
            # net = slim.separable_conv2d(net, 256, kernel_size=3, padding='same', activation_fn=None,
            #                              scope='conv2')
            # net = tf.image.resize_images(net,(32,32),method=0)
            net = slim.separable_conv2d(net, 128, kernel_size=3, padding='same', activation_fn=None,
                                         scope='conv2')
            # net = tf.image.resize_images(net,(64,64),method=0)
            net = slim.separable_conv2d(net, output_channels, kernel_size=3, padding='same', activation_fn=None,
                                         scope='conv3')

            return net

class TestModel:
    def __init__(self, is_train=True):
        self.input_image = tf.placeholder(tf.float32, [None, 256, 256, 3], "inputs")
        self.is_train = is_train

        with tf.variable_scope("BlazePose") as scope:
            self.train_digits = self.construct_net(True)
            scope.reuse_variables()
            self.pred_digits = self.construct_net(False)
        self.predictions = self.pred_digits

    def construct_net(self, is_trained=True):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding='same',
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            net = slim.conv2d(self.input_image, 32, kernel_size=3, stride=2,padding='same',
                              activation_fn=None, scope='conv1')

            net = block(net, 64, 1, 'b1') #128
            net = block(net, 128, 2, 'b2') #64
            net1 = block(net, 128, 1, 'b3') #64
            net = block(net1, 256, 2, 'b4') #32
            net2 = block(net, 256, 1, 'b5') #32
            net = block(net2, 512, 2, 'b6') #16
            net = block(net, 512, 1, 'b7') #16
            net = block(net, 512, 1, 'b8')
            net = block(net, 512, 1, 'b9')
            net = block(net, 512, 1, 'b10')
            net3 = block(net, 512, 1, 'b11')
            net = block(net3, 1024, 2, 'b12') #8
            net4 = block(net, 1024, 1, 'b13') #8
            #
            # net = slim.conv2d_transpose(net,512,3,2,padding='same',activation_fn=None,scope='upsample_1')
            net = fpn_head([net1,net2,net3,net4],64,'FPN')
            # net = upsample_head(net4,64,'heatmap_head')
            return net

