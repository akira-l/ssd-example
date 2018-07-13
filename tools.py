import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average
import cfg

# conv
def convolution(input_, shape, strides, name):
    with tf.variable_scope(name):
        weight = tf.get_variable(initializer=tf.truncated_normal(shape, 0, 1), dtype=tf.float32, name=name+'_weight')
        bias = tf.get_variable(initializer=tf.truncated_normal(shape[-1:], 0, 1), dtype=tf.float32, name=name+'_bias')
        result = tf.nn.conv2d(input_, weight, strides, padding='SAME', name=name+'_conv')
        result = tf.nn.bias_add(result, bias)
        result = batch_normalization(result, name=name+'_bn')
        result = tf.nn.relu(result, name=name+'_relu')
        return result




# fully connect
def fc(input_, out_shape, name):
    with tf.variable_scope(name+'_fc'):
        in_shape = 1
        for d in input_.get_shape().as_list()[1:]:
            in_shape *= d
        weight = tf.get_variable(initializer=tf.truncated_normal([in_shape, out_shape], 0, 1), dtype=tf.float32, name=name+'_fc_weight')
        bias = tf.get_variable(initializer=tf.truncated_normal([out_shape], 0, 1), dtype=tf.float32, name=name+'_fc_bias')
        result = tf.reshape(input_, [-1, in_shape])
        result = tf.nn.xw_plus_b(result, weight, bias, name=name+'_fc_do')
        return result





# Batch Normalization
def batch_normalization(input_, name):
    with tf.variable_scope(name):
        bn_input_shape = input_.get_shape() 
        moving_mean = tf.get_variable(name+'_mean', bn_input_shape[-1:] , initializer=tf.zeros_initializer, trainable=False)
        moving_variance = tf.get_variable(name+'_variance', bn_input_shape[-1:] , initializer=tf.ones_initializer, trainable=False)
        def mean_var_with_update():
            mean, variance = tf.nn.moments(input_, list(range(len(bn_input_shape) - 1)), name=name+'_moments')
            with tf.control_dependencies([assign_moving_average(moving_mean, mean, cfg.conv_bn_decay),assign_moving_average(moving_variance, variance, cfg.conv_bn_decay)]):
                return tf.identity(mean), tf.identity(variance)
        #mean, variance = tf.cond(tf.cast(self.isTraining, tf.bool), mean_var_with_update, lambda: (moving_mean, moving_variance))
        mean, variance = tf.cond(tf.cast(True, tf.bool), mean_var_with_update, lambda: (moving_mean, moving_variance))
        beta = tf.get_variable(name+'_beta', bn_input_shape[-1:] , initializer=tf.zeros_initializer)
        gamma = tf.get_variable(name+'_gamma', bn_input_shape[-1:] , initializer=tf.ones_initializer)
        return tf.nn.batch_normalization(input_, mean, variance, beta, gamma, cfg.conv_bn_epsilon, name+'_bn_opt')





