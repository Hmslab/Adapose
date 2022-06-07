""" Wrapper functions for TensorFlow layers.
Author: Charles R. Qi
Date: November 2016
"""

import numpy as np
import tensorflow as tf
from tflearn.layers.conv import conv_1d as conv_1d_l
from tflearn.layers.conv import avg_pool_1d as avg_pool_1d_l
from tflearn.layers.core import fully_connected as fully_connected_l
from tflearn.layers.normalization import batch_normalization as batch_normalization_l



def _variable_on_cpu(name, shape, initializer, use_fp16=False):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
      use_xavier: bool, whether to use xavier initializer
    Returns:
      Variable Tensor
    """
    
    if use_xavier:
        initializer = tf.contrib.layers.xavier_initializer()
    else:
        initializer = tf.truncated_normal_initializer(stddev=stddev)
    var = _variable_on_cpu(name, shape, initializer)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def conv1d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=1,
           padding='SAME',
           data_format='NHWC',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=None,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
  """ 1D convolution with non-linear operation.
  Args:
    inputs: 3-D tensor variable BxLxC
    num_output_channels: int
    kernel_size: int
    scope: string
    stride: int
    padding: 'SAME' or 'VALID'
    data_format: 'NHWC' or 'NCHW'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    assert(data_format=='NHWC' or data_format=='NCHW')
    if data_format == 'NHWC':
      num_in_channels = inputs.get_shape()[-1].value
    elif data_format=='NCHW':
      num_in_channels = inputs.get_shape()[1].value
    kernel_shape = [kernel_size,
                    num_in_channels, num_output_channels]
    kernel = _variable_with_weight_decay('weights',
                                         shape=kernel_shape,
                                         use_xavier=use_xavier,
                                         stddev=stddev,
                                         wd=weight_decay)
    outputs = tf.nn.conv1d(inputs, kernel,
                           stride=stride,
                           padding=padding,
                           data_format=data_format)
    biases = _variable_on_cpu('biases', [num_output_channels],
                              tf.constant_initializer(0.0))
    outputs = tf.nn.bias_add(outputs, biases, data_format=data_format)

    if bn:
      outputs = batch_norm_for_conv1d(outputs, is_training,
                                      bn_decay=bn_decay, scope='bn',
                                      data_format=data_format)

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs


def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1],
           padding='SAME',
           data_format='NHWC',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=None,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
  """ 2D convolution with non-linear operation.
  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    data_format: 'NHWC' or 'NCHW'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
      kernel_h, kernel_w = kernel_size
      assert(data_format=='NHWC' or data_format=='NCHW')
      if data_format == 'NHWC':
        num_in_channels = inputs.get_shape()[-1].value
      elif data_format=='NCHW':
        num_in_channels = inputs.get_shape()[1].value
      kernel_shape = [kernel_h, kernel_w,
                      num_in_channels, num_output_channels]
      kernel = _variable_with_weight_decay('weights',
                                           shape=kernel_shape,
                                           use_xavier=use_xavier,
                                           stddev=stddev,
                                           wd=weight_decay)
      stride_h, stride_w = stride
      outputs = tf.nn.conv2d(inputs, kernel,
                             [1, stride_h, stride_w, 1],
                             padding=padding,
                             data_format=data_format)
      biases = _variable_on_cpu('biases', [num_output_channels],
                                tf.constant_initializer(0.0))
      outputs = tf.nn.bias_add(outputs, biases, data_format=data_format)

      if bn:
        outputs = batch_norm_for_conv2d(outputs, is_training,
                                        bn_decay=bn_decay, scope='bn',
                                        data_format=data_format)

      if activation_fn is not None:
        outputs = activation_fn(outputs)
      return outputs

def conv2d_transpose(inputs,
                     num_output_channels,
                     kernel_size,
                     scope,
                     stride=[1, 1],
                     padding='SAME',
                     use_xavier=True,
                     stddev=1e-3,
                     weight_decay=0.0,
                     activation_fn=tf.nn.relu,
                     bn=False,
                     bn_decay=None,
                     is_training=None):
    """ 2D convolution transpose with non-linear operation.
    Args:
      inputs: 4-D tensor variable BxHxWxC
      num_output_channels: int
      kernel_size: a list of 2 ints
      scope: string
      stride: a list of 2 ints
      padding: 'SAME' or 'VALID'
      use_xavier: bool, use xavier_initializer if true
      stddev: float, stddev for truncated_normal init
      weight_decay: float
      activation_fn: function
      bn: bool, whether to use batch norm
      bn_decay: float or float tensor variable in [0,1]
      is_training: bool Tensor variable
    Returns:
      Variable tensor
    Note: conv2d(conv2d_transpose(a, num_out, ksize, stride), a.shape[-1], ksize, stride) == a
    """
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_h, kernel_w,
                        num_output_channels, num_in_channels]  # reversed to conv2d
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        stride_h, stride_w = stride

        # from slim.convolution2d_transpose
        def get_deconv_dim(dim_size, stride_size, kernel_size, padding):
            dim_size *= stride_size

            if padding == 'VALID' and dim_size is not None:
                dim_size += max(kernel_size - stride_size, 0)
            return dim_size

        # caculate output shape
        batch_size = inputs.get_shape()[0].value
        height = inputs.get_shape()[1].value
        width = inputs.get_shape()[2].value
        out_height = get_deconv_dim(height, stride_h, kernel_h, padding)
        out_width = get_deconv_dim(width, stride_w, kernel_w, padding)
        output_shape = [batch_size, out_height, out_width, num_output_channels]

        outputs = tf.nn.conv2d_transpose(inputs, kernel, output_shape,
                                         [1, stride_h, stride_w, 1],
                                         padding=padding)
        biases = _variable_on_cpu('biases', [num_output_channels],
                                  tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_conv2d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def conv3d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
    """ 3D convolution with non-linear operation.
    Args:
      inputs: 5-D tensor variable BxDxHxWxC
      num_output_channels: int
      kernel_size: a list of 3 ints
      scope: string
      stride: a list of 3 ints
      padding: 'SAME' or 'VALID'
      use_xavier: bool, use xavier_initializer if true
      stddev: float, stddev for truncated_normal init
      weight_decay: float
      activation_fn: function
      bn: bool, whether to use batch norm
      bn_decay: float or float tensor variable in [0,1]
      is_training: bool Tensor variable
    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_d, kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_d, kernel_h, kernel_w,
                        num_in_channels, num_output_channels]
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        stride_d, stride_h, stride_w = stride
        outputs = tf.nn.conv3d(inputs, kernel,
                               [1, stride_d, stride_h, stride_w, 1],
                               padding=padding)
        biases = _variable_on_cpu('biases', [num_output_channels],
                                  tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_conv3d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=0.0,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bn_decay=None,
                    is_training=None):
    """ Fully connected layer with non-linear operation.

    Args:
      inputs: 2-D tensor BxN
      num_outputs: int

    Returns:
      Variable tensor of size B x num_outputs.
    """
    
    with tf.variable_scope(scope) as sc:
        num_input_units = inputs.get_shape()[-1].value
        weights = _variable_with_weight_decay('weights',
                                              shape=[num_input_units, num_outputs],
                                              use_xavier=use_xavier,
                                              stddev=stddev,
                                              wd=weight_decay)
        outputs = tf.matmul(inputs, weights)
        biases = _variable_on_cpu('biases', [num_outputs],
                                  tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_fc(outputs, is_training, bn_decay, 'bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def max_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
    """ 2D max pooling.
    Args:
      inputs: 4-D tensor BxHxWxC
      kernel_size: a list of 2 ints
      stride: a list of 2 ints

    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        outputs = tf.nn.max_pool(inputs,
                                 ksize=[1, kernel_h, kernel_w, 1],
                                 strides=[1, stride_h, stride_w, 1],
                                 padding=padding,
                                 name=sc.name)
        return outputs


def avg_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
    """ 2D avg pooling.
    Args:
      inputs: 4-D tensor BxHxWxC
      kernel_size: a list of 2 ints
      stride: a list of 2 ints

    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        outputs = tf.nn.avg_pool(inputs,
                                 ksize=[1, kernel_h, kernel_w, 1],
                                 strides=[1, stride_h, stride_w, 1],
                                 padding=padding,
                                 name=sc.name)
        return outputs


def max_pool3d(inputs,
               kernel_size,
               scope,
               stride=[2, 2, 2],
               padding='VALID'):
    """ 3D max pooling.
    Args:
      inputs: 5-D tensor BxDxHxWxC
      kernel_size: a list of 3 ints
      stride: a list of 3 ints

    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_d, kernel_h, kernel_w = kernel_size
        stride_d, stride_h, stride_w = stride
        outputs = tf.nn.max_pool3d(inputs,
                                   ksize=[1, kernel_d, kernel_h, kernel_w, 1],
                                   strides=[1, stride_d, stride_h, stride_w, 1],
                                   padding=padding,
                                   name=sc.name)
        return outputs


def avg_pool3d(inputs,
               kernel_size,
               scope,
               stride=[2, 2, 2],
               padding='VALID'):
    """ 3D avg pooling.
    Args:
      inputs: 5-D tensor BxDxHxWxC
      kernel_size: a list of 3 ints
      stride: a list of 3 ints

    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_d, kernel_h, kernel_w = kernel_size
        stride_d, stride_h, stride_w = stride
        outputs = tf.nn.avg_pool3d(inputs,
                                   ksize=[1, kernel_d, kernel_h, kernel_w, 1],
                                   strides=[1, stride_d, stride_h, stride_w, 1],
                                   padding=padding,
                                   name=sc.name)
        return outputs


def batch_norm_template(inputs, is_training, scope, moments_dims_unused, bn_decay, data_format='NHWC'):
    """ Batch normalization on convolutional maps and beyond...
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow

    Args:
        inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
        is_training:   boolean tf.Varialbe, true indicates training phase
        scope:         string, variable scope
        moments_dims:  a list of ints, indicating dimensions for moments calculation
        bn_decay:      float or float tensor variable, controling moving average weight
        data_format:   'NHWC' or 'NCHW'
    Return:
        normed:        batch-normalized maps
    """
    bn_decay = bn_decay if bn_decay is not None else 0.9
    return tf.contrib.layers.batch_norm(inputs,
                                        center=True, scale=True,
                                        is_training=is_training, decay=bn_decay, updates_collections=None,
                                        scope=scope,
                                        data_format=data_format)


def batch_norm_for_fc(inputs, is_training, bn_decay, scope):
    """ Batch normalization on FC data.

    Args:
        inputs:      Tensor, 2D BxC input
        is_training: boolean tf.Varialbe, true indicates training phase
        bn_decay:    float or float tensor variable, controling moving average weight
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0, ], bn_decay)


def batch_norm_for_conv1d(inputs, is_training, bn_decay, scope, data_format):
    """ Batch normalization on 1D convolutional maps.

    Args:
        inputs:      Tensor, 3D BLC input maps
        is_training: boolean tf.Varialbe, true indicates training phase
        bn_decay:    float or float tensor variable, controling moving average weight
        scope:       string, variable scope
        data_format: 'NHWC' or 'NCHW'
    Return:
        normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0, 1], bn_decay, data_format)


def batch_norm_for_conv2d(inputs, is_training, bn_decay, scope, data_format):
    """ Batch normalization on 2D convolutional maps.

    Args:
        inputs:      Tensor, 4D BHWC input maps
        is_training: boolean tf.Varialbe, true indicates training phase
        bn_decay:    float or float tensor variable, controling moving average weight
        scope:       string, variable scope
        data_format: 'NHWC' or 'NCHW'
    Return:
        normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0, 1, 2], bn_decay, data_format)


def batch_norm_for_conv3d(inputs, is_training, bn_decay, scope):
    """ Batch normalization on 3D convolutional maps.

    Args:
        inputs:      Tensor, 5D BDHWC input maps
        is_training: boolean tf.Varialbe, true indicates training phase
        bn_decay:    float or float tensor variable, controling moving average weight
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0, 1, 2, 3], bn_decay)


def dropout(inputs,
            is_training,
            scope,
            keep_prob=0.5,
            noise_shape=None):
    """ Dropout layer.
    Args:
      inputs: tensor
      is_training: boolean tf.Variable
      scope: string
      keep_prob: float in [0,1]
      noise_shape: list of ints
    Returns:
      tensor variable
    """
    with tf.variable_scope(scope) as sc:
        outputs = tf.cond(is_training,
                          lambda: tf.nn.dropout(inputs, keep_prob, noise_shape),
                          lambda: inputs)
        return outputs

def encoder_with_convs_and_symmetry(
    in_signal,
    n_filters=[64, 128, 256, 1024],
    filter_sizes=[1],
    strides=[1],
    b_norm=True,
    b_norm_decay=0.9,
    non_linearity=tf.nn.relu,
    regularizer=None,
    weight_decay=0.001,
    symmetry=tf.reduce_max,
    dropout_prob=None,
    pool=avg_pool_1d_l,
    pool_sizes=None,
    scope=None,
    reuse=False,
    padding="same",
    verbose=False,
    closing=None,
    conv_op=conv_1d_l,
    return_layer_before_symmetry=False,
):
    if verbose:
        print("Building Encoder")

    n_layers = len(n_filters)
    filter_sizes = replicate_parameter_for_all_layers(filter_sizes, n_layers)
    strides = replicate_parameter_for_all_layers(strides, n_layers)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

    if n_layers < 2:
        raise ValueError("More than 1 layers are expected.")

    for i in range(n_layers):
        if i == 0:
            layer = in_signal

        name = "encoder_conv_layer_" + str(i)
        scope_i = expand_scope_by_name(scope, name)
        layer = conv_op(
            layer,
            nb_filter=n_filters[i],
            filter_size=filter_sizes[i],
            strides=strides[i],
            regularizer=regularizer,
            weight_decay=weight_decay,
            name=name,
            reuse=reuse,
            scope=scope_i,
            padding=padding,
        )
        if verbose:
            print(
                (
                    name,
                    "conv params = ",
                    np.prod(layer.W.get_shape().as_list())
                    + np.prod(layer.b.get_shape().as_list()),
                )
            )

        if b_norm:
            name += "_bnorm"
            scope_i = expand_scope_by_name(scope, name)
            layer = batch_normalization_l(
                layer, decay=b_norm_decay, name=name, reuse=reuse, scope=scope_i
            )
            if verbose:
                print(
                    (
                        "bnorm params = ",
                        np.prod(layer.beta.get_shape().as_list())
                        + np.prod(layer.gamma.get_shape().as_list()),
                    )
                )

        if non_linearity is not None:
            layer = non_linearity(layer)

        if pool is not None and pool_sizes is not None:
            if pool_sizes[i] is not None:
                layer = pool(layer, kernel_size=pool_sizes[i])

        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

        if verbose:
            print(layer)
            print(("output size:", np.prod(layer.get_shape().as_list()[1:]), "\n"))

        layer_before_symmetry = layer
        if symmetry is not None:
            layer = symmetry(layer, axis=1)
            if verbose:
                print(layer)

        if closing is not None:
            layer = closing(layer)
            print(layer)

        if return_layer_before_symmetry:
            return layer, layer_before_symmetry
        else:
            return layer


def decoder_with_fc_only(
    latent_signal,
    layer_sizes=[],
    b_norm=True,
    b_norm_decay=0.9,
    non_linearity=tf.nn.relu,
    regularizer=None,
    weight_decay=0.001,
    reuse=False,
    scope=None,
    dropout_prob=None,
    b_norm_finish=False,
    b_norm_decay_finish=0.9,
    verbose=False,
):
    """A decoding network which maps points from the latent space back onto the data space.
    """
    if verbose:
        print("Building Decoder")

    n_layers = len(layer_sizes)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

    if n_layers < 2:
        raise ValueError("For an FC decoder with single a layer use simpler code.")

    for i in range(0, n_layers - 1):
        name = "decoder_fc_" + str(i)
        scope_i = expand_scope_by_name(scope, name)

        if i == 0:
            layer = latent_signal

        layer = fully_connected_l(
            layer,
            layer_sizes[i],
            activation="linear",
            weights_init="xavier",
            name=name,
            regularizer=regularizer,
            weight_decay=weight_decay,
            reuse=reuse,
            scope=scope_i,
        )

        if verbose:
            print(
                (
                    name,
                    "FC params = ",
                    np.prod(layer.W.get_shape().as_list())
                    + np.prod(layer.b.get_shape().as_list()),
                )
            )

        if b_norm:
            name += "_bnorm"
            scope_i = expand_scope_by_name(scope, name)
            layer = batch_normalization_l(
                layer, decay=b_norm_decay, name=name, reuse=reuse, scope=scope_i
            )
            if verbose:
                print(
                    (
                        "bnorm params = ",
                        np.prod(layer.beta.get_shape().as_list())
                        + np.prod(layer.gamma.get_shape().as_list()),
                    )
                )

        if non_linearity is not None:
            layer = non_linearity(layer)

        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

        if verbose:
            print(layer)
            print(("output size:", np.prod(layer.get_shape().as_list()[1:]), "\n"))

    # Last decoding layer never has a non-linearity.
    name = "decoder_fc_" + str(n_layers - 1)
    scope_i = expand_scope_by_name(scope, name)
    layer = fully_connected_l(
        layer,
        layer_sizes[n_layers - 1],
        activation="linear",
        weights_init="xavier",
        name=name,
        regularizer=regularizer,
        weight_decay=weight_decay,
        reuse=reuse,
        scope=scope_i,
    )
    if verbose:
        print(
            (
                name,
                "FC params = ",
                np.prod(layer.W.get_shape().as_list())
                + np.prod(layer.b.get_shape().as_list()),
            )
        )

    if b_norm_finish:
        name += "_bnorm"
        scope_i = expand_scope_by_name(scope, name)
        layer = batch_normalization_l(
            layer, decay=b_norm_decay_finish, name=name, reuse=reuse, scope=scope_i
        )
        if verbose:
            print(
                (
                    "bnorm params = ",
                    np.prod(layer.beta.get_shape().as_list())
                    + np.prod(layer.gamma.get_shape().as_list()),
                )
            )

    if verbose:
        print(layer)
        print(("output size:", np.prod(layer.get_shape().as_list()[1:]), "\n"))

    return layer



def expand_scope_by_name(scope, name):
    """ expand tf scope by given name.
    """

    if isinstance(scope, str):
        scope += "/" + name
        return scope

    if scope is not None:
        return scope.name + "/" + name
    else:
        return scope


def replicate_parameter_for_all_layers(parameter, n_layers):
    if parameter is not None and len(parameter) != n_layers:
        if len(parameter) != 1:
            raise ValueError()
        parameter = np.array(parameter)
        parameter = parameter.repeat(n_layers).tolist()
    return parameter

def Imgcoor2Pcloud_tf(img, coor, is_demo='ITOP', init_z=None):
    """
    :param img: shape = [B * Height, Width]
    :param coor: shape= [B* J *2]  axis x,y
    :param is_demo: Dataset type
    :param init_z: for caculate the initial poses
    :return:
    """
    coor = tf.cast(coor, tf.int32)
    c = 0.0035
    frames = img.get_shape()[0].value
    assis = np.arange(frames)
    assis = np.tile(assis[:, np.newaxis, np.newaxis], [1, 15, 1])
    assis = tf.constant(assis, tf.int32)
    tmp_coor = tf.concat([assis, coor], axis=-1)

    if init_z is None:
        split1, split2, split3 = tf.split(tmp_coor, [1,1,1], axis=2)
        img_coor = tf.concat([split1, split3, split2], axis=2)
        init_z = tf.expand_dims(tf.gather_nd(img, img_coor),axis=2) # shape=[B*J]
    else:
        init_z = tf.expand_dims(init_z, axis=2)
    coor = tf.expand_dims(tf.cast(coor, tf.float32), axis=2)
    if is_demo == 'ITOP':
        pcloud = [(coor[..., 0] - 160) * c * init_z, (120 - coor[..., 1]) * c * init_z, init_z]
        pcloud = tf.concat(pcloud, axis=2)
    elif is_demo == 'demo':
        pcloud = [(0.0083 * coor[..., 0] - 1.2040) * init_z, (-0.0091 * coor[..., 1] + 1.0447) * init_z, init_z]
        pcloud = tf.concat(pcloud, axis=2)
    elif is_demo == 'NTU':
        pcloud = [(0.0027 * coor[..., 0] - 0.7032) * init_z, (-0.0027 * coor[..., 1] + 0.5673) * init_z, init_z]
        pcloud = tf.concat(pcloud, axis=2)
    return pcloud

def GetBatch2DBBoxSet_tf(pose_2d, setting):
    # sample_pixel = 8
    bbox0 = pose_2d[..., 0:1] - setting.sample_pixel
    bbox1 = pose_2d[..., 0:1] + setting.sample_pixel
    bbox2 = pose_2d[..., 1:2] - setting.sample_pixel
    bbox3 = pose_2d[..., 1:2] + setting.sample_pixel
    bbox_set = tf.concat([bbox0, bbox1, bbox2, bbox3], axis=2)
    return bbox_set

def Getcorebox_tf(joint_root,setting):
    bbox0 = joint_root[:, 0:1] - setting.zsample_pixel
    bbox1 = joint_root[:, 0:1] + setting.zsample_pixel
    bbox2 = joint_root[:, 1:2] - setting.zsample_pixel
    bbox3 = joint_root[:, 1:2] + setting.zsample_pixel
    bbox_set = tf.concat([bbox0, bbox1, bbox2, bbox3], axis=1)
    return bbox_set

def Dmap_Patch2Pcloud_tf(img, bbox_set, side_len, is_demo='ITOP'):
    c = tf.constant(0.0035, tf.float32)
    # side_len= int(16)
    xmin = tf.cast(bbox_set[..., 0], tf.int32)
    xmax = tf.cast(bbox_set[..., 1], tf.int32)
    ymin = tf.cast(bbox_set[..., 2], tf.int32)
    ymax = tf.cast(bbox_set[..., 3], tf.int32)

    for i in range(xmin.get_shape()[0].value):
        for j in range(xmin.get_shape()[1].value):
            fir_tmp = tf.reshape(tf.tile(tf.expand_dims(tf.keras.backend.arange(xmin[i, j], xmax[i, j]), 1),
                              [1, side_len]), [1, 1, -1])

            sec_tmp = tf.reshape(tf.tile(tf.keras.backend.arange(ymin[i, j], ymax[i, j]),
                              [side_len]), [1, 1, -1])
            index = tf.concat([sec_tmp[0, 0, :, tf.newaxis], fir_tmp[0, 0, :, tf.newaxis]], axis=1)
            z_tmp = tf.reshape(tf.gather_nd(img[i, ...], index), [1, 1, -1])
            fir_tmp = tf.cast(fir_tmp, tf.float32)
            sec_tmp = tf.cast(sec_tmp, tf.float32)
            if j == 0:
                fir_s = fir_tmp
                sec_s = sec_tmp
                z_s = z_tmp
            else:
                fir_s = tf.concat([fir_s, fir_tmp], axis=1)
                sec_s = tf.concat([sec_s, sec_tmp], axis=1)
                z_s = tf.concat([z_s, z_tmp], axis=1)
        if i == 0:
            fir = fir_s
            sec = sec_s
            z = z_s
        else:
            fir = tf.concat([fir, fir_s], axis=0)
            sec = tf.concat([sec, sec_s], axis=0)
            z = tf.concat([z, z_s], axis=0)
    if is_demo == 'NTU':
        pcloud_x = tf.expand_dims((0.0027 * fir - 0.7032) * z, axis=3)
        pcloud_y = tf.expand_dims((-0.0027 * sec + 0.5673) * z, axis=3)
        pcloud_z = tf.expand_dims(z, axis=3)
        pcloud = tf.concat([pcloud_x, pcloud_y, pcloud_z], axis=3)
    if is_demo == 'ITOP':
        pcloud_x = tf.expand_dims((fir - 160) * c * z, axis=3)
        pcloud_y = tf.expand_dims((120 - sec) * c * z, axis=3)
        pcloud_z = tf.expand_dims(z, axis=3)
        pcloud = tf.concat([pcloud_x, pcloud_y, pcloud_z], axis=3)
    return pcloud

def Dmap2Pcloud_tf(img, is_demo='ITOP'):
    c = tf.constant(0.0035, tf.float32)
    height = img.get_shape()[1].value
    width = img.get_shape()[2].value
    for i in range(img.get_shape()[0].value):
        fir_tmp = tf.reshape(tf.tile(tf.expand_dims(tf.keras.backend.arange(0, width), 1),
                              [1, height]), [1, -1])

        sec_tmp = tf.reshape(tf.tile(tf.keras.backend.arange(0, height),
                              [width]), [1, -1])
        index = tf.concat([sec_tmp[0, :, tf.newaxis], fir_tmp[0, :, tf.newaxis]], axis=1)
        z_tmp = tf.reshape(tf.gather_nd(img[i, ...], index), [1, -1])
        fir_tmp = tf.cast(fir_tmp, tf.float32)
        sec_tmp = tf.cast(sec_tmp, tf.float32)
        if i == 0:
            fir = fir_tmp
            sec = sec_tmp
            z = z_tmp
        else:
            fir = tf.concat([fir, fir_tmp], axis=0)
            sec = tf.concat([sec, sec_tmp], axis=0)
            z = tf.concat([z, z_tmp], axis=0)
    if is_demo == 'NTU':
        pcloud_x = tf.expand_dims((0.0027 * fir - 0.7032) * z, axis=2)
        pcloud_y = tf.expand_dims((-0.0027 * sec + 0.5673) * z, axis=2)
        pcloud_z = tf.expand_dims(z, axis=2)
        pcloud = tf.concat([pcloud_x, pcloud_y, pcloud_z], axis=2)
    if is_demo == 'ITOP':
        pcloud_x = tf.expand_dims((fir - 160) * c * z, axis=2)
        pcloud_y = tf.expand_dims((120 - sec) * c * z, axis=2)
        pcloud_z = tf.expand_dims(z, axis=2)
        pcloud = tf.concat([pcloud_x, pcloud_y, pcloud_z], axis=2)
    return pcloud

def GetPatchWeight_tf(segmentation, side_len, bbox_set):
    xmin = tf.cast(bbox_set[..., 0], tf.int32)
    xmax = tf.cast(bbox_set[..., 1], tf.int32)
    ymin = tf.cast(bbox_set[..., 2], tf.int32)
    ymax = tf.cast(bbox_set[..., 3], tf.int32)

    for i in range(xmin.get_shape()[0].value):
        for j in range(xmin.get_shape()[1].value):
            fir_tmp = tf.reshape(tf.tile(tf.expand_dims(tf.keras.backend.arange(xmin[i, j], xmax[i, j]), 1),
                                         [1, side_len]), [1, 1, -1])
            sec_tmp = tf.reshape(tf.tile(tf.keras.backend.arange(ymin[i, j], ymax[i, j]),
                                         [side_len]), [1, 1, -1])
            index = tf.concat([sec_tmp[0, 0, :, tf.newaxis], fir_tmp[0, 0, :, tf.newaxis]], axis=1)
            weight_tmp = tf.reshape(tf.gather_nd(segmentation[i, ...], index), [1, 1, -1])
            if j == 0:
                weight_s = weight_tmp
            else:
                weight_s = tf.concat([weight_s, weight_tmp], axis=1)
        if i == 0:
            weight = weight_s
        else:
            weight = tf.concat([weight, weight_s], axis=0)
    return weight


def GetEveryPcloud_tf(depth_maps, segmentation, bbox_set, setting, is_demo='ITOP'):
    result = Dmap_Patch2Pcloud_tf(depth_maps, bbox_set, int(setting.sample_pixel*2), is_demo=is_demo)
    result_weight = GetPatchWeight_tf(segmentation, int(setting.sample_pixel*2), bbox_set)
    return result, result_weight

def GetRootPcloud_tf(depth_maps, root_box, setting, is_demo='ITOP'):
    length = int(setting.zsample_pixel*2)
    tmp_box = tf.expand_dims(root_box, 1)
    pcloud = Dmap_Patch2Pcloud_tf(depth_maps, tmp_box, length, is_demo=is_demo)
    pcloud = tf.squeeze(pcloud, axis=1)
    return pcloud


def PcloudNomalization_tf(pcloud, bbox, num):
    # pcloud : tf tensor [B*J*N*3]
    # bbox: tf tensor [B*J*6]
    xmean = bbox[..., 0]  #batch_size * joint_num
    ymean = bbox[..., 1]
    zmean = bbox[..., 2]
    xlength = bbox[..., 3]
    ylength = bbox[..., 4]
    zlength = bbox[..., 5]

    new_pcloud0 = tf.expand_dims((pcloud[..., 0] - tf.tile(tf.expand_dims(xmean, axis=2), (1, 1, num))) / tf.tile(
        tf.expand_dims(xlength+1e-5, axis=2), (1, 1, num)), axis=-1)
    new_pcloud1 = tf.expand_dims((pcloud[..., 1] - tf.tile(tf.expand_dims(ymean, axis=2), (1, 1, num))) / tf.tile(
        tf.expand_dims(ylength+1e-5, axis=2), (1, 1, num)), axis=-1)
    new_pcloud2 = tf.expand_dims((pcloud[..., 2] - tf.tile(tf.expand_dims(zmean, axis=2), (1, 1, num))) / tf.tile(
        tf.expand_dims(zlength+1e-5, axis=2), (1, 1, num)), axis=-1)
    new_pcloud = tf.concat([new_pcloud0, new_pcloud1, new_pcloud2], axis=-1)
    return new_pcloud, bbox

def PcloudWholeNomalization_tf(pcloud, bbox, num):
    # pcloud : tf tensor [B*N*3]
    # bbox: tf tensor [B*J*6]
    xmean = bbox[:, 0, 0]  #batch_size * joint_num
    ymean = bbox[:, 0, 1]
    zmean = bbox[:, 0, 2]
    xlength = bbox[:, 0, 3]
    ylength = bbox[:, 0, 4]
    zlength = bbox[:, 0, 5]

    new_pcloud0 = tf.expand_dims((pcloud[..., 0] - tf.tile(tf.expand_dims(xmean, axis=1), (1, num))) / tf.tile(
        tf.expand_dims(xlength+1e-5, axis=1), (1, num)), axis=-1)
    new_pcloud1 = tf.expand_dims((pcloud[..., 1] - tf.tile(tf.expand_dims(ymean, axis=1), (1, num))) / tf.tile(
        tf.expand_dims(ylength+1e-5, axis=1), (1, num)), axis=-1)
    new_pcloud2 = tf.expand_dims((pcloud[..., 2] - tf.tile(tf.expand_dims(zmean, axis=1), (1, num))) / tf.tile(
        tf.expand_dims(zlength+1e-5, axis=1), (1, num)), axis=-1)
    new_pcloud = tf.concat([new_pcloud0, new_pcloud1, new_pcloud2], axis=-1)
    return new_pcloud, bbox

if __name__ == '__main__':
    pose_2d = tf.ones([10,15,2], tf.float32)
    bbox_set = GetBatch2DBBoxSet_tf(pose_2d)
    img = tf.ones([10, 240, 320], tf.float32)
    pcloud = Dmap_Patch2Pcloud_tf(img, bbox_set)
    print(pcloud.shape)



