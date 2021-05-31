####################################
#
#  MLB Project 2021
#
#  Module: Classification with Convolutional Neural Networks (CNN)
#  File: create_model
#
#  Created on May 30, 2021
#  All rights reserved to João Saraiva and Débora Albuquerque
#
####################################

import tensorflow as tf


def create_model(name, dimension,
                 input_shape, filters, kernel_sizes, activation_function='sigmoid', padding='same', dilation_rate=0, stride=1,
                 output_size=2, output_activation_function='sigmoid',
                 pool_size=2, pool_type='max', regularization=None, lambda_reg=0.01,
                 dropout_visible=0, dropout_hidden=[], dropout_fc=[],
                 fully_connected_layer_size=[], batch_normalization=False
                 ):
    """
    Receives a n-array of integers and returns a sequential model named after parameter name, with n hidden layers,
    each one of size indicated in each element of the array.
    The activation function in each hidden layer can be set by the activation_function parameter.
    L1 regularization can be added to all hidden layers with the coefficient given by lambda_reg,
    if the regularization parameter is set to 'l1'.
    L2 regularization can be added to all hidden layers with the coefficient given by lambda_reg,
    if the regularization parameter is set to 'l2'.
    Dropping out some input layer units can be done by setting the percentage with parameter dropout_visible.
    Dropping out some units in each hidden layer can be done by setting the percentage with parameter dropout.
    """

    if dimension == '2d':
        return create_2d_model(name=name,
                               input_shape=input_shape, filters_shape=filters, kernel_sizes=kernel_sizes, activation_function=activation_function, padding=padding, stride=stride,
                               output_size=output_size, output_activation_function=output_activation_function,
                               pool_size=pool_size, regularization=regularization, lambda_reg=lambda_reg,
                               dropout_visible=dropout_visible, dropout_hidden=dropout_hidden, dropout_fc=dropout_fc,
                               fully_connected_layer_size=fully_connected_layer_size)

    if dimension == '1d':
        return create_1d_model(name=name,
                               input_shape=input_shape, filters_shape=filters, kernel_sizes=kernel_sizes, activation_function=activation_function, padding=padding, dilation_rate=dilation_rate, stride=stride,
                               output_size=output_size, output_activation_function=output_activation_function,
                               pool_size=pool_size, pool_type=pool_type, regularization=regularization, lambda_reg=lambda_reg,
                               dropout_visible=dropout_visible, dropout_hidden=dropout_hidden, dropout_fc=dropout_fc,
                               fully_connected_layer_size=fully_connected_layer_size, batch_normalization=batch_normalization)


def create_2d_model(name,
                    input_shape, filters_shape, kernel_sizes, activation_function, padding, stride,
                    output_size, output_activation_function,
                    pool_size=2, regularization=None, lambda_reg=0.01,
                    dropout_visible=0, dropout_hidden=0, dropout_fc=0,
                    fully_connected_layer_size=0):

    model = tf.keras.Sequential(name=name)
    model.add(tf.keras.layers.InputLayer(input_shape, name='Input'))

    if dropout_visible > 0:
        model.add(tf.keras.layers.Dropout(dropout_visible, name='Dropout visible'))

    for i in range(0, len(filters_shape)):
        if regularization is None:
            model.add(tf.keras.layers.Conv2D(filters=filters_shape[i], kernel_size=kernel_sizes[i],
                                             activation=activation_function, padding=padding,
                                            ))

        if regularization == 'l1':
            model.add(tf.keras.layers.Conv2D(filters=filters_shape[i], kernel_size=kernel_sizes[i],
                                             activation=activation_function, padding=padding,
                                             kernel_regularizer=tf.keras.regularizers.l1(lambda_reg),
                                             ))

        if regularization == 'l2':
            model.add(tf.keras.layers.Conv2D(filters=filters_shape[i], kernel_size=kernel_sizes[i],
                                             activation=activation_function, padding=padding,
                                             kernel_regularizer=tf.keras.regularizers.l2(lambda_reg),
                                             ))

        if dropout_hidden > 0:
            model.add(tf.keras.layers.Dropout(dropout_hidden))

    model.add(tf.keras.layers.MaxPool2D(pool_size=pool_size, name='Max pooling'))

    if dropout_fc > 0:
        model.add(tf.keras.layers.Dropout(dropout_fc))

    model.add(tf.keras.layers.Flatten(name='Flattening'))

    if fully_connected_layer_size > 0:
        model.add(tf.keras.layers.Dense(units=fully_connected_layer_size, activation=activation_function, name="Fully connected layer"))

    if dropout_fc > 0:
        model.add(tf.keras.layers.Dropout(dropout_fc, name="Dropout of fully connected layer"))

    model.add(tf.keras.layers.Dense(output_size, activation=output_activation_function, name='Output'))

    return model


def create_1d_model(name,
                    input_shape, filters_shape, kernel_sizes, activation_function, padding, dilation_rate, stride,
                    output_size, output_activation_function,
                    pool_size, pool_type='max', regularization=None, lambda_reg=0.01,
                    dropout_visible=0, dropout_hidden=[], dropout_fc=[],
                    fully_connected_layer_size=[], batch_normalization=False):

    model = tf.keras.Sequential(name=name)
    model.add(tf.keras.layers.InputLayer(input_shape, name='Input'))

    if dropout_visible > 0:
        model.add(tf.keras.layers.Dropout(dropout_visible, name='DropoutVisible'))

    for i in range(0, len(filters_shape)):
        if regularization is None:
            model.add(tf.keras.layers.Conv1D(filters=filters_shape[i], kernel_size=kernel_sizes[i],
                                             padding=padding, strides=stride,
                                             activation=activation_function,
                                             name='Hidden'+str(i)))

        if regularization == 'l1':
            model.add(tf.keras.layers.Conv1D(filters=filters_shape[i], kernel_size=kernel_sizes[i],
                                             padding=padding, strides=stride,
                                             activation=activation_function,
                                             kernel_regularizer=tf.keras.regularizers.l1(lambda_reg),
                                             name='Hidden'+str(i)+'RegL1'+str(lambda_reg)))

        if regularization == 'l2':
            model.add(tf.keras.layers.Conv1D(filters=filters_shape[i], kernel_size=kernel_sizes[i],
                                             padding=padding, strides=stride,
                                             activation=activation_function,
                                             kernel_regularizer=tf.keras.regularizers.l2(lambda_reg),
                                             name='Hidden'+str(i)+'RegL2'+str(lambda_reg)))

        if batch_normalization is True and i == 0:
            model.add(tf.keras.layers.BatchNormalization())

        if pool_type == 'max':
            model.add(tf.keras.layers.MaxPooling1D(pool_size=pool_size[i], name='MaxPooling'+str(i)))
        if pool_type == 'avg':
            model.add(tf.keras.layers.AvgPool1D(pool_size=pool_size[i], name='AvgPooling'+str(i)))

        if dropout_hidden is not None and dropout_hidden[i] > 0:
            model.add(tf.keras.layers.Dropout(dropout_hidden[i], name='DropoutHidden'+str(i)))


    model.add(tf.keras.layers.Flatten(name='Flattening'))

    for i in range(0, len(fully_connected_layer_size)):
        model.add(tf.keras.layers.Dense(units=fully_connected_layer_size[i], activation=activation_function, name="FullyConnected"+str(i)))
        if dropout_fc is not None and dropout_fc[i] > 0:
            model.add(tf.keras.layers.Dropout(dropout_fc[i], name="DropoutFullyConnected"+str(i)))

    model.add(tf.keras.layers.Dense(output_size, activation=output_activation_function, name='Output'))

    return model
