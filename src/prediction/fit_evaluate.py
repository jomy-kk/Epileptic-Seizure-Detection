####################################
#
#  MLB Project 2021
#
#  Module: Classification with Convolutional Neural Networks (CNN)
#  File: fit_evaluate
#
#  Created on May 30, 2021
#  All rights reserved to João Saraiva and Débora Albuquerque
#
####################################

import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np


def evaluate_model(model, weights, test_inputs, test_targets, show_predictions=False):
    """
    Evaluates a given model with the given weights. Prints the test accuracy and loss.
    Provide the following:
    :param model: A valid already trained classification model, such as a CNN.
    :param weights: Weight vectors of the model after training. Give the best.
    :param test_inputs: Input vectors reserved for testing.
    :param test_targets: Target vectors reserved for testing -- growth truth.
    """
    model.load_weights(weights)
    loss, acc = model.evaluate(test_inputs, test_targets)
    if show_predictions:
        ypred = model.predict(test_inputs)
        print(ypred)
        if ypred[0][1] > 0.5:
            print("Predicted crisis correctly")
        else:
            print("Did not predict crisis correctly")

    return loss, acc


def fit_model(model, train_inputs, train_targets,
                           loss_function='sparse_categorical_crossentropy', optimizer='adam',
                           metrics=('accuracy', ), epochs=10000, batch_size=32, patience=10, validation_split=.2,
                           verbose=True):

    model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)
    if(verbose):
        model.summary()

    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience, verbose=verbose)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('best.h5', monitor='val_accuracy', verbose=verbose, save_best_only=True)
    model_train = model.fit(train_inputs, train_targets, validation_split=validation_split,
                            callbacks=[earlystop, checkpoint], epochs=epochs, batch_size=batch_size, verbose=verbose)

    return model_train


def fit_and_evaluate_model(model, train_inputs, train_targets, test_inputs, test_targets,
                           loss_function='sparse_categorical_crossentropy', optimizer='adam',
                           metrics=('accuracy', ), epochs=10000, batch_size=32, patience=10, validation_split=.2):
    """
    Trains a given model with the given optimization and loss functions for the given number of epochs.
    Stops based on the accuracy metric. An early stop is defined with a convergence patience set by patience.
    A checkpoint is defined to save only the weights from the best model, based on the validation accuracy.
    A percentage of the training examples given by validation_split is used only for validation.
    It plots the training accuracy and loss and the validation accuracy and loss against the number of epochs,
    and evaluates the best model for the given dataset.
    """

    model_train = fit_model(model, train_inputs, train_targets,
                               loss_function=loss_function, optimizer=optimizer,
                               metrics=metrics, epochs=epochs, batch_size=batch_size,
                               patience=patience, validation_split=validation_split)

    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(20, 7))

    loss_ax.set_title('Loss')
    loss_ax.plot(model_train.history['loss'], '-r', label='Train')
    loss_ax.plot(model_train.history['val_loss'], '-g', label='Validation')

    acc_ax.set_title('Accuracy')
    acc_ax.plot(model_train.history['accuracy'], '-r', label='Train')
    acc_ax.plot(model_train.history['val_accuracy'], '-g', label='Validation')

    plt.legend(loc=4)
    plt.show()

    loss, acc = evaluate_model(model, 'best.h5', test_inputs, test_targets)

    print('\nAccuracy: {}'.format(acc))
    print('Loss: {}'.format(loss))


def do_experiment(model, train_inputs, train_targets, test_inputs, test_targets,
                           loss_function='sparse_categorical_crossentropy', optimizer='adam',
                           metrics=('accuracy', ), epochs=10000, batch_size=32, patience=10, validation_split=.2):

    val_accuracies, val_losses, test_accuracies, test_losses = 0, 0, 0, 0

    for i in range(10):
        print("Run", i)
        model_train = fit_model(model, train_inputs, train_targets,
                               loss_function=loss_function, optimizer=optimizer,
                               metrics=metrics, epochs=epochs, batch_size=batch_size,
                               patience=patience, validation_split=validation_split, verbose=False)

        #val_accuracies += np.max(model_train.history['accuracy'])
        #val_losses += np.min(model_train.history['loss'])

        loss, acc = evaluate_model(model, 'best.h5', test_inputs, test_targets, show_predictions=True)
        test_losses += loss
        test_accuracies += acc

    val_accuracies, val_losses, test_accuracies, test_losses =\
        val_accuracies/10, val_losses/10, test_accuracies/10, test_losses/10

    #print('\nValidation Accuracy: {}'.format(val_accuracies))
    #print('Validation Loss: {}'.format(val_losses))
    print('\nTest Accuracy: {}'.format(test_accuracies))
    print('Test Loss: {}'.format(test_losses))
