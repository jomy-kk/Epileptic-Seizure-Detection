####################################
#
#  MLB Project 2021
#
#  Module: Classification with Convolutional Neural Networks (CNN)
#  File: init
#
#  Created on May 30, 2021
#  All rights reserved to João Saraiva and Débora Albuquerque
#
####################################

from prediction.load_dataset import *
from prediction.create_model import *
from prediction.fit_evaluate import *


# Load the dataset
train_inputs, train_targets, test_inputs, test_targets, input_shape = \
    prepare_dataset(patient=102, state="awake", test_crisis=1,
                    #feature_inputs='all',
                    #feature_inputs=['lf_hf', 'rmssd', 'csv', 's', 'cosen'],
                    feature_inputs=['sdnn', 'rmssd', 'mean', 'maxhr', 'csi', 'csv', 'car', 'sampen', 'cosen'],
                    dimensions=2, n_baseline_tests=4,
                    before_onset_minutes=15, crisis_minutes=2,
                    raw=False, raw_input_segment=None)  # seconds

print("\nCNN input shape:", input_shape)

# Create model
model = create_model(name="Developer", dimension='1d',
                     input_shape=input_shape, filters=[16, 18], kernel_sizes=[7, 7],
                     activation_function='relu',
                     padding='causal', stride=1,
                     output_size=2, output_activation_function='softmax',
                     pool_size=[5, 5], pool_type='max',
                     regularization=None, lambda_reg=0.1,
                     dropout_visible=0, dropout_hidden=[0, 0.2], dropout_fc=[0, 0],
                     fully_connected_layer_size=[128, ],
                     batch_normalization=True
                     )


# Fit and evaluate model
fit_and_evaluate_model(model, train_inputs, train_targets, test_inputs, test_targets,
                       loss_function='sparse_categorical_crossentropy', optimizer='adadelta', patience=20,
                       validation_split=0.2, batch_size=1)

print(model.predict(test_inputs, verbose=1))

exit(0)


