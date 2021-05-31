####################################
#
#  MLB Project 2021
#
#  Module: Classification with Convolutional Neural Networks (CNN)
#  File: load_dataset
#
#  Created on May 30, 2021
#  All rights reserved to João Saraiva and Débora Albuquerque
#
####################################

import json
from datetime import timedelta, datetime

from src.feature_extraction import get_patient_hrv_features, get_patient_hrv_baseline_features
from src.feature_extraction.io import __read_crisis_nni, __read_baseline_nni
import numpy as np

data_path = 'data'


def normalise_feats(features, norm='minmax'):
    if norm == 'stand':
        return (features - features.mean()) / (features.std())

    elif norm == 'minmax':
        print(features)
        return (features - features.min()) / (features.max() - features.min())


def normalise_feats_baseline(features, baseline, norm='minmax'):
    if norm == 'stand':
        return (features - baseline.mean()) / (baseline.std())

    elif norm == 'minmax':
        print(features)
        return (features - baseline.min()) / (baseline.max() - baseline.min())


def prepare_dataset(patient, state, feature_inputs, test_crisis, before_onset_minutes=15, crisis_minutes=2, dimensions=1,
                    n_baseline_tests=1, raw=False, raw_input_segment=60):
    # get metadata
    with open(data_path + '/patients.json') as metadata_file:
        metadata = json.load(metadata_file)
        patient_crises_metadata = metadata['patients'][str(patient)]['crises']

    train_inputs, train_targets, test_inputs, test_targets = [], [], [], []

    input_shape = None
    features_labels = None
    n_samples_segment = None

    if raw:
        baseline_features = __read_baseline_nni(patient, state)
        baseline_features = normalise_feats(baseline_features)
    else:
        baseline_features = get_patient_hrv_baseline_features(patient, state)
        baseline_features = normalise_feats(baseline_features, norm="stand")

    #for crisis in patient_crises_metadata.keys():
    for crisis in ("1", "2", "3", "4"):
        # features
        if raw:
            crisis_features = __read_crisis_nni(patient, crisis)
            crisis_features = normalise_feats_baseline(crisis_features, baseline_features, norm="stand")
        else:
            crisis_features = get_patient_hrv_features(patient, crisis)
            crisis_features = normalise_feats(crisis_features, norm="stand")

        # timepoints
        ictal_onset = datetime.strptime(metadata['patients'][str(patient)]['crises'][crisis]['onset'], "%d/%m/%Y %H:%M:%S")
        preictal_onset = ictal_onset - timedelta(minutes=before_onset_minutes)
        interictal_after_onset = ictal_onset + timedelta(minutes=crisis_minutes)

        # timepoints in sample numbers
        n_samples_interictal_before = len(crisis_features[crisis_features.index < preictal_onset])
        n_samples_preictal = len(crisis_features[crisis_features.index < ictal_onset]) - n_samples_interictal_before
        n_samples_ictal = len(crisis_features[crisis_features.index < interictal_after_onset]) - n_samples_interictal_before - n_samples_preictal
        n_samples_interictal_after = len(crisis_features) - n_samples_interictal_before - n_samples_preictal - n_samples_ictal

        if input_shape is None:
            n_samples_segment = n_samples_preictal + n_samples_ictal
            input_shape = n_samples_segment
            if feature_inputs == 'all':
                features_labels = crisis_features.columns.tolist()
        else:
            n_samples_segment = input_shape

        # Filter features wanted
        if feature_inputs != 'all' and not raw:
            crisis_features = crisis_features.filter(items=feature_inputs)

        # Subset 'seizure present':
        input_crisis = crisis_features.to_numpy().tolist()[
                       n_samples_interictal_before: n_samples_interictal_before + n_samples_segment]

        if int(crisis) == test_crisis:
            test_targets.append(1)
            test_inputs.append(input_crisis)
        else:
            train_targets.append(1)
            train_inputs.append(input_crisis)



    # Subset baseline:

    center = int(len(baseline_features.values) / 2)
    dev = int(n_samples_segment / 2)

    # Filter features wanted
    if feature_inputs != 'all' and not raw:
        baseline_features = baseline_features.filter(items=feature_inputs)

    # to train
    n_train_baseline_inputs = len(train_inputs)
    for i in range(n_train_baseline_inputs):
        train_targets.append(0)
    train_inputs += np.split(baseline_features.values[0 : n_samples_segment * n_train_baseline_inputs], n_train_baseline_inputs)

    # to train
    if n_baseline_tests == 1:  # create only 1 test from baseline signal
        input_baseline = baseline_features.to_numpy().tolist()[  # gets a piece in the middle
                         center - dev : center + dev + 1]
        test_targets.append(0)
        test_inputs.append(input_baseline)
    else:
        for i in range(n_baseline_tests):
            test_targets.append(0)

        test_inputs += np.split(baseline_features.values[center : center + n_samples_segment * n_baseline_tests], n_baseline_tests)

    train_inputs = np.array(train_inputs)
    train_targets = np.array(train_targets)
    test_inputs = np.array(test_inputs)
    test_targets = np.array(test_targets)

    if raw_input_segment is not None:  # split into more inputs
        sf = metadata['sampling_frequency']  # Hz
        n_samples_segment = sf * raw_input_segment  # Hz * seconds = number of samples
        div = int(len(train_inputs[0]) / n_samples_segment)

        # train
        res_inputs, res_targets = [], []
        t = 1
        for input in train_inputs:
            res_inputs += np.split(input, div)
            res_targets += [t, ] * div
            if t == 1: t = 0
            else: t = 1
        train_inputs = np.array(res_inputs)
        train_targets = np.array(res_targets)

        # test
        res_inputs, res_targets = [], []
        t = 1
        for input in test_inputs:
            res_inputs += np.split(input, div)
            res_targets += [t, ] * div
            t = 0
        test_inputs = np.array(res_inputs)
        test_targets = np.array(res_targets)

    if dimensions == 1:  # for 1D convolutional networks

        input_shape = train_inputs[:, :, 0].shape
        n_features = train_inputs.shape[2]

        dataset = dict()
        for i in range(n_features):
            dataset[features_labels[i]] = train_inputs[:, :, i]
        train_inputs = dataset

        dataset = dict()
        for i in range(n_features):
            dataset[features_labels[i]] = train_targets
        train_targets = dataset

        dataset = dict()
        for i in range(n_features):
            dataset[features_labels[i]] = test_inputs[:, :, i]
        test_inputs = dataset

        dataset = dict()
        for i in range(n_features):
            dataset[features_labels[i]] = test_targets
        test_targets = dataset

        print("\nTrain input shape:", train_inputs[features_labels[0]].shape)
        print("Train target shape:", train_targets[features_labels[0]].shape)
        print("Test input shape:", test_inputs[features_labels[0]].shape)
        print("Test target shape:", test_targets[features_labels[0]].shape)

        return train_inputs, train_targets, test_inputs, test_targets, input_shape, features_labels

    else:
        print("\nTrain input shape:", train_inputs.shape)
        print("Train target shape:", train_targets.shape)
        print("Test input shape:", test_inputs.shape)
        print("Test target shape:", test_targets.shape)

        return train_inputs, train_targets, test_inputs, test_targets, (train_inputs.shape[1], train_inputs.shape[2])


