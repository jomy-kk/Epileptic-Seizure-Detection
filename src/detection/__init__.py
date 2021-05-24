
import json
import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
from _datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

from src.feature_selection import features
from src.feature_extraction import get_patient_hrv_features, get_patient_hrv_baseline_features
data_path = './data'

#Divide crisis patient data into pre-ictal and ictal

def create_dataset (features:dict):
    with open(data_path + '/patients.json') as metadata_file:
        metadata = json.load(metadata_file)
    #dataset_X, dataset_Y = {}, {}
    dataset_x, dataset_y = [], []
    onsets = {}
    feature_label=['rms_sd', 'sd_nn', 'mean_nn', 'lf_pwr', 'hf_pwr', 'kfd', 'sampen',
       'cosen', 'lf', 'hf', 'lf_hf', 'hf_lf', 'katz_fractal_dim', 'sd1', 'sd2',
       'csi', 'csv', 's', 'rec', 'det', 'lmax', 'nn50', 'pnn50', 'sdnn',
       'rmssd', 'mean', 'var', 'hr', 'maxhr']

    before_onset_minutes = int(input("How many minutes of pre-ictal phase before crisis onset?"))
    crisis_minutes = int(input("How many minutes does the crisis last?"))

    for patient in features.keys():
        onsets[patient] = {}
        #dataset_X[patient] = {}
        #dataset_Y[patient] = {}
        for crisis in features[patient].keys():
            print(features[patient][crisis].keys())
            if any(feature in feature_label for feature in features[patient][crisis].keys()):
                onset = metadata['patients'][str(patient)]['crises'][str(crisis)]['onset']
                onset = datetime.strptime(onset, "%d/%m/%Y %H:%M:%S")
                onsets[patient][crisis] = onset

                ictal_onset = onset
                preictal_onset = ictal_onset - timedelta(minutes=before_onset_minutes)
                interictal_after_onset = ictal_onset + timedelta(minutes=crisis_minutes)

                index = features[patient][crisis].index

                n_samples_interictal_before = len(features[patient][crisis][index < preictal_onset])
                n_samples_preictal = len(features[patient][crisis][index < ictal_onset]) - n_samples_interictal_before

                n_samples_ictal = len(features[patient][crisis][index < interictal_after_onset]) - n_samples_interictal_before - n_samples_preictal
                n_samples_interictal_after = len(features[patient][crisis]) - n_samples_interictal_before - n_samples_preictal - n_samples_ictal

                assert n_samples_interictal_before + n_samples_preictal + n_samples_ictal + n_samples_interictal_after \
                       == len(features[patient][crisis])

                print("N_samples_interictal_before", n_samples_interictal_before)
                print("N_samples_preictal", n_samples_preictal)
                print("N_samples_ictal", n_samples_ictal)
                print("N_samples_interictal_after", n_samples_interictal_after)

                print(" ")

                # Subset 'interictal phase before onset':
                dataset_y += [0] * n_samples_interictal_before
                dataset_x += features[patient][crisis].to_numpy().tolist()[0:n_samples_interictal_before]

                # Subset 'pre-ital phase':
                dataset_y += [1] * n_samples_preictal
                dataset_x += features[patient][crisis].to_numpy().tolist()[n_samples_interictal_before:(n_samples_interictal_before + n_samples_preictal)]

                # Subset 'ictal phase':
                dataset_y += [2] * n_samples_ictal
                dataset_x += features[patient][crisis].to_numpy().tolist()[
                             (n_samples_interictal_before + n_samples_preictal):(n_samples_interictal_before + n_samples_preictal + n_samples_ictal)]

                # Subset 'interictal phase after onset':
                dataset_y += [0] * n_samples_interictal_after
                dataset_x += features[patient][crisis].to_numpy().tolist()[(n_samples_interictal_before + n_samples_preictal + n_samples_ictal):]
                assert len(dataset_y) == len(dataset_x)

                #dataset_X[patient][crisis] = np.array(dataset_x, dtype=np.float64)
                #dataset_Y[patient][crisis] = np.array(dataset_y, dtype=np.float64)


    print("Created dataset with", onsets.keys())
    return dataset_x, dataset_y

def save_best_params(grid_best_params):
    try:
        file_path = data_path + '/best_svm_parameters.json'
        file = open(file_path, 'w')
        json.dump(grid_best_params, file)
        print("Best parameters saved to " + file_path + " successfully.")
    except IOError:
        print("Cannot write best parameters to file. Save failed.")

def read_best_params():
    try:  # try to read a previously computed JSON file with best parameters
        file_path = data_path + '/best_svm_parameters.json'
        file = open(file_path, 'r')
        grid = json.load(file)

        print("Data from " + file_path + " was retrieved.")

        print("Best values: ", grid)

        return grid
    except IOError:  # Parameters not found, return None
        return None


def train_test_svm (features):
    dataset_x, dataset_y = create_dataset(features)
    test_size = float(input("What size do you want the test sample to have (0-1)?"))
    x_train, x_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size=test_size)

    def hyperparameter_tuning (x_train, y_train, x_test, y_test):
        # defining parameter range
        param_grid = {'C': np.arange(0.1,1,0.1).tolist() + np.arange(1,10,1).tolist() + np.arange(10,100,10).tolist() + np.arange(100,1000,100).tolist(),
                      'gamma': np.arange(1, 0.1, -0.1).tolist() + np.arange(0.1,0.01,-0.01).tolist() + np.arange(0.01, 0.001,-0.001).tolist() + np.arange(0.001,0.0001,-0.0001).tolist(),
                      'kernel': ['rbf', 'poly', 'linear','sigmoid'], 'degree': np.arange(1,5,1).tolist()}

        grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
        grid_best_params = grid

        # fitting the model for grid search
        grid.fit(x_train, y_train)

        # predicting
        grid_predictions = grid.predict(x_test)

        # print classification report
        print(classification_report(y_test, grid_predictions))

        return grid_best_params

    # Get best parameters from file. If it doesn't work, calculate best parameters
    parameters = read_best_params()

    if parameters is None or input('Do you want to recalculate best parameters? y/n ').lower() == 'y':
        print("Calculating best parameters.")
        best_params = hyperparameter_tuning(x_train, y_train, x_test, y_test)
        print("Saving best parameters.")
        parameters = {'C': best_params.best_estimator_.C, 'degree': best_params.best_estimator_.degree,
                      'kernel': best_params.best_estimator_.kernel, 'gamma': best_params.best_estimator_.gamma}
        save_best_params(parameters)

    #model implementation
    model = SVC(kernel=parameters['kernel'], gamma=parameters['gamma'], C=parameters['C'],degree=parameters['degree'])
    model = model.fit(x_train, y_train)
    y_prediction= model.predict(x_test)
    print("Y_predicted = ", y_prediction)

    # print classification report
    print(classification_report(y_test,y_prediction))
    return y_prediction


