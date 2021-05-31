
import json
import math
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm
import statistics
import pickle
import matplotlib.pyplot as plt
from _datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score, make_scorer
from sklearn.base import BaseEstimator, RegressorMixin

from src.feature_selection import features
from src.feature_extraction import get_patient_hrv_features, get_patient_hrv_baseline_features
import src.feature_extraction.io

data_path = './data'

# Wrapper for OSM to be used with cross_val_score
class SMWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, model_class, fit_intercept=True):
        self.model_class = model_class
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        if self.fit_intercept:
            X = sm.add_constant(X)
        self.model_ = self.model_class(y, X)
        self.results_ = self.model_.fit()
        return self.results_

    def predict(self, X):
        if self.fit_intercept:
            X = sm.add_constant(X)
        return self.results_.predict(X)

#Divide crisis patient data into pre-ictal and ictal

def create_dataset (features:dict):
    with open(data_path + '/patients.json') as metadata_file:
        metadata = json.load(metadata_file)
    #dataset_X, dataset_Y = {}, {}
    dataset_x, dataset_y = [], []
    onsets = {}
    feature_label=['sampen',
       'cosen', 'lf', 'hf', 'lf_hf', 'hf_lf',  'sd1', 'sd2',
       'csi', 'csv', 's', 'rec', 'det', 'lmax', 'nn50', 'pnn50', 'sdnn',
       'rmssd', 'mean', 'var', 'hr', 'maxhr', 'katz_fractal_dim']

    before_onset_minutes = int(input("How many minutes of pre-ictal phase before crisis onset?")) # 20 min
    crisis_minutes = int(input("How many minutes does the crisis last?")) # 2 min

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
                dataset_y += [1] * n_samples_ictal
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

def save_best_params(grid_best_params, patient:int):
    try:
        file_path = data_path + '/best_svm_parameters' + str(patient) + '.json'
        file = open(file_path, 'w')
        json.dump(grid_best_params, file)
        print("Best parameters saved to " + file_path + " successfully.")
    except IOError:
        print("Cannot write best parameters to file. Save failed.")

def read_best_params(patient:int):
    try:  # try to read a previously computed JSON file with best parameters
        file_path = data_path + '/best_svm_parameters' + str(patient) + '.json'
        file = open(file_path, 'r')
        grid = json.load(file)

        print("Data from " + file_path + " was retrieved.")

        print("Best values: ", grid)

        return grid
    except IOError:  # Parameters not found, return None
        return None

def train_test_svm (features, test_size):
    patient = list(features.keys())[0]
    labels = list(list(list(features.values())[0].values())[0].columns)

    dataset_x, dataset_y = create_dataset(features)
    x_train, x_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size=test_size, stratify=dataset_y)
    ns=len(y_train)
    ns0=y_train.count(0)
    ns1= y_train.count(1)
    ns2 = y_train.count(2)
    def hyperparameter_tuning (x_train, y_train, x_test, y_test, ns=ns, ns0=ns0, ns1=ns1, ns2=ns2):
        # defining parameter range

        print("Ns, ns0, ns1", ns, ns0, ns1)

        param_grid = {'C': [1, 10,20, 30,40,50],
                      'gamma': [0.001, 0.01, 0.1, 1],
                      'kernel': ['rbf'], 'class_weight':[{0: ns/(2*ns0), 1: ns/(2*ns1)}] }

        #param_grid = {'C': [0.1, 1, 10],
                      #'gamma': [0.001, 0.01, 0.1, 1],
                      #'kernel': ['rbf', 'poly', 'linear', 'sigmoid'], 'degree': np.arange(1, 3, 1).tolist(),
                      #'class_weight': [{0: ns / (2 * ns0), 1: ns / (2 * ns1)}]}
        grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3, scoring= 'f1_weighted', cv=10)
        grid_best_params = grid


        # fitting the model for grid search
        grid.fit(x_train, y_train)
        #print('grid cv results:', pd.DataFrame(grid.cv_results_))
        # predicting
        grid_predictions = grid.predict(x_test)

        # print classification report
        print(classification_report(y_test, grid_predictions))

        return grid_best_params

    # def stepwise_regression(x_train, y_train, x_test, labels):
    #     repeat = True
    #
    #     while(repeat):
    #         model = SMWrapper(sm.OLS)
    #         results = model.fit(x_train, y_train)
    #
    #         pvals = results.pvalues[1:]
    #
    #         max_pval_i = np.argmax(pvals)
    #
    #         if pvals[max_pval_i] > 0.05:
    #              # Remove max p-value element for each example i
    #             for i in range(len(x_train)):
    #                 x_train[i].pop(max_pval_i)
    #
    #             for i in range(len(x_test)):
    #                 x_test[i].pop(max_pval_i)
    #
    #             labels.pop(max_pval_i)
    #
    #         else:
    #             repeat = False
    #             print(results.summary())
    #
    #     return x_train, x_test, labels
    #
    # x_train, x_test, labels = stepwise_regression(x_train, y_train, x_test, labels)
    #
    # label_list = src.feature_extraction.io.__read_stepwise(patient)
    # if label_list is None:
    #     label_list = []
    #
    # label_list.append(labels)
    # src.feature_extraction.io.__save_stepwise(label_list, patient)

    #After finding best features
    feature_label = ['sdnn', 'mean', 'var', 'hr', 'lf', 'hf', 'lf_hf', 'hf_lf', 'csi', 'csv', 'rec', 'det', 'sampen', 'cosen']
    #'mean', 'var', 'hr', 'lf', 'hf', 'hf_lf', 'csi', 'csv', 's', 'rec', 'det', 'cosen'

    idx_to_remove = []

    for j in range(len(labels)):
        if labels[j] not in feature_label:
            idx_to_remove.append(j)

    for index in sorted(idx_to_remove, reverse=True):
        for i in range(len(x_train)):
            x_train[i].pop(index)
        for i in range(len(x_test)):
            x_test[i].pop(index)

    labels = feature_label

    # def rfe (x_train, y_train, x_test):
    #
    #     rfe = RFE(estimator=SVC(kernel='linear', C=20), n_features_to_select=19)
    #
    #     # evaluate model
    #     cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    #     n_scores = cross_val_score(rfe, x_train, y_train, scoring='f1_weighted', cv=cv, n_jobs=-1, error_score=np.nan)
    #     # report performance
    #     print('F1_weighted: %.3f (%.3f)' % (statistics.mean(n_scores), statistics.stdev(n_scores)))
    #     # fit RFE
    #     rfe.fit(x_train, y_train)
    #     # summarize all features
    #     for feature_i in range(len(x_train[0])):
    #         # feature_array = []
    #         # for x_point in range(len(x_train)):
    #         #     feature_array[x_point] = x_train[feature_i][x_point]
    #         print('Column: %d, Selected %s, Rank: %.3f' % (feature_i, rfe.support_[feature_i], rfe.ranking_[feature_i]))
    #      # transform the data
    #      #print('Before x:', x_train)
    #     x_train = rfe.transform(x_train)
    #     x_test = rfe.transform(x_test)
    #     #print('After x:', x_train)
    #     return x_train, x_test
    #
    #  #feature selection rfe
    # x_train, x_test = rfe(x_train, y_train, x_test)

    # Get best parameters from file. If it doesn't work, calculate best parameters
    parameters = read_best_params(list(features.keys())[0])

    if parameters is None or input('Do you want to recalculate best parameters? y/n ').lower() == 'y':
        print("Calculating best parameters.")
        best_params = hyperparameter_tuning(x_train, y_train, x_test, y_test)

        print("Saving best parameters.")
        parameters = {'C': best_params.best_estimator_.C, 'degree': best_params.best_estimator_.degree,
                      'kernel': best_params.best_estimator_.kernel, 'gamma': best_params.best_estimator_.gamma}
        save_best_params(parameters, list(features.keys())[0])

    #model implementation

    def classification_report_with_f1_score(y_train, y_prediction):
        print(classification_report(y_train, y_prediction))  # print classification report
        return f1_score(y_train, y_prediction)  # return accuracy score

    model = SVC(kernel=parameters['kernel'], gamma=parameters['gamma'], C=parameters['C'],degree=parameters['degree'])
    model = model.fit(x_train, y_train)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(model, x_train, y_train, scoring=make_scorer(classification_report_with_f1_score), cv=cv,
                               n_jobs=-1, error_score='raise')
    n_scores_mean = statistics.mean(n_scores)
    n_scores_stdev = statistics.stdev(n_scores)
    print('F1: %.3f (%.3f)' % (n_scores_mean, n_scores_stdev))
    y_prediction = model.predict(x_test)
    print("Y_predicted = ", y_prediction)


    # print classification report
    print(classification_report(y_test,y_prediction))
    cm = confusion_matrix(y_test, y_prediction, labels=[1,0])
    print('Confusion matrix : \n', cm)
    tp, fn, fp, tn = confusion_matrix(y_test, y_prediction, labels=[1,0]).reshape(-1)
    print('Outcome values : \n', 'true positive:', tp, 'false negative:',fn, 'false positive:', fp, 'true negative:', tn)
    print('False positive rate:', fp/(fp+tn))

    return model, n_scores_mean, labels

def train_test_svm_wrapper(features):

    run_size = int(input("How many runs do you want to perform on the dataset?"))
    test_size = float(input("What size do you want the test sample to have (0-1)?"))  #Using 0.2
    if input("Do you want to delete previous feature selection? y/n").lower() == 'y':
        src.feature_extraction.io.__delete_stepwise(list(features.keys())[0])

    f1_scores = []

    for i in range(run_size):
        model, f1_score, labels = train_test_svm(features, test_size)
        f1_scores.append(f1_score)

    mean = statistics.mean(f1_scores)
    st_dev = statistics.stdev(f1_scores) / math.sqrt(10)
    print("F1 scores: ", f1_scores)
    print("Mean: ", mean, "(", st_dev, ")")

    if input("Save model? y/n").lower() == 'y':
        src.feature_extraction.io.__save_model(model, list(features.keys())[0])
        src.feature_extraction.io.__save_labels(labels, list(features.keys())[0])


    if input("Read stepwise? y/n").lower() == 'y':
        label_list = src.feature_extraction.io.__read_stepwise(list(features.keys())[0])
        print(label_list)


    return mean, st_dev

