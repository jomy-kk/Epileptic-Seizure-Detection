####################################
#
#  MLB Project 2021
#
#  Module: Feature Selection
#  File: init
#
#  Created on May 17, 2021
#  All rights reserved to João Saraiva and Débora Albuquerque
#
####################################
import json
from _datetime import datetime

import matplotlib.pyplot as plt
import tkinter as tk
import pandas as pd

from functools import partial
from math import sqrt


from src.feature_extraction import get_patient_hrv_features, get_patient_hrv_baseline_features
from src.feature_selection.utils_signal_processing import *

data_path = './data'


def get_features_from_patients(patients: list, crises: list):
    res = dict()
    for patient in patients:
        features = dict()
        for crisis in crises:
            features[crisis] = clean_outliers(get_patient_hrv_features(patient, crisis))
            features[crisis] = normalise_feats(features[crisis])
        res[patient] = features
    return res

def get_baseline_from_patients(patients: list, state : str):
    res = dict()
    for patient in patients:
        features = dict()
        for state in state:
            features[state] = clean_outliers(get_patient_hrv_baseline_features(patient, state))
            features[state] = normalise_feats(features[state])
        res[patient] = features
    return res


def handle_click(_options):
    print(_options)


CBC = dict()

class Table:
    """
    features: Dictionary of dictionaries of lists. First dimension corresponds to each patient, associated by its key
                Second dimension corresponds to each patient's crisis, associated to its key.
                Third dimension corresponds to the existing extracted features for that patient-crisis pair.
    """

    def __init__(self, root, features, checkbox_controls):
        # do a header for the table
        header = set()
        for patient in features.values():
            for crisis in patient.values():
                for feature in crisis.columns:
                    header.add(feature)
            #if baseline:
                #for feature in state.columns:
                    #header.add('baseline'+ feature)
        self.feature_labels = list(header)

        # initialize checkbox control variables
        for p in patients:
            checkbox_controls[p] = dict()
            for c in crises:
                checkbox_controls[p][c] = dict()
                for f in self.feature_labels:
                    checkbox_controls[p][c][f] = tk.IntVar(root)

        # save first column width
        width = 30

        # insert header
        self.e = tk.Entry(root, width=width, fg='black', font=('Arial', 10, 'bold'), justify='right')
        self.e.grid(row=0, column=0)
        self.e.insert(tk.END, "Patient / Crisis")
        for j in range(len(self.feature_labels)):
            self.e = tk.Entry(root, width=width, fg='black', font=('Arial', 10))
            self.e.grid(row=j+1, column=0)
            self.e.insert(tk.END, self.feature_labels[j])

        # creating body
        m = 1
        for patient in features.keys():
            for crisis in features[patient].keys():
                n = 0
                self.e = tk.Entry(root, width=5, fg='black', font=('Arial', 10, 'bold'))
                self.e.grid(row=n, column=m)
                self.e.insert(tk.END, str(patient) + '/' + str(crisis))
                n += 1
                for feature in self.feature_labels:
                    self.e = tk.Checkbutton(root, variable=checkbox_controls[patient][crisis][feature])
                    self.e.grid(row=n, column=m)
                    if feature in features[patient][crisis].columns:
                        checkbox_controls[patient][crisis][feature].set(1)
                    else:
                        checkbox_controls[patient][crisis][feature].set(0)
                    n += 1
                m += 1

        # create (dis)select all buttons
        n = 1
        for f in self.feature_labels:
            print(f)
            tk.Button(root, text="(De)Select All", command=partial(self.de_select_all, f)).grid(row=n, column=m+1)
            n += 1

    def de_select_all(self, f):
        print(CBC[patients[0]][crises[0]][f])
        if CBC[patients[0]][crises[0]][f].get() == 0:
            new_value = 1
            print("Selecting all", f, "features")
        else:
            new_value = 0
            print("Deselecting all", f, "features")

        for p in CBC.values():
            for c in p.values():
                c[f].set(new_value)

#def get_full_baseline(patients):
    #baseline_awake = get_baseline_from_patients(patients, "awake")
    #baseline_asleep = get_baseline_from_patients(patients, "asleep")

    #for patient in patients:
        #if baseline_awake[patient] is not None:
            #if baseline_asleep[patient] is not None:
                #baseline_awake[patient].update(baseline_asleep[patient])


    #return baseline_awake

patients = [106]
crises = [2,3,4]
state = "awake"
features = get_features_from_patients(patients, crises)
#baseline = get_full_baseline(patients)

# create root window
#root = tk.Tk()
#root.title("Feature Selection")

# create table
#t = Table(root, features, CBC)
#root.mainloop()

def convert_date_time(date_time: str):
    date, time = date_time.split(' ')
    d, m, a = date.split('/')
    return a + '-' + m + '-' + d + ' ' + time


def inspect_features(features):
    header = set()
    for patient in features.values():
        for crisis in patient.values():
            for feature in crisis.columns:
                header.add(feature)
    feature_labels = list(header)

    colors = ['#00bfc2', '#5756d6', '#fada5e', '#62d321', '#fe9b29']
    # color for each payient-crisis pair is given by: patientID - 100 + crisisID - 2

    n_features = len(feature_labels)
    n_subplots_per_side = int(sqrt(n_features))

    # get onsets
    with open(data_path + '/patients.json') as metadata_file:
        metadata = json.load(metadata_file)

    background_color = (0, 0, 0)
    rolling_avg = False
    if input("Do you want to do rolling average? y/n ").lower() == 'y':
        rolling_avg = True
        n_avg = int(input("Number of points around each point for rolling average: "))


    fig = plt.figure(figsize=(20, 20), facecolor=background_color)
    for i in range(n_features):
        print(i,"/",n_subplots_per_side,"/",n_features)
        ax = plt.subplot(n_subplots_per_side + 1, n_subplots_per_side + 1, i+1, facecolor=background_color)
        ax.grid(color='white', linestyle='--', linewidth=0.35)
        feature_label = feature_labels[i]
        plt.title(feature_label, color='white')

        reference_onset = None
        for p in features.keys():
            for c in features[p].keys():
                if feature_label in features[p][c].keys():
                    onset = metadata['patients'][str(p)]['crises'][str(c)]['onset']
                    onset = datetime.strptime(onset, "%d/%m/%Y %H:%M:%S")
                    if reference_onset is None:
                        reference_onset = onset
                        time_axis = features[p][c].index
                    else:
                        difference = onset - reference_onset
                        if onset > reference_onset:
                            time_axis = features[p][c].index - difference
                        else:
                            time_axis = features[p][c].index + difference

        #for p in features.keys():
         #   for c in features[p].keys():
          #      if feature_label in features[p][c].keys():
                    feature = features[p][c][feature_label]
                    feature_rolling = features[p][c][feature_label]
                    if not rolling_avg:
                        plt.plot(time_axis, feature.values, '.', label=str(p) + '/' + str(c), markersize=1.2,
                                 alpha=0.7)  # .values discards time axis
                        plt.xlabel('time', color='white')
                    else:
                        for j in range(len(feature) - 2 * n_avg):
                            feature_rolling[j + n_avg] = feature.values[j:j+n_avg*2].mean()

                        # Tendency line
                        plt.plot(time_axis, feature_rolling.values, '-', label=str(p)+'/'+str(c), markersize=1.2, alpha=0.7)  # .values discards time axis

                        plt.xlabel('time', color='white')

        # draw onset vertical line
        plt.axvline(x=reference_onset, color='red', linewidth=0.4)
        leg = plt.legend(loc='best', facecolor=background_color)
        for line, text in zip(leg.get_lines(), leg.get_texts()):
            text.set_color('white')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.6, hspace=0.6)
    plt.show()

inspect_features(features)



