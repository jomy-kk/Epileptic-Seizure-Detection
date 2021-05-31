import json
import pandas as pd
import pickle
import os

data_path = './data'

with open(data_path + '/patients.json') as metadata_file:
    metadata = json.load(metadata_file)


def assert_patient(patient): assert 101 <= int(patient) <= 111, "Patient number should be between 101 and 111."


def assert_crisis(crisis): assert int(crisis) > 0, "Crisis must be a positive integer."


def assert_state(state): assert state == 'awake' or state == 'asleep', "State should be either 'awake' or 'asleep'."


def __get_patient_numbers():
    return metadata['patients'].keys()


def __get_patient_crises_numbers(patient: int):
    assert_patient(patient)
    return metadata['patients'][str(patient)]['crises'].keys()


def __read_crisis_nni(patient: int, crisis: int):
    assert_patient(patient)
    assert_crisis(crisis)
    try:
        file_path = '/' + str(patient) + '/nni_crisis_' + str(crisis)
        data = pd.read_hdf(data_path + file_path)
        print("Data from " + file_path + " was retrieved.")
        return data
    except IOError:
        file_path = '/' + str(patient) + '/nni_crisis_' + str(crisis)
        print("That patient/crisis pair does not exist. None was returned.")
        return None


def __read_baseline_nni(patient: int, state: str):
    assert_patient(patient)
    assert_state(state)
    try:
        file_path = '/' + str(patient) + '/nni_baseline_' + str(state)
        data = pd.read_hdf(data_path + file_path)
        print("Data from " + file_path + " was retrieved.")
        return data
    except IOError:
        print("That patient/baseline pair does not exist. None was returned.")
        return None


def __save_crisis_hrv_features(patient: int, crisis: int, features: pd.DataFrame):
    assert_patient(patient)
    assert_crisis(crisis)
    try:
        file_path = '/' + str(patient) + '/hrv_crisis_' + str(crisis)
        features.to_hdf(data_path + file_path, 'features', mode='a')
        print("Written in " + file_path + " was successful.")
    except IOError:
        print("That patient/baseline pair cannot be created. Save failed.")


def __save_baseline_hrv_features(patient: int, state: str, features: pd.DataFrame):
    assert_patient(patient)
    assert_state(state)
    try:
        file_path = '/' + str(patient) + '/hrv_baseline_' + str(state)
        features.to_hdf(data_path + file_path, 'features', mode='a')
        print("Written in " + file_path + " was successful.")
    except IOError:
        print("That patient/crisis pair cannot be created. Save failed.")


def __read_crisis_hrv_features(patient: int, crisis: int):
    assert_patient(patient)
    assert_crisis(crisis)
    try:  # try to read a previously computed HDF containing the features
        file_path = '/' + str(patient) + '/hrv_crisis_' + str(crisis)
        data = pd.read_hdf(data_path + file_path)
        print("Data from " + file_path + " was retrieved.")

        print("Data",data)

        return data
    except IOError:  # HDF not found, return None
        return None


def __read_baseline_hrv_features(patient: int, state: str):
    assert_patient(patient)
    assert_state(state)
    try:  # try to read a previously computed HDF containing the features
        file_path = '/' + str(patient) + '/hrv_baseline_' + str(state)
        data = pd.read_hdf(data_path + file_path)
        print("Data from " + file_path + " was retrieved.")
        return data
    except IOError:  # HDF not found, return None
        return None

def __save_model(model, patient : int):
    try:
        file_path = data_path + '/model_' + str(patient) + '.p'
        pickle.dump(model, open(file_path,'wb'))
        print("Written in " + file_path + " was successful.")
    except IOError:
        print("Model could not be saved. Save failed.")

def __load_model(patient:int):
    try:
        file_path = data_path + '/model_' + str(patient) + '.p'
        model = pickle.load(open(file_path, 'rb'))
        print("Loaded model from " + file_path + " successfully.")
        return model
    except IOError:
        print("Model could not be loaded.")
        return None

def __save_labels(labels, patient:int):
    try:
        file_path = data_path + '/labels_' + str(patient) + '.json'
        file = open(file_path, 'w')
        json.dump(labels, file)
        print("Labels saved to " + file_path + " successfully.")
    except IOError:
        print("Cannot write labels to file. Save failed.")

def __read_labels(patient:int):
    try:  # try to read a previously computed JSON file with best parameters
        file_path = data_path + '/labels_' + str(patient) + '.json'
        file = open(file_path, 'r')
        labels = json.load(file)

        print("Data from " + file_path + " was retrieved.")

        print("Labels: ", labels)

        return labels
    except IOError:  # Parameters not found, return None
        return None

def __save_stepwise(features, patient:int):
    try:
        file_path = data_path + '/stepwise_' + str(patient) + '.json'
        file = open(file_path, 'w')
        json.dump(features, file)
        print("Labels saved to " + file_path + " successfully.")
    except IOError:
        print("Cannot write labels to file. Save failed.")

def __read_stepwise(patient:int):
    try:  # try to read a previously computed JSON file with best parameters
        file_path = data_path + '/stepwise_' + str(patient) + '.json'
        file = open(file_path, 'r')
        features = json.load(file)

        print("Data from " + file_path + " was retrieved.")

        return features
    except IOError:  # Parameters not found, return None
        return None

def __delete_stepwise(patient:int):
    try:
        file_path = data_path + '/stepwise_' + str(patient) + '.json'
        os.remove(file_path)
        print("Data from " + file_path + " removed.")

        return True
    except IOError:  # Parameters not found, return None
        print("No file found.")
        return None
