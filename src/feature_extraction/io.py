import json
import pandas as pd

data_path = 'data'

with open(data_path + '/patients.json') as metadata_file:
    metadata = json.load(metadata_file)


def assert_patient(patient): assert patient >= 101 and patient <= 111, "Patient number should be between 101 and 111."


def assert_crisis(crisis): assert crisis > 0, "Crisis must be a positive integer."


def assert_state(state): assert state == 'awake' or state == 'asleep', "State should be either 'awake' or 'asleep'."


def __get_patient_numbers():
    return metadata['patients'].keys()


def __get_patient_crises_numbers(patient: int):
    return metadata['patients'][str(patient)].keys()


def __read_crisis_nni(patient: int, crisis: int):
    assert_patient(patient)
    assert_crisis(crisis)
    try:
        file_path = '/Patient' + str(patient) + '/nni_Crise ' + str(crisis) + '_hospital'
        data = pd.read_hdf(data_path + file_path)
        print("Data from " + file_path + " was retreived.")
        return data
    except IOError:
        print("That patient/crisis pair does not exist. None was returned.")
        return None


def __save_crisis_hrv_features(patient: int, crisis: int, features: pd.DataFrame):
    assert_patient(patient)
    assert_crisis(crisis)
    try:
        file_path = '/Patient' + str(patient) + '/crisis' + str(crisis) + '_hrv_features'
        features.to_hdf(data_path + file_path, 'features', mode='a')
        print("Written in " + file_path + " was successful.")
    except IOError:
        print("That patient/crisis pair cannot be created. Save failed.")


def __read_crisis_hrv_features(patient: int, crisis: int):
    assert_patient(patient)
    assert_crisis(crisis)
    try:  # try to read a previously computed HDF containing the features
        file_path = '/Patient' + str(patient) + '/crisis' + str(crisis) + '_hrv_features'
        data = pd.read_hdf(data_path + file_path)
        print("Data from " + file_path + " was retrieved.")
        return data
    except IOError:  # HDF not found, compute the features
        return None

