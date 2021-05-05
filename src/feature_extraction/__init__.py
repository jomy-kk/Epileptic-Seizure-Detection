import os

import numpy as np
import pandas as pd
from multipledispatch import dispatch

from feature_extraction.FrequencyFeaturesCalculator import FrequencyFeaturesCalculator
from feature_extraction.HRVFeaturesCalculator import HRVFeaturesCalculator
from feature_extraction.KatzFeaturesCalculator import KatzFeaturesCalculator
from feature_extraction.PointecareFeaturesCalculator import PointecareFeaturesCalculator
from feature_extraction.RQAFeaturesCalculator import RQAFeaturesCalculator
from feature_extraction.TimeFeaturesCalculator import TimeFeaturesCalculator


data_path = 'data'


def assert_patient(patient): assert patient >= 101 and patient <= 111, "Patient number should be between 101 and 111."


def assert_crisis(crisis): assert crisis > 0, "Crisis must be a positive integer."


def assert_state(state): assert state == 'awake' or state == 'asleep', "State should be either 'awake' or 'asleep'."


#@dispatch(np.array, int, _time=bool, _frequency=bool, _pointecare=bool, _katz=bool, _rqa=bool)
def extract_segment_hrv_features(nni_segment, sampling_frequency, _time=False, _frequency=False, _pointecare=False,
                                 _katz=False,
                                 _rqa=False):
    """
    Method 1: Get all features of a group or groups.
    Given an nni segment and its sampling frequency, extracts and returns all the features of the groups marked as True.
    :param nni_segment: Sequence of nni samples.
    :param sampling_frequency: Sampling frequency (in Hertz) of the nni segment.
    :param _time: Pass as True to compute all time domain features.
    :param _frequency: Pass as True to compute all frequency domain features.
    :param _pointecare: Pass as True to compute all pointecare features.
    :param _katz: Pass as True to compute all katz features.
    :param _rqa: Pass as True to compute all recurrent quantitative analysis features.
    :return extracted_features: An np.hstack with all the requested features.
    """
    extracted_features = np.hstack(())

    if _time:
        from feature_extraction.TimeFeaturesCalculator import TimeFeaturesCalculator
        features_calculator = TimeFeaturesCalculator(nni_segment, sampling_frequency)
        extracted_features = np.hstack((extracted_features,
                                        features_calculator.get_mean(),
                                        features_calculator.get_var(),
                                        features_calculator.get_rmssd(),
                                        features_calculator.get_sdnn(),
                                        features_calculator.get_nn50(),
                                        features_calculator.get_pnn50()
                                        ))

    if _frequency:
        from feature_extraction.FrequencyFeaturesCalculator import FrequencyFeaturesCalculator
        features_calculator = FrequencyFeaturesCalculator(nni_segment, sampling_frequency)
        extracted_features = np.hstack((extracted_features,
                                        features_calculator.get_lf(),
                                        features_calculator.get_hf(),
                                        features_calculator.get_lf_hf(),
                                        ))

    if _pointecare:
        from feature_extraction.PointecareFeaturesCalculator import PointecareFeaturesCalculator
        features_calculator = PointecareFeaturesCalculator(nni_segment)
        extracted_features = np.hstack((extracted_features,
                                        features_calculator.get_sd1(),
                                        features_calculator.get_sd2(),
                                        features_calculator.get_csi(),
                                        features_calculator.get_csv(),
                                        ))

    if _katz:
        from feature_extraction.KatzFeaturesCalculator import KatzFeaturesCalculator
        features_calculator = KatzFeaturesCalculator(nni_segment)
        extracted_features = np.hstack((extracted_features,
                                        features_calculator.get_katz_fractal_dim(),
                                        ))

    if _rqa:
        from feature_extraction.RQAFeaturesCalculator import RQAFeaturesCalculator
        features_calculator = RQAFeaturesCalculator(nni_segment)
        extracted_features = np.hstack((extracted_features,
                                        features_calculator.get_rec(),
                                        features_calculator.get_det(),
                                        features_calculator.get_lmax(),
                                        ))

    del features_calculator
    return extracted_features


#@dispatch(np.array, int, list)
def extract_segment_hrv_features(nni_segment, sampling_frequency, needed_features: list):
    """
    Method 2: Specify which features are needed. More inefficient.
    Given an nni segment and its sampling frequency, extracts and returns the requested needed features.
    :param nni_segment: Sequence of nni samples.
    :param sampling_frequency: Sampling frequency (in Hertz) of the nni segment.
    :param needed_features: List containing the needed features in strings. Any feature is possible if defined in
    the HRVFeaturesCalculator classes.
    :return features: A list of the requested feature values.
    """

    # Auxiliary procedure
    def __get_hrv_features(calculator: HRVFeaturesCalculator, needed_features: list):
        """
        Given an HRV features calculator, gathers a list of computed features.
        :param calculator: A defined HRVFeaturesCalculator.
        :param needed_features: List containing the needed features in strings.
        :return: A list of the requested feature values.
        """
        features = []
        for needed_feature in needed_features:
            assert isinstance(needed_feature, str)
            assert hasattr(calculator, 'get_' + needed_feature)
            features.append(getattr(calculator, 'get_' + needed_feature)())
        return features

    extracted_features = np.hstack(())

    features_calculator = TimeFeaturesCalculator(nni_segment, sampling_frequency)
    extracted_features = np.hstack((extracted_features, __get_hrv_features(features_calculator, needed_features)))

    features_calculator = FrequencyFeaturesCalculator(nni_segment, sampling_frequency)
    extracted_features = np.hstack((extracted_features, __get_hrv_features(features_calculator, needed_features)))

    features_calculator = PointecareFeaturesCalculator(nni_segment)
    extracted_features = np.hstack((extracted_features, __get_hrv_features(features_calculator, needed_features)))

    features_calculator = KatzFeaturesCalculator(nni_segment)
    extracted_features = np.hstack((extracted_features, __get_hrv_features(features_calculator, needed_features)))

    features_calculator = RQAFeaturesCalculator(nni_segment)
    extracted_features = np.hstack((extracted_features, __get_hrv_features(features_calculator, needed_features)))

    del features_calculator
    return extracted_features


def segment_nni_signal(nni_signal, n_samples_segment):
    n_segments = int(len(nni_signal) / n_samples_segment)
    segmented_nni = np.array_split(nni_signal['nni'], n_segments)
    date_time_indexes = list(nni_signal['nni'].keys())
    segmented_date_time = np.array_split(date_time_indexes, n_segments)

    return segmented_nni, segmented_date_time


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


def extract_patient_hrv_features(n_samples_segment: int, patient: int, crises=None,
                                 _time=False, _frequency=False, _pointecare=False, _katz=False, _rqa=False,
                                 needed_features: list = None,
                                 _save=True):
    """
    Extracts features of some or all crises of a given patient.

    :param n_samples_segment: Integer number of seconds for each segment.
    :param patient: Integer number of the patient.
    :param crises: Integer number of the crisis of the patient, or
                    a list of integers of multiple crises of the patient, or
                    None to extract features of all crises of the patient.

    :param _time: Pass as True to compute all time domain features.
    :param _frequency: Pass as True to compute all frequency domain features.
    :param _pointecare: Pass as True to compute all pointecare features.
    :param _katz: Pass as True to compute all katz features.
    :param _rqa: Pass as True to compute all recurrent quantitative analysis features.

    :param needed_features: List containing the needed features in strings. Any feature is possible if defined in
    the HRVFeaturesCalculator classes.

    :param _save: Pass as True to save the features in a HDF file.
    :return: A pd.Dataframe containing the extracted features in each column by segments in rows (for a single crisis),
             or a dictionary with one element for each crisis of the patient, identified by the crisis number. Each of
             these elements is a pd.Dataframe.

    Note: The computed features are the union between the _time, _frequency, _pointecare, _katz, _rqa groups and the
    individual features specified in needed_features.
    """

    segment_time = 16  # seconds
    sf = 4  # Hz
    segment_samples = segment_time * sf  # 64 samples

    def __extract_crisis_hrv_features(patient, crisis):  # TODO: Union with needed_features
        nni_signal = __read_crisis_nni(patient, crisis)
        segmented_nni, segmented_date_time = segment_nni_signal(nni_signal, n_samples_segment)
        features = pd.DataFrame(columns=columns)
        for i, segment, segment_times in zip(range(len(segmented_nni)), segmented_nni,
                                             segmented_date_time):
            feat = extract_segment_hrv_features(segment, _time=_time, _frequency=_frequency, _pointecare=_pointecare,
                                                _katz=_katz, _rqa=_rqa)
            features.loc[i] = feat  # add a line
            t = segment_times[0] + ((segment_times[-1] - segment_times[0]) / 2)  # time in the middle of the segment
            features = features.rename({i: t}, axis='index')
        if _save:
            __save_crisis_hrv_features(patient, crisis, features)
        return features

    if crises is None:
        if input("You are about to extract features from all crisis of patient " + str(
                patient) + ". Are you sure? y/n").lower() == 'n':
            return
        # extract features for all crisis of the given patient
        crises = get_patient_crises_numbers(patient)
        features_set = {}
        for crisis in crises:
            features_set[crisis] = __extract_crisis_hrv_features(patient, crisis)
        return features_set

    elif isinstance(crises, list) and len(crises) > 0:  # extract features for a specific crisis of the given patient
        features_set = {}
        for crisis in crises:
            features_set[crisis] = __extract_crisis_hrv_features(patient, crisis)
        return features_set

    elif isinstance(crises, int):
        return __extract_crisis_hrv_features(patient, crises)

    else:
        raise Exception("'crises' should be an integer (for 1 crisis extraction), a list of integers (for multiple "
                        "crisis extraction), or None to extract features of all crises of the patient.")


def extract_hrv_features_all_patients(n_samples_segment: int, _save=True):
    """
    Extracts features of all crises of all patients.
    :param n_samples_segment:
    :param _save:
    :return: A dictionary with one element for each patient, identified by its number. Each element is another
    dictionary with one element for each crisis of that patient, identified by the crisis number. Each of these elements
    are a pd.Dataframe containing the extracted features in each column by segments in rows.
    """
    all_patient_numbers = __get_patient_numbers()
    patients_set = {}
    for patient in all_patient_numbers:
        patients_set[patient] = extract_patient_hrv_features(n_samples_segment, patient, None, _save=_save)

    return patients_set


def get_patient_hrv_features(patient: int, crisis: int):
    """
    Returns the features of the given patient-crisis pair.
    :param patient:  Integer number of the patient.
    :param crisis: Integer number of the crisis of the patient.
    :return:
    """
    assert_patient(patient)
    assert_crisis(crisis)
    try:  # try to read a previously computed HDF containing the features
        file_path = '/Patient' + str(patient) + '/crisis' + str(crisis) + '_hrv_features'
        data = pd.read_hdf(data_path + file_path)
        print("Data from " + file_path + " was retrieved.")
        return data
    except IOError:  # HDF not found, compute the features
        if input("The HRV features for this patient/crisis pair were never computed before. Would you like to compute them now? y/n").lower() == 'y':
            n_samples_segment = input("Number of samples per segment = ")
            needed_features = input("Which features to compute (separated by spaces, or 'all') = ")
            if needed_features == 'all':
                needed_features = None
            else:
                needed_features = needed_features.split(sep=' ')

            print("Extracting features...")
            features = extract_patient_hrv_features(int(n_samples_segment), patient, crisis,
                                         needed_features=needed_features, _save=True)
            print("Feature extraction saved.")
            return features

        else:
            return None


def __get_patient_numbers():
    directories = [name for name in os.listdir(data_path) if name[0] != '.']
    return [name.split('Patient')[1] for name in directories]

