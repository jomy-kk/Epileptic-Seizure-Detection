import numpy as np
import pandas as pd

from src.feature_extraction.FrequencyFeaturesCalculator import FrequencyFeaturesCalculator
from src.feature_extraction.HRVFeaturesCalculator import HRVFeaturesCalculator
from src.feature_extraction.KatzFeaturesCalculator import KatzFeaturesCalculator
from src.feature_extraction.PointecareFeaturesCalculator import PointecareFeaturesCalculator
from src.feature_extraction.RQAFeaturesCalculator import RQAFeaturesCalculator
from src.feature_extraction.TimeFeaturesCalculator import TimeFeaturesCalculator
from src.feature_extraction.COSenFeaturesCalculator import COSenFeaturesCalculator
import src.feature_extraction.io


def extract_segment_hrv_features(nni_segment, sampling_frequency, _time=False, _frequency=False, _pointecare=False,
                                 _katz=False, _rqa=False, _cosen=False, m=None, g=None):
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
    :param _cosen: Pass as True to compute all COSen features.
    :param m: m parameter for the COSen calculator referring to the embedding dimension. It should be an integer, usually between 1-5.
    :param g: g parameter for the COSen calculator used to calculate de vector comparison distance (r). Value in percentage.Usually between 01-0.5.
    :return extracted_features: An np.hstack with all the requested features.
    """
    extracted_features = []

    if _time:
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
        features_calculator = FrequencyFeaturesCalculator(nni_segment, sampling_frequency)
        extracted_features = np.hstack((extracted_features,
                                        features_calculator.get_lf(),
                                        features_calculator.get_hf(),
                                        features_calculator.get_lf_hf(),
                                        features_calculator.get_hf_lf(),
                                        ))

    if _pointecare:
        features_calculator = PointecareFeaturesCalculator(nni_segment)
        extracted_features = np.hstack((extracted_features,
                                        features_calculator.get_sd1(),
                                        features_calculator.get_sd2(),
                                        features_calculator.get_csi(),
                                        features_calculator.get_csv(),
                                        ))

    if _katz:
        features_calculator = KatzFeaturesCalculator(nni_segment)
        extracted_features = np.hstack((extracted_features,
                                        features_calculator.get_katz_fractal_dim(),
                                        ))

    if _rqa:
        features_calculator = RQAFeaturesCalculator(nni_segment)
        extracted_features = np.hstack((extracted_features,
                                        features_calculator.get_rec(),
                                        features_calculator.get_det(),
                                        features_calculator.get_lmax(),
                                        ))

    if _cosen:
        features_calculator = COSenFeaturesCalculator(nni_segment, m, g)
        extracted_features = np.hstack((extracted_features,
                                        features_calculator.get_sampen(),
                                        features_calculator.get_cosen(),
                                        ))

    del features_calculator
    return extracted_features


def extract_segment_some_hrv_features(nni_segment, sampling_frequency, needed_features: list, m=None, g=None):
    """
    Method 2: Specify which features are needed. More inefficient.
    Given an nni segment and its sampling frequency, extracts and returns the requested needed features.
    :param nni_segment: Sequence of nni samples.
    :param sampling_frequency: Sampling frequency (in Hertz) of the nni segment.
    :param needed_features: List containing the needed features in strings. Any feature is possible if defined in
    the HRVFeaturesCalculator classes.
   :param m: m parameter for the COSen calculator referring to the embedding dimension. It should be an integer, usually between 1-5.
    :param g: g parameter for the COSen calculator used to calculate de vector comparison distance (r). Value in percentage.Usually between 01-0.5.
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
        labels = []
        for needed_feature in needed_features:
            assert isinstance(needed_feature, str)
            if hasattr(calculator, 'get_' + needed_feature):
                features.append(getattr(calculator, 'get_' + needed_feature)())
                labels.append(calculator.labels[needed_feature])
        return features, labels

    # Main segment
    extracted_features = []
    labels = []

    features_calculator = TimeFeaturesCalculator(nni_segment, sampling_frequency)
    f, l = extracted_features, __get_hrv_features(features_calculator, needed_features)
    extracted_features = np.hstack((extracted_features, f))
    labels += l

    features_calculator = FrequencyFeaturesCalculator(nni_segment, sampling_frequency)
    f, l = extracted_features, __get_hrv_features(features_calculator, needed_features)
    extracted_features = np.hstack((extracted_features, f))
    labels += l

    features_calculator = PointecareFeaturesCalculator(nni_segment)
    f, l = extracted_features, __get_hrv_features(features_calculator, needed_features)
    extracted_features = np.hstack((extracted_features, f))
    labels += l

    features_calculator = KatzFeaturesCalculator(nni_segment)
    f, l = extracted_features, __get_hrv_features(features_calculator, needed_features)
    extracted_features = np.hstack((extracted_features, f))
    labels += l

    features_calculator = RQAFeaturesCalculator(nni_segment)
    f, l = extracted_features, __get_hrv_features(features_calculator, needed_features)
    extracted_features = np.hstack((extracted_features, f))
    labels += l

    features_calculator = COSenFeaturesCalculator(nni_segment, m, g)
    f, l = extracted_features, __get_hrv_features(features_calculator, needed_features)
    extracted_features = np.hstack((extracted_features, f))
    labels += l

    del features_calculator
    return extracted_features


def segment_nni_signal(nni_signal, n_samples_segment, n_samples_overlap=0):
    date_time_indexes = list(nni_signal['nni'].keys())
    step = n_samples_segment - n_samples_overlap
    segmented_nni = [nni_signal[i: i + n_samples_segment] for i in range(0, len(nni_signal)-n_samples_segment, step)]
    segmented_date_time = [date_time_indexes[i: i + n_samples_segment] for i in range(0, len(date_time_indexes)-n_samples_segment, step)]
    print("Divided signal into " + str(len(segmented_nni)) + " samples.")
    assert len(segmented_nni) == len(segmented_date_time)
    return segmented_nni, segmented_date_time


def extract_patient_hrv_features(segment_time: int, patient: int, crises=None, state="awake", segment_overlap_time=0,
                                 _time=False, _frequency=False, _pointecare=False, _katz=False, _rqa=False, _cosen=False,
                                 m=None, g=None, baseline = False,
                                 needed_features: list = None,
                                 _save=True):
    """
    Extracts features of some or all crises of a given patient.

    :param segment_time: Integer number of seconds for each segment.
    :param patient: Integer number of the patient.
    :param crises: Integer number of the crisis of the patient, or
                    a list of integers of multiple crises of the patient, or
                    None to extract features of all crises of the patient.

    :param _time: Pass as True to compute all time domain features.
    :param _frequency: Pass as True to compute all frequency domain features.
    :param _pointecare: Pass as True to compute all pointecare features.
    :param _katz: Pass as True to compute all katz features.
    :param _rqa: Pass as True to compute all recurrent quantitative analysis features.
    :param _cosen: Pass as True to compute all COSen features.
    :param m: m parameter for the COSen calculator referring to the embedding dimension. It should be an integer, usually between 1-5.
    :param g: g parameter for the COSen calculator used to calculate de vector comparison distance (r). Value in percentage.Usually between 01-0.5.

    :param needed_features: List containing the needed features in strings. Any feature is possible if defined in
    the HRVFeaturesCalculator classes.

    :param _save: Pass as True to save the features in a HDF file.
    :return: A pd.Dataframe containing the extracted features in each column by segments in rows (for a single crisis),
             or a dictionary with one element for each crisis of the patient, identified by the crisis number. Each of
             these elements is a pd.Dataframe.

    Note: The computed features are the union between the _time, _frequency, _pointecare, _katz, _rqa groups and the
    individual features specified in needed_features.
    """

    sf = src.feature_extraction.io.metadata['sampling_frequency']  # Hz
    n_samples_segment = segment_time * sf
    n_samples_overlap = segment_overlap_time * sf

    # Get labels for the features
    labels = []
    # groups
    if _time:
        labels += list(TimeFeaturesCalculator.labels.values())
    if _frequency:
        labels += list(FrequencyFeaturesCalculator.labels.values())
    if _pointecare:
        labels += list(PointecareFeaturesCalculator.labels.values())
    if _katz:
        labels += list(KatzFeaturesCalculator.labels.values())
    if _rqa:
        labels += list(RQAFeaturesCalculator.labels.values())
    if _cosen:
        labels += list(COSenFeaturesCalculator.labels.values())
    # individual features
    if needed_features is not None:
        for f in needed_features:
            if f in labels:  # if this feature was already part of a group, it was already extracted
                needed_features.remove(f)  # remove to prevent duplicates
            else:
                labels.append(f)

    # Auxiliary procedure
    def __extract_crisis_hrv_features(patient, crisis):
        nni_signal = src.feature_extraction.io.__read_crisis_nni(patient, crisis)
        segmented_nni, segmented_date_time = segment_nni_signal(nni_signal, n_samples_segment, n_samples_overlap=n_samples_overlap)
        features = pd.DataFrame(columns=labels)

        # complete group features
        for i, segment, segment_times in zip(range(len(segmented_nni)), segmented_nni,
                                             segmented_date_time):
            # group features
            extracted_features = extract_segment_hrv_features(segment, sf, _time=_time, _frequency=_frequency,
                                                              _pointecare=_pointecare, _katz=_katz, _rqa=_rqa, _cosen=_cosen)

            # individual features
            if needed_features is not None:
                extracted_features = np.hstack(
                    (extracted_features, extract_segment_some_hrv_features(segment, sf, needed_features)))

            features.loc[i] = extracted_features  # add a line
            t = segment_times[0] + ((segment_times[-1] - segment_times[0]) / 2)  # time in the middle of the segment
            features = features.rename({i: t}, axis='index')

        if _save:
            src.feature_extraction.io.__save_crisis_hrv_features(patient, crisis, features)

        return features

    def __extract_baseline_hrv_features(patient, state):
        nni_signal = src.feature_extraction.io.__read_baseline_nni(patient, state)
        segmented_nni, segmented_date_time = segment_nni_signal(nni_signal, n_samples_segment,
                                                                n_samples_overlap=n_samples_overlap)
        features = pd.DataFrame(columns=labels)

        # complete group features
        for i, segment, segment_times in zip(range(len(segmented_nni)), segmented_nni,
                                             segmented_date_time):
            # group features
            extracted_features = extract_segment_hrv_features(segment, sf, _time=_time, _frequency=_frequency,
                                                              _pointecare=_pointecare, _katz=_katz, _rqa=_rqa,
                                                              _cosen=_cosen)

            # individual features
            if needed_features is not None:
                extracted_features = np.hstack(
                    (extracted_features, extract_segment_some_hrv_features(segment, sf, needed_features)))

            features.loc[i] = extracted_features  # add a line
            t = segment_times[0] + ((segment_times[-1] - segment_times[0]) / 2)  # time in the middle of the segment
            features = features.rename({i: t}, axis='index')

        if _save:
            src.feature_extraction.io.__save_baseline_hrv_features(patient, state, features)

        return features

    # Extract baseline
    if baseline:
       features_baseline= __extract_baseline_hrv_features(patient, state)
       return features_baseline

    # Main segment
    if crises is None:
        if input("You are about to extract features from all crisis of patient " + str(
                patient) + ". Are you sure? y/n").lower() == 'n':
            return
        # extract features for all crisis of the given patient
        crises = src.feature_extraction.io.__get_patient_crises_numbers(patient)
        features_set = {}
        for crisis in crises:
            # check if it already exists
            features = src.feature_extraction.io.__read_crisis_hrv_features(patient, crisis)
            if features is not None:
                if input("An HRV features HDF5 file for this patient's crisis " + str(
                        crisis) + " was found with the features " + str(
                        list(features.columns)) + ".\nDiscard and recompute? y/n").lower() == 'y':
                    print("Recomputing...")
                    features_set[crisis] = __extract_crisis_hrv_features(patient, crisis)
                else:
                    print("Returning the features founded.")
                    features_set[crisis] = features

        return features_set

    elif isinstance(crises, list) and len(crises) > 0:  # extract features for a specific crisis of the given patient
        features_set = {}
        for crisis in crises:
            # check if it already exists
            features = src.feature_extraction.io.__read_crisis_hrv_features(patient, crisis)
            if features is not None:
                if input("An HRV features HDF5 file for this patient's crisis " + str(crisis) + " was found with the features " + str(
                        list(features.columns)) + ".\nDiscard and recompute? y/n").lower() == 'y':
                    print("Recomputing...")
                    features_set[crisis] = __extract_crisis_hrv_features(patient, crisis)
                else:
                    print("Returning the features founded.")
                    features_set[crisis] = features

        return features_set

    elif isinstance(crises, int):
        # check if it already exists
        features = src.feature_extraction.io.__read_crisis_hrv_features(patient, crises)
        if features is not None:
            if input("An HRV features HDF5 file for this patient/crisis was found with the features " + str(list(features.columns)) + ".\nDiscard and recompute? y/n").lower() == 'y':
                print("Recomputing...")
                return __extract_crisis_hrv_features(patient, crises)
            else:
                print("Returning the features founded.")
                return features

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
    all_patient_numbers = src.feature_extraction.io.__get_patient_numbers()
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
    features = src.feature_extraction.io.__read_crisis_hrv_features(patient, crisis)

    if features is None:  # HDF not found, compute the features
        if input(
                "The HRV features for this patient/crisis pair were never computed before. Would you like to compute them now? y/n").lower() == 'y':
            segment_seconds = input("Seconds per segment = ")
            needed_features = input("Which features to compute (separated by spaces, or 'all') = ")
            if needed_features == 'all':
                needed_features = None
            else:
                needed_features = needed_features.split(sep=' ')

            m, g = None, None

            for f in COSenFeaturesCalculator.labels:
                if (needed_features is None) or (f in needed_features):  # meio que estupido, mas reduz o codigo xD
                    m = input("For the COSen features, give a m: ")
                    g = input("For the COSen features, give a g: ")
                    break

            print("Extracting features...")
            features = extract_patient_hrv_features(int(segment_seconds), patient, crisis, m=m, g=g,
                                                    needed_features=needed_features, _save=True)
            print("Feature extraction saved.")
            return features

        else:
            return None
    else:
        return features

def get_patient_hrv_baseline_features(patient: int, state: str):
    """
    Returns the features of the given patient-baseline pair.
    :param patient:  Integer number of the patient.
    :param string: String of the state of the patient: awake/asleep.
    :return:
    """

    features = src.feature_extraction.io.__read_baseline_hrv_features(patient, state)

    if features is None:  # HDF not found, compute the features
        if input(
                "The HRV features for this patient/crisis pair were never computed before. Would you like to compute them now? y/n").lower() == 'y':
            segment_seconds = input("Seconds per segment = ")
            needed_features = input("Which features to compute (separated by spaces, or 'all') = ")
            if needed_features == 'all':
                needed_features = None
            else:
                needed_features = needed_features.split(sep=' ')

            m, g = None, None

            for f in COSenFeaturesCalculator.labels:
                if (needed_features is None) or (f in needed_features):  # meio que estupido, mas reduz o codigo xD
                    m = input("For the COSen features, give a m: ")
                    g = input("For the COSen features, give a g: ")
                    break

            print("Extracting features...")
            features = extract_patient_hrv_features(int(segment_seconds), patient, None, state=state, baseline=True, m=m, g=g,
                                                    needed_features=needed_features, _save=True)
            print("Feature extraction saved.")
            return features

        else:
            return None
    else:
        return features
