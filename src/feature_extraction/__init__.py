import numpy as np
from multipledispatch import dispatch

from feature_extraction.FrequencyFeaturesCalculator import FrequencyFeaturesCalculator
from feature_extraction.HRVFeaturesCalculator import HRVFeaturesCalculator
from feature_extraction.KatzFeaturesCalculator import KatzFeaturesCalculator
from feature_extraction.PointecareFeaturesCalculator import PointecareFeaturesCalculator
from feature_extraction.RQAFeaturesCalculator import RQAFeaturesCalculator
from feature_extraction.TimeFeaturesCalculator import TimeFeaturesCalculator


@dispatch(np.array, int, _time=bool, _frequency=bool, _pointecare=bool, _katz=bool, _rqa=bool)
def extract_hrv_features(nni_segment, sampling_frequency, _time=False, _frequency=False, _pointecare=False, _katz=False,
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

    return extracted_features


@dispatch(np.array, int, needed_features=list)
def extract_hrv_features(nni_segment, sampling_frequency, needed_features: list):
    """
    Method 2: Specify which features are needed. More inefficient.
    Given an nni segment and its sampling frequency, extracts and returns the requested needed features.
    :param nni_segment: Sequence of nni samples.
    :param sampling_frequency: Sampling frequency (in Hertz) of the nni segment.
    :param needed_features: List containing the needed features in strings. Any feature is possible if defined in
    TimeFeaturesCalculator.
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

    return extracted_features

