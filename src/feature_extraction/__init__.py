import numpy as np

from feature_extraction.HRVFeaturesCalculator import HRVFeaturesCalculator


# @private
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


def extract_hrv_features_calculator(nni_segment, sampling_frequency, needed_features: list):
    """
    Given an nni segment and its sampling frequency, extracts and returns the requested needed features.
    :param nni_segment: Sequence of nni samples.
    :param sampling_frequency: Sampling frequency (in Hertz) of the nni segment.
    :param needed_features: List containing the needed features in strings. Any feature is possible if defined in
    TimeFeaturesCalculator.
    :return features: A list of the requested feature values.
    """
    from feature_extraction.TimeFeaturesCalculator import TimeFeaturesCalculator
    features_calculator = TimeFeaturesCalculator(nni_segment, sampling_frequency)
    return __get_hrv_features(features_calculator, needed_features)


def extract_hrv_frequency_features(nni_segment, sampling_frequency, needed_features: list):
    """
    Given an nni segment and its sampling frequency, extracts and returns the requested needed features.
    :param nni_segment: Sequence of nni samples.
    :param sampling_frequency: Sampling frequency (in Hertz) of the nni segment.
    :param needed_features: List containing the needed features in strings. Any feature is possible if defined in
    TimeFeaturesCalculator.
    :return features: A list of the requested feature values.
    """
    from feature_extraction.FrequencyFeaturesCalculator import FrequencyFeaturesCalculator
    features_calculator = FrequencyFeaturesCalculator(nni_segment, sampling_frequency)
    features = []
    for needed_feature in needed_features:
        assert isinstance(needed_feature, str)
        assert hasattr(features_calculator, 'get_' + needed_feature)
        features.append(getattr(features_calculator, 'get_' + needed_feature)())
    return features


def extract_hrv_features(nni_segment, sampling_frequency, _time=False, _frequency=False, _pointecare=False, _katz=False,
                         _rqa=False):
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
