import numpy as np

from feature_extraction.TimeFeature import TimeFeaturesCalculator


def extract_hrv_time_features(nni_segment, sampling_frequency, needed_features: list):
    """
    Given an nni segment and its sampling frequency, extracts and returns the requested needed features.
    :param nni_segment: Sequence of nni samples.
    :param sampling_frequency: Sampling frequency (in Hertz) of the nni segment.
    :param needed_features: List containing the needed features in strings. Any feature is possible if defined in
    TimeFeaturesCalculator.
    :return features: A list of the requested feature values.
    """
    time_features_calculator = TimeFeaturesCalculator(nni_segment, sampling_frequency)
    features = []
    for needed_feature in needed_features:
        assert isinstance(needed_feature, str)
        assert hasattr(time_features_calculator, 'get_' + needed_feature)
        features.append(getattr(time_features_calculator, 'get_' + needed_feature)())
    return features


def extract_hrv_features(nni_segment, sampling_frequency, _time=False, _frequency=False, _pointecare=False, _katz=False,
                         _rqa=False):
    extracted_features = np.hstack(())

    if _time:
        from feature_extraction.TimeFeature import TimeFeaturesCalculator
        time_features = TimeFeaturesCalculator(nni_segment, sampling_frequency)
        extracted_features = np.hstack((extracted_features,
                                        time_features.get_mean(),
                                        time_features.get_var(),
                                        time_features.get_rmssd(),
                                        time_features.get_sdnn(),
                                        time_features.get_nn50(),
                                        time_features.get_pnn50()
                                        ))

    if _frequency:
        from feature_extraction.FrequencyFeature import FrequencyFeaturesCalculator
        time_features = FrequencyFeaturesCalculator(nni_segment, sampling_frequency)
        extracted_features = np.hstack((extracted_features,
                                        time_features.get_lf(),
                                        time_features.get_hf(),
                                        time_features.get_lf_hf(),
                                        ))

    if _pointecare:
        from feature_extraction.PointecareFeature import PointecareFeaturesCalculator
        time_features = PointecareFeaturesCalculator(nni_segment)
        extracted_features = np.hstack((extracted_features,
                                        time_features.get_sd1(),
                                        time_features.get_sd2(),
                                        time_features.get_csi(),
                                        time_features.get_csv(),
                                        ))

    if _katz:
        from feature_extraction.KatzFeature import KatzFeaturesCalculator
        time_features = KatzFeaturesCalculator(nni_segment)
        extracted_features = np.hstack((extracted_features,
                                        time_features.get_katz_fractal_dim(),
                                        ))

    if _rqa:
        from feature_extraction.RQAFeature import RQAFeaturesCalculator
        time_features = RQAFeaturesCalculator(nni_segment)
        extracted_features = np.hstack((extracted_features,
                                        time_features.get_rec(),
                                        time_features.get_det(),
                                        time_features.get_lmax(),
                                        ))

    return extracted_features
