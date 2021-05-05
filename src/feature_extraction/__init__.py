import numpy as np

from feature_extraction.TimeFeature import TimeFeaturesCalculator


def extract_hrv_time_features(nni_signal, sampling_frequency, needed_features: list = []):
    time_features_calculator = TimeFeaturesCalculator('', nni_signal, sampling_frequency)
    features = []
    for needed_feature in needed_features:
        assert isinstance(needed_feature, str)
        assert hasattr(time_features_calculator, 'get_' + needed_feature)
        features.append(getattr(time_features_calculator, 'get_' + needed_feature)())
    return features

def extract_hrv_features(nni_signal, sampling_frequency, _time=False, _frequency=False, _pointecare=False, _katz=False,
                         _rqa=False):
    extracted_features = np.hstack(())
    extracted_features_labels = []

    if _time:
        from feature_extraction.TimeFeature import TimeFeaturesCalculator
        time_features = TimeFeaturesCalculator('All time features', nni_signal, sampling_frequency)
        mean, var, rmssd, sdnn, nn50, pnn50 = time_features.get_mean(), time_features.get_var(), \
                                              time_features.get_rmssd(), time_features.get_sdnn(), \
                                              time_features.get_nn50(), time_features.get_pnn50()

        extracted_features = np.hstack((extracted_features, float(mean), float(var), float(rmssd), float(sdnn),
                                        float(nn50), float(pnn50)))
        extracted_features_labels += [str(mean), str(var), str(rmssd), str(sdnn), str(nn50), str(pnn50), ]

    if _frequency:
        from feature_extraction.FrequencyFeature import FrequencyFeaturesCalculator
        time_features = FrequencyFeaturesCalculator('', nni_signal, sampling_frequency)
        extracted_features = np.hstack((extracted_features,
                                        time_features.get_lf(),
                                        time_features.get_hf(),
                                        time_features.get_lf_hf(),
                                        ))

    if _pointecare:
        from feature_extraction.PointecareFeature import PointecareFeaturesCalculator
        time_features = PointecareFeaturesCalculator('', nni_signal, )
        extracted_features = np.hstack((extracted_features,
                                        time_features.get_sd1(),
                                        time_features.get_sd2(),
                                        time_features.get_csi(),
                                        time_features.get_csv(),
                                        ))

    if _katz:
        from feature_extraction.KatzFeature import KatzFeaturesCalculator
        time_features = KatzFeaturesCalculator('', nni_signal, )
        extracted_features = np.hstack((extracted_features,
                                        time_features.get_katz_fractal_dim(),
                                        ))

    if _rqa:
        from feature_extraction.RQAFeature import RQAFeaturesCalculator
        time_features = RQAFeaturesCalculator('', nni_signal, )
        extracted_features = np.hstack((extracted_features,
                                        time_features.get_rec(),
                                        time_features.get_det(),
                                        time_features.get_lmax(),
                                        ))

    return extracted_features
