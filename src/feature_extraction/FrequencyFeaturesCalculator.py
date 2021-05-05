import numpy as np
from scipy.signal import welch

from feature_extraction.HRVFeaturesCalculator import HRVFeaturesCalculator


class FrequencyFeaturesCalculator(HRVFeaturesCalculator):

    def __init__(self, nni_signal, sampling_frequency):
        super().__init__('linear', nni_signal)
        self.sf = sampling_frequency
        # Labels
        self.labels = {'lf': 'Low Frequency Power (LF)', 'hf': 'High Frequency Power (HF)',
                       'lf_hf': 'LF/HF', 'hf_lf': 'HF/LF'}

    #@private
    def __spectral_density(self):
        self.frequency_dist, self.power_dist = welch(self.nni, fs=self.sf, scaling='density')
        self.total_power = np.sum(self.power_dist)

    def get_lf(self):
        if not hasattr(self, 'lf'):
            if not hasattr(self, 'frequency_dist'):
                self.__spectral_density()
            low_freq_band = np.argwhere(self.frequency_dist < 0.15).reshape(-1)
            self.lf = np.sum(self.power_dist[low_freq_band]) / self.total_power
        return self.lf

    def get_hf(self):
        if not hasattr(self, 'hf'):
            if not hasattr(self, 'frequency_dist'):
                self.__spectral_density()
            high_freq_band = np.argwhere(self.frequency_dist > 0.4).reshape(-1)
            self.hf = np.sum(self.power_dist[high_freq_band]) / self.total_power
        return self.hf

    def get_lf_hf(self):
        return self.get_lf() / self.get_hf()

    def get_hf_lf(self):
        return self.get_hf() / self.get_lf()
