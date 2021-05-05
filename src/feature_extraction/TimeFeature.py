import numpy as np

from feature_extraction.Feature import Feature
from feature_extraction.HRVFeaturesCalculator import HRVFeaturesCalculator


class TimeFeaturesCalculator(HRVFeaturesCalculator):

    def __init__(self, name, nni_signal, sampling_frequency):
        super().__init__(name, 'linear', nni_signal)
        self.sf = sampling_frequency

    def get_nn50(self):
        if not hasattr(self, 'nn50'):
            self.nn50 = len(np.argwhere(abs(np.diff(self.nni)) > 0.05 * self.sf))
        return Feature(self.nn50, 'NN50')

    def get_pnn50(self):
        if not hasattr(self, 'pnn50'):
            self.pnn50 = self.get_nn50() / len(self.nni)
        return Feature(self.pnn50, 'PPN50')

    def get_sdnn(self):
        if not hasattr(self, 'sdnn'):
            self.sdnn = self.nni.std()
        return Feature(self.sdnn, 'Std Dev NNI')

    def get_rmssd(self):
        if not hasattr(self, 'rmssd'):
            self.rmssd = ((np.diff(self.nni) ** 2).mean()) ** 0.5
        return Feature(self.rmssd, 'Root Mean Std Dev NNI')

    def get_mean(self):
        if not hasattr(self, 'mean'):
            self.mean = self.nni.mean()
        return Feature(self.mean, 'Mean NNI')

    def get_var(self):
        if not hasattr(self, 'var'):
            self.var = self.nni.var()
        return Feature(self.var, 'Variance NNI')
