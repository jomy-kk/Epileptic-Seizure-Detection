import numpy as np

from src.feature_extraction.HRVFeaturesCalculator import HRVFeaturesCalculator


class TimeFeaturesCalculator(HRVFeaturesCalculator):
    
    def __init__(self, nni_signal, sampling_frequency):
        super().__init__('linear', nni_signal)
        self.sf = sampling_frequency

    # Labels
    labels = {'nn50': 'NN50', 'pnn50': 'PPN50', 'sdnn': 'Std Dev NNI', 'rmssd': 'Root Mean Std Dev NNI',
                   'mean': 'Mean NNI', 'var': 'Variance NNI', 'hr': 'Mean HR', 'maxhr': 'maximum beats per minute'}

    def get_nn50(self):
        if not hasattr(self, 'nn50'):
            self.nn50 = len(np.argwhere(abs(np.diff(self.nni.to_numpy(),axis=0)) > 0.05 * self.sf))
        return self.nn50

    def get_pnn50(self):
        if not hasattr(self, 'pnn50'):
            self.pnn50 = self.get_nn50() / len(self.nni)
        return self.pnn50

    def get_sdnn(self):
        if not hasattr(self, 'sdnn'):
            self.sdnn = np.std(self.nni.to_numpy())
        return self.sdnn

    def get_rmssd(self):
        if not hasattr(self, 'rmssd'):
            self.rmssd = ((np.diff(self.nni.to_numpy(),axis=0) ** 2).mean()) ** 0.5
        return self.rmssd

    def get_mean(self):
        if not hasattr(self, 'mean'):
            self.mean = np.mean(self.nni.to_numpy())
        return self.mean

    def get_var(self):
        if not hasattr(self, 'var'):
            self.var = np.var(self.nni.to_numpy())
        return self.var

    def get_hr(self):
        if not hasattr(self, 'hr'):
            self.hr = 60*1000/np.mean(self.nni.to_numpy()) #60000 ms (1 minute) / average nni
        return self.hr

    def get_maxhr(self):
        if not hasattr(self, 'maxhr'):
            self.maxhr = np.amax(60*1000/self.nni.to_numpy())
        return self.maxhr
