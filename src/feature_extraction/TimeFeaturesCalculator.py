import numpy as np

from feature_extraction.FeatureSequence import FeatureSequence
from feature_extraction.HRVFeaturesCalculator import HRVFeaturesCalculator


class TimeFeaturesCalculator(HRVFeaturesCalculator):
    
    def __init__(self, nni_signal, sampling_frequency):
        super().__init__('linear', nni_signal)
        
        self.sf = sampling_frequency
    
    # Labels
    nn50_label  = 'NN50'
    pnn50_label = 'PPN50'
    sdnn_label  = 'Std Dev NNI'
    rmssd_label = 'Root Mean Std Dev NNI'
    mean_label  = 'Mean NNI'
    var_label   = 'Variance NNI'
    

    def get_nn50(self):
        if not hasattr(self, 'nn50'):
            self.nn50 = len(np.argwhere(abs(np.diff(self.nni)) > 0.05 * self.sf))
        return self.nn50

    def get_pnn50(self):
        if not hasattr(self, 'pnn50'):
            self.pnn50 = self.get_nn50() / len(self.nni)
        return self.pnn50

    def get_sdnn(self):
        if not hasattr(self, 'sdnn'):
            self.sdnn = self.nni.std()
        return self.sdnn

    def get_rmssd(self):
        if not hasattr(self, 'rmssd'):
            self.rmssd = ((np.diff(self.nni) ** 2).mean()) ** 0.5
        return self.rmssd

    def get_mean(self):
        if not hasattr(self, 'mean'):
            self.mean = self.nni.mean()
        return self.mean

    def get_var(self):
        if not hasattr(self, 'var'):
            self.var = self.nni.var()
        return self.var
