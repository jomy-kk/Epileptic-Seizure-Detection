import math
import numpy as np

from feature_extraction.HRVFeaturesCalculator import HRVFeaturesCalculator


class PointecareFeaturesCalculator(HRVFeaturesCalculator):

    def __init__(self, nni_signal):
        super().__init__('non-linear', nni_signal)
        # Labels
        self.labels = {'sd1': 'SD1', 'sd2': 'SD2', 'csi': 'CSI (SD2/SD1)', 'csv': 'CSV'}

    def get_sd1(self):
        if not hasattr(self, 'sd1'):
            x1 = np.asarray(self.nni[:-1])
            x2 = np.asarray(self.nni[1:])
            self.sd1 = np.std(np.subtract(x1, x2) / np.sqrt(2))
        return self.sd1

    def get_sd2(self):
        if not hasattr(self, 'sd2'):
            x1 = np.asarray(self.nni[:-1])
            x2 = np.asarray(self.nni[1:])
            self.sd2 = np.std(np.add(x1, x2) / np.sqrt(2))
        return self.sd1

    def get_csi(self):
        return self.get_sd2() / self.get_sd1()

    def get_csv(self):
        if not hasattr(self, 'csv'):
            self.csv = math.log10(self.get_sd1() * self.get_sd2())
        return self.csv
