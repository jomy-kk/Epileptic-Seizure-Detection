import math
import numpy as np

from src.feature_extraction.HRVFeaturesCalculator import HRVFeaturesCalculator


class KatzFeaturesCalculator(HRVFeaturesCalculator):
    """
    http://tux.uis.edu.co/geofractales/articulosinteres/PDF/waveform.pdf
    """

    def __init__(self, nni_signal):
        super().__init__('non-linear', nni_signal)

    # Labels
    labels = {'katz_fractal_dim': 'Katz fractal dimension'}

    def get_katz_fractal_dim(self):
        if not hasattr(self, 'katz_fractal_dim'):
            for j in range(len(self.nni)):
                d_kfd = np.max([((1 - j) ** 2 + (self.nni.iloc[0,0] - self.nni.iloc[j,0]) ** 2) ** 0.5 for j in range(len(self.nni))])
                l_kfd = np.sum([(1 + (self.nni.iloc[i,0] - self.nni.iloc[i + 1,0]) ** 2) ** 0.5 for i in range(len(self.nni) - 1)])
            self.katz_fractal_dim = math.log10(l_kfd) / math.log10(d_kfd)
        return self.katz_fractal_dim
