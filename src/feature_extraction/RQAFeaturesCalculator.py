import math
import numpy as np

from feature_extraction.HRVFeaturesCalculator import HRVFeaturesCalculator


class RQAFeaturesCalculator(HRVFeaturesCalculator):
    """
    http://tux.uis.edu.co/geofractales/articulosinteres/PDF/waveform.pdf
    """

    def __init__(self, nni_signal):
        super().__init__('non-linear', nni_signal)
        self.__rr()

    # Labels
    labels = {'rec': 'REC', 'det': 'Determinant', 'lmax': 'Maximum L'}

    # @private
    def __rr(self):
        if not hasattr(self, 'rr'):
            self.rr = np.zeros((len(self.nni), len(self.nni)))
            for i in range(len(self.nni)):
                for j in range(len(self.nni)):
                    self.rr[i, j] = abs(self.nni[j] - self.nni[i])
            self.rr = np.heaviside((self.rr.mean() - self.rr), 0.5)
        return self.rr

    # @private
    def __diag_det_lmax(self):
        dim = len(self.__rr())
        assert dim == len(self.__rr()[0])
        return_grid = [[] for total in range(2 * len(self.__rr()) - 1)]
        for row in range(len(self.__rr())):
            for col in range(len(self.__rr()[row])):
                return_grid[row + col].append(self.__rr()[col][row])
        diags = [i for i in return_grid if 0.0 not in i and len(i) > 2]
        if len(diags) >= 1:
            diag_points = np.sum(np.hstack(diags))

            long_diag = np.max([len(diag) for diag in diags])
        else:
            diag_points = 0
            long_diag = 0

        return diag_points / (np.sum(self.__rr())), long_diag

    def get_rec(self):
        if not hasattr(self, 'rec'):
            self.rec = self.__rr().sum() / (len(self.nni) ** 2)
        return self.rec

    def get_det(self):
        if not hasattr(self, 'det'):
            self.det, self.lmax = self.__diag_det_lmax()
        return self.det

    def get_lmax(self):
        if not hasattr(self, 'lmax'):
            self.det, self.lmax = self.__diag_det_lmax()
        return self.lmax


