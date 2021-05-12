import math
import numpy as np

from src.feature_extraction.HRVFeaturesCalculator import HRVFeaturesCalculator


class COSenFeaturesCalculator(HRVFeaturesCalculator):

    def __init__(self, nni_signal):
        super().__init__('non-linear', nni_signal)

    # Labels
    labels = {'se': 'Sample Entropy', 'cosen': 'COSen: Coefficient of Sample Entropy'}

    #@private
    def distance(self, X, i, j):
        dist = np.abs(X[i] - X[j])
        return np.max(dist)

    # @private
    def get_entropyb(self, 'entropyb', m, g):
        N = len(self.nni)
        X = []

        # Get sequences of m matches
        for i in range(0,N - m - 1):
            X_m_i = self.nni[i:i+m-1]
            X.append(X_m_i)
        X = np.asarray(X)

        r = g * np.std(X)

        B_vector = []

        for i in range(0, N - m):
            Bi = 0
            for j in range(0, N - m):
                if j != i:
                    if distance(X, i, j) <= r:
                        Bi = Bi + 1

            Bi = Bi / (N - m - 1)
            B_vector.append(Bi)

        B_vector = np.asarray(B_vector)

        Bm = 1/(N - m) * B_vector.sum()






