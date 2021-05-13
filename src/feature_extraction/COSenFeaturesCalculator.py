import math
import numpy as np

from src.feature_extraction.HRVFeaturesCalculator import HRVFeaturesCalculator


class COSenFeaturesCalculator(HRVFeaturesCalculator):

    def __init__(self, nni_signal, m, g):
        super().__init__('non-linear', nni_signal)
        self.m = m
        self.g = g

    # Labels
    labels = {'sampen': 'Sample Entropy', 'cosen': 'COSen: Coefficient of Sample Entropy'}

    # @private
    def get_r(self, m):
        N = len(self.nni)
        X = []

        # Get sequences of m matches
        for i in range(0, N - m + 1):
            X_m_i = self.nni[i:i + m]
            X.append(X_m_i)
        X = np.asarray(X)

        r = self.g * np.std(X)
        return r

    # @private
    def get_distance(self, X, i, j):
        dist = np.abs(X[i] - X[j])
        max= np.max(dist)
        return max

    # @private
    def get_entropyb(self):
        N = len(self.nni)
        X = []

        # Get sequences of m matches
        for i in range(0,N - self.m + 1):
            X_m_i = self.nni[i:i+self.m]
            X.append(X_m_i)
        X = np.asarray(X)

        B_vector = []

        r = self.get_r(self.m)

        for i in range(0, N - self.m):
            Bi = 0
            for j in range(0, N - self.m):
                if j != i:
                    if self.get_distance(X, i, j) <= r:
                        Bi = Bi + 1

            Bi = Bi / (N - self.m - 1)
            B_vector.append(Bi)

        B_vector = np.asarray(B_vector)
        self.Bm = 1/(N - self.m) * B_vector.sum()
        return self.Bm

    # @private
    def get_entropya(self):
        N = len(self.nni)
        X = []

        # Get sequences of m matches
        for i in range(0, N - self.m):
            X_m_i = self.nni[i:i + self.m + 1]
            X.append(X_m_i)
        X = np.asarray(X)

        A_vector = []
        r= self.get_r(self.m + 1)
        for i in range(0, N - self.m - 1):
            Ai = 0
            for j in range(0, N - self.m - 1):
                if j != i:
                    if self.get_distance(X, i, j) <= r:
                        Ai = Ai + 1

            Ai = Ai / (N - self.m - 1)
            A_vector.append(Ai)

        A_vector = np.asarray(A_vector)

        self.Am = 1 / (N - self.m) * A_vector.sum()

        return self.Am

    def get_sampen(self):
        self.sampen= math.log(self.Bm/self.Am)
        return self.sampen

    def get_cosen(self):
        self.cosen= self.sampen + math.log(2*self.get_r(self.m)) - math.log(self.nni.mean())
        return self.cosen








