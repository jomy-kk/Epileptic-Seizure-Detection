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

    # Auxiliary private procedures

    # @private
    def __r(self, m):
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
    def __distance(self, X, i, j):
        dist = np.abs(X[i] - X[j])
        max= np.max(dist)
        return max

    # @private
    def get_sampen(self):
        if not hasattr(self,'sampen'):
            N = len(self.nni)
            Xa = []
            Xb = []

            # Get sequences of m matches
            for i in range(0, N - self.m + 1):
                X_m_i_b = self.nni[i:i+self.m]
                Xb.append(X_m_i_b)

                if i < N - self.m:
                    X_m_i_a = self.nni[i:(i+self.m+1)]
                    Xa.append(X_m_i_a)
            Xb = np.asarray(Xb)
            Xa = np.asarray(Xa)
            B_sum = 0
            A_sum = 0

            ra= self.__r(self.m + 1)
            rb= self.__r(self.m)

            for i in range(0, N - self.m):
                Bi = 0
                Ai = 0
                for j in range(0, N - self.m):
                    if j != i:
                        if self.__distance(Xb, i, j) <= rb:
                            Bi = Bi + 1
                        if i < N - self.m - 1 and j < N - self.m - 1:
                            if self.__distance(Xa, i, j) <= ra:
                                Ai = Ai + 1

                Bi = Bi / (N - self.m - 1)
                B_sum = B_sum + Bi
                #if i <= N - self.m - 1:
                Ai = Ai / (N - self.m - 1)
                A_sum = A_sum + Ai

            self.Bm = 1/(N - self.m) * B_sum
            self.Am = 1/(N - self.m) * A_sum
            self.sampen = math.log(self.Bm / self.Am)
        return self.sampen

    # @private
    def __entropyb(self):
        N = len(self.nni)
        X = []

        # Get sequences of m matches
        for i in range(0, N - self.m + 1):
            X_m_i = self.nni[i:i+self.m]
            X.append(X_m_i)
        X = np.asarray(X)

        B_sum = 0

        r = self.__r(self.m)

        B_sum = 0
        for i in range(0, N - self.m):
            Bi = 0
            for j in range(0, N - self.m):
                if j != i:
                    if self.__distance(X, i, j) <= r:
                        Bi = Bi + 1

            Bi = Bi / (N - self.m - 1)
            B_sum = B_sum + Bi

        self.Bm = 1/(N - self.m) * B_sum
        return self.Bm

    # @private
    def __entropya(self):
        N = len(self.nni)
        X = []

        # Get sequences of m matches
        for i in range(0, N - self.m):
            X_m_i = self.nni[i:i + self.m + 1]
            X.append(X_m_i)
        X = np.asarray(X)

        A_sum = 0
        r= self.__r(self.m + 1)
        for i in range(0, N - self.m - 1):
            Ai = 0
            for j in range(0, N - self.m - 1):
                if j != i:
                    if self.__distance(X, i, j) <= r:
                        Ai = Ai + 1

            Ai = Ai / (N - self.m - 1)
            A_sum = A_sum + Ai

        self.Am = 1 / (N - self.m) * A_sum

        return self.Am

    # Methods to publicly call

    def get_sampen_backup(self):
        self.sampen= math.log(self.__entropyb()/self.__entropya())
        return self.sampen

    def get_cosen(self):
        self.cosen= self.get_sampen() + math.log(2 * self.__r(self.m)) - math.log(self.nni.mean())
        return self.cosen

