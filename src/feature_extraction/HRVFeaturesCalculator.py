import numpy as np

class HRVFeaturesCalculator:
    def __init__(self, name: str, type: str, nni_signal: np.array):
        self.name = name
        assert type == 'linear' or type == 'non-linear'
        self.type = type
        self.nni = nni_signal

    def __repr__(self):
        return self.name
