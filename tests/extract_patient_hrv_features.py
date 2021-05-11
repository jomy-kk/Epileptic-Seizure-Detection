import unittest

from src.feature_extraction import *


class MyTestCase(unittest.TestCase):
    data_path = '../data'
    def test_something(self):
        res = extract_patient_hrv_features(15, 101, crises=1, _time=True)
        print(res['NN50']['2019-02-28 11:15:16.549354130'])
        assert (res['2019-02-28 11:15:16.549354130']['NN50'] == 551.640729)



if __name__ == '__main__':
    unittest.main()
