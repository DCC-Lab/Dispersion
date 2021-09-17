import unittest
import numpy as np

class TestFFTAndPulses(unittest.TestCase):
    def test01CreateTimeArray(self):
        """ Can I create an array properly? Does it include the limiting points?
        """
        T = 5000e-15
        N = 5000
        time = np.linspace(-T, T, N)
        self.assertIsNotNone(time)
        self.assertEqual(time[0],-T)
        self.assertEqual(time[N-1],T)
    def test02CreateElectricFieldArray(self):
        """ Can I create an electric field array? Is it large enough that the field is zero at
        ath edges? """
        T = 5000e-15
        N = 5000
        sigma = 100e-15
        t = np.linspace(-T, T, N)
        field = np.exp(-t*t/(sigma*sigma))
        self.assertIsNotNone(field)
        self.assertTrue(abs(field[0]) < 1e-6 )
        self.assertTrue(abs(field[N-1]) < 1e-6 )


if __name__ == '__main__':
    unittest.main()
