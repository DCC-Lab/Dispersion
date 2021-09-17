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
    def test03FourierTransformCalibration(self):
        """ If I understand FFTs, then the max frequency is f_max = 1 / 2 * T  and ∆f = 1/(2*T*N)
        It starts at zero, goes to f_max-df, then -f_max to -df"""
        T = 160e-15
        N = 16
        t = np.linspace(-T, T, N)
        dt = 2*T/N
        f_max = 1/2/dt
        df = 1/(2*T)

        frequencies = np.fft.fftfreq(N, dt)
        self.assertEqual(frequencies[0], 0)
        self.assertEqual(frequencies[1], df)
        self.assertEqual(frequencies[-1], -df)
        self.assertEqual(frequencies[int(N/2)], -f_max)
        self.assertEqual(frequencies[int(N/2)-1], f_max-df)

    def test04MoreFourierTransformCalibration(self):
        """ So I should be able to reconstruct the frequencies if I understand well """
        T = 160e-15
        N = 16
        t = np.linspace(-T, T, N)
        dt = 2*T/N
        f_max = 1/2/dt
        df = 1/(2*T)

        frequencies = np.fft.fftfreq(N, dt)
        myFrequencies = np.concatenate((np.linspace(0, f_max-df, int(N/2)-1), np.linspace(-f_max, -df, int(N/2))) )
        self.assertTrue(frequencies.all() == myFrequencies.all())

    def test05FourierTransformNormalization(self):
        """ Back and forth should give us the original field """
        
        T = 200e-15
        N = 128
        sigma = (100e-15)
        t = np.linspace(-T, T, N)
        dt = 2*T/N
        f_max = 1/2/dt
        df = 1/(2*T)

        temporalField = np.exp(-t*t/(sigma*sigma))
        fourierField = np.fft.fft(temporalField)
        tranformedField = np.fft.ifft(fourierField)
        self.assertTrue(temporalField.all() == tranformedField.all())

if __name__ == '__main__':
    unittest.main()
