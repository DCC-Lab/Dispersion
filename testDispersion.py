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
        """ FFT-iFFT Back and forth should give us the original field """

        T = 1000e-15 # if this is too large, it does not work
        N = 256      # if this is too laarge, it does not work. Probably rounding error
        sigma = (100e-15)
        t = np.linspace(-T, T, N)
        dt = 2*T/N
        f_max = 1/2/dt
        df = 1/(2*T)

        temporalField = np.exp(-t*t/(sigma*sigma))
        fourierField = np.fft.fftshift(np.fft.fft(temporalField))
        retranformedField = np.fft.ifft(np.fft.ifftshift(fourierField))
        self.assertTrue(temporalField.all() == retranformedField.all())

    def test06FourierTransformAnOffsetInTimeIsAPhaseShiftInFrequencySpace(self):
        """ I know that if I shift somethign in time by to, the spectrum will have a linear
        phase added of -2pi*to*f"""

        T = 5000e-15 
        N = 5000      
        sigma = (100e-15)
        t = np.linspace(-T, T, N)
        dt = 2*T/N
        f_max = 1/2/dt
        df = 1/(2*T)
        to = 10e-15

        temporalField = np.exp(-(t*t)/(sigma*sigma))
        originalFourierField = np.fft.fft(temporalField)

        shiftedTemporalField = np.exp(-(t-to)*(t-to)/(sigma*sigma))
        shiftedFourierField = np.fft.fft(shiftedTemporalField)

        phaseDifference = np.angle(shiftedFourierField/originalFourierField)
        self.assertAlmostEqual(phaseDifference[0], 0)
        self.assertAlmostEqual(phaseDifference[1], -df*to*2*np.pi, 4)
        frequencies = np.fft.fftfreq(N, dt)
        self.assertAlmostEqual(phaseDifference.all(), (-frequencies*to*2*np.pi).all(), 4)

    def test07GetPulseDurationWithRMSwidthofPulse(self):
        """I want to see if calculating the RMS width of the pulse gives the same value as the input
        duration parameter"""

        T = 10000e-15
        N = 5000
        sigma = (200e-15)
        t = np.linspace(-T, T, N)

        temporalField = np.exp(-(t * t) / (2 * sigma * sigma))
        mu = t.dot(temporalField / temporalField.sum())
        mu2 = np.power(t, 2).dot(temporalField / temporalField.sum())
        RMS = np.sqrt(mu2 - mu ** 2)
        self.assertTrue(RMS == sigma)

    def test08timeBandwithProductOfPulseAndFFT(self):
        """I want to see if multiplying the FWHM of the time pulse and frequency pulse gives correct
        result, which is 0.44 for FWHM width of a gaussian"""

        T = 10000e-15
        N = 5000
        sigma = (100e-15)
        t = np.linspace(-T, T, N)
        dt = 2 * T / N
        frequencies = np.fft.fftfreq(N, dt)

        temporalIntensity = np.exp(-(t * t) / (2 * sigma * sigma))
        temporalField = np.sqrt(temporalIntensity)
        FourierField = np.fft.fft(temporalField)
        FourierIntensity = abs(FourierField)**2
        mu = frequencies.dot(FourierIntensity / FourierIntensity.sum())
        mu2 = np.power(frequencies, 2).dot(FourierIntensity/ FourierIntensity.sum())
        var = mu2 - mu ** 2
        RMSfreq = np.sqrt(var)


        self.assertAlmostEqual(RMSfreq*sigma*2.35482*2.35482, 0.44, 2)

        """I tried making this initially with the RMS width of a time-bandwidth product = 0.5, however
        I could not make it work. From my understanding FWHM = 2.35482*RMS, so I don't see how the product of RMS width
        can give a larger value."""

if __name__ == '__main__':
    unittest.main()
