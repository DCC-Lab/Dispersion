import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import numpy as np
from scipy.signal import hilbert, chirp
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


# Shortcuts
I = complex(0, 1)
π = np.pi
c = 3e8


class Pulse:
    def __init__(self, 𝛕, 𝜆ₒ):
        N = 1024 * 16
        S = 40

        self.N = N
        self.𝛕ₒ = 𝛕
        self.𝜆ₒ = 𝜆ₒ
        self.kₒ = 2 * π / 𝜆ₒ
        self.𝝎ₒ = self.kₒ * c
        self.fₒ = self.𝝎ₒ / 2 / π

        self.time = np.linspace(-𝛕 * S, 𝛕 * S, N)

        self.fieldInTime = np.exp(-(self.time * self.time) / (𝛕 * 𝛕)) * np.cos(
            self.𝝎ₒ * self.time
        )
        self.distancePropagated = 0

    @property
    def dt(self):
        return self.time[1] - self.time[0]

    @property
    def frequencies(self):
        return np.fft.fftfreq(len(self.fieldInTime), self.dt)

    @property
    def wavelengths(self):
        return c / (self.frequencies + 0.01)  # avoid c/0

    @property
    def spectrum(self):
        return np.fft.fft(self.fieldInTime)

    @property
    def spectralWidth(self):
        frequencies = self.frequencies
        positiveFrequencies = np.extract(frequencies > 0, frequencies)
        amplitudes = np.extract(frequencies > 0, abs(self.spectrum))

        return self.rms(positiveFrequencies, amplitudes)

    @property
    def temporalWidth(self):
        return self.rms(self.time, self.envelope)

    def rms(self, x, y):
        sumY = np.sum(y)
        meanX = np.sum(x * y) / sumY
        meanX2 = np.sum(x * x * y) / sumY
        return np.sqrt(meanX2 - meanX * meanX)

    @property
    def timeBandwidthProduct(self):
        return 2 * π * self.spectralWidth * self.temporalWidth

    @property
    def envelope(self):
        return np.abs(self.analyticSignal)

    def instantFrequency(self):
        # Extract envelope and carrier
        analyticSignal = self.analyticSignal

        instantEnvelope = np.abs(analyticSignal)
        instantPhase = np.unwrap(np.angle(analyticSignal))
        instantFrequency = np.diff(instantPhase) * 1 / self.dt

        instantFrequency = np.extract(instantEnvelope[0:-1] > 0.001, instantFrequency)
        instantTime = np.extract(instantEnvelope[0:-1] > 0.001, self.time)
        instantEnvelope = np.extract(instantEnvelope[0:-1] > 0.001, instantEnvelope)

        return instantTime, instantEnvelope, instantPhase, instantFrequency

    @property
    def analyticSignal(self):
        analyticSignal = hilbert(self.fieldInTime.real)

        # Center maximum at t=0
        maxIndex = np.argmax(np.abs(analyticSignal))
        centerIndex = len(analyticSignal) // 2
        deltaRoll = centerIndex - maxIndex
        analyticSignal = np.roll(analyticSignal, deltaRoll)
        return analyticSignal

    def propagate(self, d, indexFct=None):
        if indexFct is None:
            indexFct = self.bk7
        𝜙 = np.array([2 * π / 𝜆 * indexFct(abs(𝜆)) * d for 𝜆 in self.wavelengths])

        phaseFactor = np.exp(-I * 𝜙)
        field = np.fft.fft(self.fieldInTime)
        field *= phaseFactor
        field = np.fft.ifft(field)

        self.fieldInTime = field
        self.distancePropagated += d

        return self.time, field

    def plotEnvelope(self, axis=None):
        timeIsPs = self.time * 1e12
        plt.plot(timeIsPs, self.envelope, 'k-')

        (
            instantTime,
            instantEnvelope,
            instantPhase,
            instantFrequency,
        ) = self.instantFrequency()
        normalizedFrequency = (
            -(instantFrequency - self.𝝎ₒ) / (5 * 2 * π * self.spectralWidth) + 0.33
        )

        if axis is None:
            axis = plt.gca()

        instantTimeInPs = instantTime * 1e12
        hsv = cm.get_cmap("hsv", 256)
        M = 128
        step = len(instantTimeInPs) // M
        for i in range(0, len(instantTimeInPs) - step, step):
            t1 = instantTimeInPs[i]
            t2 = instantTimeInPs[i + step]
            f  = normalizedFrequency[i + step // 2]
            e1 = instantEnvelope[i]
            e2 = instantEnvelope[i + step]
            axis.add_patch(
                Polygon([(t1, 0), (t1, e1), (t2, e2), (t2, 0)], facecolor=hsv(f))
            )

    def silica(self, wavelength):
        x = wavelength * 1e6
        if x < 0.3:
            x = 0.3
        elif x > 2.5:
            x = 2.5
        return (
            1
            + 1.03961212 / (1 - 0.00600069867 / x**2)
            + 0.231792344 / (1 - 0.0200179144 / x**2)
            + 1.01046945 / (1 - 103.560653 / x**2)
        ) ** 0.5

    def bk7(self, wavelength):
        x = wavelength * 1e6
        if x < 0.3:
            x = 0.3
        elif x > 2.5:
            x = 2.5
        n = (
            1
            + 1.03961212 / (1 - 0.00600069867 / x**2)
            + 0.231792344 / (1 - 0.0200179144 / x**2)
            + 1.01046945 / (1 - 103.560653 / x**2)
        ) ** 0.5
        return n

    def sf10(self, wavelength):
        x = wavelength * 1e6
        if x < 0.3:
            x = 0.3
        elif x > 2.5:
            x = 2.5
        return (
            1
            + 1.62153902 / (1 - 0.0122241457 / x**2)
            + 0.256287842 / (1 - 0.0595736775 / x**2)
            + 1.64447552 / (1 - 147.468793 / x**2)
        ) ** 0.5


if __name__ == "__main__":

    pulse = Pulse(𝛕=100e-15, 𝜆ₒ=800e-9)

    print("#\td\t∆t[ps]\t∆𝝎[THz]\tProduct")
    for j in range(10):
        print("{0}\t{1:0.3f}\t{2:0.3f}\t{3:0.3f}\t{4:0.3f}".format(j, pulse.distancePropagated, pulse.temporalWidth*1e12, 2*π*pulse.spectralWidth*1e-12, pulse.timeBandwidthProduct))

        pulse.plotEnvelope()
        plt.ylim(0, 1)
        plt.xlim(-5*pulse.𝛕ₒ*1e12, 5*pulse.𝛕ₒ*1e12)
        plt.savefig("fig-{0:02d}.png".format(j))
        plt.clf()

        pulse.propagate(10e-2)
