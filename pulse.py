import numpy as np
from scipy.signal import hilbert, chirp
from materials import *

# You could import materials from Raytracing:
# from raytracing.materials import *
# print(Material.all())

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

        dt = 1/self.fₒ/16
        T = S * self.𝛕ₒ
        N = int(2*T / dt)
        print(N)
        t = np.linspace(-T, T, N)
        self.field = np.exp(-(t * t) / (𝛕 * 𝛕)) * np.cos(self.𝝎ₒ * t)
        self.time = t
        self.distancePropagated = 0

    def generateTimeSteps(self, N, S):
        return np.linspace(-self.𝛕ₒ * S, self.𝛕ₒ * S, N)

    @property
    def dt(self):
        return self.time[1] - self.time[0]

    @property
    def frequencies(self):
        return np.fft.fftfreq(len(self.field), self.dt)

    @property
    def wavelengths(self):
        return c / (self.frequencies + 0.01)  # avoid c/0

    @property
    def spectrum(self):
        return np.fft.fft(self.field)

    @property
    def spectralWidth(self):
        frequencies = self.frequencies
        positiveFrequencies = np.extract(frequencies > 0, frequencies)
        amplitudes = np.extract(frequencies > 0, abs(self.spectrum))

        return self.rms(positiveFrequencies, amplitudes)

    @property
    def temporalWidth(self):
        return self.rms(self.time, self.fieldEnvelope)

    def rms(self, x, y):
        sumY = np.sum(y)
        meanX = np.sum(x * y) / sumY
        meanX2 = np.sum(x * x * y) / sumY
        return np.sqrt(meanX2 - meanX * meanX)

    @property
    def timeBandwidthProduct(self):
        return 2 * π * self.spectralWidth * self.temporalWidth

    @property
    def fieldEnvelope(self):
        return np.abs(self.analyticSignal)

    def instantRadFrequency(self):
        # Extract envelope and carrier
        analyticSignal = self.analyticSignal

        instantEnvelope = np.abs(analyticSignal)
        instantPhase = np.unwrap(np.angle(analyticSignal))
        instantRadFrequency = np.diff(instantPhase) * 1 / self.dt

        instantRadFrequency = np.extract(
            instantEnvelope[0:-1] > 0.001, instantRadFrequency
        )
        instantTime = np.extract(instantEnvelope[0:-1] > 0.001, self.time)
        instantPhase = np.extract(instantEnvelope[0:-1] > 0.001, instantPhase)
        instantEnvelope = np.extract(instantEnvelope[0:-1] > 0.001, instantEnvelope)

        return instantTime, instantEnvelope, instantPhase, instantRadFrequency

    @property
    def analyticSignal(self):
        analyticSignal = hilbert(self.field.real)

        # Center maximum at t=0
        maxIndex = np.argmax(np.abs(analyticSignal))
        centerIndex = len(analyticSignal) // 2
        deltaRoll = centerIndex - maxIndex
        analyticSignal = np.roll(analyticSignal, deltaRoll)
        return analyticSignal

    def propagate(self, d, indexFct=None):
        if indexFct is None:
            indexFct = bk7

        if np.mean(self.field[0:10]) > 2e-2:
            print("Warning: temporal field reaching edges")

        𝜙 = np.array([2 * π / 𝜆 * indexFct(abs(𝜆)) * d for 𝜆 in self.wavelengths])

        phaseFactor = np.exp(I * 𝜙)
        field = np.fft.fft(self.field)
        field *= phaseFactor
        field = np.fft.ifft(field)

        self.field = field.real
        self.distancePropagated += d

        return self.time, field


if __name__ == "__main__":
    from viewer import *

    # All adjustable parameters below
    pulse = Pulse(𝛕=10e-15, 𝜆ₒ=800e-9)

    # Material properties and distances, steps
    material = bk7
    totalDistance = 0.3e-2
    steps = 20

    # What to display on graph in addition to envelope?
    adjustTimeScale = True
    showCarrier = True
    showChirpColour = True
    # Save graph? (set to None to not save)
    filenameTemplate = "fig-{0:03d}.png" # Can use PDF but PNG for making movies with Quicktime Player

    # End adjustable parameters

    viewer = Viewer(pulse, "Propagation in {0}".format(material.__name__))
    viewer.beginPlot()

    print("#\td[mm]\t∆t[ps]\t∆𝝎[THz]\tProduct")
    stepDistance = totalDistance / steps
    for j in range(steps):
        print(
            "{0}\t{1:.1f}\t{2:0.3f}\t{3:0.3f}\t{4:0.3f}".format(
                j,
                pulse.distancePropagated * 1e3,
                pulse.temporalWidth * 1e12,
                2 * π * pulse.spectralWidth * 1e-12,
                pulse.timeBandwidthProduct,
            )
        )

        viewer.draw(None, showChirpColour, showCarrier, adjustTimeScale, filenameTemplate.format(j))
        pulse.propagate(stepDistance, material)

    viewer.endPlot()

