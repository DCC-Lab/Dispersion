import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

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
    def __init__(self, 𝛕, 𝜆ₒ, S=40):
        self.𝛕ₒ = 𝛕
        self.𝜆ₒ = 𝜆ₒ
        self.kₒ = 2 * π / 𝜆ₒ
        self.𝝎ₒ = self.kₒ * c
        self.fₒ = self.𝝎ₒ / 2 / π

        dt = 1/self.fₒ/16
        T = S * self.𝛕ₒ
        N = int(2*T / dt)
        t = np.linspace(-T, T, N, endpoint=True)
        self.field = np.exp(-(t * t) / (𝛕 * 𝛕)) * np.cos(self.𝝎ₒ * t)
        self.time = t
        self.distancePropagated = 0
        self.globalPhase = 0

    @property
    def dt(self):
        return self.time[1] - self.time[0]

    @property
    def frequencies(self):
        return np.fft.fftfreq(len(self.time), self.dt)

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

    def doPropagation(self, totalDistance, indexFct=None, steps=20):
        stepDistance = totalDistance / steps
        
        print("#\td[mm]\t∆t[ps]\t∆𝝎[THz]\tProduct")
        for j in range(steps):
            print(
                "{0}\t{1:.3f}\t{2:0.4f}\t{3:0.4f}\t{4:0.3f}".format(
                    j,
                    self.distancePropagated * 1e3,
                    self.temporalWidth * 1e12,
                    2 * π * self.spectralWidth * 1e-12,
                    self.timeBandwidthProduct,
                )
            )

            self.propagate(stepDistance, material)


    def propagate(self, d, indexFct):
        𝜙 = np.array([2 * π / 𝜆 * indexFct(abs(𝜆)) * d for 𝜆 in self.wavelengths])

        phaseFactor = np.exp(I * 𝜙 )

        field = np.fft.fft(self.field)
        field *= phaseFactor
        field = np.fft.ifft(field)

        self.field = field.real
        self.distancePropagated += d

        return self.time, field


if __name__ == "__main__":
    # All adjustable parameters below
    pulse = Pulse(𝛕=50e-15, 𝜆ₒ=800e-9)

    # Material properties and distances, steps
    material = bk7
    totalDistance = 1
    steps = 100

    # End adjustable parameters

    pulse.doPropagation(totalDistance, indexFct=material, steps=steps)
