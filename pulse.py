import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import numpy as np
from scipy.signal import hilbert, chirp
from matplotlib import colormaps
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

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
        self.S = S
        self.𝛕ₒ = 𝛕
        self.𝜆ₒ = 𝜆ₒ
        self.kₒ = 2 * π / 𝜆ₒ
        self.𝝎ₒ = self.kₒ * c
        self.fₒ = self.𝝎ₒ / 2 / π

        t = self.generateTimeSteps(N, S)
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
            indexFct = self.bk7
        self.indexFct = indexFct

        if np.mean(self.fieldEnvelope[0:10]) > 2e-4:
            self.S += self.S
            self.N += self.N
            self.time = self.generateTimeSteps(self.N, self.S)
            self.field = np.pad(self.field, (int(self.N/4),), 'constant', constant_values=(0,))
            print("Warning: temporal field reaching edges")


        𝜙 = np.array([2 * π / 𝜆 * indexFct(abs(𝜆)) * d for 𝜆 in self.wavelengths])

        phaseFactor = np.exp(I * 𝜙)
        field = np.fft.fft(self.field)
        field *= phaseFactor
        field = np.fft.ifft(field)

        self.field = field
        self.distancePropagated += d

        return self.time, field

    def setupPlot(self, title=""):
        plt.style.use(
            "https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")
        plt.title(title)
        plt.xlabel("Time [ps]")
        plt.ylabel("Amplitude [arb.u.]")
        plt.ylim(-1, 1)

        axis = plt.gca()
        axis.text(
            0.05,
            0.95,
            "Distance = {2:.0f} mm\n$\Delta t$ = {0:.0f} fs\n$\Delta \omega \\times \Delta t$ = {1:0.2f}".format(
                self.temporalWidth * 1e15,
                self.timeBandwidthProduct,
                self.distancePropagated * 1e3,
            ),
            transform=axis.transAxes,
            fontsize=14,
            verticalalignment="top",
        )

    def drawTemporalAndSpectral(self, title=""):
        fig, (axTime, axFreq) = plt.subplots(1, 2, figsize=(12, 5))

        plt.sca(axTime)
        self.setupPlot(title)
        self.drawEnvelope(axTime)
        self.drawChirpColour(axTime)

        self.drawSpectrum(axFreq)
        axFreq.set_title("Spectre")

        plt.tight_layout()
        return fig, axTime, axFreq

    def tearDownPlot(self):
        plt.close("all")
        

    def drawEnvelope(self, axis=None):
        if axis is None:
            axis = plt.gca()

        timeIsPs = self.time * 1e12
        axis.plot(timeIsPs, self.fieldEnvelope, "k-")

    def drawField(self, axis=None):
        if axis is None:
            axis = plt.gca()

        (
            instantTime,
            instantEnvelope,
            instantPhase,
            instantRadFrequency,
        ) = self.instantRadFrequency()

        timeIsPs = instantTime * 1e12
        axis.plot(timeIsPs, instantEnvelope * np.cos(instantPhase), "k-",  linewidth=0.5)

    def drawChirpColour(self, axis=None):
        if axis is None:
            axis = plt.gca()

        (
            instantTime,
            instantEnvelope,
            instantPhase,
            instantRadFrequency,
        ) = self.instantRadFrequency()

        # We want green for the center frequency (+0.33)
        normalizedFrequencyForColor = (instantRadFrequency - self.𝝎ₒ) / (
            5 * 2 * π * self.spectralWidth
        ) + 0.33

        hsv = colormaps["hsv"]
        M = 128

        instantTimeInPs = instantTime * 1e12
        step = len(instantTimeInPs) // M
        for i in range(0, len(instantTimeInPs) - step, step):
            t1 = instantTimeInPs[i]
            t2 = instantTimeInPs[i + step]
            c = normalizedFrequencyForColor[i + step // 2]
            e1 = instantEnvelope[i]
            e2 = instantEnvelope[i + step]
            axis.add_patch(
                Polygon([(t1, 0), (t1, e1), (t2, e2), (t2, 0)], facecolor=hsv(c))
            )

    def drawSpectrum(self, axis=None):
        if axis is None:
            axis = plt.gca()

        plt.style.use(
            "https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

        frequencies = self.frequencies
        spectrum = self.spectrum
        amplitudes = abs(spectrum)
        amplitudes /= amplitudes.max()

        positiveFreqs = np.extract(frequencies > 0, frequencies)
        positiveAmps = np.extract(frequencies > 0, amplitudes)
        positiveSpectrum = np.extract(frequencies > 0, spectrum)

        freqsInTHz = positiveFreqs * 1e-12
        axis.plot(freqsInTHz, positiveAmps, "k-")
        axis.set_xlabel("Frequency [THz]")
        axis.set_ylabel("Amplitude [arb.u.]")

        fₒInTHz = self.fₒ * 1e-12
        Δf = self.spectralWidth * 1e-12
        axis.set_xlim(fₒInTHz - 5 * Δf, fₒInTHz + 5 * Δf)

        # Spectral phase via group delay on the raw positive-frequency spectrum.
        # angle(S[k+1]*conj(S[k])) gives the wrapped phase step per bin. Unwrapping
        # the dPhase array is robust because the GVD variation between consecutive
        # bins is tiny (~0.002 rad). Subtracting the mean removes the group delay,
        # and cumsum recovers the GVD (quadratic) phase.
        mask = positiveAmps > 0.2
        maskedSpectrum = positiveSpectrum[mask]
        maskedFreqs = freqsInTHz[mask]

        dPhase = np.angle(maskedSpectrum[1:] * np.conj(maskedSpectrum[:-1]))
        dPhase = np.unwrap(dPhase)
        dPhase -= np.mean(dPhase)
        envPhase = np.concatenate(([0], np.cumsum(dPhase)))

        linearFit = np.polyfit(maskedFreqs, envPhase, 1)
        envPhase -= np.polyval(linearFit, maskedFreqs)

        axPhase = axis.twinx()
        axPhase.plot(maskedFreqs, envPhase, "r-")
        axPhase.set_ylabel("Phase [rad]")
        axPhase.set_ylim(-2*π,  2*π)
        axPhase.tick_params(axis="y")

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

    def water(self, wavelength):
        x = wavelength * 1e6
        if x < 0.3:
            x = 0.3
        elif x > 2.5:
            x = 2.5

        x2 = x * x
        n = (
            1
            + 5.672526103e-1 / (1 - 5.085550461e-3 / x2)
            + 1.736581125e-1 / (1 - 1.814938654e-2 / x2)
            + 2.121531502e-2 / (1 - 2.617260739e-2 / x2)
            + 1.138493213e-1 / (1 - 1.073888649e1 / x2)
        ) ** 0.5
        return n


if __name__ == "__main__":


    # All adjustable parameters below
    pulse = Pulse(𝛕=5e-15, 𝜆ₒ=805e-9) # 𝛕 must be the gaissian parameter in electric field
    # pulse = Pulse(𝛕=1.4142*151e-15, 𝜆ₒ=1045e-9) # 𝛕 must be the gaissian parameter in electric field

    # Material propertiues and distances, steps
    material = pulse.silica
    totalDistance = 0.001
    steps = 400

    # What to display on graph in addition to envelope?
    adjustTimeScale = True
    showCarrier = True
    showChirpColour = True

    # Save graph? (set to None to not save)
    filenameTemplate = "fig-{0:02d}.png" # Can use PDF but PNG for making movies with Quicktime Player

    # End adjustable parameters


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

        # fig, axTime , _ = pulse.drawTemporalAndSpectral()

        # # 𝛕 = pulse.temporalWidth*1e12
        # axTime.set_xlim(-1, 1)
        axTime = None
        pulse.setupPlot("Propagation in {0}".format(material.__func__.__name__))
        pulse.setupPlot()
        pulse.drawEnvelope(axTime)

        if showChirpColour:
            pulse.drawChirpColour(axTime)
    
        if showCarrier:
            pulse.drawField(axTime)

        if adjustTimeScale:
            𝛕 = pulse.temporalWidth*1e12
            plt.xlim(-5*𝛕, 5*𝛕)
        
        plt.draw()
        plt.pause(0.001)

        if filenameTemplate is not None:
            plt.savefig(filenameTemplate.format(j), dpi=300)
        pulse.tearDownPlot()

        pulse.propagate(stepDistance, material)
