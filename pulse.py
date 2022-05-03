import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import numpy as np
from scipy.signal import hilbert, chirp
from matplotlib import cm
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

        if np.mean(self.field[0:10]) > 2e-2:
            print("Warning: temporal field reaching edges")

        𝜙 = np.array([2 * π / 𝜆 * indexFct(abs(𝜆)) * d for 𝜆 in self.wavelengths])

        phaseFactor = np.exp(I * 𝜙)
        field = np.fft.fft(self.field)
        field *= phaseFactor
        field = np.fft.ifft(field)

        self.field = field
        self.distancePropagated += d

        return self.time, field

    def beginPlot(self, title=""):
        plt.style.use(
            "https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle"
        )
        plt.title(title)
        plt.xlabel("Time [ps]")
        plt.ylabel("Field amplitude [arb.u.]")

    def endPlot(self):
        plt.show()

    def beginPulse(self, axis=None):
        if axis is None:
            axis = plt.gca()

        title = axis.get_title()
        xlabel = axis.get_xlabel()
        ylabel = axis.get_ylabel()

        axis.clear()

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

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

    def endPulse(self, filenameTemplate):
        if filenameTemplate is not None:
            plt.savefig(filenameTemplate.format(j), dpi=300)

    def draw(self, axis=None, showChirp=True, showCarrier=False, adjustTimeScale=True, filenameTemplate=None):
        pulse.beginPulse(axis)
        pulse.drawEnvelope()
        pulse.drawChirpColour()
    
        if showCarrier:
            pulse.drawField()
            plt.ylim(-1,1)

        if adjustTimeScale:
            𝛕 = pulse.temporalWidth*1e12
            plt.xlim(-5*𝛕, 5*𝛕)
        
        plt.draw()
        plt.pause(0.001)
        pulse.endPulse(filenameTemplate)

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
        axis.plot(timeIsPs, instantEnvelope * np.cos(instantPhase), "k-")

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

        hsv = cm.get_cmap("hsv", 64)
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

def silica(wavelength):
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

def bk7(wavelength):
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

def sf10(wavelength):
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

def water(wavelength):
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


class Viewer:
    def __init__(self, pulse, title):
        self.pulse = pulse

        self.figure = None
        self.temporalAxis = None
        self.spectralAxis = None
        self.title = title
        self.xlabel = "Time [ps]"
        self.ylabel = "Field amplitude [arb.u.]"

    def beginPlot(self):
        self.figure = plt.subplot()

        plt.style.use(
            "https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle"
        )
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)

    def endPlot(self):
        plt.show()

    def beginPulse(self, axis=None):
        if axis is None:
            axis = plt.gca()
        else:
            axis = self.temporalAxis

        if axis is not None:
            title = axis.get_title()
            xlabel = axis.get_xlabel()
            ylabel = axis.get_ylabel()
            axis.clear()

        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)

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

    def endPulse(self, filenameTemplate):
        if filenameTemplate is not None:
            plt.savefig(filenameTemplate.format(j), dpi=300)

    def draw(self, axis=None, showChirp=True, showCarrier=False, adjustTimeScale=True, filenameTemplate=None):
        self.beginPulse(axis)
        self.drawEnvelope()
        self.drawChirpColour()
    
        if showCarrier:
            self.drawField()
            plt.ylim(-1,1)

        if adjustTimeScale:
            𝛕 = self.pulse.temporalWidth*1e12
            plt.xlim(-5*𝛕, 5*𝛕)
        
        plt.draw()
        plt.pause(0.001)
        self.endPulse(filenameTemplate)

    def drawEnvelope(self, axis=None):
        if axis is None:
            axis = plt.gca()

        timeIsPs = self.pulse.time * 1e12
        axis.plot(timeIsPs, self.pulse.fieldEnvelope, "k-")

    def drawField(self, axis=None):
        if axis is None:
            axis = plt.gca()

        (
            instantTime,
            instantEnvelope,
            instantPhase,
            instantRadFrequency,
        ) = self.pulse.instantRadFrequency()

        timeIsPs = instantTime * 1e12
        axis.plot(timeIsPs, instantEnvelope * np.cos(instantPhase), "k-")

    def drawChirpColour(self, axis=None):
        if axis is None:
            axis = plt.gca()

        (
            instantTime,
            instantEnvelope,
            instantPhase,
            instantRadFrequency,
        ) = self.pulse.instantRadFrequency()

        # We want green for the center frequency (+0.33)
        normalizedFrequencyForColor = (instantRadFrequency - self.pulse.𝝎ₒ) / (
            5 * 2 * π * self.pulse.spectralWidth
        ) + 0.33

        hsv = cm.get_cmap("hsv", 64)
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

if __name__ == "__main__":


    # All adjustable parameters below
    pulse = Pulse(𝛕=100e-15, 𝜆ₒ=800e-9)

    # Material propertiues and distances, steps
    material = pulse.bk7
    totalDistance = 0.3
    steps = 20

    # What to display on graph in addition to envelope?
    adjustTimeScale = False
    showCarrier = True
    showChirpColour = True
    # Save graph? (set to None to not save)
    filenameTemplate = "fig-{0:02d}.png" # Can use PDF but PNG for making movies with Quicktime Player

    # End adjustable parameters

    viewer = Viewer(pulse, "Propagation in {0}".format(material.__func__.__name__))
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

        viewer.draw(showChirpColour, showCarrier, adjustTimeScale, filenameTemplate)
        pulse.propagate(stepDistance, material)

    viewer.endPlot()

