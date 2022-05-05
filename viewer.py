from pulse import *
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

class Viewer:
    def __init__(self, pulse, title):
        self.pulse = pulse

        self.figure = None
        self.temporalAxis = None
        self.spectralAxis = None
        self.title = title
        self.xlabel = "Time [ps]"
        self.ylabel = "Field amplitude [arb.u.]"

        self.showChirp = True
        self.showCarrier = False
        self.adjustTimeScale = True

    def showPropagation(self, totalDistance, material, steps,):
        self.pulse.doPropagation(totalDistance, material, steps=steps)

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
        axis.set_ylim(0,1)

        axis.text(
            0.05,
            0.95,
            "Distance = {2:.1f} mm\n$\Delta t$ = {0:.0f} fs\n$\Delta \omega \\times \Delta t$ = {1:0.2f}".format(
                self.pulse.temporalWidth * 1e15,
                self.pulse.timeBandwidthProduct,
                self.pulse.distancePropagated * 1e3,
            ),
            transform=axis.transAxes,
            fontsize=14,
            verticalalignment="top",
        )

    def endPulse(self, filename):
        if filename is not None:
            plt.savefig(filename, dpi=300)

    def draw(self, axis=None, showChirp=True, showCarrier=False, adjustTimeScale=True, filename=None):
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
        self.endPulse(filename)

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
    from viewer import *

    # All adjustable parameters below
    pulse = Pulse(𝛕=5e-15, 𝜆ₒ=800e-9, S=100)

    # Material properties and distances, steps
    material = bk7
    totalDistance = 1e-3
    steps = 400

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
            "{0}\t{1:.2f}\t{2:0.3f}\t{3:0.3f}\t{4:0.3f}".format(
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

