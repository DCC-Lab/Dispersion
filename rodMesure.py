import numpy as np
from pulse import Pulse
import matplotlib.pyplot as plt


I = complex(0, 1)
π = np.pi
c = 2.998e8

class Dispersion(Pulse):
    def __init__(self, 𝜏, λo, material=None):
        super().__init__(𝜏, λo)
        self.material = material
        if material is None:
            self.material = self.sf57
        self.𝜏 = 𝜏  # s
        self.𝝺o = 𝝺o  # m

        self.𝝳𝛎 = 15  # cm-1
        self.𝝳𝝺 = 1e7/self.𝝳𝛎  # nm
        self.𝝳𝝺 *= 1e-9
        self.𝝳t = np.sqrt(2*self.𝜏**2)
        self.groupDelay = self.𝝳t/self.𝝳𝝺  # s/m
        print(self.groupDelay)

    def phase(self, d, indexFct=None):
        if indexFct is None:
            indexFct = self.material
        𝜙 = np.array([2 * π / 𝜆 * indexFct(abs(𝜆)) * d for 𝜆 in np.fft.fftshift(self.wavelengths)])
        print(np.round((self.wavelengths[501]-self.wavelengths[500])*10**9,3))
        phaseFactor = np.exp(I * 𝜙)
        field = np.fft.fft(self.field)
        field *= phaseFactor
        field = np.fft.ifft(field)
        plt.plot(field)

    def phaseSpeed(self):
        pass


if __name__ == "__main__":
    pulse = Dispersion(100e-15, 1045e-9)
    pulse.phase(0.005)
    # plt.show()
