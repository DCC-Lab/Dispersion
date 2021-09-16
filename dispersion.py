import matplotlib.pyplot as plt
import numpy as np
from sympy import *


class Pulse:
    def __init__(self, duration, TL, wavelength):
        self.duration = duration
        self.TL = TL
        self.wavelength = wavelength
        self.time = np.linspace(-5000e-15, 5000e-15, 5000)
        self.IntensityTime = 1/((self.duration / 2.35482)*np.sqrt(2*np.pi))*np.exp(-self.time**2 / (2*(self.duration / 2.35482)**2))
        self.IntensityTimeTL = 1 / ((self.duration / 2.35482) * np.sqrt(2 * np.pi)) * np.exp(-self.time ** 2 / (2 * (self.duration / self.TL / 2.35482) ** 2))
        self.EfieldTime = np.sqrt(self.IntensityTimeTL)


    @property
    def frequencySpectra(self):
        self.EfieldFrequency = np.fft.fftshift(np.fft.fft(self.EfieldTime))
        self.IntensityFrequency = self.EfieldFrequency**2
        self.frequency  = np.fft.fftshift(np.fft.fftfreq(self.EfieldFrequency.shape[0],self.time[1]-self.time[0]))
        self.FWHMfreq = 2 * np.abs(self.frequency[np.argmin(np.abs(0.5*np.max(self.IntensityFrequency)-self.IntensityFrequency))])
        self.FWHMwavelength = self.wavelength**2 * self.FWHMfreq/3e8
        return self.frequency, self.EfieldFrequency, self.IntensityFrequency, self.FWHMfreq, self.FWHMwavelength


    def propagate(self, material, thickness):
        c = 299792458
        shiftedFreq = [i + c/self.wavelength for i in self.frequency]
        wavelengths = [c*10**6/j for j in shiftedFreq]

        x = Symbol('x')
        if material == "silica":
            n = (1 + 1.03961212 / (1 - 0.00600069867 / x ** 2) + 0.231792344 / (1 - 0.0200179144 / x ** 2) + 1.01046945 / (1 - 103.560653 / x ** 2))**.5
        elif material == "BK7":
            n = 1.2
        else:
            raise ValueError("chosen material not available")

        n = lambdify(x, n)  #input is in micrometers
        phaseTerm = [exp(i*n(c*10**6/i)*thickness*-1j/c) for i in shiftedFreq]
        EfieldFrequencyPropagated = [a*b for a,b in zip(self.EfieldFrequency, phaseTerm)]
        EfieldTimeOut = abs(np.fft.ifft(np.fft.ifftshift(EfieldFrequencyPropagated)))
        IntensityOut = EfieldTimeOut**2
        mu = self.time.dot(IntensityOut / IntensityOut.sum())
        mu2 = np.power(self.time, 2).dot(IntensityOut / IntensityOut.sum())
        var = mu2 - mu ** 2
        print("FWHM of new pulse : " + str(2.35482 * sqrt(var) * 10 ** 15))
        plt.plot(self.time, IntensityOut, '-r', self.time, self.IntensityTime, '.b')

        plt.show()
        # GVD = (self.wavelength*10**6)**3/(2*np.pi*(c)**2)*f.diff(x, 2) * 10**21

    def plotPulses(self):
        fig, axs = plt.subplots(2,1, figsize=(6,8))

        axs[0].set_title('Time domain')
        axs[0].plot(self.time, self.IntensityTime, label="Wavelength = %.1f nm"%(self.wavelength*10**9))
        axs[0].axvline(-self.duration*0.5, color='r', alpha=0.5, label='FWHM = %.1f fs'%(self.duration*1e15))
        axs[0].axvline(self.duration*0.5, color='r', alpha=0.5)

        axs[0].set_xlim(-350e-15, 350e-15)
        #axs[0].set_ylim(0,1.3)
        axs[0].set_xlabel('Time (sec)')

        axs[1].set_title('Frequency domain')
        axs[1].plot(self.frequency,self.IntensityFrequency)

        axs[1].axvline(-self.FWHMfreq*0.5, color='g', alpha=0.5, label='FWHM is %.1f THz or %0.2f nm'%(self.FWHMfreq*10**-12, self.FWHMwavelength*10**9))
        axs[1].axvline( self.FWHMfreq*0.5, color='g', alpha=0.5)

        axs[1].set_xlim(-1.0e13,1.0e13)
        #axs[1].set_ylim(0,30000)
        axs[1].set_xlabel('Frequency (Hz)')

        for ax in axs:
            ax.legend(fontsize=8)
            ax.set_ylabel('Electric field intensity (arbitrary units)')

        plt.tight_layout()
        plt.savefig('time-bandwidth-product.png', dpi=200)
        plt.show()

pulseTest = Pulse(90e-15, 1.13, 800e-9)
pulseTest.frequencySpectra
pulseTest.propagate("BK7", 0.15)
#pulseTest.plotPulses()