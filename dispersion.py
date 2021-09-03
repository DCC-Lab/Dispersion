import matplotlib.pyplot as plt
import numpy as np
from sympy import *


class Pulse:
    def __init__(self, duration, TL, wavelength):
        self.duration = duration
        self.TL = TL
        self.wavelength = wavelength

    def timeProfile(self):
        time = np.linspace(-1000e-15, 1000e-15, 4000)
        IntensityTime = 1/((self.duration / 2.35482)*np.sqrt(2*np.pi))*np.exp(-(time)**2 / (2*(self.duration / 2.35482)**2))
        IntensityTimeNonTL = 1/((self.duration / 2.35482)*np.sqrt(2*np.pi))*np.exp(-time**2 / (2*(self.duration / self.TL / 2.35482)**2))
        EfieldTime = np.sqrt(IntensityTimeNonTL)
        return time, IntensityTime, EfieldTime


    def spectralWidth(self, time, timeSpectra):
        EfieldFrequency = np.abs(np.fft.fftshift(np.fft.fft(timeSpectra)))
        IntensityFrequency = EfieldFrequency**2
        frequency  = np.fft.fftshift(np.fft.fftfreq(EfieldFrequency.shape[0],time[1]-time[0]))
        FWHMfreq = 2 * np.abs(frequency[ np.argmin(np.abs(0.5*np.max(IntensityFrequency)-IntensityFrequency))])
        FWHMwavelength = self.wavelength**2 * FWHMfreq/3e8
        return frequency, EfieldFrequency, IntensityFrequency, FWHMfreq, FWHMwavelength


    def dispersion(self, frequencies, time, EfieldFrequency, IntensityTime, thickness, FWHMwavelength):
        c = 299792458
        shiftedFreq = [i + c/self.wavelength for i in frequencies]
        wavelengths = [c*10**6/j for j in shiftedFreq]
        x = Symbol('x')
        f = (1 + 1.03961212 / (1 - 0.00600069867 / x ** 2) + 0.231792344 / (1 - 0.0200179144 / x ** 2) + 1.01046945 / (1 - 103.560653 / x ** 2))**.5
        GVD = (self.wavelength*10**6)**3/(2*np.pi*(c)**2)*f.diff(x, 2) * 10**21
        GVD = lambdify(x, GVD)  #input is in micrometers
        phaseTerm = [(exp(thickness*10**3 * GVD(c*10**6/(i + c/self.wavelength))*i**2*-1j/2)) for i in frequencies]
        Eout = [x*y for x,y in zip(phaseTerm, EfieldFrequency)]

        EfieldTimeOut = np.fft.fftshift(np.abs(np.fft.ifft(np.fft.ifftshift(Eout))))

        plt.plot(time, EfieldTimeOut)
        #plt.xlim((self.wavelength-6*FWHMwavelength)*10**6, (self.wavelength+6*FWHMwavelength)*10**6)
        #plt.ylim(0.9, 1.8)
        plt.show()


    def plotPulses(self, time, IntensityTime, frequency, IntensityFrequency, FWHMfreq, FWHMwavelength):
        fig, axs = plt.subplots(2,1, figsize=(6,8))

        axs[0].set_title('Time domain')
        axs[0].plot(time, IntensityTime, label="Wavelength = %.1f nm"%(self.wavelength*10**9))
        axs[0].axvline(-self.duration*0.5, color='r', alpha=0.5, label='FWHM = %.1f fs'%(self.duration*1e15))
        axs[0].axvline(self.duration*0.5, color='r', alpha=0.5)

        axs[0].set_xlim(-350e-15, 350e-15)
        #axs[0].set_ylim(0,1.3)
        axs[0].set_xlabel('Time (sec)')

        axs[1].set_title('Frequency domain')
        axs[1].plot(frequency,IntensityFrequency)

        axs[1].axvline(-FWHMfreq*0.5, color='g', alpha=0.5, label='FWHM is %.1f THz or %0.2f nm'%(FWHMfreq*10**-12, FWHMwavelength*10**9))
        axs[1].axvline( FWHMfreq*0.5, color='g', alpha=0.5)

        axs[1].set_xlim(-1.0e13,1.0e13)
        #axs[1].set_ylim(0,30000)
        axs[1].set_xlabel('Frequency (Hz)')

        for ax in axs:
            ax.legend(fontsize=8)
            ax.set_ylabel('Electric field intensity (arbitrary units)')

        plt.tight_layout()
        plt.savefig('time-bandwidth-product.png', dpi=200)
        plt.show()

pulseTest = Pulse(120e-15, 1, 1050e-9)
(t, It, Et) = pulseTest.timeProfile()
(f, Ef, If, w1, w2) = pulseTest.spectralWidth(t, Et)
pulseTest.dispersion(f, t, Ef, It, 0.02, w2)
pulseTest.plotPulses(t, It, f, If, w1, w2)