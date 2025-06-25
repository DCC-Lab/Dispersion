import matplotlib.pyplot as plt
import numpy as np
from sympy import *


# This file is deprecated.  Use pulse.py instead.
class Pulse:
    def __init__(self, duration, TL, wavelength):
        print("This file is deprecated.  Use pulse.py instead and its Pulse class.")
        self.duration = duration
        self.TL = TL
        self.wavelength = wavelength
        self.time = np.linspace(-10000e-15, 10000e-15, 5000) #find a way to define with wavelength and duration
        self.IntensityTime = 1/((self.duration / 2.35482)*np.sqrt(2*np.pi))*np.exp(-self.time**2 / (2*(self.duration / 2.35482)**2))
        self.EfieldTime = np.sqrt(self.IntensityTime)*np.exp(1j/(4*(self.duration / 2.35482)**2)*np.sqrt(self.TL**2 - 1)*self.time**2)
        self.IntensityTimePropagated = []
        self.durationPropagated = 0
        self.EfieldFrequency = None


    @property
    def frequencySpectra(self):
        self.EfieldFrequency = np.fft.fftshift(np.fft.fft(self.EfieldTime))
        self.IntensityFrequency = abs(self.EfieldFrequency)**2
        self.frequency  = np.fft.fftshift(np.fft.fftfreq(self.EfieldFrequency.shape[0],self.time[1]-self.time[0]))
        self.FWHMfreq = 2 * np.abs(self.frequency[np.argmin(np.abs(0.5*np.max(self.IntensityFrequency)-self.IntensityFrequency))])
        self.FWHMwavelength = self.wavelength**2 * self.FWHMfreq/3e8
        print("FWHM of initial pulse : " + str(self.duration * 10 ** 15) + " fs")
        return self.frequency, self.EfieldFrequency, self.IntensityFrequency, self.FWHMfreq, self.FWHMwavelength


    def propagate(self, material, thickness):
        c = 299792458
        shiftedFreq = [i + c/self.wavelength for i in self.frequency]
        wavelengths = [c*10**6/j for j in shiftedFreq]

        x = Symbol('x')
        if material == "silica":
            n = (1 + 1.03961212 / (1 - 0.00600069867 / x ** 2) + 0.231792344 / (1 - 0.0200179144 / x ** 2) + 1.01046945 / (1 - 103.560653 / x ** 2))**.5
        elif material == "BK7":
            n = (1 + 1.03961212 / (1 - 0.00600069867 / x ** 2) + 0.231792344 / (1 - 0.0200179144 / x ** 2) + 1.01046945 / (1 - 103.560653 / x ** 2)) ** .5
        elif material == "SF10":
            n = (1 + 1.62153902 / (1 - 0.0122241457 / x ** 2) + 0.256287842 / (1 - 0.0595736775 / x ** 2) + 1.64447552 / (1 - 147.468793 / x ** 2)) ** .5
        else:
            raise ValueError("chosen material not available")


        n = lambdify(x, n)  #input is in micrometers
        nValues = [n(c * 10 ** 6 / i) for i in shiftedFreq]
        phaseTerm = [exp(i*n(c*10**6/i)*2*np.pi*thickness*-1j/c) for i in shiftedFreq]
        #phaseTerm = [exp(i/i *0* -1j) for i in shiftedFreq]
        self.EfieldFrequency = [a*b for a,b in zip(self.EfieldFrequency, phaseTerm)]
        EfieldTimeOut = np.fft.ifft(np.fft.ifftshift(self.EfieldFrequency))
        self.IntensityTimePropagated = abs(EfieldTimeOut)**2
        mu = self.time.dot(self.IntensityTimePropagated / self.IntensityTimePropagated.sum())
        mu2 = np.power(self.time, 2).dot(self.IntensityTimePropagated / self.IntensityTimePropagated.sum())
        var = mu2 - mu ** 2
        self.durationPropagated = 2.35482 * sqrt(var)

        #print("FWHM of new pulse : " + str("%.2f" %self.durationPropagated) + " fs")
        #plt.plot(self.time, self.IntensityTime, "r", self.time, self.IntensityTimePropagated, "k-")
        #plt.show()
        
        # GVD = (self.wavelength*10**6)**3/(2*np.pi*(c)**2)*f.diff(x, 2) * 10**21

    def plotInitialPulse(self):
        fig, axs = plt.subplots(2,1, figsize=(6,6))

        axs[0].set_title('Time domain')
        axs[0].plot(self.time, self.IntensityTime, label="Wavelength = %.1f nm"%(self.wavelength*10**9))
        axs[0].axvline(-self.duration*0.5, color='r', alpha=0.5, label='FWHM = %.1f fs'%(self.duration*1e15))
        axs[0].axvline(self.duration*0.5, color='r', alpha=0.5)

        axs[0].set_xlim(-2*self.duration, 2*self.duration)
        #axs[0].set_ylim(0,1.3)
        axs[0].set_xlabel('Time (sec)')

        axs[1].set_title('Frequency domain')
        axs[1].plot(self.frequency,self.IntensityFrequency)

        axs[1].axvline(-self.FWHMfreq*0.5, color='g', alpha=0.5, label='FWHM is %.1f THz or %0.2f nm'%(self.FWHMfreq*10**-12, self.FWHMwavelength*10**9))
        axs[1].axvline( self.FWHMfreq*0.5, color='g', alpha=0.5)

        axs[1].set_xlim(-3*self.FWHMfreq,3*self.FWHMfreq)
        #axs[1].set_ylim(0,30000)
        axs[1].set_xlabel('Frequency (Hz)')

        for ax in axs:
            ax.legend(fontsize=8)
            ax.set_ylabel('Electric field intensity (arbitrary units)')

        plt.tight_layout()
        plt.savefig('time-bandwidth-product.png', dpi=200)
        plt.show()

    def plotPropagatedPulse(self):
        print("FWHM of propagated pulse : %.1f fs"%(self.durationPropagated*1e15))
        peakIndex = np.argmax(self.IntensityTimePropagated)
        PropagatedShift = int(-1*(len(self.IntensityTimePropagated)/2 - peakIndex))
        PropagatedCentered = np.append(self.IntensityTimePropagated[PropagatedShift:], self.IntensityTimePropagated[:PropagatedShift])

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.set_title('Time domain')
        ax.plot(self.time, self.IntensityTime/max(self.IntensityTime), label="Original pulse \nDuration = %.1f fs"%(self.duration*1e15))
        ax.plot(self.time, PropagatedCentered/max(self.IntensityTime), label="Propagated pulse \nDuration = %.1f fs"%(self.durationPropagated*1e15))
        ax.set_xlim(-2*float(self.durationPropagated), 2*float(self.durationPropagated))
        ax.set_xlabel('Time (sec)')
        ax.legend(fontsize=8)
        ax.set_ylabel('Electric field intensity (arbitrary units)')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    pulseTest = Pulse(120e-15, 2, 1045e-9)
    pulseTest.frequencySpectra
    pulseTest.plotInitialPulse()
    #pulseTest.propagate("silica", 0.15)
    pulseTest.propagate("SF10", 0.05)
    pulseTest.plotPropagatedPulse()
