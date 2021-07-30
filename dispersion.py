import matplotlib.pyplot as plt
import numpy as np


class Pulse:
    def __init__(self, duration, TL, wavelength):
        self.duration = duration
        self.TL = TL
        self.wavelength = wavelength

    def timeProfile(self):
        time = np.linspace(-10000e-15, 10000e-15, 40000)
        IntensityTime = np.exp(-time**2 / (2*(self.duration / 2.35482)**2))
        IntensityTimeNonTL = np.exp(-time**2 / (2*(self.duration / self.TL / 2.35482)**2))
        EfieldTime = np.sqrt(IntensityTimeNonTL)
        return time, IntensityTime, EfieldTime


    def spectralWidth(self, time, timeSpectra):
        EfieldFrequency = np.abs(np.fft.fftshift(np.fft.fft(timeSpectra)))
        IntensityFrequency = EfieldFrequency**2
        frequency  = np.fft.fftshift(np.fft.fftfreq(EfieldFrequency.shape[0],time[1]-time[0]))

        FWHMfreq = 2 * np.abs(frequency[ np.argmin(np.abs(0.5*np.max(IntensityFrequency)-IntensityFrequency))])
        FWHMwavelength = self.wavelength**2 * FWHMfreq/3e8
        return frequency, IntensityFrequency, FWHMfreq, FWHMwavelength


    def dispersion(self, time, IntensityTime, glass, thickness, FWHMwavelength):
        pass


    def plotPulses(self, time, IntensityTime, frequency, IntensityFrequency, FWHMfreq, FWHMwavelength):
        fig, axs = plt.subplots(2,1, figsize=(6,8))

        axs[0].set_title('Time domain')
        axs[0].plot(time,IntensityTime, label="Wavelength = %.1f nm"%(self.wavelength*10**9))
        axs[0].axvline(-self.duration*0.5, color='r', alpha=0.5, label='FWHM = %.1f fs'%(self.duration*1e15))
        axs[0].axvline( self.duration*0.5, color='r', alpha=0.5)

        axs[0].set_xlim(-350e-15, 350e-15)
        axs[0].set_ylim(0,1.3)
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

pulseTest = Pulse(120e-15, 1, 1050e-6)
(t, It, Et) = pulseTest.timeProfile()
(f, If, w1, w2) = pulseTest.spectralWidth(t, Et)
pulseTest.plotPulses(t, It, f, If, w1, w2)