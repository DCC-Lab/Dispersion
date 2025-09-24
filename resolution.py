import numpy as np
import math
from pulse import Pulse
import matplotlib.pyplot as plt


# review how to compute the TBP 
TBP = 0.44  # time-bandwidth product for gaussian pulse shape (Fourier transformed limited)
TBP = 0.315  # time-bandwidth product for sech^2 pulse shape (Fourier transformed limited)
𝜏0 = 6*10**-12  # for picosecond pulses with the picoTRAIN laser (goal)
𝜏1 = 100*10**-15  # for femtosecond pulses with Ti:Saph (current)
c = 2.998*10**8  # light speed in m/s
𝝺m = 805*10**-9  # wavelength of interest for CARS on myelin with 1045 nm fixed output
𝝺f = 1045*10**-9  # wavelength of the fixed Ti:Saph output

def convert(wave):  # wavenumber to and from wavelength
    return 10**7/wave

𝛎f = convert(𝝺f*10**9)  # wavenumber of the fixed output at 1045 nm
𝝺w = convert(3350+𝛎f) # wavelength of interest for CARS on water with 1045 nm fixed output

def sech2(x):
    return 1/(math.cosh(x)**2)

print(f'{int(𝝺w)} nm')

def 𝚫𝛎(pulse):  # frequential bandwidth
    return TBP/pulse

𝚫𝛎0 = 𝚫𝛎(𝜏0)  # for 6 ps
𝚫𝛎1 = 𝚫𝛎(𝜏1)  # for 100 fs

def 𝚫𝝺(𝝺, fbw):  # wavelength bandwidth in nm
    return 𝝺**2/c*fbw*10**9

𝚫𝝺0 = 𝚫𝝺(𝝺m, 𝚫𝛎0)  # for 6 ps
𝚫𝝺1 = 𝚫𝝺(𝝺f, 𝚫𝛎1)  # for 100 fs

print(np.round(𝚫𝝺0/2,2), np.round(𝚫𝝺1/2,2))

# try again in wavenumber intervals
𝚫𝛎0 = convert(𝝺m*10**9-𝚫𝝺0/2)-convert(𝝺m*10**9+𝚫𝝺0/2)  # wavenumber bandwidth in cm-1 for a ps pulse
𝚫𝛎1 = convert(𝝺m*10**9-𝚫𝝺1/2)-convert(𝝺m*10**9+𝚫𝝺1/2)  # wavenumber bandwidth in cm-1 for a fs pulse

print(round(𝚫𝛎0,2), round(𝚫𝛎1,2))


# what if I try getting the wavenumber interval directly
def WNBandwidth(freq):
    return freq/(c*10**2)

𝚫𝛎0 = WNBandwidth(𝚫𝛎(𝜏0))
𝚫𝛎1 = WNBandwidth(𝚫𝛎(𝜏1))

print(round(𝚫𝛎0,2), round(𝚫𝛎1,2))  # we get the same intervals, which makes sense


# we need the temporal resolution of the detection system
# we can compute d𝜏/d𝝺 which is a function of 𝝺 representing the group delay slope at different wavelength [ps/nm or fs/nm]
# group delay is d𝜏/d𝝺=d𝜏/dω*dω/d𝝺 using ω=2𝛑c/𝝺

# what would be an equivalent of the temporal resolution with focusing CARS?


# pulse = Pulse(𝛕=100e-15, 𝜆ₒ=805e-9)
pulse = Pulse(6e-12, 1045e-9)  # picoTRAIN as resolution reference
# plt.plot(pulse.spectrum)
plt.plot(pulse.fieldEnvelope)
plt.show()