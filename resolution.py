import numpy as np
import math


# review how to compute the TBP 
TBP = 0.44  # time-bandwidth product for gaussian pulse shape
𝜏0 = 6*10**-12  # for picosecond pulses with the picoTRAIN laser (goal)
𝜏1 = 100*10**-15  # for femtosecond pulses with Ti:Saph (current)
c = 2.998*10**8  # light speed in m/s
𝝺m = 805*10**-9  # wavelength of interest for CARS on myelin with 1045 nm fixed output
𝝺f = 1045*10**-9  # wavelength of the fixed Ti:Saph output

def convert(wave):  # wavenumber to and from wavelength
    return 10**7/wave

𝛎f = convert(𝝺f*10**9)  # wavenumber of the fixed output at 1045 nm
𝝺w = convert(3200+𝛎f) # wavelength of interest for CARS on water with 1045 nm fixed output

print(f'{int(𝝺w)} nm')

def 𝚫𝛎(pulse):  # frequential bandwidth
    return TBP/pulse

𝚫𝛎0 = 𝚫𝛎(𝜏0)  # for 6 ps
𝚫𝛎1 = 𝚫𝛎(𝜏1)  # for 100 fs

def 𝚫𝝺(𝝺, fbw):  # wavelength bandwidth in nm
    return 𝝺**2/c*fbw*10**9

𝚫𝝺0 = 𝚫𝝺(𝝺m, 𝚫𝛎0)  # for 6 ps
𝚫𝝺1 = 𝚫𝝺(𝝺m, 𝚫𝛎1)  # for 100 fs

print(np.round(𝚫𝝺0/2,2), np.round(𝚫𝝺1/2,2))

# wanted length of Ti:Saph pulse elongated with glass rod in ps
𝜏f = 𝚫𝝺1*𝜏0/𝚫𝝺0*10**12

print(f'{int(𝜏f)} ps')  # makes no sense, way too big

# try again in wavenumber intervals
𝚫𝛎0 = convert(𝝺m*10**9-𝚫𝝺0/2)-convert(𝝺m*10**9+𝚫𝝺0/2)  # wavenumber bandwidth in cm-1 for a ps pulse
𝚫𝛎1 = convert(𝝺m*10**9-𝚫𝝺1/2)-convert(𝝺m*10**9+𝚫𝝺1/2)  # wavenumber bandwidth in cm-1 for a fs pulse

print(round(𝚫𝛎0,2), round(𝚫𝛎1,2))

𝜏f = 𝚫𝛎1*𝜏0/𝚫𝛎0*10**12

print(f'{int(𝜏f)} ps')  # we get the exact same result of 360 ps


# what if I try getting the wavenumber interval directly
def WNBandwidth(freq):
    return freq/(c*10**2)

𝚫𝛎0 = WNBandwidth(𝚫𝛎(𝜏0))
𝚫𝛎1 = WNBandwidth(𝚫𝛎(𝜏1))

print(round(𝚫𝛎0,2), round(𝚫𝛎1,2))  # we get the same intervals, which makes sense

# So it seems we can't realistically get the same spectral resolution with the femtosecond laser as with the picosecond one. What minimal resolution could we then expect?
# This is considering all wavelengths will be continously and linearly chirped within the pulse, which is not actually true.
# But then, we get different resolutions depending if we synchronize with the beginning or the end of the pulse?
# How else can we measure the spectral resolution? What was used in Pegoraro paper?

# print(f'{int(𝜏f*10**-12/𝜏1)}x')
# This method makes no sense because the chirp makes it so the whole bandwidth is no longer mixed everywhere
# we still need a certain band of comparison...