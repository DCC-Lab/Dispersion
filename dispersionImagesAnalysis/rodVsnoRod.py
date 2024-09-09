from matplotlib import pyplot as plt
from pandas import *

x_noRod = [793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 817, 819, 821, 823, 825, 830, 840]
x_Rod = [793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 815, 817, 819, 821, 825]
x_ref = [793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809]
wn_norod = [(1/i - 1/1045)*10000000 for i in x_noRod]
wn_rod = [(1/i - 1/1045)*10000000 for i in x_Rod]
wn_ref = [(1/i - 1/1045)*10000000 for i in x_ref]

noRodMyelin = read_csv("norod-myelinMean.csv")["Mean"]
noRodBG = read_csv("norod-bgMean.csv")["Mean"]
RodMyelin = read_csv("rod-myelinMean.csv")["Mean"]
RodBG = read_csv("rod-bgMean.csv")["Mean"]
y_ref = [15, 15, 20, 55, 70, 120, 145, 150, 160, 165, 170, 190, 200, 180, 100, 20, 15]

def normalize(x):
    return [(i-min(x))/(max(x)-min(x)) for i in x]

plt.plot(wn_rod, RodMyelin, "k-", label="Myelin signal, SF-CARS")
plt.plot(wn_rod, RodBG, "k--", label="Background signal, SF-CARS")
#plt.plot(wn_ref, normalize(y_ref), "r-", label="Approximate expected \nmyelin spectra")
plt.xlabel("Wavenumber [cm-1]")
plt.ylabel("Intensity [-]")
plt.legend()
plt.show()