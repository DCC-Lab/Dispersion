from matplotlib import pyplot as plt

x = [50.6, 50.62, 50.64, 50.66, 50.68, 50.7]
x2 = [2920, 2940, 2960, 2980, 3000, 3020]
myelin = [85, 146, 219, 234, 185, 114]
bg = [111, 180, 203, 169, 99, 67]

plt.plot(x2[::-1], myelin, "k-", label="Myelin signal")
plt.plot(x2[::-1], bg, "k--", label="Background signal")
# plt.plot(x, myelin, "k-", label="Myelin signal, SF-CARS")
# plt.plot(x, bg, "k--", label="Background signal, SF-CARS")
plt.xlabel("Wavenumber [cm-1]")
# plt.xlabel("Micrometer stage delay value [mm]")
plt.ylabel("Intensity [-]")
plt.legend()
a = 0.99994
print(round(a, 3))
plt.show()
