import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import pi
import os

directory = os.path.dirname(os.path.realpath(__file__))


def cosineModel(x, amplitude, omega):
    return amplitude * np.cos(omega, x)


test = np.loadtxt(f"{directory}/rawData/experimentThree.txt", delimiter="\t")

angles, intensity = np.loadtxt(
    f"{directory}/rawData/experimentThree.txt", delimiter="\t", unpack=True
)

angles = angles * pi / 180

test = angles[:]
test2 = intensity[:]
test3 = curve_fit(cosineModel, test, test2)

plt.errorbar(angles, intensity)
plt.show()
