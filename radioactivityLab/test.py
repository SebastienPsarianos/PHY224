import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

print(
    np.loadtxt(
        "radioactivityLab/Barium137_20220929AM_20min20sec.txt",
        skiprows=2,
        delimiter="\t",
        unpack=True,
    )
)

print(np.arange(0, 1201, 20))

def decay_function(time, exponent):
    return 1/2 ^{}