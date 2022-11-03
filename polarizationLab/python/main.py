import numpy as np
from scipy.optimize import curve_fit
import os

directory = os.path.dirname(os.path.realpath(__file__))

experimentOneFile = open('experimentOne.txt', 'w')
experimentTwoFile = open('experimentTwo.txt', 'w')

# expOneAngles, expOneAngles = np.loadtxt(f"{directory}/rawData/experimentOne.txt", delimiter=',', unpack=True)
# expTwoAngles, expTwoAngles = np.loadtxt(f"{directory}/rawData/experimentTwo.txt", delimiter=',', unpack=True)
# expThreeAngles, expThreeAngles = np.loadtxt(f"{directory}/rawData/experimentThree.txt", delimiter=',', unpack=True)

