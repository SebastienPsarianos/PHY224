import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

# Exercise 1: Malus' Law
# Defining Malus' Law Model


def cosineModel(angle, initialIntensity):
    return initialIntensity * ((np.cos(angle)) ** 2)


sensorPosition1, lightIntensity1 = \
    np.loadtxt('exerciseOne.txt', delimiter="\t", skiprows=2, unpack=True)

popt1, pcov1 = curve_fit(cosineModel, sensorPosition1, lightIntensity1)
pvar1 = np.sqrt(np.diag(pcov1))
print("The initial intensity is " + str(popt1[0]) + " ± " + str(pvar1[0]))

# Plotting the data
plt.errorbar(sensorPosition1, lightIntensity1, yerr=0.01, xerr=0.00001,
             color='blue', fmt='o', label='Experimental Data')
plt.plot(sensorPosition1, cosineModel(sensorPosition1, popt1[0]), color='red',
         label='Theoretical Model')
plt.xlabel('Angle (rad)')
plt.ylabel('Light Intensity (V)')
plt.legend()
plt.show()
plt.cla()

plt.errorbar(np.cos(sensorPosition1) ** 2, lightIntensity1, yerr=0.01,
             xerr=(2 * (abs(np.sin(sensorPosition1)) * 0.00001) /
                   np.cos(sensorPosition1) ** 2), color='blue', fmt='o',
             label='Experimental Data')
plt.plot(np.cos(sensorPosition1) ** 2, cosineModel(sensorPosition1, popt1[0]),
         color='red', label='Theoretical Model')
plt.xlabel('Cosine of Angle Squared')
plt.ylabel('Light Intensity (V)')
plt.legend()
plt.show()
plt.cla()

plt.scatter(sensorPosition1, cosineModel(sensorPosition1, popt1[0]) -
            lightIntensity1, label='Residual Plot for the Intensity vs Angle')
plt.xlabel('Angle (rad)')
plt.ylabel('Light Intensity Difference (V)')
plt.legend()
plt.show()
plt.cla()

plt.scatter(np.cos(sensorPosition1) ** 2, cosineModel(sensorPosition1, popt1[0])
            - lightIntensity1, label='Residual Plot for the Intensity vs the '
                                     'Cosine of the Angle Squared')
plt.xlabel('Cosine of Angle Squared')
plt.ylabel('Light Intensity Difference (V)')
plt.legend()
plt.show()
plt.cla()

# Exercise 2: Three Polarizers
# Defining Model


def intensity(angle, firstIntensity):
    return firstIntensity / 4 * (np.sin(np.pi / 2 + 2 * angle) ** 2)


sensorPosition2, lightIntensity2 = \
    np.loadtxt('exerciseTwo.txt', delimiter="\t", skiprows=2, unpack=True)

popt2, pcov2 = curve_fit(intensity, sensorPosition2, lightIntensity2)
pvar2 = np.sqrt(np.diag(pcov2))
print("The intensity passing through the first polarizer is " + str(popt2[0])
      + " ± " + str(pvar2[0]))

plt.errorbar(sensorPosition2, lightIntensity2, yerr=0.01, xerr=0.00001,
             color='blue', fmt='o', label='Experimental Data')
plt.plot(sensorPosition2, intensity(sensorPosition2, popt2[0]), color='red',
         label='Theoretical Model')
plt.xlabel('Angle (rad)')
plt.ylabel('Light Intensity (V)')
plt.legend()
plt.show()

plt.scatter(sensorPosition2, intensity(sensorPosition2, popt2[0]) -
            lightIntensity2, label='Residual Plot for the Intensity vs Angle')
plt.xlabel('Angle (rad)')
plt.ylabel('Light Intensity Difference (V)')
plt.legend()
plt.show()
plt.cla()


def characterize(y: any, func: any, u: any) -> float:
    """Return the chi^2 metric to determine how well a model function fits a
    given set of data using the measured data <y>, the prediction with the model
    <func> and the uncertainty on each measurement's dependent data <u>.
    """
    value = 0

    for i in range(np.size(y)):
        value += ((y[i] - func[i]) ** 2) / (u ** 2)
        i += 1

    return value / (np.size(y) - 1)


print(characterize(lightIntensity1, cosineModel(sensorPosition1, popt1[0]),
                   0.01))
print(characterize(lightIntensity2, intensity(sensorPosition2, popt2[0]), 0.01))
