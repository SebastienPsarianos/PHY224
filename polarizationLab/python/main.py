import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

directory = os.path.dirname(os.path.realpath(__file__))

############
# FUNCTIONS
############
def convertMeasuredToIncidence(measurement):
    return np.radians(215 - measurement)


# Error propagation
def divisionPropogation(val1, val2, err1, err2):
    return np.sqrt(((err1 ** 2) / val1) + ((err2 ** 2) / val2))


def tanPropagation(val, err):
    return err * (1 / np.cos(val)) ** 2


# Models
def parallelReflectanceCoeffModel(theta1, theta2, n2):
    """
        Model used to calculate parallel reflectance coefficient.
        Based off of fresnels equation.
        Uses a n1 value of 1 for the air.
        <n2> (index of refraction for acrylic) is theoretically calculated with
            Brewster's angle
        <theta1> (angle of incidence) is experimentally measured
        <theta2> (angle of reflectance) is determined theoretically with snells
            law
    """
    return (np.cos(theta2) - n2 * np.cos(theta1)) / (
        np.cos(theta2) + n2 * np.cos(theta1)
    )


def perpendicularReflectanceCoeffModel(theta1, theta2, n2):
    """
        Model used to calculate perpendicular reflectance coefficient.
        Based off of fresnels equation.
        Uses a n1 value of 1 for the air.
        <n2> (index of refraction for acrylic) is theoretically calculated with
            Brewster's angle
        <theta1> (angle of incidence) is experimentally measured
        <theta2> (angle of reflectance) is determined theoretically with snells
            law
    """
    return (np.cos(theta1) - n2 * np.cos(theta2)) / (
        np.cos(theta1) + n2 * np.cos(theta2)
    )


def calculateTheta2(theta1, n2):
    """
        Uses snells law to calculate theta2 in terms of theta1 and n2
    """
    return np.arcsin(np.sin(theta1) / n2)


def removeInconsistentPoints(firstIntensities, secondIntensities, thetas):
    i = 5
    while i < len(firstIntensities):
        firstIntensitiesAvg = np.sum(firstIntensities[i - 5 : i]) / 5
        secondIntensitiesAvg = np.sum(secondIntensities[i - 5 : i]) / 5
        if (
            firstIntensities[i] - firstIntensitiesAvg
            < -firstIntensitiesAvg / 10
        ) or (
            secondIntensities[i] - secondIntensitiesAvg
            < -secondIntensitiesAvg / 10
        ):

            firstIntensities = np.delete(firstIntensities, i)
            secondIntensities = np.delete(secondIntensities, i)

            thetas = np.delete(thetas, i)
            break
        i += 1
    else:
        return firstIntensities, secondIntensities, thetas

    firstIntensities, secondIntensities, thetas = removeInconsistentPoints(
        firstIntensities, secondIntensities, thetas
    )
    return firstIntensities, secondIntensities, thetas


############
# CONSTANTS
############
intensityUncertainty = 0.000005
angleUncertainty = 0.01

polarizedAngles, polarizedIntensities = np.loadtxt(
    f"{directory}/rawData/experimentThreePolarized.txt",
    unpack=True,
    delimiter="\t",
)

_, unpolarizedIntensities = np.loadtxt(
    f"{directory}/rawData/experimentThreeUnpolarized.txt",
    unpack=True,
    delimiter="\t",
)

polarizedAngles = convertMeasuredToIncidence(polarizedAngles)

# Plot pre-cleanup ratio
plt.errorbar(
    polarizedAngles[:250],
    polarizedIntensities[:250] / unpolarizedIntensities[:250],
    xerr=angleUncertainty,
    yerr=divisionPropogation(
        polarizedIntensities[:250],
        unpolarizedIntensities[:250],
        intensityUncertainty,
        intensityUncertainty,
    ),
    capsize=2,
    c="k",
    marker="o",
    linestyle="",
    ecolor="k",
    elinewidth=0.25,
    capthick=0.5,
    barsabove=True,
    markersize=2,
)
plt.xlabel("Incidence Angle (radians)")
plt.ylabel("ratio of polarized and non-polarized intensities ")
plt.savefig(f"{directory}/../figures/figure10.pdf")
plt.cla()


# Plot pre-cleanup intensities
plt.errorbar(
    polarizedAngles[:250],
    polarizedIntensities[:250],
    label="Polarized Intensity measurement",
    xerr=angleUncertainty,
    yerr=intensityUncertainty,
    capsize=2,
    c="k",
    marker="o",
    linestyle="",
    ecolor="k",
    elinewidth=0.25,
    capthick=0.5,
    barsabove=True,
    markersize=2,
)
plt.errorbar(
    polarizedAngles[:250],
    unpolarizedIntensities[:250],
    xerr=angleUncertainty,
    yerr=intensityUncertainty,
    label="Non-Polarized Intensity Measurement",
    capsize=2,
    c="r",
    marker="o",
    linestyle="",
    ecolor="k",
    elinewidth=0.25,
    capthick=0.5,
    barsabove=True,
    markersize=2,
)
plt.xlabel("Incidence Angle (radians)")
plt.ylabel("Measured Intensity (V)")
plt.legend()
plt.savefig(f"{directory}/../figures/figure12.pdf")
plt.cla()


polarizedIntensities, unpolarizedIntensities, angles = removeInconsistentPoints(
    polarizedIntensities[:250],
    unpolarizedIntensities[:250],
    polarizedAngles[:250],
)

ratio = polarizedIntensities / unpolarizedIntensities

plt.errorbar(
    angles,
    ratio,
    xerr=angleUncertainty,
    yerr=divisionPropogation(
        polarizedIntensities,
        unpolarizedIntensities,
        intensityUncertainty,
        intensityUncertainty,
    ),
    capsize=2,
    c="k",
    marker="o",
    linestyle="",
    ecolor="k",
    elinewidth=0.25,
    capthick=0.5,
    barsabove=True,
    markersize=2,
)
plt.xlabel("Incidence Angle (radians)")
plt.ylabel("Ratio of polarized and non-polarized intensities ")
plt.savefig(f"{directory}/../figures/figure11.pdf")
plt.cla()

plt.errorbar(
    angles,
    polarizedIntensities,
    label="Polarized Intensity measurement",
    yerr=intensityUncertainty,
    xerr=angleUncertainty,
    capsize=2,
    c="k",
    marker="o",
    linestyle="",
    ecolor="k",
    elinewidth=0.25,
    capthick=0.5,
    barsabove=True,
    markersize=2,
)
plt.errorbar(
    angles,
    unpolarizedIntensities,
    label="Non-Polarized Intensity Measurement",
    yerr=intensityUncertainty,
    xerr=angleUncertainty,
    capsize=2,
    c="r",
    marker="o",
    linestyle="",
    ecolor="k",
    elinewidth=0.25,
    capthick=0.5,
    barsabove=True,
    markersize=2,
)
plt.xlabel("Incidence Angle (radians)")
plt.ylabel("Measured Intensity (V)")
plt.legend()
plt.savefig(f"{directory}/../figures/figure13.pdf")
plt.cla()

# ####################################################
# # Calculating values based on angle of minimal ratio
# ####################################################
brewstersMin = angles[np.argmin(ratio)]

# # Calculating theoretical n2
n2Min = np.tan(brewstersMin)  # n1 is 1
n2MinErr = tanPropagation(n2Min, angleUncertainty)

# # Calculating theoretical theta2
theta2Min = calculateTheta2(angles, n2Min)

# # Plotting reflectance coefficents
plt.plot(
    angles,
    perpendicularReflectanceCoeffModel(angles, theta2Min, n2Min),
    label="Perpendicular reflectance coefficient",
)
plt.plot(
    angles,
    parallelReflectanceCoeffModel(angles, theta2Min, n2Min),
    label="Parallel reflectance coefficient",
)
plt.xlabel("Incidence Angle (radians)")
plt.ylabel("Coefficient Value")
plt.legend()
plt.savefig(f"{directory}/../figures/figure14.pdf")
plt.cla()

print("Final calculated values")
print(f"Brewster's Angle: {brewstersMin} ± {angleUncertainty}.")
print(f"Acrylic index of refraction {n2Min} ± {n2MinErr}.")
