import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

directory = os.path.dirname(os.path.realpath(__file__))

############
# FUNCTIONS
############
# TODO Figure out initial incidence angle from lab measurements
def convertMeasuredToIncidence(measurement):
    return np.radians(180 - measurement + 50)


# TODO figure out angle calculation based on results from lab
def calculateAngles(thetaPolarized, thetaUnpolarized):
    thetaPolarized = convertMeasuredToIncidence(thetaPolarized)
    thetaUnpolarized = convertMeasuredToIncidence(thetaUnpolarized)

    return thetaPolarized


# Error propogation
def divisionPropogation(val1, val2, err1, err2):
    return np.sqrt(((err1 ** 2) / val1) + ((err2 ** 2) / val2))


def tanPropagation(val, err):
    return err * (1 / np.cos(val)) ** 2


def arctanPropagation(val, err):
    return err / (1 + val ** 2)


# Models
def parallelReflectanceCoeffModel(theta1, theta2, n2):
    """
        Model used to calculate parallel reflectance.
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
        Model used to calculate parallel reflectance.
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


def REMOVEMODEL(theta1, n2):
    return (
        parallelReflectanceCoeffModel(theta1, calculateTheta2(theta1, n2), n2)
        ** 2
    )


############
# CONSTANTS
############
intensityUncertainty = 0.000005
angleUncertainty = 0.000005

polarizedAngles, polarizedIntensities = np.loadtxt(
    f"{directory}/rawData/experimentThree.txt", unpack=True, delimiter="\t"
)

unpolarizedAngles, unpolarizedIntensities = np.loadtxt(
    f"{directory}/rawData/experimentThree.txt", unpack=True, delimiter="\t"
)


# TODO update ratio calculation with new data
# TODO figure out ratio error propogation (ratioError)
angles = calculateAngles(polarizedAngles, unpolarizedAngles)
ratio = REMOVEMODEL(angles, 0.9)
ratio[ratio < 0] = 0


##################################################
# Calculating values based on curve_fit regression
##################################################
n2Reg, n2RegCov = curve_fit(REMOVEMODEL, angles, ratio, p0=[0.9])
n2RegErr = n2Reg[0] * np.sqrt(n2RegCov[0][0])
brewstersReg = np.arctan(n2Reg[0])  # with n1 = 1
brewstersRegErr = arctanPropagation(brewstersReg, n2RegErr)
theta2Reg = calculateTheta2(angles, n2Reg)

plt.errorbar(
    angles,
    ratio,
    # yerr=ratioError,
    xerr=angleUncertainty,
    capsize=1,
    c="r",
    label="Experimental Data",
    marker="o",
    linestyle="",
    ecolor="k",
    elinewidth=0.25,
    capthick=0.5,
    barsabove=True,
    markersize=2,
)
plt.plot(
    angles, REMOVEMODEL(angles, n2Reg[0]), c="k", label="Model using curve fit"
)
plt.legend()
plt.xlabel("Incidence Angle (radians)")
plt.ylabel("Parallel reflectance")
plt.savefig(f"{directory}/../figures/intensityVsIncidenceAngle.pdf")
plt.cla()

# Plotting reflectance coefficients
plt.plot(
    angles,
    perpendicularReflectanceCoeffModel(angles, theta2Reg, n2Reg),
    label="Perpendicular reflectance coefficient",
)
plt.plot(
    angles,
    parallelReflectanceCoeffModel(angles, theta2Reg, n2Reg),
    label="Parallel reflectance coefficient",
)
plt.xlabel("Incidence Angle (radians)")
plt.ylabel("Coefficient Value")
plt.legend()
plt.savefig(f"{directory}/../figures/regressionReflectanceCoefficient.pdf")
plt.cla()


####################################################
# Calculating values based on angle of minimal ratio
####################################################
brewstersMin = angles[np.argmin(ratio)]

# Calculating theoretical n2
n2Min = np.tan(brewstersMin)  # n1 is 1
n2MinErr = tanPropagation(n2Min, angleUncertainty)

# Calculating theoretical theta2
theta2Min = calculateTheta2(angles, n2Min)

# Plotting reflectance coefficents
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
plt.savefig(f"{directory}/../figures/minReflectanceCoefficient.pdf")
plt.cla()


print("Values calculated by ratio minimization")
print(f"Brewster's Angle: {brewstersMin} ± {angleUncertainty}.")
print(f"Acrylic index of refraction {n2Min} ± {n2MinErr}.")
print("")
print("------------------------------------------------------")
print("")
print("Values calculated from the scipy curve_fit regression:")
print(f"Brewster's Angle: {brewstersReg} ± {brewstersRegErr}.")
print(f"Acrylic index of refraction {n2Reg[0]} ± {n2RegErr}.")
