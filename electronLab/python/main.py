import numpy as np
from scipy.optimize import curve_fit
from scipy.constants import mu_0, e, m_e
import matplotlib.pyplot as plt
import os

directory = os.path.dirname(os.path.realpath(__file__))

############
# Constants
############

coilRadius = 0.311
coilTurns = 130

# k is defined:
# k =  1 ( 4 ) (3/2) (mu_0)n
#     -- ( - )       -------
#     √2 ( 5 )          R
k = ((1 / np.sqrt(2)) * ((4 / 5) ** (3 / 2)) * (mu_0) * coilTurns) / coilRadius

# Square root of the charge mass ratio using scipy values
literatureChargeMassRatioSqrt = np.sqrt(e / m_e)

# Factor of 3 due to one measurement on each side of the scale and measuring the
# middle of the scale
# Divide by 200 to convert from diameter in cm to radius in m
rulerUncertainty = 0.05 * 3 / 200

currentUncertaintyConstants = (0.18, 3, 0.02)
voltageUncertaintyConstants = (0.002, 1000, 0.0006)

#############
# Functions #
##############
def reducedChiSquared(x, y, yerr, model, modelParams):
    return (
        1
        / (len(y) - len(modelParams))
        * np.sum(((y - model(x, *modelParams)) / yerr) ** 2)
    )


def multimeterUncertainty(
    value, valuePercentage, multimeterRange, rangePercentage
):
    return (value * valuePercentage + multimeterRange * rangePercentage) / 100


################################
# Model for external field (3).
################################
def externalField(r, b, B):
    return b * np.sqrt(149.820) / r - B


########################
# Experimental Analysis
########################
currents, currentDiameters = np.loadtxt(
    f"{directory}/variedCurrent.csv", unpack=True, delimiter=","
)

voltages, voltageDiameters = np.loadtxt(
    f"{directory}/variedVoltage.csv", unpack=True, delimiter=","
)

# Dividing by 200 to convert from diameter in cm to radius in m
currentRadius = currentDiameters / 200
voltageRadius = voltageDiameters / 200
coilFieldArray = ((4 / 5) ** (3 / 2)) * mu_0 * coilTurns * currents / coilRadius


# Models for ratio
def variedCurrent(I, a):
    return 1 / (a * k * ((I - externalFitVariables[1] / k) / np.sqrt(149.820)))


def variedVoltage(v, a):
    return 1 / (a * k * ((1.510946 - externalFitVariables[1] / k) / np.sqrt(v)))


###############################################################################
externalFitVariables, externalFitCovariance = curve_fit(
    externalField, currentRadius, coilFieldArray
)

currentFitVariables, currentFitCovariance = curve_fit(
    variedCurrent, currents, currentRadius, p0=[literatureChargeMassRatioSqrt],
)

# Varied voltage fit
voltageFitVariables, voltageFitCovariance = curve_fit(
    variedVoltage, voltages, voltageRadius, p0=[literatureChargeMassRatioSqrt],
)

# Plot to determine the external magnetic field
plt.errorbar(
    currents,
    coilFieldArray,
    multimeterUncertainty(currents, *currentUncertaintyConstants),
    ((4 / 5) ** (3 / 2))
    * mu_0
    * coilTurns
    / coilRadius
    * multimeterUncertainty(currents, *currentUncertaintyConstants),
    capsize=4,
    c="r",
    label="Experimental Data",
    marker="o",
    linestyle="",
    ecolor="k",
    elinewidth=0.25,
    capthick=0.5,
    barsabove=True,
    markersize=5,
)
plt.plot(
    currents,
    externalField(currentRadius, *externalFitVariables),
    label="Line of Best Fit",
)
plt.xlabel("Current through the Helmholtz Coils (Amps)")
plt.ylabel("Magnetic Field of Helmholtz Coils (T)")
plt.legend()
plt.savefig(f"{directory}/../figures/externalMagneticGraph.pdf")
plt.cla()

# Residuals Plot for external field
plt.scatter(
    currents,
    coilFieldArray - externalField(currentRadius, *externalFitVariables),
)
plt.xlabel("Current through the Helmholtz Coils (Amps)")
plt.savefig(f"{directory}/../figures/externalMagneticResiduals.pdf")
plt.cla()


# Varied current plotting
plt.errorbar(
    currents,
    currentRadius,
    rulerUncertainty,
    multimeterUncertainty(currents, *currentUncertaintyConstants),
    capsize=4,
    c="r",
    label="Experimental Data",
    marker="o",
    linestyle="",
    ecolor="k",
    elinewidth=0.25,
    capthick=0.5,
    barsabove=True,
    markersize=5,
)
plt.plot(
    currents,
    variedCurrent(currents, currentFitVariables),
    label="Best fit curve",
)
plt.plot(
    currents,
    variedCurrent(currents, literatureChargeMassRatioSqrt),
    label="Theoretical Model",
)
plt.xlabel("Current through the Helmholtz Coils (Amps)")
plt.ylabel("Radius of the circular electron beam (m)")
plt.legend()
plt.savefig(f"{directory}/../figures/variedCurrentGraph.pdf")
plt.cla()

# Residuals Plot for varied current
plt.scatter(
    currents, currentRadius - variedCurrent(currents, currentFitVariables)
)
plt.xlabel("Current through the Helmholtz Coils (Amps)")
plt.savefig(f"{directory}/../figures/variedCurrentResiduals.pdf")
plt.cla()

# Varied voltage plotting
plt.errorbar(
    voltages,
    voltageRadius,
    rulerUncertainty,
    multimeterUncertainty(voltages, *voltageUncertaintyConstants),
    capsize=4,
    c="r",
    label="Experimental Data",
    marker="o",
    linestyle="",
    ecolor="k",
    elinewidth=0.25,
    capthick=0.5,
    barsabove=True,
    markersize=5,
)
plt.plot(
    voltages, variedVoltage(voltages, voltageFitVariables), label="Linear fit",
)
plt.plot(
    voltages,
    variedVoltage(voltages, literatureChargeMassRatioSqrt),
    label="Theoretical Model",
)
plt.xlabel("Potential Difference across the electron gun anode (Volts)")
plt.ylabel("Radius of the circular electron beam (m)")
plt.legend()
plt.savefig(f"{directory}/../figures/variedVoltageGraph.pdf")
plt.cla()

# Residuals Plot for varied current
plt.scatter(
    voltages, voltageRadius - variedVoltage(voltages, voltageFitVariables)
)
plt.xlabel("Potential Difference across the electron gun anode (Volts)")
plt.savefig(f"{directory}/../figures/variedVoltageResiduals.pdf")
plt.cla()

print("###### Fit Variables ######")
print(
    f"Current Fit square root electron charge mass ratio: {currentFitVariables[0] ** 2} ± {2 * currentFitVariables[0] * np.sqrt(currentFitCovariance[0][0])}"
)
print(
    f"Voltage Fit square root electron charge mass ratio: {voltageFitVariables[0] ** 2} ± {2 * currentFitVariables[0] * np.sqrt(voltageFitCovariance[0][0])}"
)
print(
    f"Calculated external electric field: {externalFitVariables[1]} ± {np.sqrt(externalFitCovariance[1][1])}"
)
print(f"Literature Value: {literatureChargeMassRatioSqrt ** 2}")
print("\n")
print("###### Fit Quality ######")
print(
    f"""Current Fit reduced chi-squared value: {
    reducedChiSquared(
        currents,
        currentRadius,
        rulerUncertainty,
        variedCurrent,
        currentFitVariables,
    )}"""
)
print(
    f"""Voltage Fit reduced chi-squared value: {
    reducedChiSquared(
        voltages,
        voltageRadius,
        rulerUncertainty,
        variedVoltage,
        voltageFitVariables,
    )}"""
)
