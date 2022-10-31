import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import mu_0, elementary_charge, m_e
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
literatureChargeMassRatioSqrt = np.sqrt(elementary_charge / m_e)

# Divide by 200 to convert from diameter in cm to radius in m
rulerUncertainty = 0.05 / 200

# One measurement on each side of the scale and measuring the middle of
# the scale
radiusMeasurementUncertainty = rulerUncertainty * 3

currentUncertaintyConstants = (0.01, 0.02, 3)
voltageUncertaintyConstants = (0.01, 0.02, 3)

#############
# Functions #
##############
def reducedChiSquared(x, y, yerr, model, modelParams):
    return (
        1
        / (len(y) - len(modelParams))
        * np.sum(((y - model(x, *modelParams)) / yerr) ** 2)
    )


def multimeterUncertainty(value, percentage, res, multiplier):
    return value * percentage / 100 + res * multiplier


#########
# Models
#########
def variedCurrent(I, a, Io):
    return (a * k * (I + Io * (1 / np.sqrt(2))) / np.sqrt(149.820)) ** -1


def variedVoltage(v, a, Io):
    return (a * k * (1.510946 + Io * (1 / np.sqrt(2))) / np.sqrt(v)) ** -1


########################
# Experimental Analysis
########################

currents, currentDiameters = np.loadtxt(
    f"{directory}/variedCurrent.csv", unpack=True, delimiter=","
)


voltages, voltageDiameters = np.loadtxt(
    f"{directory}/variedVoltage.csv", unpack=True, delimiter=","
)

# Dividing by 200 to convert from diameter in cm to radius in cm
currentRadius = currentDiameters / 200
voltageRadius = voltageDiameters / 200

# Varied current fit
currentFitVariables, currentFitCovariance = curve_fit(
    variedCurrent,
    currents,
    currentRadius,
    p0=[literatureChargeMassRatioSqrt, 0],
)

# Varied voltage fit
voltageFitVariables, voltageFitCovariance = curve_fit(
    variedVoltage,
    voltages,
    voltageRadius,
    p0=[literatureChargeMassRatioSqrt, 0],
)

# Varied current plotting
plt.errorbar(
    currents,
    currentRadius,
    radiusMeasurementUncertainty,
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
plt.plot(currents, variedCurrent(currents, *currentFitVariables))
plt.xlabel("Current through the Helmholtz Coils (Amps)")
plt.ylabel("Radius of the circular electron beam (m)")
plt.savefig(f"{directory}/../figures/variedCurrentGraph.pdf")
plt.cla()

# Varied voltage plotting
plt.errorbar(
    voltages,
    voltageRadius,
    radiusMeasurementUncertainty,
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
plt.plot(voltages, variedVoltage(voltages, *voltageFitVariables))
plt.xlabel("Potential Difference of the electron gun anode (Volts)")
plt.ylabel("Radius of the circular electron beam (m)")
plt.savefig(f"{directory}/../figures/variedVoltageGraph.pdf")
plt.cla()

print("###### Fit Variables ######")
print(
    f"Current Fit square root electron charge mass ratio: {currentFitVariables[0]}"
)
print(
    f"Voltage Fit square root electron charge mass ratio: {voltageFitVariables[0]}"
)
print(f"Current fit background electric field:  {currentFitVariables[1] / k}")
print(f"Voltage fit background electric field:  {voltageFitVariables[1] / k}")
print("\n")
print("###### Fit Quality ######")
print(
    f"""Current Fit reduced chi-squared value: {
    reducedChiSquared(
        currents,
        currentRadius,
        radiusMeasurementUncertainty,
        variedCurrent,
        currentFitVariables,
    )}"""
)
print(
    f"""Voltage Fit reduced chi-squared value: {
    reducedChiSquared(
        voltages,
        voltageRadius,
        radiusMeasurementUncertainty,
        variedVoltage,
        voltageFitVariables,
    )}"""
)
