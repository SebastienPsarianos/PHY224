import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from statistics import mean
import os

directory = os.path.dirname(os.path.realpath(__file__))

# Defining the constant
constant = (18 * (6 / 1000) * np.pi * 0.00001827 ** (3 / 2)) / (
    np.sqrt(2) * np.sqrt(9.8) * np.sqrt(875.3 - 1.204)
)


# Uncertainty in voltage readings and error propagation
def uncertaintyVoltage(reading, range):
    return 0.006 * reading + 0.001 * range


def uncertaintyElementaryOne(
    velocityOne, voltageOne, uncertaintyOne, uncertaintyTwo
):
    return np.sqrt(
        (3 * constant / 2 * np.sqrt(abs(velocityOne)) / voltageOne) ** 2
        * uncertaintyOne ** 2
        + (-1 * constant * abs(velocityOne) ** (3 / 2) / voltageOne ** 2) ** 2
        * uncertaintyTwo ** 2
    )


def uncertaintyElementaryTwo(
    velocityOne,
    velocityTwo,
    voltageTwo,
    uncertaintyOne,
    uncertaintyTwo,
    uncertaintyThree,
):
    return np.sqrt(
        (
            (3 * constant / 2 * np.sqrt(abs(velocityOne)))
            + (
                constant
                / 2
                * velocityTwo
                / voltageTwo
                / np.sqrt(abs(velocityOne))
            )
        )
        ** 2
        * uncertaintyOne ** 2
        + (constant * np.sqrt(abs(velocityOne)) / voltageTwo) ** 2
        * uncertaintyTwo ** 2
        + (
            (-1 * constant * abs(velocityOne) ** (3 / 2) / voltageTwo ** 2)
            - (
                constant
                * velocityTwo
                * np.sqrt(abs(velocityOne))
                / voltageTwo ** 2
            )
        )
        ** 2
        * uncertaintyThree ** 2
    )


# Reading the data
sampleDataFileNames = os.listdir(f"{directory}/rawData/")
sampleDataFileNames.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

samples = []
for fileName in sampleDataFileNames:
    fallingPositions = []
    risingPositions = []
    test = open(f"{directory}/rawData/{fileName}")
    lines = test.readlines()
    for line in lines:
        risingPosition, fallingPosition = line.strip("\n").split(",")
        if risingPosition != "":
            risingPositions.append(int(risingPosition))
        if fallingPosition != "":
            fallingPositions.append(int(fallingPosition))
    test.close()
    samples.append(
        (
            np.array(risingPositions) / 540000,
            np.array(fallingPositions) / 540000,
        )
    )

stoppingVoltage, voltageUp = np.loadtxt(
    f"{directory}/voltages.csv", delimiter=",", unpack=True
)
stoppingVoltageUncertainty = uncertaintyVoltage(stoppingVoltage, 1100)
voltageUpUncertainty = uncertaintyVoltage(voltageUp, 1100)


# Finding the velocities
def linear(time, velocity, intercept):
    return time * velocity + intercept


velocityUp = []
velocityUpUncertainty = []
velocityTerminal = []
velocityTerminalUncertainty = []

for position in samples:
    timeUp = np.linspace(
        0.1, 0.1 * len(position[0]), len(position[0]), endpoint=True
    )
    timeTerminal = np.linspace(
        0.1, 0.1 * len(position[1]), len(position[1]), endpoint=True
    )

    poptUp, pcovUp = curve_fit(linear, timeUp, position[0])
    pvarUp = np.sqrt(np.diag(pcovUp))
    poptTerminal, pcovTerminal = curve_fit(linear, timeTerminal, position[1])
    pvarTerminal = np.sqrt(np.diag(pcovTerminal))

    velocityUp.append(poptUp[0])
    velocityUpUncertainty.append(pvarUp[0])
    velocityTerminal.append(poptTerminal[0])
    velocityTerminalUncertainty.append(pvarTerminal[0])


# Finding the elementary charges
# Method 1
def elementaryOne(velocityOne, voltageOne):
    return constant * velocityOne ** (3 / 2) / voltageOne


elementaryChargeOne = []
for i in range(len(velocityTerminal)):
    elementaryChargeOne.append(
        elementaryOne(abs(velocityTerminal[i]), stoppingVoltage[i])
    )
    i += 1

elementaryChargeOne = np.array(elementaryChargeOne)


# Method 2
def elementaryTwo(velocityOne, velocityTwo, voltageTwo):
    return (
        constant
        * (abs(velocityOne) + velocityTwo)
        * abs(velocityOne) ** (1 / 2)
        / voltageTwo
    )


elementaryChargeTwo = []
for i in range(len(velocityUp)):
    elementaryChargeTwo.append(
        elementaryTwo(velocityTerminal[i], velocityUp[i], voltageUp[i])
    )
    i += 1

elementaryChargeTwo = np.array(elementaryChargeTwo)

# Finding uncertainty in individual elementary charge measurements

elementaryChargeOneUncertainty = []
for i in range(len(velocityTerminal)):
    elementaryChargeOneUncertainty.append(
        uncertaintyElementaryOne(
            velocityTerminal[i],
            stoppingVoltage[i],
            velocityTerminalUncertainty[i],
            stoppingVoltageUncertainty[i],
        )
    )

elementaryChargeTwoUncertainty = []
for i in range(len(velocityTerminal)):
    elementaryChargeTwoUncertainty.append(
        uncertaintyElementaryTwo(
            velocityTerminal[i],
            velocityUp[i],
            voltageUp[i],
            velocityTerminalUncertainty[i],
            velocityUpUncertainty[i],
            voltageUpUncertainty[i],
        )
    )


# Average uncertainty
def averageUncertainty(uncertainties):
    num = 0
    for uncertainty in uncertainties:
        num += uncertainty ** 2
    return np.sqrt(num) / len(uncertainties)


print(
    "Using method 1, the elementary charge was determined to be "
    + str(mean(elementaryChargeOne))
    + " ± "
    + str(averageUncertainty(elementaryChargeOneUncertainty))
)

print(
    "Using method 2, the elementary charge was determined to be "
    + str(mean(elementaryChargeTwo))
    + " ± "
    + str(averageUncertainty(elementaryChargeTwoUncertainty))
)

# Generating histograms

plt.hist(elementaryChargeOne, density=True, bins=40)
plt.show()
plt.cla()

plt.hist(elementaryChargeTwo, density=True, bins=40)
plt.show()
