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

#############################
# Uncertainty / Propagation #
#############################

# Uncertainty in voltage readings and error propagation
def uncertaintyVoltage(reading, ran):
    return 0.006 * reading + 0.001 * ran


def uncertaintyChargeMethodOne(
    velocityOne, voltageOne, uncertaintyOne, uncertaintyTwo
):
    return np.sqrt(
        (3 * constant / 2 * np.sqrt(abs(velocityOne)) / voltageOne) ** 2
        * uncertaintyOne ** 2
        + (-1 * constant * abs(velocityOne) ** (3 / 2) / voltageOne ** 2) ** 2
        * uncertaintyTwo ** 2
    )


def uncertaintyChargeMethodTwo(
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


#############################
# Charge Calculation Models #
#############################
def chargeMethodOne(velocityOne, voltageOne):
    return constant * velocityOne ** (3 / 2) / voltageOne


# Method 2
def chargeMethodTwo(velocityOne, velocityTwo, voltageTwo):
    return (
        constant
        * (abs(velocityOne) + velocityTwo)
        * abs(velocityOne) ** (1 / 2)
        / voltageTwo
    )


# General Linear Model
def linear(time, velocity, intercept):
    return time * velocity + intercept


sampleDataFileNames = os.listdir(f"{directory}/rawData/")
sampleDataFileNames.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

samples = []
for fileName in sampleDataFileNames:
    fallingPositions = np.array([])
    risingPositions = np.array([])
    test = open(f"{directory}/rawData/{fileName}")
    lines = test.readlines()
    for line in lines:
        risingPosition, fallingPosition = line.strip("\n").split(",")
        if risingPosition != "":
            risingPositions = np.append(
                risingPositions, int(risingPosition) / 540000
            )
        if fallingPosition != "":
            fallingPositions = np.append(
                fallingPositions, int(fallingPosition) / 540000
            )
    test.close()
    samples.append((risingPositions, fallingPositions))

stoppingVoltage, voltageUp = np.loadtxt(
    f"{directory}/voltages.csv", delimiter=",", unpack=True
)
stoppingVoltageUncertainty = uncertaintyVoltage(stoppingVoltage, 1100)
voltageUpUncertainty = uncertaintyVoltage(voltageUp, 1100)

upwardsVelocities = np.array([])
upwardsVelocityUncertainties = np.array([])
terminalVelocities = np.array([])
terminalVelocityUncertainties = np.array([])

for position in samples:
    # Calculating upwards velocity and its uncertainty
    upwardsTime = np.linspace(
        0.1, 0.1 * len(position[0]), len(position[0]), endpoint=True
    )
    (upwardsVelocity, _), pcovUp = curve_fit(linear, upwardsTime, position[0])
    upwardsVelocityUncertainty, _ = np.sqrt(np.diag(pcovUp))
    upwardsVelocities = np.append(upwardsVelocities, upwardsVelocity)
    upwardsVelocityUncertainties = np.append(
        upwardsVelocityUncertainties, upwardsVelocityUncertainty
    )

    # Calculating terminal velocity and its uncertainty
    terminalTime = np.linspace(
        0.1, 0.1 * len(position[1]), len(position[1]), endpoint=True
    )
    (terminalVelocity, _), pcovTerminal = curve_fit(
        linear, terminalTime, position[1]
    )
    terminalVelocityUncertainty, _ = np.sqrt(np.diag(pcovTerminal))
    terminalVelocities = np.append(terminalVelocities, terminalVelocity)
    terminalVelocityUncertainties = np.append(
        terminalVelocityUncertainties, terminalVelocityUncertainty
    )

elementaryChargeOne = chargeMethodOne(abs(terminalVelocities), stoppingVoltage)
elementaryChargeTwo = chargeMethodTwo(
    terminalVelocities, upwardsVelocities, voltageUp
)

# # Finding uncertainty in individual elementary charge measurements
elementaryChargeOneUncertainty = uncertaintyChargeMethodOne(
    terminalVelocities,
    stoppingVoltage,
    terminalVelocityUncertainties,
    stoppingVoltageUncertainty,
)
elementaryChargeTwoUncertainty = uncertaintyChargeMethodTwo(
    terminalVelocities,
    upwardsVelocities,
    voltageUp,
    terminalVelocityUncertainties,
    upwardsVelocityUncertainties,
    voltageUpUncertainty,
)

print(min(elementaryChargeOne / (1.6 * 10 ** (-19))))
