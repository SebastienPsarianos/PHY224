import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

directory = "/".join(__file__.split("/")[:-1])


####################################
### 1. Reading and plotting data
####################################

rocketTime, rocketPosition, rocketUncertainty = np.loadtxt(
    f"{directory}/rocket.csv", delimiter=",", encoding="utf-8", unpack=True,
)

plt.xlabel("Time since launch")
plt.ylabel("Rocket Position")
plt.title("Rocket position vs Time since Launch for Saturn V rocket")

plt.errorbar(
    rocketTime,
    rocketPosition,
    yerr=rocketUncertainty,
    label="Raw Data",
    capsize=4,
    c="r",
    marker="o",
    linestyle="",
    ecolor="k",
    elinewidth=0.25,
    capthick=0.5,
    barsabove=True,
    markersize=5,
)
plt.legend()
plt.draw()
plt.savefig(f"{directory}/graphs/rocketPositionTime.pdf")


####################################
### 2. Estimating the speed
####################################

# Calculate Average Speed for every Time interval using:
# ΔD/Δt
# And put each value in a numpy array
avgSpeeds = np.array([])
for i in range(len(rocketTime)):
    if i != 0:
        avgSpeeds = np.append(
            avgSpeeds,
            (rocketPosition[i] - rocketPosition[i - 1])
            / (rocketTime[i] - rocketTime[i - 1]),
        )

# Calcualte mean based on avg speed intervals
meanSpeed = np.sum(avgSpeeds) / len(avgSpeeds)

stdDeviation = np.sqrt(
    np.sum((avgSpeeds - meanSpeed) ** 2) / (len(rocketTime) - 1)
)

stdUncertainty = stdDeviation / np.sqrt(len(rocketTime))

print("2. Estimating the speed")
print(f"Mean Speed: {meanSpeed}")
print(f"Standard Uncertainty: {stdUncertainty}")
print("\n")


####################################
### 3. Linear Regression
####################################

avgTime = np.sum(rocketTime) / len(rocketTime)
avgPosition = np.sum(rocketPosition) / len(rocketPosition)

linearRegressionSpeed = np.sum(
    (rocketTime - avgTime) * (rocketPosition - rocketTime)
) / np.sum((rocketTime - avgTime) ** 2)
linearRegressionInitialPosition = avgPosition - linearRegressionSpeed * avgTime

print("3. Linear Regression")
print(f"Calculated u: {linearRegressionSpeed}")
print(f"Calculated initial d: {linearRegressionInitialPosition}")
print("\n")


####################################
### 4. Plotting the prediction
####################################


def distance_model(time, dZero, velocity):
    return dZero + time * velocity


plt.xlabel("Time since launch")
plt.ylabel("Rocket Position")
plt.title(
    "Rocket position vs Time since Launch for Saturn V rocket with\n analytical best fit"
)

plt.plot(
    rocketTime,
    distance_model(
        rocketTime, linearRegressionInitialPosition, linearRegressionSpeed
    ),
    label="Analytic Best Fit",
    linewidth="0.5",
    c="b",
)
plt.legend()
plt.draw()
plt.savefig(f"{directory}/graphs/rocketPositionVTimeWithAnalyticFit.pdf")


####################################
### 5. Characterizing the fit
####################################


def calculateChiSquared(xData, yData, uncertainty, fitYData, *parameters):
    return (1 / (len(xData) - len(parameters))) * np.sum(
        (yData - fitYData(xData, *parameters)) ** 2 / uncertainty ** 2
    )


chiSquared = calculateChiSquared(
    rocketTime,
    rocketPosition,
    rocketUncertainty,
    distance_model,
    linearRegressionInitialPosition,
    linearRegressionSpeed,
)
print("5. Characterizing the fit")
print(f"Calculated XR squared: {chiSquared}")
print("\n")


####################################
### 6. Curve Fitting
####################################

popt, pcov = curve_fit(
    distance_model,
    rocketTime,
    rocketPosition,
    sigma=rocketUncertainty,
    absolute_sigma=True,
)
curveFitChiSquared = calculateChiSquared(
    rocketTime, rocketPosition, rocketUncertainty, distance_model, *popt
)

print("6. Curve Fitting")
print(f"Curve Fit d_0 value: {popt[0]} +- {np.sqrt(pcov[0][0])}")
print(f"Curve Fit u value: {popt[1]}, +- {np.sqrt(pcov[1][1])}")
print(f"Chi Squared: {curveFitChiSquared}")
print("\n")

plt.xlabel("Time since launch")
plt.ylabel("Rocket Position")
plt.title(
    "Rocket position vs Time since Launch for Saturn V rocket with analytic\n and curve_fit best fit models"
)

plt.plot(
    rocketTime,
    distance_model(rocketTime, *popt),
    label="curve_fit best fit model",
    linewidth="0.5",
    c="g",
)
plt.legend()
plt.draw()

plt.savefig(
    f"{directory}/graphs/rocketPositionVTimeWithAnalyticAndCurvefitFit.pdf"
)


####################################
### 7. Feather Drop Experiment
####################################

featherTime, featherPosition, featherUncertainty = np.loadtxt(
    f"{directory}/feather.csv", delimiter=",", unpack=True
)


def fallingModel(times, initialDisplacement, initialVelocity, acceleration):
    return (
        initialDisplacement
        + initialVelocity * times
        + acceleration * times ** 2 / 2
    )


popt, pcov = curve_fit(
    fallingModel,
    featherTime,
    featherPosition,
    sigma=featherUncertainty,
    absolute_sigma=True,
    p0=(1.75, 0, 1.4),
)

print("7. Feather Drop Experiment")
print(f"Initial Displacement: {popt[0]}m +- {pcov[0][0]}m")
print(f"Initial Velocity: {popt[1]}m/s +- {pcov[1][1]}m/s")
print(f"Acceleration: {popt[2]}m/s^2 +- {pcov[2][2]}m/s^2")

plt.cla()

plt.xlabel("Time since drop (s)")
plt.ylabel("Feather Position (m)")
plt.title("Feather Position vs Time with curve_fit best fit model")

plt.errorbar(
    featherTime,
    featherPosition,
    yerr=featherUncertainty,
    label="Raw Data",
    capsize=4,
    c="r",
    marker="o",
    linestyle="",
    ecolor="k",
    elinewidth=0.25,
    capthick=0.5,
    barsabove=True,
    markersize=5,
)
plt.plot(
    featherTime,
    fallingModel(featherTime, *popt),
    label="curve_fit best fit model",
    linewidth="0.5",
    c="g",
)
plt.legend()
plt.savefig(f"{directory}/graphs/featherPositionVTime.pdf")

