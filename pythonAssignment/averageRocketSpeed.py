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

plt.errorbar(
    rocketTime, rocketPosition, yerr=rocketUncertainty, label="Raw Data"
)
plt.legend()
plt.draw()
plt.savefig(f"{directory}/rocketPositionTime.png")


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

print("Estimating the speed \n")
print(f"Mean Speed: {meanSpeed}")
print(f"Standard Uncertainty: {stdUncertainty}")
print("\n\n")


####################################
### 3. Linear Regression
####################################

avgTime = np.sum(rocketTime) / len(rocketTime)
avgPosition = np.sum(rocketPosition) / len(rocketPosition)

linearRegressionSpeed = np.sum(
    (rocketTime - avgTime) * (rocketPosition - rocketTime)
) / np.sum((rocketTime - avgTime) ** 2)
linearRegressionInitialPosition = avgPosition - linearRegressionSpeed * avgTime

print("Linear Regression\n")
print(f"Calculated u: {linearRegressionSpeed}")
print(f"Calculated initial d: {linearRegressionInitialPosition}")
print("\n\n")


####################################
### 4. Plotting the prediction
####################################


def distance_model(time, dZero, velocity):
    return dZero + time * velocity


plt.plot(
    rocketTime,
    distance_model(
        rocketTime, linearRegressionInitialPosition, linearRegressionSpeed
    ),
    label="Analytic Linear Regression",
)
plt.legend()
plt.draw()
plt.savefig(f"{directory}/rocketPositionTimeWithLinearRegression.png")


####################################
### 5. Characterizing the fit
####################################
def calculateXRSquared(xData, yData, uncertainty, fitYData, *parameters):
    return (1 / (len(xData) - len(parameters))) * np.sum(
        (yData - fitYData(xData, *parameters)) ** 2 / uncertainty ** 2
    )


print("Characterizing the fit\n")
print(
    f"Calculated XR squared: {calculateXRSquared(rocketTime,rocketPosition,rocketUncertainty,distance_model,linearRegressionInitialPosition,linearRegressionSpeed)}"
)
print("\n\n")

####################################
### 6. Curve Fitting
####################################
popt, pcov = curve_fit(distance_model, rocketTime, rocketPosition)

print("Curve Fitting\n")
print(
    f"Curve Fit initial d value: {popt[0]}, Uncertainty: {np.sqrt(pcov[0][0])}"
)
print(f"Curve Fit u value: {popt[1]}, Uncertainty: {np.sqrt(pcov[1][1])}")

plt.cla()
plt.plot(
    rocketTime,
    distance_model(rocketTime, *popt),
    label="curve_fit Linear Regression",
)
plt.legend()
plt.draw()

plt.savefig(f"{directory}/rocketPossitionWithMultipleRegressions.png")

