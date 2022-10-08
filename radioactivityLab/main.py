import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from math import log

# Defining the model functions.
def linear_model(values, a, b) -> any:
    return a * values + b


def exponential_model(values, a, b) -> any:
    return b * np.exp(a * values)


# Reading the data into Python.
_, total = np.loadtxt(
    "radioactivityLab/Barium137_20220929AM_20min20sec.txt",
    delimiter="\t",
    skiprows=2,
    unpack=True,
)
_, background = np.loadtxt(
    "radioactivityLab/Background20220929AM-20min20s.txt",
    delimiter="\t",
    skiprows=2,
    unpack=True,
)

# Calculating the mean background radiation and subtracting it from the data.
mean_background = np.average(background[1])
sample_count = total - mean_background
# Changing all values less than or equal to 0 to 10^-20 to avoid mathematical
#   issues with log domain for linearization
sample_count[sample_count <= 0] = 10 ** -20

# Calculating the uncertainty for each point.
def error_propagation(uncertainty_one, uncertainty_two) -> any:
    """Return the uncertainty of the sample.
    """
    return np.sqrt(uncertainty_one + uncertainty_two)


sample_count_uncertainty = error_propagation(total, background)

# Convert the count data into rates.
def count_rate(events, sample_time) -> tuple:
    """Return the count rate and its uncertainty.
    """
    return events / sample_time, np.sqrt(events) / sample_time


sample_rate, sample_rate_uncertainty = count_rate(sample_count, 20)

# Performing the linear regression on the data using the linear model.

time = np.arange(20, 1201, 20)
popt1, pcov1 = curve_fit(
    linear_model, time, np.log(sample_count), p0=(np.log(1 / 2) / 156, 0)
)
print(
    "The half-life for the linear regression method is "
    + str(-1 / popt1[0] * np.log(2))
    + "s."
)

# Performing the nonlinear regression on the data using the nonlinear model.

popt2, pcov2 = curve_fit(
    exponential_model, time, sample_count, p0=(np.log(1 / 2) / 156, 0)
)
print(
    "The half-life for the nonlinear regression method is "
    + str(-1 / popt2[0] * np.log(2))
    + "s."
)

# Plotting the raw data, models and theoretical curve.
plt.figure(figsize=(8, 6))
plt.scatter(time, sample_count, color="green", label="measured data")
plt.plot(
    time,
    exponential_model(time, popt1[0], np.exp(popt1[1])),
    label="linear model",
)
plt.plot(
    time, exponential_model(time, popt2[0], popt2[1]), label="exponential model"
)
plt.plot(time, np.exp(time * np.log(1 / 2) / 156), label="theoretical curve")
plt.errorbar(
    time, sample_count, yerr=sample_count_uncertainty, marker="o", ls="", lw=1
)
plt.xlabel("Time (s)")
plt.ylabel("Sample Count")
plt.title("Title")
plt.legend()
plt.show()

# Plotting using a logarithmic y-axis.

plt.figure(figsize=(8, 6))
plt.plot(
    time,
    exponential_model(time, popt1[0], np.exp(popt1[1])),
    label="linear model",
)
plt.plot(
    time, exponential_model(time, popt2[0], popt2[1]), label="exponential model"
)
plt.plot(time, np.exp(time * np.log(1 / 2) / 156), label="theoretical curve")
plt.errorbar(
    time, sample_count, yerr=sample_count_uncertainty, marker="o", ls="", lw=1
)
plt.xlabel("Time (s)")
plt.ylabel("Sample Count")
plt.yscale("log")
plt.title("Title")
plt.legend()
plt.show()


def characterize(y: any, func: any, u: any) -> float:
    """Return the chi^2 metric to determine how well a model function fits a
    given set of data using the measured data <y>, the prediction with the model
    <func> and the uncertainty on each measurement's dependent data <u>.
    """
    value = 0

    for i in range(np.size(y)):
        value += ((y[i] - func[i]) ** 2) / (u[i] ** 2)
        i += 1

    return value / (np.size(y) - 2)


############
# Histograms
############
# Bar model option
plt.cla()
plt.bar(
    np.arange(10, 1201, 20),
    sample_count,
    20,
    linewidth=1,
    color="r",
    edgecolor="k",
    yerr=sample_count_uncertainty,
    ecolor="k",
    capsize=3,
)

plt.xlim(0, 1200)
plt.ylim(0, 700)
plt.xlabel("Time (s)")
plt.ylabel("Sample Count")
plt.show()

# Step option
plt.cla()
plt.stairs(sample_count, np.arange(0, 1201, 20))
plt.xlim(0, 1200)
plt.ylim(0, 700)
plt.xlabel("Time (s)")
plt.ylabel("Sample Count")
plt.show()
